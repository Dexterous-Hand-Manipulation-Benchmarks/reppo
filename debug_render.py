import jax
import jax.numpy as jnp
import numpy as np

# Force JAX to initialize its backends (finding both CPU and GPU)
_ = jax.devices() 

# Compatibility patch for Orbax and current JAX versions
if not hasattr(jax.monitoring, 'record_scalar'):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None
if not hasattr(jax.monitoring, 'record_event_duration_secs'):
    jax.monitoring.record_event_duration_secs = lambda *args, **kwargs: None

import imageio
import os
import re
import wandb
import argparse
import math

from omegaconf import OmegaConf
from src.jaxrl.reppo import ReppoConfig, make_init, make_policy
import orbax.checkpoint as ocp
from flax.serialization import from_state_dict, to_state_dict
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser
from src.env_utils.jax_wrappers import LogWrapper, ClipAction, NormalizeVec, DexmachinaGymnaxWrapper, make_dexmachina_env

def get_wandb_run_id(ckpt_dir):
    """Attempt to find the WandB run ID from the Hydra output directory structure."""
    base_dir = os.path.dirname(os.path.abspath(ckpt_dir))
    wandb_dir = os.path.join(base_dir, "wandb")
    
    if os.path.exists(wandb_dir):
        for folder in os.listdir(wandb_dir):
            if folder.startswith("run-"):
                return folder.split("-")[-1]
    return None

# =====================================================================
# ARGUMENT PARSING
# =====================================================================
parser = get_common_argparser()
parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the checkpoints directory")
parser.add_argument("--wandb_project", type=str, default="dexmachina", help="WandB project name")
args, _ = parser.parse_known_args()

ckpt_dir = os.path.abspath(args.ckpt_dir)
if not os.path.exists(ckpt_dir):
    raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")

# =====================================================================
# LOAD HYDRA CONFIG FROM RUN DIRECTORY
# =====================================================================
base_dir = os.path.dirname(os.path.abspath(ckpt_dir))
hydra_config_path = os.path.join(base_dir, ".hydra", "config.yaml")

if os.path.exists(hydra_config_path):
    print(f">>> Loading original Hydra config from {hydra_config_path}")
    hydra_cfg = OmegaConf.load(hydra_config_path)
    reppo_cfg = ReppoConfig(**hydra_cfg.hyperparameters)
else:
    raise FileNotFoundError(f"Could not find Hydra config at {hydra_config_path}. Check your ckpt_dir path.")

# =====================================================================
# WANDB INITIALIZATION (RESUME RUN)
# =====================================================================
wandb_project = args.wandb_project
if hasattr(hydra_cfg, "wandb") and hasattr(hydra_cfg.wandb, "project"):
    wandb_project = hydra_cfg.wandb.project

run_id = get_wandb_run_id(ckpt_dir)
if run_id is None:
    raise ValueError(f">>> Error: Could not auto-detect WandB run ID from {ckpt_dir}.")

print(f">>> Resuming WandB run ID: {run_id}")
wandb.init(project=args.wandb_project, id=run_id, resume="must")

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================
# Store original trained_num_envs before overriding for eval
trained_num_envs = hydra_cfg.hyperparameters.get("num_envs", 512)
# Override specific parameters for evaluation rendering
hydra_cfg.hyperparameters.num_envs = 1

# Ensure evaluation records video using the front camera with a visualizer
if "env" in hydra_cfg:
    hydra_cfg.env.num_envs = 1
    hydra_cfg.env.record_video = True
    hydra_cfg.env.render_camera = "front"
    hydra_cfg.env.scene_kwargs = hydra_cfg.env.get("scene_kwargs", {})
    hydra_cfg.env.scene_kwargs["use_visualizer"] = True
    hydra_cfg.env.camera_kwargs = {
        "front": {'res': (256, 256), 'pos': (0.0, -1.6, 2.2), 'lookat': (0.0, -0.1, 1.2), 'fov': 30}
    }

print(">>> Initializing evaluation environment via make_dexmachina_env...")
base_env = make_dexmachina_env(hydra_cfg.env, hydra_cfg.hyperparameters)
base_env = DexmachinaGymnaxWrapper(base_env)

# CRITICAL: Sync the evaluation loop safely!
hp_dict = OmegaConf.to_container(hydra_cfg.hyperparameters, resolve=True)
hp_dict["max_episode_steps"] = getattr(base_env, "max_episode_steps", hp_dict.get("max_episode_steps", 1000))
reppo_cfg = ReppoConfig(**hp_dict)

# Apply the exact same wrappers used during training
env = LogWrapper(base_env, hydra_cfg.hyperparameters.num_envs)
env = ClipAction(env)
if reppo_cfg.normalize_env:
    env = NormalizeVec(env)

rng = jax.random.PRNGKey(0)

# =====================================================================
# BUILD THE ARCHITECTURE SKELETON ONCE
# =====================================================================
print(">>> Initializing model architecture skeleton...")
_, _, sample_state = env.reset(key=rng)

state_cls = type(sample_state)
if not hasattr(state_cls, "unwrapped"):
    setattr(state_cls, "unwrapped", lambda self: self)
if not hasattr(state_cls, "set_env_state"):
    setattr(state_cls, "set_env_state", lambda self, new_state: new_state)

init_fn = make_init(reppo_cfg, env, env_params=None)
rng, init_key = jax.random.split(rng)

vmap_init = jax.vmap(init_fn)
base_train_state = vmap_init(jax.random.split(init_key, 1))

def tile_env_dims(path, x):
    path_str = str(path)
    if "last_" in path_str and isinstance(x, (jnp.ndarray, np.ndarray)):
        if x.ndim > 1 and x.shape[1] == 1:
            return jnp.repeat(x, trained_num_envs, axis=1)
    return x

target_state = jax.tree_util.tree_map_with_path(tile_env_dims, base_train_state)

# =====================================================================
# GET AND SORT ORBAX CHECKPOINTS
# =====================================================================
checkpoint_paths = [
    os.path.join(ckpt_dir, d) 
    for d in os.listdir(ckpt_dir) 
    if os.path.isdir(os.path.join(ckpt_dir, d)) and "checkpoint_seed_" in d
]

def extract_step(filename):
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1)) if match else -1

checkpoint_paths.sort(key=extract_step)

print(f">>> Found {len(checkpoint_paths)} Orbax checkpoints. Beginning evaluation loop...")

# =====================================================================
# EVALUATION LOOP
# =====================================================================
wandb.define_metric("eval_step")
wandb.define_metric("render_video", step_metric="eval_step")

latest_video_path = None
checkpointer = ocp.StandardCheckpointer()

for ckpt_path in checkpoint_paths:
    step_num = extract_step(ckpt_path)
    print(f"\n>>> Loading Orbax checkpoint for step {step_num}...")

    # 1. Perfectly matched Guided Restore (Positional argument!)
    restored_state = checkpointer.restore(ckpt_path, target_state)

    # 2. Strip the num_seeds=1 dimension
    final_state = jax.tree.map(
        lambda x: x[0] if getattr(x, 'ndim', 0) > 0 and x.shape[0] == 1 else x, 
        restored_state
    )

    # 3. Build the policy
    policy = make_policy(final_state)
    
    print(f">>> resetting environment for step {step_num}...", flush=True)
    rng, reset_key, eval_key = jax.random.split(rng, 3)
    
    # FIX 2: Pass the loaded normalization state into env.reset!
    norm_state = final_state.last_env_state if reppo_cfg.normalize_env else None
    obs, critic_obs, env_state = env.reset(reset_key, norm_state)
    
    frames = []
    for _ in range(reppo_cfg.max_episode_steps):
        eval_key, act_key, step_key = jax.random.split(eval_key, 3)
        
        # FIX 3: The NormalizeVec wrapper handles everything natively.
        action_jax, _ = policy(act_key, obs)
        
        obs, critic_obs, env_state, reward, done, info = env.step(
            step_key, env_state, action_jax
        )
        
        frame = base_env.render()
            
        if frame is not None:
            if hasattr(frame, "__array__"):
                frame = np.array(frame)
            frames.append(frame)
            
    if frames:
        video_array = np.array(frames)
        if video_array.ndim == 5:
            video_array = video_array.squeeze(1)
            
        vid_filename = f"eval_video_step_{step_num}.mp4"
        imageio.mimsave(vid_filename, video_array, fps=30)
        
        print(f">>> uploading {vid_filename} to wandb at step {step_num}...")
        wandb.log({
            "eval_step": step_num,
            "render_video": wandb.Video(vid_filename, fps=30, format="mp4")
        })

        latest_video_path = vid_filename
    else:
        print(f">>> error: no frames captured for step {step_num}.")

# =====================================================================
# AUTOMATIC SUMMARY LOGGING (OVERVIEW PAGE RENDER)
# =====================================================================
if latest_video_path and os.path.exists(latest_video_path):
    print(">>> setting final evaluation render on wandb summary overview...")
    wandb.run.summary["render_video"] = wandb.Video(latest_video_path, fps=30, format="mp4")

wandb.finish()
print("\n>>> All checkpoints evaluated and uploaded successfully!")