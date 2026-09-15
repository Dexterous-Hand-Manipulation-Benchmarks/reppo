"""
Render rollouts from saved reppo checkpoints in the DexMachina environment.

Fixes over the previous version:
  - vid_filename NameError
  - wandb.Video channel ordering (expects T,C,H,W for ndarrays)
  - curriculum / actuate_object no longer disabled (that removed the virtual
    object controller entirely, changing the environment dynamics)
  - VOC gains are restored per checkpoint instead of left at init or zeroed
  - early-reset disabled for eval so videos are not truncated mid-motion
  - malformed checkpoint dirs are filtered instead of sorting to the front
  - consistent env.reset signature
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

# Force JAX to initialize its backends (finding both CPU and GPU)
_ = jax.devices()

# Compatibility patch for Orbax and current JAX versions
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None
if not hasattr(jax.monitoring, "record_event_duration_secs"):
    jax.monitoring.record_event_duration_secs = lambda *args, **kwargs: None

import os
import re
import math
import argparse

import wandb
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from src.jaxrl.reppo import ReppoConfig, make_init, make_policy
from src.env_utils.jax_wrappers import (
    LogWrapper,
    ClipAction,
    NormalizeVec,
    DexmachinaGymnaxWrapper,
    make_dexmachina_env,
)
from dexmachina.envs.constructors import get_common_argparser


def get_wandb_run_id(ckpt_dir):
    """Find the WandB run ID from the Hydra output directory structure."""
    base_dir = os.path.dirname(os.path.abspath(ckpt_dir))
    wandb_dir = os.path.join(base_dir, "wandb")
    if os.path.exists(wandb_dir):
        for folder in sorted(os.listdir(wandb_dir)):
            if folder.startswith("run-"):
                return folder.split("-")[-1]
    return None


def extract_step(path):
    match = re.search(r"step_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


# =====================================================================
# ARGUMENT PARSING
# =====================================================================
parser = get_common_argparser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--wandb_project", type=str, default="dexmachina")
parser.add_argument("--wandb_entity", type=str, default="schrapsi-kit")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--voc_mode",
    type=str,
    default="replay",
    choices=["replay", "zero", "manual", "init"],
    help=(
        "How to set the virtual object controller gains for each checkpoint. "
        "'replay' advances the curriculum to the checkpoint's epoch (correct for "
        "fixed schedules, approximate for adaptive ones). 'zero' evaluates fully "
        "unassisted. 'manual' uses --eval_kp/--eval_kv/--eval_fr. 'init' leaves "
        "the gains at their initial values."
    ),
)
parser.add_argument(
    "--dump_eval_npy",
    action="store_true",
    help=(
        "Also write eval_ep0.npy (obj_state + demo_state) next to each checkpoint "
        "so dexmachina.eval.compute_add can produce the canonical ADD/AUC numbers. "
        "Use with --voc_mode zero for unassisted (paper-comparable) metrics."
    ),
)
parser.add_argument(
    "--max_videos",
    type=int,
    default=0,
    help=(
        "Render at most N checkpoints, evenly spaced, always including the FIRST "
        "and LAST. 0 = render all. Each video costs ~40s (200 frames at 500x500 "
        "plus encode and upload), so 33 checkpoints is ~22 minutes -- more than a "
        "30-minute queue slot allows on top of training."
    ),
)
parser.add_argument("--eval_kp", type=float, default=0.0)
parser.add_argument("--eval_kv", type=float, default=0.0)
parser.add_argument("--eval_fr", type=float, default=50.0)
args, _ = parser.parse_known_args()

ckpt_dir = os.path.abspath(args.ckpt_dir)
if not os.path.exists(ckpt_dir):
    raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")

# =====================================================================
# LOAD HYDRA CONFIG FROM RUN DIRECTORY
# =====================================================================
base_dir = os.path.dirname(ckpt_dir)
hydra_config_path = os.path.join(base_dir, ".hydra", "config.yaml")
if not os.path.exists(hydra_config_path):
    raise FileNotFoundError(
        f"Could not find Hydra config at {hydra_config_path}. Check your ckpt_dir path."
    )

print(f">>> Loading original Hydra config from {hydra_config_path}")
hydra_cfg = OmegaConf.load(hydra_config_path)
OmegaConf.set_struct(hydra_cfg, False)

# =====================================================================
# WANDB INITIALIZATION (RESUME RUN)
# =====================================================================
run_id = get_wandb_run_id(ckpt_dir)
if run_id is None:
    raise ValueError(f">>> Could not auto-detect WandB run ID from {ckpt_dir}.")

print(f">>> Resuming WandB run ID: {run_id}, Entity: {args.wandb_entity}")
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    id=run_id,
    resume="must",
)

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================
# Keep the training-time env size around: it is needed to convert a global
# step count back into a curriculum epoch, and to tile the restored state.
trained_num_envs = int(hydra_cfg.hyperparameters.get("num_envs", 512))
trained_num_steps = int(hydra_cfg.hyperparameters.get("num_steps", 32))

hydra_cfg.hyperparameters.num_envs = 1

# IMPORTANT: do NOT disable `curriculum` / `actuate_object` here.
# In make_dexmachina_env those flags are what give the object its virtual
# controller joints (actuated=True) and per-env gain support
# (batch_dofs_info=True). Turning them off produces a structurally different
# environment from the one the policy trained in. The decay schedule does not
# run during eval anyway, because nothing calls set_curriculum in the rollout
# loop -- we advance it explicitly below, once per checkpoint.
hydra_cfg.env.curriculum = True
hydra_cfg.env.actuate_object = True

# Let every episode run to its natural end so videos are not cut short.
# NOTE: BaseEnv's `is_eval` flag also drops the last env from _step_env_idxs,
# which would leave zero stepped envs at num_envs=1. So we disable early reset
# via the thresholds instead of setting is_eval.
hydra_cfg.env.is_eval = False
hydra_cfg.env.early_reset_threshold = 0.0
hydra_cfg.env.aux_reset_thres = [0.0, 0.0, 0.0]

hydra_cfg.env.num_envs = 1
hydra_cfg.env.record_video = True
hydra_cfg.env.render_camera = "front"
hydra_cfg.env.scene_kwargs = hydra_cfg.env.get("scene_kwargs", {})
hydra_cfg.env.scene_kwargs["use_visualizer"] = True
# BaseEnv._add_camera forces (500, 500) when num_envs < 3, so this res is a hint.
hydra_cfg.env.camera_kwargs = {
    "front": {
        "res": (256, 256),
        "pos": (0.0, -1.6, 2.2),
        "lookat": (0.0, -0.1, 1.2),
        "fov": 30,
    }
}

print(">>> Initializing evaluation environment via make_dexmachina_env...")
base_env = make_dexmachina_env(hydra_cfg.env, hydra_cfg.hyperparameters)
base_env = DexmachinaGymnaxWrapper(
    env=base_env,
    num_steps=hydra_cfg.hyperparameters.num_steps,
    episode_length=hydra_cfg.env.get("max_episode_steps", None),
    clip_obs=hydra_cfg.env.get("clip_obs", math.inf),
    clip_actions=hydra_cfg.env.get("clip_actions", math.inf),
    asymmetric_obs=hydra_cfg.env.get("asymmetric_obs", True),
    # gains are set explicitly per checkpoint by set_voc_gains() below; the
    # wrapper must not also step the curriculum schedule while we render
    advance_curriculum=False,
)

inner_env = getattr(base_env, "env", base_env)
inner_env = getattr(inner_env, "unwrapped", inner_env)
has_curriculum = getattr(inner_env, "use_curriculum", False) and (
    getattr(inner_env, "curriculum", None) is not None
)
if not has_curriculum:
    print(
        ">>> WARNING: env reports no active curriculum. The object may not be "
        "actuated, which means the virtual object controller is absent and the "
        "dynamics differ from training."
    )

hp_dict = OmegaConf.to_container(hydra_cfg.hyperparameters, resolve=True)
hp_dict["max_episode_steps"] = getattr(
    base_env, "max_episode_steps", hp_dict.get("max_episode_steps", 1000)
)
reppo_cfg = ReppoConfig(**hp_dict)

env = LogWrapper(base_env, hydra_cfg.hyperparameters.num_envs)
env = ClipAction(env)
if reppo_cfg.normalize_env:
    env = NormalizeVec(env)

rng = jax.random.PRNGKey(0)


# =====================================================================
# VOC GAIN HANDLING
# =====================================================================
def set_voc_gains(step_num):
    """Put the virtual object controller into the state the checkpoint saw.

    A checkpoint trained at kp=40 will look broken if rendered with no
    assistance at all, so matching the gains matters for mid-training videos.
    'replay' is exact for the fixed schedule and approximate for the adaptive
    ones (their decay depended on reward statistics we no longer have).
    """
    if not has_curriculum:
        return None

    curriculum = inner_env.curriculum

    if args.voc_mode == "init":
        pass

    elif args.voc_mode == "zero":
        for k in curriculum.decay_terms:
            curriculum.curr_gains[k] = 0.0
            curriculum.curr_gains_lower[k] = 0.0
        curriculum.reset_object_gains()

    elif args.voc_mode == "manual":
        manual = {"kp": args.eval_kp, "kv": args.eval_kv, "fr": args.eval_fr}
        for k in curriculum.decay_terms:
            curriculum.curr_gains[k] = manual[k]
            curriculum.curr_gains_lower[k] = manual[k]
        curriculum.reset_object_gains()

    elif args.voc_mode == "replay":
        # Reset to the initial gains, then step the schedule forward to the
        # epoch this checkpoint corresponds to.
        curriculum.curr_gains = curriculum.init_gains.copy()
        curriculum.curr_gains_lower = curriculum.init_gains.copy()
        curriculum.num_epoch_since_last_decay = 0
        curriculum.num_epoch_since_zero = 0
        curriculum.gain_history = []
        for dq in curriculum.rew_deques.values():
            dq.clear()
        curriculum.ep_lens.clear()

        # Curriculum epochs are a normalised progress counter now (see
        # DexmachinaGymnaxWrapper.__init__), NOT the reppo iteration count.
        # step_num is a training env-step count, so scale it the same way.
        total_env_steps = int(hydra_cfg.hyperparameters.total_time_steps)
        max_epochs = int(hydra_cfg.env.get("curriculum_max_epochs", 10000))
        target_epoch = (step_num * max_epochs) // max(1, total_env_steps)
        for epoch in range(1, target_epoch + 1):
            curriculum.set_curriculum(epoch)
        curriculum.reset_object_gains()

    return curriculum.get_current_gains()


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
base_train_state = jax.vmap(init_fn)(jax.random.split(init_key, 1))


def tile_env_dims(path, x):
    """Restored checkpoints were saved at trained_num_envs; eval runs at 1."""
    if "last_" in str(path) and isinstance(x, (jnp.ndarray, np.ndarray)):
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
# Drop anything without a parseable step so it does not sort to the front.
skipped = [p for p in checkpoint_paths if extract_step(p) < 0]
for p in skipped:
    print(f">>> Skipping unparseable checkpoint dir: {os.path.basename(p)}")
checkpoint_paths = sorted(
    (p for p in checkpoint_paths if extract_step(p) >= 0), key=extract_step
)

if not checkpoint_paths:
    raise ValueError(f"No usable checkpoints found in {ckpt_dir}")

if args.max_videos and 0 < args.max_videos < len(checkpoint_paths):
    n_all = len(checkpoint_paths)
    idxs = sorted({round(i * (n_all - 1) / (args.max_videos - 1))
                   for i in range(args.max_videos)}) if args.max_videos > 1 else [n_all - 1]
    checkpoint_paths = [checkpoint_paths[i] for i in idxs]
    print(f">>> --max_videos {args.max_videos}: rendering {len(checkpoint_paths)} "
          f"of {n_all} checkpoints (steps: "
          f"{[extract_step(p) for p in checkpoint_paths]})")

print(f">>> Found {len(checkpoint_paths)} Orbax checkpoints. Beginning evaluation loop...")

# =====================================================================
# EVALUATION LOOP
# =====================================================================
# Checkpoint names are training env-step counts, so log against the same
# x-axis training used (global/env_step). Logging against a separate
# "eval_step" made wandb fall back to its default _step axis for the gain
# metrics, which is why the charts switched away from env steps mid-run.
wandb.define_metric("global/env_step")
# Re-assert the FULL step-metric binding set from the training run. Calling
# define_metric("global/env_step") on a resumed run re-registers the step metric
# and can clear the glob bindings established by the original process, which left
# train/* (and others) plotted against wandb's auto-incrementing _step axis
# instead of env steps. Re-declaring every pattern here is idempotent and keeps
# the axes correct after debug_render attaches.
#
# NOTE: render-time gains go under render/*, NOT env/* or reppo/*. Training now
# logs the live curriculum gains as env/kp etc.; since debug_render resumes the
# same run, reusing those names would overwrite the real training curve with the
# renderer's forced gains (all 0.0 under --voc_mode zero).
for _pattern in ("global/*", "train/*", "eval/*", "env/*", "perf/*",
                 "reppo/*", "render/*"):
    wandb.define_metric(_pattern, step_metric="global/env_step")
wandb.define_metric("render_video", step_metric="global/env_step")

checkpointer = ocp.StandardCheckpointer()

for ckpt_path in checkpoint_paths:
    step_num = extract_step(ckpt_path)
    print(f"\n>>> Loading Orbax checkpoint for step {step_num}...")

    restored_state = checkpointer.restore(ckpt_path, target_state)

    # Strip the num_seeds=1 leading dimension
    final_state = jax.tree.map(
        lambda x: x[0] if getattr(x, "ndim", 0) > 0 and x.shape[0] == 1 else x,
        restored_state,
    )

    gains = set_voc_gains(step_num)
    if gains is not None:
        print(f">>> VOC gains ({args.voc_mode}): {gains}")

    policy = make_policy(final_state)

    print(f">>> Resetting environment for step {step_num}...", flush=True)
    rng, reset_key, eval_key = jax.random.split(rng, 3)

    norm_state = final_state.last_env_state if reppo_cfg.normalize_env else None
    obs, critic_obs, env_state = env.reset(reset_key, norm_state)

    frames = []
    obj_states, demo_states = [], []
    obj_entity = getattr(inner_env, "object", None)
    for _ in range(reppo_cfg.max_episode_steps):
        eval_key, act_key, step_key = jax.random.split(eval_key, 3)
        action_jax, _ = policy(act_key, obs)
        # read the demo index BEFORE stepping, matching eval_rl_games.py
        if args.dump_eval_npy and obj_entity is not None:
            t = int(inner_env.episode_length_buf[0].item())
            t = min(t, obj_entity.demo_states.shape[0] - 1)
            demo_states.append(obj_entity.demo_states[t].cpu().numpy())
        obs, critic_obs, env_state, reward, done, info = env.step(
            step_key, env_state, action_jax
        )
        if args.dump_eval_npy and obj_entity is not None:
            state = torch.cat(
                [obj_entity.root_pos, obj_entity.root_quat, obj_entity.dof_pos], dim=-1
            )
            obj_states.append(state.cpu().numpy())

        frame = base_env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    if args.dump_eval_npy and obj_states:
        # compute_add.py wants demo_state (T, 8) and obj_state (T, n_envs, 8)
        npy_path = os.path.join(ckpt_path, "eval_ep0.npy")
        np.save(
            npy_path,
            {
                "obj_state": np.stack(obj_states),
                "demo_state": np.stack(demo_states),
                "voc_mode": args.voc_mode,
            },
        )
        print(f">>> Wrote {npy_path} (voc_mode={args.voc_mode})")

    if not frames:
        print(f">>> ERROR: no frames captured for step {step_num}.")
        continue

    video_array = np.stack(frames)          # (T, 1, H, W, C) from the wrapper
    if video_array.ndim == 5:
        video_array = video_array.squeeze(1)  # -> (T, H, W, C)
    if video_array.ndim != 4:
        print(f">>> ERROR: unexpected frame shape {video_array.shape}, skipping.")
        continue
    if video_array.shape[-1] == 4:
        video_array = video_array[..., :3]
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)
    # wandb.Video expects (time, channel, height, width) for ndarray input
    video_array = np.transpose(video_array, (0, 3, 1, 2))

    print(f">>> Uploading video ({video_array.shape}) at step {step_num}...")
    log_data = {
        "global/env_step": step_num,
        "render_video": wandb.Video(video_array, fps=args.fps, format="mp4"),
    }
    if gains is not None:
        log_data.update({f"render/{k}": v for k, v in gains.items()})
    wandb.log(log_data)

wandb.finish()
print("\n>>> All checkpoints evaluated and uploaded successfully!")