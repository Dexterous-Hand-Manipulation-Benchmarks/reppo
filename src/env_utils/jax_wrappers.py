from functools import partial
from typing import Any, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import wrap_for_brax_training, Wrapper

# Imports for the Dexmachina Jax Wrapper
import sys
import math
import random
import torch
import numpy as np
import jax.numpy as jnp
from typing import Dict
from gymnax.environments.environment import EnvState, EnvParams
from torch.utils.dlpack import to_dlpack as pt_to_dlpack, from_dlpack as pt_from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack, from_dlpack as jax_from_dlpack

# Import DexMachina specific modules
import os
import genesis as gs
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser
from omegaconf import OmegaConf

# Fixed set of per-step diagnostic keys pulled from DexMachina's `extras["log"]`
# dict and threaded through the JAX wrapper stack as ordinary `info` entries
# (e.g. reach wandb as "eval/pos_dist", "eval/con_rew", ...). Kept as an
# explicit, static list (rather than forwarding `extras["log"]` verbatim)
# because `io_callback` and the LogWrapper/NormalizeVec carry state need a
# fixed pytree structure known at JAX-trace time.
DEX_EXTRA_LOG_KEYS = (
    "pos_dist",
    "rot_dist",
    "arti_dist",
    "task_rew",
    "con_rew",
    "imi_rew",
    "bc_rew",
    "keypoint_dist",
    "well_track",
    "success_2cm",
    "success_5cm",
    # ADD (mean per-vertex distance to the demo pose, metres), per object part.
    # Same definition as dexmachina/eval/compute_add.py. Requires
    # env.log_add_metric=true.
    "add",
    "add_top",
    "add_bottom",
    # Contact diagnostics (see RewardModule.compute_matched_contact_reward):
    # con_rew is bounded above by con_rew_ceiling, which depends on how many
    # (part, link) pairs the demo actually wants in contact. Compare con_rew
    # against the ceiling, not against contact_rew_weight.
    "con_active_frac",
    "con_dist_active",
    "con_rew_ceiling",
    # Curriculum assistive gains, from Curriculum.get_current_gains(). These are
    # scalars broadcast to every env by _host_step. Logged from the TRAINING env
    # only (see DEX_GAIN_KEYS) -- the eval env has them permanently zeroed, so an
    # eval/kp panel would be a flat zero line.
    "kp",
    "kv",
    "fr",
    "kp_lower",
    "kv_lower",
    "fr_lower",
)

# Subset of the above that describes the curriculum's assistive gains rather than
# the policy's behaviour. Logged as env/* from the training env to match the panel
# names FastTD3 produces (its logger maps raw "kp" -> "env/kp"), and skipped in
# evaluation_fn.
DEX_GAIN_KEYS = ("kp", "kv", "fr", "kp_lower", "kv_lower", "fr_lower")

# Entries in extras["log"] that CANNOT be forwarded: `episode_rew` is
# `cumulative_task_rew[reset_env_ids]`, so it has shape (n_reset,) and is only
# present on steps where something reset -- both the varying length and the
# intermittency break the io_callback's fixed output structure.
DEX_LOG_KEY_BLOCKLIST = frozenset({"episode_rew"})


def discover_dex_log_keys(base_env, num_envs):
    """Every per-env / scalar entry the env publishes in extras["log"].

    RewardModule generates several keys by f-string (conrew_{side}_{part},
    matched_condist_{side}_{part}, {side}_control_force, grad/{term}), so a
    hardcoded list goes stale whenever the reward config changes. Read them off a
    real post-warmup step instead. DEX_EXTRA_LOG_KEYS is unioned in so the
    known-good set is always present even if discovery comes up short.
    """
    found = set()
    log = getattr(base_env, "extras", {}).get("log", {}) or {}
    for key, val in log.items():
        if key in DEX_LOG_KEY_BLOCKLIST:
            continue
        if isinstance(val, torch.Tensor):
            if val.ndim == 0 or val.shape[0] == num_envs:
                found.add(key)
        elif isinstance(val, (int, float, bool)):
            found.add(key)
    keys = tuple(sorted(found | set(DEX_EXTRA_LOG_KEYS)))
    skipped = sorted(set(log) - found - DEX_LOG_KEY_BLOCKLIST)
    print(f"[ENV] forwarding {len(keys)} extras['log'] keys to wandb")
    if skipped:
        print(f"[ENV] skipped (shape not per-env): {skipped}")
    return keys

# Subset of the above that ADD-AUC is computed from during eval.
DEX_ADD_KEY = "add"

# Thresholds (metres) for the ADD accuracy-vs-threshold curve whose area is the
# reported AUC. Kept identical to dexmachina/eval/compute_add.py's
# np.arange(0.01, 0.1, 0.01) so online and offline numbers line up.
DEX_ADD_THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        if isinstance(env_or_name, str):
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
        else:
            self.env = env_or_name
        self.reward_scale = reward_scale
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True
        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        print(self.dict_obs_key)
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        if self.dict_obs:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return obs, critic_obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return (
            obs,
            critic_obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            {},
        )


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[chex.Array, environment.EnvState]:
        obs, critic_obs, env_state = self.env.reset(key)
        # some wrapped envs (e.g. DexmachinaGymnaxWrapper) add extra fixed-shape
        # scalar-per-env keys into `info` on every step() call (see
        # `extra_log_keys` / DEX_EXTRA_LOG_KEYS in jax_wrappers.py). Seed matching
        # zero placeholders here so `state.info`'s pytree structure stays
        # identical between reset() and every subsequent step() - required
        # since `state` is carried through jax.lax.scan.
        extra_keys = getattr(self.env, "extra_log_keys", ())
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self.num_envs,)),
            episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((self.num_envs,)),
            returned_episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            timestep=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            truncated=jnp.ones((self.num_envs,), dtype=jnp.float32),
            info={
                "returned_episode": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "returned_episode_returns": jnp.zeros((self.num_envs,)),
                "timestep": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                "returned_episode_lengths": jnp.zeros(
                    (self.num_envs,), dtype=jnp.int32
                ),
                **{k: jnp.zeros((self.num_envs,), dtype=jnp.float32) for k in extra_keys},
            },
        )
        return obs, critic_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        info["returned_episode_returns"] = (
            state.returned_episode_returns * (1 - done) + new_episode_return * done
        )
        info["returned_episode_lengths"] = (
            state.returned_episode_lengths * (1 - done) + new_episode_length * done
        )
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            truncated=env_state.info["truncation"],
            info=info,
        )
        return obs, critic_obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(
        self,
        env_name,
        backend="generalized",
        episode_length=1000,
        reward_scaling=1.0,
        terminate=True,
    ):
        env = envs.get_environment(
            env_name=env_name, backend=backend, terminate_when_unhealthy=terminate
        )
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self.env = env
        self.action_size = self.env.action_size
        self.observation_size = (self.env.observation_size,)
        self.default_params = ()
        self.reward_scaling = reward_scaling

    def reset(self, key):
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action):
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state.obs,
            next_state,
            next_state.reward * self.reward_scaling,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        ), spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        )

    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )


class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _init_state(self, key):
        obs, critic_obs, env_state = self.env.reset(key)
        return NormalizeVecObsEnvState(
            mean=jnp.mean(obs, axis=0),
            var=jnp.var(obs, axis=0),
            critic_mean=jnp.mean(critic_obs, axis=0),
            critic_var=jnp.var(critic_obs, axis=0),
            count=obs.shape[0],
            env_state=env_state,
        )

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        if params is not None:
            mean = params.mean
            var = params.var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            count = params.count
        else:
            mean = jnp.mean(obs, axis=0)
            var = jnp.var(obs, axis=0)
            critic_mean = jnp.mean(critic_obs, axis=0)
            critic_var = jnp.var(critic_obs, axis=0)
            count = obs.shape[0]
        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
        )

    def step(self, key, state, action):
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )

        new_mean, new_var = self._compute_stats(state.mean, state.var, state.count, obs)
        new_critic_mean, new_critic_var = self._compute_stats(
            state.critic_mean, state.critic_var, state.count, critic_obs
        )

        new_count = state.count + obs.shape[0]

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
            reward,
            done,
            info,
        )


# --- Zero-Copy DLPack Memory Converters for Dexmachina Jax Wrapper ---
def torch_to_jax(pt_tensor: torch.Tensor) -> jax.Array:
    """Converts a PyTorch GPU Tensor to a JAX GPU Array without copying memory."""
    if not pt_tensor.is_contiguous():
        pt_tensor = pt_tensor.contiguous()
    return jax_from_dlpack(pt_to_dlpack(pt_tensor))


def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    """Converts a JAX GPU Array to a PyTorch GPU Tensor without copying memory."""
    return pt_from_dlpack(jax_to_dlpack(jax_array))

@struct.dataclass
class DexmachinaState:
    time: jnp.ndarray
    truncated: jnp.ndarray  # Added for NormalizeVec wrapper
    info: Dict[str, jnp.ndarray]


class DexmachinaGymnaxWrapper(Environment):
    """Adapts DexMachina Genesis PyTorch environment for JAX/Gymnax pipelines."""

    def __init__(
        self,
        env,
        num_steps: int,
        episode_length: int | None = None,
        clip_obs: float = math.inf,
        clip_actions: float = math.inf,
        asymmetric_obs: bool = True,
        curriculum_max_epochs: int = 10000,
        total_env_steps: int = 0,
        advance_curriculum: bool = True,
    ):
        super().__init__()
        self.env = env
        self.num_envs = env.num_envs
        self.obs_dim = env.num_obs
        self.action_dim = env.num_actions
        self.state_dim = getattr(env, "num_states", self.obs_dim)

        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self.asymmetric_obs = asymmetric_obs

        self.env_type = "dexmachina"
        # DexMachina already derives the horizon from the clip string:
        # constructors.py does `ep_len = end - start` (box-30-230 -> 200,
        # mixer-0-150 -> 150) and stores it as env_cfg['episode_length'].
        # Prefer that over any configured value so switching clips does not
        # silently run a wrong-length eval rollout.
        env_ep_len = int(getattr(env, "max_episode_length", 0) or 0)
        if episode_length is None:
            if env_ep_len <= 0:
                raise ValueError(
                    "max_episode_steps not given and the env exposes no "
                    "max_episode_length to derive it from."
                )
            self.max_episode_steps = env_ep_len
        else:
            self.max_episode_steps = int(episode_length)
            if env_ep_len > 0 and env_ep_len != self.max_episode_steps:
                print(
                    f"[ENV] WARNING: configured max_episode_steps="
                    f"{self.max_episode_steps} != the clip's own length "
                    f"{env_ep_len}; using the clip's. Set max_episode_steps: null "
                    f"to derive it automatically."
                )
                self.max_episode_steps = env_ep_len
        print(f"[ENV] max_episode_steps = {self.max_episode_steps} (from clip)")
        # exposed so LogWrapper.reset() can seed matching zero placeholders
        # for these keys in its carried `state.info` (see DEX_EXTRA_LOG_KEYS)
        self.extra_log_keys = DEX_EXTRA_LOG_KEYS

        # ------------------------------------------------------------------
        # Curriculum epoch counter.
        #
        # This used to be `epoch = total_host_steps // num_steps`, i.e. one epoch
        # per rollout. With num_steps=32 that gave only 762 epochs across a 50M
        # step run, and DexMachina's curriculum is denominated in epochs:
        # wait_epochs=100, a hardcoded 40-epoch minimum between decays, and
        # zero_epoch. 762 epochs allows at most ~19 decays; 152 (a 10M run)
        # allows 3, so the curriculum could never actually run.
        #
        # FastTD3 instead maps training progress linearly onto a fixed epoch
        # budget (fast_td3/train.py: `epoch = global_step * max_epochs //
        # total_timesteps`), so the counter always ticks max_epochs times
        # regardless of rollout length. Mirror that: the epoch-denominated
        # curriculum parameters then mean the same thing they do for them.
        # ------------------------------------------------------------------
        self._total_host_steps = 0
        self._curriculum_max_epochs = int(curriculum_max_epochs)
        # one host step advances every env by one step
        self._host_steps_total = max(1, int(total_env_steps) // max(1, self.num_envs))
        self._current_epoch = -1
        # Offline tools (debug_render.py) set the gains explicitly per checkpoint
        # and must NOT have the schedule advanced underneath them. They also run
        # at num_envs=1 with no total_env_steps, which would make the progress
        # ratio explode. Refuse to drive the curriculum unless we were actually
        # told how long the run is.
        self._advance_curriculum = bool(advance_curriculum) and int(total_env_steps) > 0
        if bool(advance_curriculum) and int(total_env_steps) <= 0:
            print("[CURR] total_env_steps not provided; curriculum advancement disabled.")


        # Force Genesis to compile its physics kernels on the main thread 
        # before JAX takes over and locks the GPU context.
        print("Warming up Genesis and PyTorch kernels...")
        self.env.reset()
        dummy_actions = torch.zeros((self.num_envs, self.action_dim), device='cuda:0')
        self.env.step(dummy_actions)
        torch.cuda.synchronize()
        print("Genesis warmup complete!")

        # extras["log"] is populated now, so the forwarded key set can be read
        # off the real env rather than hardcoded. Must happen before JAX traces
        # anything, since it fixes the io_callback's output width.
        self.extra_log_keys = discover_dex_log_keys(
            getattr(self.env, "unwrapped", self.env), self.num_envs
        )

    # ---------------------------------------------------------
    # HOST FUNCTIONS (Run standard PyTorch + DLPack outside JIT)
    # ---------------------------------------------------------
    def _host_reset(self):
        obs_dict, _ = self.env.reset()
        obs, critic_obs = self._process_obs(obs_dict)
        if obs is None:
            raise ValueError("DexMachina reset returned None for policy observations!")
        if critic_obs is None:
            critic_obs = obs
            
        torch.cuda.synchronize()
        return torch_to_jax(obs), torch_to_jax(critic_obs)

    def _host_step(self, actions_jax):
        actions_pt = jax_to_torch(actions_jax)
        if not math.isinf(self._clip_actions):
            actions_pt = torch.clamp(actions_pt, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions_pt)

        # -------------------------------------------------------------
        # Advance the curriculum epoch. `_advance_curriculum` is a per-instance
        # property: True on the training wrapper, False on the evaluation wrapper
        # and on debug_render's. Evaluation used to share this env and so needed a
        # per-call `is_eval` flag threaded through every wrapper's step() and into
        # the io_callback; with a dedicated eval env that distinction is fixed at
        # construction time.
        # -------------------------------------------------------------
        if self._advance_curriculum:
            self._total_host_steps += 1
        epoch = (
            self._total_host_steps * self._curriculum_max_epochs
        ) // self._host_steps_total
        if self._advance_curriculum and epoch > self._current_epoch:
            self._current_epoch = epoch
            self.env.set_curriculum(epoch)          # ignore the bool
            c = self.env.curriculum
            if epoch % 500 == 0:
                _, reason = c.determine_decay(epoch)   # read-only, safe to call
                print(f"[Curriculum] epoch={epoch} gains={c.get_current_gains()} "
                    f"deques={ {k: len(v) for k, v in c.rew_deques.items()} } "
                    f"ep_lens={len(c.ep_lens)} since_decay={c.num_epoch_since_last_decay} "
                    f"| {reason}")

        # Periodic status check (every 1000 host steps)
        base = getattr(self.env, "unwrapped", self.env)
        if self._advance_curriculum and self._total_host_steps % 100 == 1:
            n_nan = base.nan_envs.sum().item()
            print(f"[DBG] nan_envs={n_nan}/{base.num_envs} "
                f"ep_len_mean={base.episode_length_buf.float().mean().item():.1f} "
                f"obj_z_min={base.object.root_pos[:,2].min().item():.3f}")
            if n_nan > 0:
                for side in ['left', 'right']:
                    r = base.robots[side]
                    print(f"   {side}: nan={r.get_nan_envs().sum().item()} "
                        f"|qpos|max={r.entity.get_dofs_position().abs().max().item():.2f}")
                print(f"   obj: nan={base.object.get_nan_envs().sum().item()}")
        # -------------------------------------------------------------

        obs, critic_obs = self._process_obs(obs_dict)

        if obs is None:
            raise ValueError("DexMachina step returned None!")
        if critic_obs is None:
            critic_obs = obs

        if rew is None:
            rew = torch.zeros(self.num_envs, dtype=torch.float32, device='cuda:0')
        elif not isinstance(rew, torch.Tensor):
            rew = torch.tensor(rew, dtype=torch.float32, device='cuda:0')

        if terminated is None:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device='cuda:0')
        elif not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, dtype=torch.bool, device='cuda:0')
        else:
            terminated = terminated.to(torch.bool)

        if truncated is None:
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device='cuda:0')
        elif not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool, device='cuda:0')
        else:
            truncated = truncated.to(torch.bool)

        # -------------------------------------------------------------
        # Pull the fixed set of DEX_EXTRA_LOG_KEYS out of extras["log"] into a
        # single (num_envs, K) tensor - missing keys (e.g. a reward term that's
        # disabled this run) fall back to zeros rather than changing shape, so
        # the io_callback's output structure stays fixed across the whole run.
        # -------------------------------------------------------------
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        extra_cols = []
        for key in self.extra_log_keys:
            val = log.get(key, None)
            if val is None:
                col = torch.zeros(self.num_envs, dtype=torch.float32, device='cuda:0')
            elif isinstance(val, torch.Tensor):
                val = val.to(torch.float32)
                if val.ndim > 0 and val.shape[0] != self.num_envs:
                    # shape changed since discovery -- log zeros rather than
                    # crashing the whole run inside an io_callback
                    col = torch.zeros(self.num_envs, dtype=torch.float32, device='cuda:0')
                    extra_cols.append(col)
                    continue
                if val.ndim > 1 and val.shape[0] == self.num_envs:
                    # per-env but with trailing dims (e.g. per-link distances):
                    # average them down rather than raising in reshape()
                    col = val.reshape(self.num_envs, -1).mean(dim=-1)
                else:
                    col = val.reshape(self.num_envs)
            else:
                col = torch.full((self.num_envs,), float(val), dtype=torch.float32, device='cuda:0')
            extra_cols.append(col)
        extra_metrics = torch.stack(extra_cols, dim=-1)

        done, truncation = dexmachina_episode_end_flags(terminated, truncated)

        torch.cuda.synchronize()

        return (
            torch_to_jax(obs),
            torch_to_jax(critic_obs),
            torch_to_jax(rew),
            torch_to_jax(done),
            torch_to_jax(truncation),
            torch_to_jax(extra_metrics),
        )

    # ---------------------------------------------------------
    # JAX FUNCTIONS (JIT-Safe Callbacks)
    # ---------------------------------------------------------
    def reset(self, key, params=None):
        """JIT-safe reset using io_callback."""
        # Define expected memory shapes for JAX compiler
        result_shapes = (
            jax.ShapeDtypeStruct((self.num_envs, self.obs_dim), jnp.float32),
            jax.ShapeDtypeStruct((self.num_envs, self.state_dim), jnp.float32),
        )

        obs_jax, critic_jax = jax.experimental.io_callback(
            self._host_reset,
            result_shapes
        )
        
        # Initialize truncation states
        dummy_state = DexmachinaState(
            time=jnp.zeros(self.num_envs, dtype=jnp.float32),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            info={
                "steps": jnp.zeros(self.num_envs, dtype=jnp.int32),
                "truncation": jnp.zeros(self.num_envs, dtype=jnp.float32)
            }
        )
        return obs_jax, critic_jax, dummy_state

    def step(self, key, state, action):
        """JIT-safe step using io_callback."""
        result_shapes = (
            jax.ShapeDtypeStruct((self.num_envs, self.obs_dim), jnp.float32),      # obs
            jax.ShapeDtypeStruct((self.num_envs, self.state_dim), jnp.float32),    # critic_obs
            jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),                   # rew
            jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),                     # done
            jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),                   # truncated
            jax.ShapeDtypeStruct((self.num_envs, len(self.extra_log_keys)), jnp.float32),  # extra log metrics
        )

        obs_jax, critic_jax, rew_jax, dones_jax, truncated_jax, extra_metrics_jax = jax.experimental.io_callback(
            self._host_step,
            result_shapes,
            action,
        )

        # Update the dummy state with step counts and truncation
        new_state = DexmachinaState(
            time=state.time + 1.0,
            truncated=truncated_jax,
            info={
                "steps": state.info["steps"] + 1,
                "truncation": truncated_jax
            }
        )

        # Return Gymnax format: obs, critic_obs, state, reward, done, info
        # Pass truncation + DexMachina's per-step diagnostic metrics back in info dict
        info = {key: extra_metrics_jax[:, i] for i, key in enumerate(self.extra_log_keys)}
        return obs_jax, critic_jax, new_state, rew_jax, dones_jax, info
    
    def render(self):
        """
        Renders using DexMachina's native floating camera.
        Adapted for JAX/Gymnax.
        """
        # self.env is already the DexMachina BaseEnv, no need to peel wrappers!
        if hasattr(self.env, "_floating_camera") and self.env._floating_camera is not None:
            try:
                # 1. Render and grab just the RGB frame from the tuple
                out = self.env._floating_camera.render(segmentation=False)
                frame = out[0] if isinstance(out, tuple) else out

                # 2. Move to CPU and convert to NumPy
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()

                # 3. Drop the Alpha channel if it's RGBA (WandB requires RGB)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]

                # Note on Batch Dimensions: 
                # FlashSAC requires (1, H, W, C). 
                # If reppo expects a standard image (H, W, C), comment the next two lines out!
                if frame.ndim == 3:
                    frame = frame[np.newaxis, ...]

                return frame
            
            except Exception as e:
                print(f"[DEBUG] Camera extraction failed: {e}")

        # Ultimate Fallback (Returns a black screen so the run doesn't crash)
        return np.zeros((1, 256, 256, 3), dtype=np.uint8)

    def _process_obs(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Process observations safely in PyTorch before converting to JAX."""
        # Extract policy observation
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict  # Fallback if it's a raw tensor

        if not math.isinf(self._clip_obs):
            obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # Extract critic state safely
        critic_obs = obs
        if self.asymmetric_obs:
            if isinstance(obs_dict, dict):
                if "critic" in obs_dict:
                    critic_obs = obs_dict["critic"]
                elif "state" in obs_dict:
                    critic_obs = obs_dict["state"]
                elif "states" in obs_dict:
                    critic_obs = obs_dict["states"]

        return obs, critic_obs

    @property
    def default_params(self):
        return gymnax.EnvParams()
    
    @property
    def num_actions(self):
        return self.action_dim

    @property
    def num_obs(self):
        return self.obs_dim

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_dim,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=(self.obs_dim,)), \
               Box(low=-float("inf"), high=float("inf"), shape=(self.state_dim,))

_GENESIS_INITIALIZED = False


def dexmachina_episode_end_flags(terminated, truncated):
    """(done, truncation) for REPPO from DexMachina's reset flags: every episode end is terminal.

    DexMachina sets terminated == time_out == True for every reset (time-out at the end of the clip, object drop,
    early reset, NaN env), so its flags cannot mark a real truncation. The task ends with the demonstration and the
    observation contains t/T, so the end of the clip is a true terminal as well. Reporting truncation = 0 makes
    REPPO use done = 1 at every end: the lambda target is the last reward with no bootstrap, and the transition stays
    in the critic loss (both are skipped for truncated transitions). Before, every end was treated as truncated, so
    a drop never taught the critic that the rewards stop there.
    """
    done = terminated | truncated
    return done, torch.zeros_like(done, dtype=torch.float32)


def _genesis_init_once(seed: int):
    """gs.init() is process-global and raises if called twice.

    make_dexmachina_env() is now called more than once per process (a training
    env and a separate evaluation env), so the init has to be idempotent.
    """
    global _GENESIS_INITIALIZED
    if _GENESIS_INITIALIZED:
        return
    gs.init(backend=gs.gpu, logging_level="warning", seed=seed)  # also seeds random / numpy / torch
    _GENESIS_INITIALIZED = True


def make_dexmachina_env(cfg_env, cfg_hyperparameters, eval_mode=False, num_envs=None, *, seed: int):
    """Build a DexMachina BaseEnv.

    eval_mode=True builds a SEPARATE, smaller env for evaluation, mirroring what
    FastTD3's train.py does. Differences from the training env:
      * is_eval=True     -> _get_dones() returns `timeout, timeout` BEFORE the
                            object_fell_off check, so eval episodes always run the
                            full clip. Sharing one env meant eval episodes died at
                            ~40/200 steps and every eval metric was averaged over
                            the easy opening frames only, with a bias that shifted
                            as the policy got better at surviving.
      * early_reset_threshold=0 and rand_init_ratio=0 -> no extra termination, no
                            phase jitter, so eval is deterministic in length.
      * randomization off.
      * the curriculum's assistive gains are forced to zero and never advanced, so
        eval always measures the policy alone.
    The object stays actuated=True (unlike FastTD3, which sets actuated=False):
    keeping the virtual DOFs present with zero force range matches the training
    dynamics more closely than removing them.
    """
    _ = jax.devices()
    _genesis_init_once(seed)
    
    env_dict = OmegaConf.to_container(cfg_env, resolve=True)
    parser = get_common_argparser()
    fake_args, _ = parser.parse_known_args([]) 

    num_iterations = int(
        cfg_hyperparameters.total_time_steps // (cfg_hyperparameters.num_steps * cfg_hyperparameters.num_envs)
    )
    
    overrides = {
        # the run seed for dexmachina (args.seed); dexmachina's own default is 42
        "seed": int(seed),
        "num_envs": int(num_envs) if num_envs else cfg_hyperparameters.num_envs,
        # NOTE: constructors.py never reads `horizon`; the real episode length is
        # derived from the clip string. Kept only so fake_args has the attribute.
        "horizon": cfg_env.get("max_episode_steps", None),
        "learning_rate": cfg_hyperparameters.lr,
        # curriculum epochs are a normalised progress counter now, NOT the
        # number of reppo iterations - see DexmachinaGymnaxWrapper.__init__
        "max_epochs": int(cfg_env.get("curriculum_max_epochs", 10000)),
        "use_retarget_contact": True,
        "early_reset_threshold": 0.0,
        "aux_reset_thres": [0.0, 0.0, 0.0],
        "use_rl_games": True,
        **env_dict, 
    }

    print(f"[ENV] reppo iterations = {num_iterations}, "
          f"curriculum max_epochs = {int(cfg_env.get('curriculum_max_epochs', 10000))} "
          f"(normalised progress counter, independent of iteration count)")

    for key, value in overrides.items():
        setattr(fake_args, key, value)
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                setattr(fake_args, sub_k, sub_v)

    env_kwargs = get_all_env_cfg(fake_args, device='cuda:0')

    for key, value in env_dict.items():
        if isinstance(value, dict) and isinstance(env_kwargs.get(key), dict):
            env_kwargs[key].update(value)

    # Print AFTER the yaml merge above. get_all_env_cfg() returns dexmachina's
    # own defaults (and mangles upper/lower_ratios into {'kp': 'kp', ...} by
    # iterating a dict); the yaml values only take effect once merged, so
    # printing earlier shows numbers that are not the ones actually used.
    cc = env_kwargs['curriculum_cfg']
    print(f"[CURR] schedule={cc['schedule']} zero_epoch={cc['zero_epoch']} "
        f"wait={cc['wait_epochs']} interval={cc['interval']} upper={cc['upper_ratios']} "
        f"rew_thresholds={cc['rew_thresholds']} "
        f"min_achieved_len={cc.get('min_achieved_len', '0.85*max_ep_len')}")
    oc = list(env_kwargs['object_cfgs'].values())[0]
    print(f"[OBJ] actuated={oc.get('actuated')} kp={oc.get('kp')} "
        f"batch_dofs={env_kwargs['env_cfg']['scene_kwargs'].get('batch_dofs_info')}")

    if env_dict.get("curriculum", False):
        for obj_cfg in env_kwargs.get("object_cfgs", {}).values():
            if isinstance(obj_cfg, dict):
                obj_cfg["actuated"] = True
            else:
                setattr(obj_cfg, "actuated", True)
        
        env_cfg_dict = env_kwargs.setdefault("env_cfg", {})
        scene_kwargs = env_cfg_dict.setdefault("scene_kwargs", {})
        scene_kwargs["batch_dofs_info"] = True

    # get_all_env_cfg() only knows about dexmachina's own argparse keys, so the
    # ADD-logging flags have to be injected into env_cfg explicitly.
    env_cfg_dict = env_kwargs.setdefault("env_cfg", {})
    env_cfg_dict["log_add_metric"] = bool(env_dict.get("log_add_metric", False))
    env_cfg_dict["add_num_verts"] = int(env_dict.get("add_num_verts", 500))

    if eval_mode:
        env_cfg_dict["is_eval"] = True
        env_cfg_dict["early_reset_threshold"] = 0.0
        env_cfg_dict["early_reset_aux_thres"] = dict(con=0.0, imi=0.0, bc=0.0)
        env_cfg_dict["rand_init_ratio"] = 0.0
        env_cfg_dict["record_video"] = False
        rand_cfg = env_kwargs.setdefault("rand_cfg", {})
        if isinstance(rand_cfg, dict):
            rand_cfg["randomize"] = False

    env = BaseEnv(**env_kwargs)
    # Building the env reseeds the global RNGs with 42 (dexmachina's curriculum; numpy also in the ADD vertex
    # sampling). Seed again, so the env's later draws from them (random start frames, randomisation) depend on the
    # run seed. REPPO's own randomness comes from JAX keys and is not affected.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if eval_mode and getattr(env, "use_curriculum", False) and env.curriculum is not None:
        # unassisted physics, permanently -- nothing advances this curriculum
        c = env.curriculum
        for k in c.decay_terms:
            c.curr_gains[k] = 0.0
            c.curr_gains_lower[k] = 0.0
        c.reset_object_gains()
        # BaseEnv.step() consults this on every reset; setting it once here means
        # eval episode ends never enter the curriculum's deques, which is what the
        # per-call `is_eval` flag used to guarantee.
        env.record_curriculum = False
        print("[EVAL ENV] assistive gains zeroed permanently, curriculum bookkeeping off")
    return env
