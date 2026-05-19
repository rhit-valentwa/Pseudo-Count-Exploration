"""
Double DQN + pseudo-count exploration for Atari, with optional temporal bonus.

This file supports:
    - CTS pseudo-count bonus via modules.cts
    - PixelCNN pseudo-count bonus via modules.pixel_cnn
    - Optional temporal prediction bonus via modules.temporal_bonus

Key setup:
    - N-step Double DQN target: online network selects action at s_{t+n}, target network evaluates it.
    - CTS receives a single Atari frame, not a 4-frame stack.
    - PixelCNN receives the stacked post-step observation.
    - Temporal bonus receives current frame stack + action + next frame.
    - Sticky actions via repeat_action_probability=0.25 when supported.
    - No life-loss terminal handling.
    - DQN uses 4-frame stacks for Q-learning.
    - Optional n-step returns via --n-step, default 5.
    - Replay stores single uint8 frames and lazily builds n-step stacked transitions.

Expected project layout:
    atari_dqn.py
    setup.py
    modules/
        __init__.py
        cts.py
        cts_core.pyx / compiled cts_core.so
        pixel_cnn.py
        temporal_bonus.py

Build CTS first:
    python setup.py build_ext --inplace

CTS only:
    python atari_dqn.py --env ALE/Freeway-v5 --density cts --beta 0.05 \
      --sticky-action-prob 0.25 --no-compile --log-freq 1000 --steps 500000

CTS + temporal:
    python atari_dqn.py --env ALE/Freeway-v5 --density cts --beta 0.05 \
      --use-temporal --temporal-beta 0.01 --temporal-lr 1e-4 \
      --sticky-action-prob 0.25 --no-compile --log-freq 1000 --steps 500000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    env_id: str = "ALE/Freeway-v5"
    total_steps: int = 500_000
    buffer_capacity: int = 50_000
    batch_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step: int = 1
    train_start: int = 10_000
    train_freq: int = 4
    target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000_000

    # State-density intrinsic reward.
    density: str = "cts"  # choices: cts, pixelcnn, none
    beta: float = 0.05

    n_frames: int = 4
    log_freq: int = 1_000
    graphs_dir: str = "graphs"
    log_dir: str = "logs"
    run_name: str | None = None
    checkpoint_freq: int = 50_000

    seed: int = 0
    use_compile: bool = True
    sticky_action_prob: float = 0.25
    clip_combined_reward: bool = False


# =============================================================================
# Replay buffer: uint8 single-frame storage + lazy stacking
# =============================================================================

class LazyStackReplay:
    """
    Circular replay buffer that stores individual uint8 frames and builds
    n-frame stacks when sampling.
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: Tuple[int, int],
        n_frames: int,
        device: torch.device,
    ) -> None:
        h, w = frame_shape
        self.capacity = int(capacity)
        self.n_frames = int(n_frames)
        self.h = int(h)
        self.w = int(w)
        self.device = device
        self.ptr = 0
        self.size = 0

        pin = device.type == "cuda"
        self.frames = torch.empty(
            (self.capacity, self.h, self.w),
            dtype=torch.uint8,
            pin_memory=pin,
        )
        self.actions = torch.empty(self.capacity, dtype=torch.int64, pin_memory=pin)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32, pin_memory=pin)
        self.dones = torch.empty(self.capacity, dtype=torch.bool, pin_memory=pin)

    def __len__(self) -> int:
        return self.size

    def add(self, frame_u8: np.ndarray, action: int, reward: float, done: bool) -> None:
        self.frames[self.ptr].copy_(torch.from_numpy(frame_u8))
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, n_step: int = 1, gamma: float = 0.99):
        """
        Sample an n-step Double DQN transition.

        Returns:
            obs:            (B, n_frames, H, W), float32 in [0, 1]
            actions:        (B,), action at time t
            n_step_rewards: (B,), discounted reward from t through t+n_step-1,
                            truncated at episode end
            next_obs:       (B, n_frames, H, W), stack ending at t+n_step
            n_step_dones:   (B,), 1 if episode ended within the n-step window
        """
        n = self.n_frames
        n_step = int(n_step)

        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        if self.size <= n + n_step:
            raise ValueError("Not enough replay data to sample an n-step stacked transition.")

        start = 0 if self.size < self.capacity else self.ptr

        # Pick t so that:
        #   current stack [t-n+1, ..., t] exists
        #   reward window [t, ..., t+n_step-1] exists
        #   bootstrap stack ending at t+n_step exists
        t = np.random.randint(n - 1, self.size - n_step, size=batch_size)
        cur = torch.from_numpy(((t + start) % self.capacity).astype(np.int64))

        # ------------------------------------------------------------------
        # Current observation stack: frames [t-n+1, ..., t]
        # ------------------------------------------------------------------
        obs_offsets = np.arange(-(n - 1), 1)
        obs_logical = t[:, None] + obs_offsets[None, :]
        obs_physical = (obs_logical + start) % self.capacity

        obs_flat = torch.from_numpy(obs_physical.reshape(-1).astype(np.int64))
        obs_frames_u8 = self.frames.index_select(0, obs_flat).view(
            batch_size, n, self.h, self.w
        )
        obs_done_window = self.dones.index_select(0, obs_flat).view(batch_size, n)

        # ------------------------------------------------------------------
        # Bootstrap observation stack: frames [t+n_step-n+1, ..., t+n_step]
        # ------------------------------------------------------------------
        next_offsets = np.arange(n_step - (n - 1), n_step + 1)
        next_logical = t[:, None] + next_offsets[None, :]
        next_physical = (next_logical + start) % self.capacity

        next_flat = torch.from_numpy(next_physical.reshape(-1).astype(np.int64))
        next_frames_u8 = self.frames.index_select(0, next_flat).view(
            batch_size, n, self.h, self.w
        )
        next_done_window = self.dones.index_select(0, next_flat).view(batch_size, n)

        # ------------------------------------------------------------------
        # Reward/done window: transitions [t, ..., t+n_step-1]
        # ------------------------------------------------------------------
        reward_offsets = np.arange(n_step)
        reward_logical = t[:, None] + reward_offsets[None, :]
        reward_physical = (reward_logical + start) % self.capacity

        reward_flat = torch.from_numpy(reward_physical.reshape(-1).astype(np.int64))
        rewards_window_cpu = self.rewards.index_select(0, reward_flat).view(
            batch_size, n_step
        )
        dones_window_cpu = self.dones.index_select(0, reward_flat).view(
            batch_size, n_step
        )

        actions_cpu = self.actions.index_select(0, cur)

        dev = self.device

        obs_frames = obs_frames_u8.to(dev, non_blocking=True)
        next_frames = next_frames_u8.to(dev, non_blocking=True)
        obs_done = obs_done_window.to(dev, non_blocking=True)
        next_done = next_done_window.to(dev, non_blocking=True)

        actions = actions_cpu.to(dev, non_blocking=True)
        rewards_window = rewards_window_cpu.to(dev, non_blocking=True)
        dones_window = dones_window_cpu.to(dev, non_blocking=True)

        # ------------------------------------------------------------------
        # Zero out stacked frames that cross episode boundaries.
        # ------------------------------------------------------------------
        if n >= 2:
            obs_terms = obs_done[:, : n - 1]
            obs_prior = obs_terms.flip(1).cummax(dim=1).values.flip(1)
            obs_keep = torch.cat(
                [
                    ~obs_prior,
                    torch.ones(batch_size, 1, dtype=torch.bool, device=dev),
                ],
                dim=1,
            )

            next_terms = next_done[:, : n - 1]
            next_prior = next_terms.flip(1).cummax(dim=1).values.flip(1)
            next_keep = torch.cat(
                [
                    ~next_prior,
                    torch.ones(batch_size, 1, dtype=torch.bool, device=dev),
                ],
                dim=1,
            )
        else:
            obs_keep = torch.ones(batch_size, n, dtype=torch.bool, device=dev)
            next_keep = torch.ones(batch_size, n, dtype=torch.bool, device=dev)

        obs = obs_frames.to(torch.float32).mul_(1.0 / 255.0)
        next_obs = next_frames.to(torch.float32).mul_(1.0 / 255.0)

        obs.mul_(obs_keep[:, :, None, None])
        next_obs.mul_(next_keep[:, :, None, None])

        # ------------------------------------------------------------------
        # Discounted n-step reward, truncated at the first terminal.
        # ------------------------------------------------------------------
        discounts = torch.tensor(
            [gamma ** k for k in range(n_step)],
            dtype=torch.float32,
            device=dev,
        )

        dones_float = dones_window.to(torch.float32)

        # Keep reward k only if there was no done before k.
        # The reward on the terminal transition itself is kept.
        prior_done = torch.cumsum(dones_float, dim=1) - dones_float
        reward_keep = (prior_done <= 0.0).to(torch.float32)

        n_step_rewards = (rewards_window * discounts[None, :] * reward_keep).sum(dim=1)
        n_step_dones = dones_window.any(dim=1).to(torch.float32)

        return obs, actions, n_step_rewards, next_obs, n_step_dones

# =============================================================================
# Double DQN network
# =============================================================================

def build_qnetwork(n_actions: int, n_frames: int = 4) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 7 * 7, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, n_actions),
    )


# =============================================================================
# Environment
# =============================================================================

def make_env(env_id: str, seed: int, sticky_action_prob: float):
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing

    gym.register_envs(ale_py)

    try:
        env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=sticky_action_prob,
        )
    except TypeError:
        env = gym.make(env_id, frameskip=1)
        try:
            env.unwrapped.ale.setFloat("repeat_action_probability", sticky_action_prob)
        except Exception:
            print(
                "Warning: could not set repeat_action_probability. "
                "Your ALE/Gymnasium version may not support this option here.",
                flush=True,
            )

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    env.action_space.seed(seed)
    return env


# =============================================================================
# Bonus module selection
# =============================================================================

def make_state_bonus_fn(density: str, beta: float, n_frames: int, h: int, w: int):
    if density == "none":
        return None

    if density == "cts":
        from modules.cts import make_cts_bonus
        return make_cts_bonus((h, w), beta=beta)

    if density == "vae":
        from modules.vae import make_vae_pg_bonus
        return make_vae_pg_bonus((n_frames, h, w), beta=beta)

    if density == "pixelcnn":
        from modules.pixel_cnn import make_pixelcnn_bonus
        return make_pixelcnn_bonus((n_frames, h, w), beta=beta)

    raise ValueError(f"Unknown density model: {density}")

# =============================================================================
# Logging utilities
# =============================================================================

def make_run_dir(config: TrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_env = config.env_id.replace("/", "_").replace("-", "_")

    if config.run_name:
        run_name = config.run_name
    else:
        parts = [timestamp, safe_env, config.density]
        parts.append(f"seed{config.seed}")
        run_name = "_".join(parts)

    run_dir = os.path.join(config.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    return run_dir


def init_progress_csv(run_dir: str) -> str:
    path = os.path.join(run_dir, "progress.csv")
    fields = [
        "step",
        "epsilon",
        "episodes",
        "avg_ext_20ep",
        "avg_intr_20ep",
        "avg_state_intr_20ep",
        "current_ep_ext",
        "current_ep_intr",
        "current_ep_state_intr",
        "replay_size",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    return path


def append_progress_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: str,
    step: int,
    online_raw: torch.nn.Module,
    target_raw: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "online_state_dict": online_raw.state_dict(),
            "target_state_dict": target_raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


# =============================================================================
# Training
# =============================================================================

def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Density model: {config.density}", flush=True)
    print(f"Sticky action probability: {config.sticky_action_prob}", flush=True)
    print("Monte Carlo return mixing: disabled", flush=True)

    run_dir = make_run_dir(config)
    progress_csv = init_progress_csv(run_dir)
    print(f"Logging to: {run_dir}", flush=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    env = make_env(config.env_id, config.seed, config.sticky_action_prob)
    h, w = env.observation_space.shape
    n_actions = int(env.action_space.n)

    online_raw = build_qnetwork(n_actions, config.n_frames).to(device)
    target_raw = build_qnetwork(n_actions, config.n_frames).to(device)
    target_raw.load_state_dict(online_raw.state_dict())
    target_raw.eval()

    for p in target_raw.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(online_raw.parameters(), lr=config.learning_rate)

    online_net: torch.nn.Module = online_raw
    target_net: torch.nn.Module = target_raw

    if config.use_compile and hasattr(torch, "compile"):
        try:
            online_net = torch.compile(online_raw, mode="default")
            target_net = torch.compile(target_raw, mode="default")
            print("torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}", flush=True)
            online_net = online_raw
            target_net = target_raw

    replay = LazyStackReplay(
        capacity=config.buffer_capacity,
        frame_shape=(h, w),
        n_frames=config.n_frames,
        device=device,
    )

    state_bonus_fn = make_state_bonus_fn(
        density=config.density,
        beta=config.beta,
        n_frames=config.n_frames,
        h=h,
        w=w,
    )

    pin = device.type == "cuda"
    obs_pin = torch.empty(
        (1, config.n_frames, h, w),
        dtype=torch.uint8,
        pin_memory=pin,
    )
    stack_u8 = np.empty((config.n_frames, h, w), dtype=np.uint8)
    stack_float = np.empty((config.n_frames, h, w), dtype=np.float32)
    inv255 = np.float32(1.0 / 255.0)

    frame_deque: Deque[np.ndarray] = deque(maxlen=config.n_frames)

    def reset_deque(frame: np.ndarray) -> None:
        safe = frame.copy()
        frame_deque.clear()
        for _ in range(config.n_frames):
            frame_deque.append(safe)

    def fill_stack(out: np.ndarray) -> None:
        for idx, frame in enumerate(frame_deque):
            out[idx] = frame

    ep_returns: List[float] = []
    ep_intrinsic: List[float] = []
    ep_state_intrinsic: List[float] = []

    ep_return = 0.0
    ep_intr = 0.0
    ep_state_intr = 0.0

    first_frame, _ = env.reset(seed=config.seed)
    reset_deque(first_frame)

    for step in range(1, config.total_steps + 1):
        frac = min(1.0, step / config.epsilon_decay_steps)
        epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

        fill_stack(stack_u8)

        if random.random() < epsilon:
            action = int(env.action_space.sample())
        else:
            obs_pin[0].copy_(torch.from_numpy(stack_u8))
            with torch.no_grad():
                obs_t = obs_pin.to(device, non_blocking=True).to(torch.float32).mul_(inv255)
                action = int(online_net(obs_t).argmax(dim=1).item())

        before_action_frame = frame_deque[-1]
        next_frame, extrinsic, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # ------------------------------------------------------------------
        # State-density intrinsic bonus: CTS / PixelCNN / none
        # ------------------------------------------------------------------
        state_bonus = 0.0
        if state_bonus_fn is not None:
            if config.density == "cts":
                # Pseudo-count paper CTS: density over current single frame.
                state_bonus, _ = state_bonus_fn.bonus_and_update(next_frame)
            elif config.density == "vae":
                for i in range(config.n_frames - 1):
                    stack_u8[i] = frame_deque[i + 1]
                stack_u8[-1] = next_frame
                stack_float[:] = stack_u8
                stack_float *= inv255
                state_bonus, _ = state_bonus_fn.bonus_and_update(stack_float)
            elif config.density in ("pixelcnn", "vae"):
                # PixelCNN and VAE use the post-step 4-frame stack.
                for i in range(config.n_frames - 1):
                    stack_u8[i] = frame_deque[i + 1]
                stack_u8[-1] = next_frame
                stack_float[:] = stack_u8
                stack_float *= inv255
                state_bonus, _ = state_bonus_fn.bonus_and_update(stack_float)
                
        intr_bonus = float(state_bonus)
        total_reward = float(extrinsic) + intr_bonus
        if config.clip_combined_reward:
            total_reward = float(np.clip(total_reward, -1.0, 1.0))

        replay.add(before_action_frame, action, total_reward, done)

        ep_return += float(extrinsic)
        ep_intr += intr_bonus
        ep_state_intr += float(state_bonus)

        if done:
            ep_returns.append(ep_return)
            ep_intrinsic.append(ep_intr)
            ep_state_intrinsic.append(ep_state_intr)

            ep_return = 0.0
            ep_intr = 0.0
            ep_state_intr = 0.0

            reset_frame, _ = env.reset()
            reset_deque(reset_frame)
        else:
            frame_deque.append(next_frame.copy())

        # ------------------------------------------------------------------
        # N-step Double DQN update. No Monte Carlo return mixing.
        # ------------------------------------------------------------------
        if (
            step >= config.train_start
            and step % config.train_freq == 0
            and len(replay) > config.n_frames + config.n_step
        ):
            obs_b, act_b, rew_b, next_b, done_b = replay.sample(
                config.batch_size,
                n_step=config.n_step,
                gamma=config.gamma,
            )

            with torch.no_grad():
                next_actions = online_net(next_b).argmax(dim=1)
                next_q = target_net(next_b).gather(1, next_actions[:, None]).squeeze(1)
                bootstrap_gamma = config.gamma ** config.n_step
                target = rew_b + bootstrap_gamma * next_q * (1.0 - done_b)

            current_q = online_net(obs_b).gather(1, act_b[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(current_q, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_raw.parameters(), 10.0)
            optimizer.step()

        if step % config.target_update_freq == 0:
            target_raw.load_state_dict(online_raw.state_dict())

        if step % config.log_freq == 0:
            avg_ext = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            avg_intr = float(np.mean(ep_intrinsic[-20:])) if ep_intrinsic else 0.0
            avg_state_intr = float(np.mean(ep_state_intrinsic[-20:])) if ep_state_intrinsic else 0.0

            row = {
                "step": step,
                "epsilon": epsilon,
                "episodes": len(ep_returns),
                "avg_ext_20ep": avg_ext,
                "avg_intr_20ep": avg_intr,
                "avg_state_intr_20ep": avg_state_intr,
                "current_ep_ext": ep_return,
                "current_ep_intr": ep_intr,
                "current_ep_state_intr": ep_state_intr,
                "replay_size": len(replay),
            }
            append_progress_csv(progress_csv, row)

            print(
                f"step={step:>9,}  "
                f"eps={epsilon:.3f}  "
                f"episodes={len(ep_returns):>5}  "
                f"avg_ext_20ep={avg_ext:>8.2f}  "
                f"avg_intr_20ep={avg_intr:.4f}  "
                f"state_intr={avg_state_intr:.4f}  ",
                flush=True,
            )


    final_ckpt_path = os.path.join(run_dir, "checkpoint_final.pt")
    save_checkpoint(final_ckpt_path, config.total_steps, online_raw, target_raw, optimizer, config)

    returns_path = os.path.join(run_dir, "episode_returns.npz")
    np.savez(
        returns_path,
        extrinsic_returns=np.asarray(ep_returns, dtype=np.float32),
        intrinsic_returns=np.asarray(ep_intrinsic, dtype=np.float32),
        state_intrinsic_returns=np.asarray(ep_state_intrinsic, dtype=np.float32),
    )

    print(f"Saved final checkpoint: {final_ckpt_path}", flush=True)
    print(f"Saved episode returns: {returns_path}", flush=True)

    env.close()
    plot_training_curves(ep_returns, ep_intrinsic, config.env_id, config.graphs_dir)


# =============================================================================
# Plotting
# =============================================================================

def plot_training_curves(
    returns: List[float],
    intrinsic: List[float],
    env_id: str,
    graphs_dir: str,
    window: int = 20,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping graph", flush=True)
        return

    if not returns:
        print("No completed episodes; skipping graph", flush=True)
        return

    os.makedirs(graphs_dir, exist_ok=True)
    episodes = np.arange(1, len(returns) + 1)

    def rolling_mean(data: List[float]) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if len(arr) == 0:
            return arr
        k = min(window, len(arr))
        filt = np.ones(k, dtype=np.float32) / k
        padded = np.pad(arr, (k - 1, 0), mode="edge")
        return np.convolve(padded, filt, mode="valid")

    safe_env = env_id.replace("/", "_").replace("-", "_")
    path = os.path.join(graphs_dir, f"{safe_env}_{len(returns)}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Double DQN + Pseudo-count Bonus - {env_id}", fontsize=13)

    ax1.plot(episodes, returns, alpha=0.25, linewidth=0.7)
    ax1.plot(episodes, rolling_mean(returns), linewidth=1.8, label=f"{window}-ep avg")
    ax1.set_ylabel("Extrinsic return")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, intrinsic, alpha=0.25, linewidth=0.7)
    ax2.plot(episodes, rolling_mean(intrinsic), linewidth=1.8, label=f"{window}-ep avg")
    ax2.set_ylabel("Intrinsic bonus sum")
    ax2.set_xlabel("Episode")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {path}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Double DQN + pseudo-count exploration"
    )

    parser.add_argument("--env", default=TrainConfig.env_id)
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--buffer-capacity", type=int, default=TrainConfig.buffer_capacity)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--n-step", type=int, default=TrainConfig.n_step)
    parser.add_argument("--train-start", type=int, default=TrainConfig.train_start)
    parser.add_argument("--train-freq", type=int, default=TrainConfig.train_freq)
    parser.add_argument("--target-update-freq", type=int, default=TrainConfig.target_update_freq)
    parser.add_argument("--epsilon-start", type=float, default=TrainConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainConfig.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=TrainConfig.epsilon_decay_steps)

    parser.add_argument("--density", choices=["cts", "pixelcnn", "vae", "none"], default=TrainConfig.density)
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)

    parser.add_argument("--n-frames", type=int, default=TrainConfig.n_frames)
    parser.add_argument("--log-freq", type=int, default=TrainConfig.log_freq)
    parser.add_argument("--graphs-dir", default=TrainConfig.graphs_dir)
    parser.add_argument("--log-dir", default=TrainConfig.log_dir)
    parser.add_argument("--run-name", default=TrainConfig.run_name)
    parser.add_argument("--checkpoint-freq", type=int, default=TrainConfig.checkpoint_freq)

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--sticky-action-prob", type=float, default=TrainConfig.sticky_action_prob)
    parser.add_argument("--clip-combined-reward", action="store_true")
    parser.add_argument("--no-compile", action="store_true")

    args = parser.parse_args()

    return TrainConfig(
        env_id=args.env,
        total_steps=args.steps,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
        train_start=args.train_start,
        train_freq=args.train_freq,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        density=args.density,
        beta=args.beta,
        n_frames=args.n_frames,
        log_freq=args.log_freq,
        graphs_dir=args.graphs_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        use_compile=not args.no_compile,
        sticky_action_prob=args.sticky_action_prob,
        clip_combined_reward=args.clip_combined_reward,
    )

if __name__ == "__main__":
    train(parse_args())