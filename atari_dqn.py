"""
DQN + PixelCNN pseudo-count exploration bonus for Atari games.

Uses prediction-gain bonus from pixel_cnn.py as the density model over stacked
grayscale frames. See pixel_cnn.py for details on the log-space pseudo-count.

Install:
    pip install torch numpy gymnasium[atari] ale-py matplotlib
    AutoROM --accept-license   # downloads Atari ROMs

Run:
    python atari_dqn.py                      # Pong, 2 M steps
    python atari_dqn.py --env ALE/Breakout-v5 --steps 5_000_000
"""

from __future__ import annotations

import argparse
import collections
import os
import random
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pixel_cnn import make_pixelcnn_bonus


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # uint8 storage: 4× smaller than float32; normalized to [0,1] on sample
        self.obs      = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions  = torch.zeros(capacity, dtype=torch.long)
        self.rewards  = torch.zeros(capacity, dtype=torch.float32)
        self.dones    = torch.zeros(capacity, dtype=torch.float32)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr]      = torch.from_numpy((obs * 255).astype(np.uint8))
        self.next_obs[self.ptr] = torch.from_numpy((next_obs * 255).astype(np.uint8))
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = random.sample(range(self.size), batch_size)
        return (
            self.obs[idx].to(self.device, dtype=torch.float32).div(255.0),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_obs[idx].to(self.device, dtype=torch.float32).div(255.0),
            self.dones[idx].to(self.device),
        )


# ---------------------------------------------------------------------------
# Q-network (standard DQN CNN)
# ---------------------------------------------------------------------------

def build_qnetwork(n_actions: int, n_frames: int = 4) -> torch.nn.Module:
    class QNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(n_frames, 32, kernel_size=8, stride=4), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),        torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),        torch.nn.ReLU(),
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(64 * 7 * 7, 512), torch.nn.ReLU(),
                torch.nn.Linear(512, n_actions),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.fc(self.conv(x).flatten(1))

    return QNet()


# ---------------------------------------------------------------------------
# Environment construction helpers
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int, n_frames: int = 4):
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    gym.register_envs(ale_py)
    env = gym.make(env_id, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, stack_size=n_frames)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env_id: str = "ALE/Pong-v5",
    total_steps: int = 2_000_000,
    buffer_capacity: int = 50_000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    train_start: int = 10_000,
    train_freq: int = 4,
    target_update_freq: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: int = 500_000,
    beta: float = 0.05,
    n_frames: int = 4,
    log_freq: int = 10_000,
    graphs_dir: str = "graphs",
    seed: int = 0,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = make_env(env_id, seed, n_frames)
    obs_shape: Tuple[int, ...] = tuple(env.observation_space.shape)   # (n_frames, 84, 84)
    n_actions: int = env.action_space.n

    online_net = build_qnetwork(n_actions, n_frames).to(device)
    target_net = build_qnetwork(n_actions, n_frames).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)

    buffer = ReplayBuffer(buffer_capacity, obs_shape, device)

    bonus_fn = make_pixelcnn_bonus(obs_shape, beta=beta)

    # Logging
    ep_returns: List[float] = []
    ep_intrinsic: List[float] = []
    step_log: List[int] = []
    ep_return = 0.0
    ep_intr   = 0.0

    obs, _ = env.reset(seed=seed)
    obs_arr = np.array(obs, dtype=np.float32)

    for step in range(1, total_steps + 1):
        # Epsilon-greedy action
        frac = min(1.0, step / epsilon_decay_steps)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                t = torch.from_numpy(obs_arr).unsqueeze(0).to(device)
                action = int(online_net(t).argmax(dim=1).item())

        next_obs, extrinsic, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs_arr = np.array(next_obs, dtype=np.float32)

        # Pseudo-count intrinsic bonus on the next observation
        intr_bonus, _ = bonus_fn.bonus_and_update(next_obs_arr)
        augmented_reward = float(extrinsic) + intr_bonus

        buffer.add(obs_arr, action, augmented_reward, next_obs_arr, done)

        ep_return += float(extrinsic)
        ep_intr   += intr_bonus

        if done:
            ep_returns.append(ep_return)
            ep_intrinsic.append(ep_intr)
            ep_return = 0.0
            ep_intr   = 0.0
            obs, _ = env.reset()
            obs_arr = np.array(obs, dtype=np.float32)
        else:
            obs_arr = next_obs_arr

        # Train
        if step >= train_start and step % train_freq == 0:
            obs_b, act_b, rew_b, next_b, done_b = buffer.sample(batch_size)
            with torch.no_grad():
                next_q = target_net(next_b).max(dim=1).values
                target = rew_b + gamma * next_q * (1.0 - done_b)
            current_q = online_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % log_freq == 0 and ep_returns:
            avg_ext  = float(np.mean(ep_returns[-20:]))
            avg_intr = float(np.mean(ep_intrinsic[-20:]))
            step_log.append(step)
            print(
                f"step={step:>8,}  eps={epsilon:.3f}  "
                f"avg_extrinsic(20ep)={avg_ext:>8.2f}  "
                f"avg_intrinsic(20ep)={avg_intr:.4f}"
            )

    env.close()
    _plot(ep_returns, ep_intrinsic, env_id, graphs_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    returns: List[float],
    intrinsic: List[float],
    env_id: str,
    graphs_dir: str,
    window: int = 20,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping graph")
        return

    os.makedirs(graphs_dir, exist_ok=True)

    eps = np.arange(1, len(returns) + 1)

    def roll(data: List[float]) -> np.ndarray:
        arr = np.array(data, dtype=np.float32)
        k = np.ones(window) / window
        return np.convolve(np.pad(arr, (window - 1, 0), mode="edge"), k, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    safe = env_id.replace("/", "_").replace("-", "_")
    fig.suptitle(f"DQN + Pseudo-Count — {env_id}", fontsize=13)

    ax1.plot(eps, returns,      alpha=0.2, color="steelblue",  linewidth=0.6)
    ax1.plot(eps, roll(returns), color="steelblue",  linewidth=1.8, label=f"{window}-ep avg")
    ax1.set_ylabel("Extrinsic return")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, intrinsic,      alpha=0.2, color="darkorange", linewidth=0.6)
    ax2.plot(eps, roll(intrinsic), color="darkorange", linewidth=1.8, label=f"{window}-ep avg")
    ax2.set_ylabel("Intrinsic bonus (sum)")
    ax2.set_xlabel("Episode")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(graphs_dir, f"{safe}_{len(returns)}ep.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--beta",  type=float, default=0.05)
    parser.add_argument("--seed",  type=int, default=0)
    args = parser.parse_args()

    train(env_id=args.env, total_steps=args.steps, beta=args.beta, seed=args.seed)
