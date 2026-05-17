"""
DQN + PixelCNN pseudo-count exploration bonus for Atari games (optimized).

Algorithmic behaviour is unchanged from the original; the changes target
throughput and memory.

Key optimizations
-----------------
1. uint8 end-to-end. The env returns uint8 frames, the buffer stores uint8,
   and conversion to normalized float32 happens *once on the GPU* at sample
   time. The original did float -> uint8 -> float on every transition.
2. Lazy frame stacking. The buffer stores single 84x84 frames and assembles
   n-frame stacks on sample. ~n-fold less memory; or, equivalently, lets the
   buffer be n times larger for the same RAM budget.
3. Pinned host buffers + non_blocking H2D transfers.
4. Reusable pinned tensor + numpy scratch for action selection (no per-step
   allocation in the hot loop).
5. cudnn.benchmark + TF32 enabled.
6. torch.compile on the Q-network where available, with safe fallback.
7. Huber (smooth-L1) loss instead of MSE - standard for DQN, more robust.
8. np.random.randint sampling (vs. random.sample(range(...))).
9. Target network has requires_grad=False (skips autograd bookkeeping).

Drop-in replacement: same CLI, same `pixel_cnn` interface.

Install:
    pip install torch numpy gymnasium[atari] ale-py matplotlib
    AutoROM --accept-license

Run:
    python atari_dqn.py
    python atari_dqn.py --env ALE/Breakout-v5 --steps 5_000_000
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pixel_cnn import make_pixelcnn_bonus


# ============================================================================
# Replay buffer - lazy frame-stacking, uint8 storage
# ============================================================================

class LazyStackReplay:
    """Single-frame circular replay; assembles n-frame stacks on sample.

    Versus storing pre-stacked observations this saves a factor of `n` in
    memory and removes the float<->uint8 round-trip the original code
    performed on every transition. Stacks crossing an episode boundary have
    the prior-episode frames zero-masked (well-known acceptable approximation
    used by all major DQN implementations).
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: Tuple[int, int],
        n_frames: int,
        device: torch.device,
    ) -> None:
        H, W = frame_shape
        self.cap = capacity
        self.n = n_frames
        self.H, self.W = H, W
        self.device = device
        self.ptr = 0
        self.size = 0

        pin = device.type == "cuda"
        self.frames    = torch.empty((capacity, H, W), dtype=torch.uint8,   pin_memory=pin)
        self.actions   = torch.empty(capacity,         dtype=torch.int64,   pin_memory=pin)
        self.rewards   = torch.empty(capacity,         dtype=torch.float32, pin_memory=pin)
        self.terminals = torch.empty(capacity,         dtype=torch.bool,    pin_memory=pin)

    def __len__(self) -> int:
        return self.size

    def add(self, frame_u8: np.ndarray, action: int, reward: float, terminal: bool) -> None:
        """Store the frame the agent saw at this step plus the action it took,
        the resulting reward, and whether the episode ended."""
        self.frames[self.ptr].copy_(torch.from_numpy(frame_u8))
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.terminals[self.ptr] = terminal
        self.ptr = (self.ptr + 1) % self.cap
        if self.size < self.cap:
            self.size += 1

    def sample(self, batch_size: int):
        n, cap, H, W = self.n, self.cap, self.H, self.W
        # 0 = oldest in logical order; size-1 = newest written.
        start = 0 if self.size < cap else self.ptr
        # t in [n-1, size-2]: needs (n-1) history before t plus a frame at t+1.
        t = np.random.randint(n - 1, self.size - 1, size=batch_size)
        # n+1 frames per sample: cols [0..n-1] = obs, cols [1..n] = next_obs.
        offsets  = np.arange(-(n - 1), 2)
        log_win  = t[:, None] + offsets
        phys_win = (log_win + start) % cap
        flat     = torch.from_numpy(phys_win.reshape(-1).astype(np.int64))
        cur_phys = torch.from_numpy(((t + start) % cap).astype(np.int64))

        # CPU gather, then a single H2D transfer per tensor.
        frames_u8   = self.frames.index_select(0, flat).view(batch_size, n + 1, H, W)
        term_window = self.terminals.index_select(0, flat).view(batch_size, n + 1)
        actions_cpu = self.actions.index_select(0, cur_phys)
        rewards_cpu = self.rewards.index_select(0, cur_phys)
        dones_cpu   = self.terminals.index_select(0, cur_phys)

        dev = self.device
        frames_gpu = frames_u8.to(dev, non_blocking=True)
        term_gpu   = term_window.to(dev, non_blocking=True)
        actions    = actions_cpu.to(dev, non_blocking=True)
        rewards    = rewards_cpu.to(dev, non_blocking=True)
        dones      = dones_cpu.to(dev, non_blocking=True).to(torch.float32)

        # Episode-boundary mask via a vectorized right-cumulative OR.
        #   - For obs cols [0..n-1]: col j is "prior episode" iff a terminal
        #     occurs in cols [j..n-2]. Col n-1 is always the current episode.
        #   - For next_obs cols [1..n]: col j is "prior episode" iff a terminal
        #     occurs in cols [j..n-1]. Col n is always the current episode.
        if n >= 2:
            obs_t      = term_gpu[:, : n - 1]                              # (B, n-1)
            obs_prior  = obs_t.flip(1).cummax(dim=1).values.flip(1)
            obs_keep   = torch.cat(
                [~obs_prior, torch.ones(batch_size, 1, dtype=torch.bool, device=dev)],
                dim=1,
            )                                                              # (B, n)

            nxt_t      = term_gpu[:, 1:n]                                  # (B, n-1)
            nxt_prior  = nxt_t.flip(1).cummax(dim=1).values.flip(1)
            nxt_keep   = torch.cat(
                [~nxt_prior, torch.ones(batch_size, 1, dtype=torch.bool, device=dev)],
                dim=1,
            )                                                              # (B, n)
        else:
            obs_keep = nxt_keep = torch.ones(batch_size, n, dtype=torch.bool, device=dev)

        # uint8 -> float32 in [0, 1] on GPU; one fused convert + scale.
        obs = frames_gpu[:, :n].to(torch.float32).mul_(1.0 / 255.0)
        nxt = frames_gpu[:, 1:].to(torch.float32).mul_(1.0 / 255.0)
        obs.mul_(obs_keep.unsqueeze(-1).unsqueeze(-1))
        nxt.mul_(nxt_keep.unsqueeze(-1).unsqueeze(-1))
        return obs, actions, rewards, nxt, dones


# ============================================================================
# Q-network
# ============================================================================

def build_qnetwork(n_actions: int, n_frames: int = 4) -> torch.nn.Module:
    """Standard DQN CNN. Sequential is friendlier to torch.compile than a
    custom forward, and inplace ReLUs save a small amount of memory."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(n_frames, 32, kernel_size=8, stride=4), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),       torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),       torch.nn.ReLU(inplace=True),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 7 * 7, 512),                       torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, n_actions),
    )


# ============================================================================
# Environment - single uint8 frames; we do the stacking ourselves
# ============================================================================

def make_env(env_id: str, seed: int):
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing

    gym.register_envs(ale_py)
    env = gym.make(env_id, frameskip=1)
    # scale_obs=False -> uint8 frames; we normalize on GPU at sample time.
    # No FrameStackObservation -> we manage the stack so the buffer can
    # store single frames.
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env.action_space.seed(seed)
    return env


# ============================================================================
# Training loop
# ============================================================================

def train(
    env_id: str = "ALE/Pong-v5",
    total_steps: int = 50_000,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    train_start: int = 10_000,
    train_freq: int = 4,
    target_update_freq: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 1_000_000,
    beta: float = 0.05,
    density_model: str = "pixelcnn",
    n_frames: int = 4,
    log_freq: int = 10_000,
    graphs_dir: str = "graphs",
    seed: int = 0,
    use_compile: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    env = make_env(env_id, seed)
    H, W = env.observation_space.shape          # (84, 84)
    n_actions: int = env.action_space.n

    # Keep the *raw* modules around so state_dict transfer works regardless of
    # whether the network has been wrapped by torch.compile.
    online_raw = build_qnetwork(n_actions, n_frames).to(device)
    target_raw = build_qnetwork(n_actions, n_frames).to(device)
    target_raw.load_state_dict(online_raw.state_dict())
    target_raw.eval()
    for p in target_raw.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(online_raw.parameters(), lr=learning_rate)

    online_net, target_net = online_raw, target_raw
    if use_compile and hasattr(torch, "compile"):
        try:
            online_net = torch.compile(online_raw, mode="default")
            target_net = torch.compile(target_raw, mode="default")
        except Exception as e:
            print(f"torch.compile disabled: {e}")
            online_net, target_net = online_raw, target_raw

    buffer = LazyStackReplay(buffer_capacity, (H, W), n_frames, device)
    if density_model == "cts":
        from cts import make_cts_bonus
        bonus_fn = make_cts_bonus((n_frames, H, W), beta=beta)
    else:
        bonus_fn = make_pixelcnn_bonus((n_frames, H, W), beta=beta)
    print(f"Density model: {density_model}")

    # Reusable host-side scratch buffers (no per-step allocation in the hot loop)
    pin       = device.type == "cuda"
    obs_pin   = torch.empty((1, n_frames, H, W), dtype=torch.uint8, pin_memory=pin)
    stack_u8  = np.empty((n_frames, H, W), dtype=np.uint8)
    bonus_buf = np.empty((n_frames, H, W), dtype=np.float32)
    inv255    = np.float32(1.0 / 255.0)

    # Frame deque for action-time stacking (stores defensive copies)
    deque_frames: deque = deque(maxlen=n_frames)

    def reset_deque(frame_u8: np.ndarray) -> None:
        safe = frame_u8.copy()
        deque_frames.clear()
        for _ in range(n_frames):
            deque_frames.append(safe)

    def fill_stack(out: np.ndarray) -> None:
        for i, f in enumerate(deque_frames):
            out[i] = f

    # Episode logging
    ep_returns:   List[float] = []
    ep_intrinsic: List[float] = []
    ep_return = 0.0
    ep_intr   = 0.0

    f0, _ = env.reset(seed=seed)
    reset_deque(f0)

    for step in range(1, total_steps + 1):
        # --- Action selection -------------------------------------------------
        frac    = min(1.0, step / epsilon_decay_steps)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            fill_stack(stack_u8)
            obs_pin[0].copy_(torch.from_numpy(stack_u8))
            with torch.no_grad():
                obs_t  = obs_pin.to(device, non_blocking=True).float().mul_(inv255)
                action = int(online_net(obs_t).argmax(dim=1).item())

        # --- Environment step -------------------------------------------------
        before_action_frame = deque_frames[-1]   # safe ref; never mutated in place
        next_frame, extrinsic, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Build the post-step stack for the bonus: oldest n-1 frames + new frame
        for i in range(n_frames - 1):
            stack_u8[i] = deque_frames[i + 1]
        stack_u8[-1] = next_frame
        bonus_buf[:] = stack_u8                  # uint8 -> float32 cast
        bonus_buf *= inv255
        intr_bonus, _ = bonus_fn.bonus_and_update(bonus_buf)

        # Store transition (the agent stored what it actually saw before acting)
        buffer.add(before_action_frame, action, float(extrinsic) + intr_bonus, done)

        ep_return += float(extrinsic)
        ep_intr   += intr_bonus

        if done:
            ep_returns.append(ep_return)
            ep_intrinsic.append(ep_intr)
            ep_return = 0.0
            ep_intr   = 0.0
            f0, _ = env.reset()
            reset_deque(f0)
        else:
            deque_frames.append(next_frame.copy())

        # --- Train ------------------------------------------------------------
        if step >= train_start and step % train_freq == 0 and len(buffer) > n_frames:
            obs_b, act_b, rew_b, nxt_b, done_b = buffer.sample(batch_size)
            with torch.no_grad():
                next_q = target_net(nxt_b).max(dim=1).values
                target = rew_b + gamma * next_q * (1.0 - done_b)
            current_q = online_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current_q, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_raw.parameters(), 10.0)
            optimizer.step()

        if step % target_update_freq == 0:
            target_raw.load_state_dict(online_raw.state_dict())

        if step % log_freq == 0 and ep_returns:
            avg_ext  = float(np.mean(ep_returns[-20:]))
            avg_intr = float(np.mean(ep_intrinsic[-20:]))
            print(
                f"step={step:>8,}  eps={epsilon:.3f}  "
                f"avg_extrinsic(20ep)={avg_ext:>8.2f}  "
                f"avg_intrinsic(20ep)={avg_intr:.4f}"
            )

    env.close()
    _plot(ep_returns, ep_intrinsic, env_id, graphs_dir)


# ============================================================================
# Plotting
# ============================================================================

def _plot(returns, intrinsic, env_id, graphs_dir, window=20):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping graph")
        return

    os.makedirs(graphs_dir, exist_ok=True)
    eps = np.arange(1, len(returns) + 1)

    def roll(data):
        arr = np.array(data, dtype=np.float32)
        k = np.ones(window) / window
        return np.convolve(np.pad(arr, (window - 1, 0), mode="edge"), k, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    safe = env_id.replace("/", "_").replace("-", "_")
    fig.suptitle(f"DQN + Pseudo-Count - {env_id}", fontsize=13)

    ax1.plot(eps, returns,      alpha=0.2, color="steelblue",  linewidth=0.6)
    ax1.plot(eps, roll(returns), color="steelblue",  linewidth=1.8, label=f"{window}-ep avg")
    ax1.set_ylabel("Extrinsic return"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(eps, intrinsic,      alpha=0.2, color="darkorange", linewidth=0.6)
    ax2.plot(eps, roll(intrinsic), color="darkorange", linewidth=1.8, label=f"{window}-ep avg")
    ax2.set_ylabel("Intrinsic bonus (sum)"); ax2.set_xlabel("Episode")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(graphs_dir, f"{safe}_{len(returns)}ep.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Graph saved to {path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   default="ALE/Freeway-v5")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--beta",  type=float, default=0.10)
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--density", choices=["pixelcnn", "cts"], default="pixelcnn")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (use it if you hit a compile-time error).")
    args = parser.parse_args()

    train(
        env_id=args.env,
        total_steps=args.steps,
        beta=args.beta,
        seed=args.seed,
        density_model=args.density,
        use_compile=not args.no_compile,
    )