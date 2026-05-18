from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionedForwardModel(nn.Module):
    def __init__(self, n_actions: int, n_frames: int = 4) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_frames = int(n_frames)

        self.action_embed = nn.Embedding(self.n_actions, 84 * 84)

        self.net = nn.Sequential(
            nn.Conv2d(n_frames + 1, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4),
        )

    def forward(self, obs_stack: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b = obs_stack.shape[0]
        action_plane = self.action_embed(actions).view(b, 1, 84, 84)
        x = torch.cat([obs_stack, action_plane], dim=1)
        return self.net(x)[..., :84, :84]


class ActionAgnosticForwardModel(nn.Module):
    def __init__(self, n_frames: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4),
        )

    def forward(self, obs_stack: torch.Tensor) -> torch.Tensor:
        return self.net(obs_stack)[..., :84, :84]


class ActionAdvantageTemporalBonus:
    """
    Temporal bonus based on action usefulness for prediction.

    Trains two models:
      1. action-conditioned: p(x_{t+1} | stack_t, a_t)
      2. action-agnostic:    p(x_{t+1} | stack_t)

    Bonus:
      beta * max(error_agnostic - error_conditioned, 0)

    This suppresses reward for background motion that action does not explain.
    """

    def __init__(
        self,
        n_actions: int,
        n_frames: int = 4,
        beta: float = 0.01,
        lr: float = 1e-4,
        device: torch.device | str | None = None,
        train_every: int = 1,
        grad_clip: float = 10.0,
        loss_type: str = "bce",
        normalize_bonus: bool = True,
        ema_decay: float = 0.99,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.n_frames = int(n_frames)
        self.beta = float(beta)
        self.train_every = int(train_every)
        self.grad_clip = float(grad_clip)
        self.loss_type = loss_type
        self.normalize_bonus = bool(normalize_bonus)
        self.ema_decay = float(ema_decay)
        self.step = 0

        self.conditioned = ActionConditionedForwardModel(
            n_actions=n_actions,
            n_frames=n_frames,
        ).to(self.device)

        self.agnostic = ActionAgnosticForwardModel(
            n_frames=n_frames,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.conditioned.parameters()) + list(self.agnostic.parameters()),
            lr=lr,
        )

        self.adv_ema: float | None = None
        self.adv2_ema: float | None = None

    def _preprocess_stack(self, obs_stack_u8: Any) -> torch.Tensor:
        arr = np.asarray(obs_stack_u8)

        if arr.ndim != 3:
            raise ValueError(f"Expected stack shape (n_frames,H,W), got {arr.shape}")

        if arr.shape[0] != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {arr.shape[0]}")

        if arr.shape[-2:] != (84, 84):
            raise ValueError(f"Expected 84x84 frames, got {arr.shape[-2:]}")

        if arr.dtype == np.uint8:
            x_np = arr.astype(np.float32) / 255.0
        else:
            x_np = arr.astype(np.float32)
            if x_np.max() > 1.5:
                x_np *= 1.0 / 255.0
            x_np = np.clip(x_np, 0.0, 1.0)

        x = torch.from_numpy(x_np).to(self.device)
        return x.unsqueeze(0)

    def _preprocess_next(self, next_frame_u8: Any) -> torch.Tensor:
        arr = np.asarray(next_frame_u8)

        if arr.ndim == 3:
            arr = arr[-1]

        if arr.ndim != 2:
            raise ValueError(f"Expected next frame shape (H,W), got {arr.shape}")

        if arr.shape != (84, 84):
            raise ValueError(f"Expected 84x84 next frame, got {arr.shape}")

        if arr.dtype == np.uint8:
            y_np = arr.astype(np.float32) / 255.0
        else:
            y_np = arr.astype(np.float32)
            if y_np.max() > 1.5:
                y_np *= 1.0 / 255.0
            y_np = np.clip(y_np, 0.0, 1.0)

        y = torch.from_numpy(y_np).to(self.device)
        return y.unsqueeze(0).unsqueeze(0)

    def _loss(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(torch.sigmoid(pred_logits), target, reduction="mean")

        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                pred_logits,
                target,
                reduction="mean",
            )

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _update_running_stats(self, value: float) -> Tuple[float, float]:
        if self.adv_ema is None:
            self.adv_ema = value
            self.adv2_ema = value * value
        else:
            d = self.ema_decay
            self.adv_ema = d * self.adv_ema + (1.0 - d) * value
            self.adv2_ema = d * self.adv2_ema + (1.0 - d) * value * value

        var = max(float(self.adv2_ema - self.adv_ema * self.adv_ema), 1e-8)
        return float(self.adv_ema), float(np.sqrt(var))

    def bonus_and_update(
        self,
        obs_stack_u8: Any,
        action: int,
        next_frame_u8: Any,
    ) -> Tuple[float, Dict[str, float]]:
        self.step += 1

        obs = self._preprocess_stack(obs_stack_u8)
        target = self._preprocess_next(next_frame_u8)
        act = torch.tensor([int(action)], dtype=torch.long, device=self.device)

        self.conditioned.eval()
        self.agnostic.eval()

        with torch.no_grad():
            pred_cond = self.conditioned(obs, act)
            pred_agn = self.agnostic(obs)

            err_cond = float(self._loss(pred_cond, target).detach().item())
            err_agn = float(self._loss(pred_agn, target).detach().item())

        raw_advantage = max(err_agn - err_cond, 0.0)

        mean, std = self._update_running_stats(raw_advantage)
        if self.normalize_bonus:
            normalized_advantage = max((raw_advantage - mean) / std, 0.0)
            bonus = self.beta * normalized_advantage
        else:
            normalized_advantage = raw_advantage
            bonus = self.beta * raw_advantage

        train_loss = np.nan
        if self.step % self.train_every == 0:
            self.conditioned.train()
            self.agnostic.train()

            pred_cond = self.conditioned(obs, act)
            pred_agn = self.agnostic(obs)

            loss_cond = self._loss(pred_cond, target)
            loss_agn = self._loss(pred_agn, target)

            # Train both models fairly on the same target.
            loss = loss_cond + loss_agn

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.conditioned.parameters()) + list(self.agnostic.parameters()),
                    self.grad_clip,
                )

            self.optimizer.step()
            train_loss = float(loss.detach().item())

        info = {
            "error_conditioned": float(err_cond),
            "error_agnostic": float(err_agn),
            "raw_advantage": float(raw_advantage),
            "normalized_advantage": float(normalized_advantage),
            "bonus": float(bonus),
            "train_loss": float(train_loss),
        }

        return float(bonus), info


def make_temporal_bonus(
    n_actions: int,
    n_frames: int = 4,
    beta: float = 0.01,
    lr: float = 1e-4,
    device: torch.device | str | None = None,
    train_every: int = 1,
    grad_clip: float = 10.0,
    loss_type: str = "bce",
    normalize_bonus: bool = True,
    ema_decay: float = 0.99,
) -> ActionAdvantageTemporalBonus:
    return ActionAdvantageTemporalBonus(
        n_actions=n_actions,
        n_frames=n_frames,
        beta=beta,
        lr=lr,
        device=device,
        train_every=train_every,
        grad_clip=grad_clip,
        loss_type=loss_type,
        normalize_bonus=normalize_bonus,
        ema_decay=ema_decay,
    )