"""
PixelCNN density model for pseudo-count exploration.

Uses prediction-gain bonus (log-space) instead of raw probabilities,
because joint image probabilities are numerically too small to store as floats.

    PG(x) = log rho'(x) - log rho(x)   (improvement in log-likelihood after one gradient step)
    bonus  = beta * sqrt(max(PG, 0))

Plug into atari_dqn.py by replacing the bonus_fn line:

    # old
    bonus_fn = make_image_bonus(obs_shape, beta=beta)

    # new
    from pixel_cnn import make_pixelcnn_bonus
    bonus_fn = make_pixelcnn_bonus(obs_shape, beta=beta)

Install: pip install torch   (already required by atari_dqn.py)
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Number of quantization levels per pixel. 8 keeps the softmax small while
# retaining enough resolution for pseudo-count discrimination.
N_LEVELS = 8


# ---------------------------------------------------------------------------
# Masked convolution
# ---------------------------------------------------------------------------

class MaskedConv2d(nn.Conv2d):
    """Conv2d with raster-scan autoregressive mask."""

    def __init__(self, mask_type: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        kh, kw = self.weight.shape[2:]
        mask = torch.zeros_like(self.weight)
        mask[:, :, :kh // 2] = 1
        mask[:, :, kh // 2, :kw // 2] = 1
        if mask_type == "B":
            mask[:, :, kh // 2, kw // 2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.mask, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


# ---------------------------------------------------------------------------
# PixelCNN model
# ---------------------------------------------------------------------------

class PixelCNNModel(nn.Module):
    """
    Small PixelCNN for single-channel 84x84 frames.
    Outputs per-pixel logits over N_LEVELS quantization bins.
    """

    def __init__(
        self,
        n_levels: int = N_LEVELS,
        n_filters: int = 64,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            MaskedConv2d("A", 1, n_filters, kernel_size=7, padding=3),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers += [
                MaskedConv2d("B", n_filters, n_filters, kernel_size=7, padding=3),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(n_filters, n_levels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, n_levels, H, W)

    def log_prob(self, x_float: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
        """Mean per-pixel log-likelihood (scalar)."""
        logits = self.forward(x_float)
        return -F.cross_entropy(logits, x_q, reduction="mean")


# ---------------------------------------------------------------------------
# Density model wrapper
# ---------------------------------------------------------------------------

class PixelCNNDensity:
    """
    Wraps PixelCNNModel as a density model for use with PredictionGainBonus.

    obs_shape should be the full stacked-frame shape, e.g. (4, 84, 84).
    Only the most recent frame (last channel) is modelled, which is fast and
    sufficient for novelty detection — consecutive frames are highly correlated.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_levels: int = N_LEVELS,
        n_filters: int = 64,
        n_layers: int = 4,
        lr: float = 5e-4,
        device: str | None = None,
    ) -> None:
        self.n_levels = n_levels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.h, self.w = obs_shape[-2], obs_shape[-1]
        self.model = PixelCNNModel(n_levels, n_filters, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _prepare(self, obs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[-1]  # most recent frame from stack
        x_f = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)     # (1,1,H,W)
        x_q = torch.from_numpy(
            (arr * (self.n_levels - 1)).astype(np.int64)
        ).unsqueeze(0).to(self.device)                                             # (1,H,W)
        return x_f, x_q

    def log_prob(self, obs: Any) -> float:
        x_f, x_q = self._prepare(obs)
        with torch.no_grad():
            return float(self.model.log_prob(x_f, x_q).item())

    def log_prob_after_step(self, obs: Any) -> Tuple[float, float]:
        """
        Return (log_prob_before, log_prob_after) for one gradient step on obs.
        The model is actually updated — call this in place of separate log_prob + update.
        """
        x_f, x_q = self._prepare(obs)
        with torch.no_grad():
            lp_before = float(self.model.log_prob(x_f, x_q).item())
        self.model.train()
        loss = -self.model.log_prob(x_f, x_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            lp_after = float(self.model.log_prob(x_f, x_q).item())
        return lp_before, lp_after


# ---------------------------------------------------------------------------
# Bonus
# ---------------------------------------------------------------------------

class PredictionGainBonus:
    """
    Intrinsic bonus via prediction gain (log-space pseudo-count).

        PG(x)  = log rho'(x) - log rho(x)
        bonus  = beta * sqrt(max(PG, 0))

    High PG  → model improved a lot → state was novel → large bonus.
    Low PG   → model barely improved → state is familiar → small bonus.
    """

    def __init__(self, density: PixelCNNDensity, beta: float = 0.05) -> None:
        self.density = density
        self.beta = beta

    def bonus_and_update(self, obs: Any) -> Tuple[float, float]:
        """Return (bonus, prediction_gain) and update the density model."""
        lp_before, lp_after = self.density.log_prob_after_step(obs)
        pg = max(0.0, lp_after - lp_before)
        return self.beta * pg ** 0.5, pg


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_pixelcnn_bonus(
    obs_shape: Tuple[int, ...],
    beta: float = 0.05,
    n_filters: int = 64,
    n_layers: int = 4,
    device: str | None = None,
) -> PredictionGainBonus:
    """
    Drop-in replacement for make_image_bonus() from main.py.

    Example in atari_dqn.py:
        from pixel_cnn import make_pixelcnn_bonus
        bonus_fn = make_pixelcnn_bonus(obs_shape, beta=beta)
    """
    density = PixelCNNDensity(obs_shape, n_filters=n_filters, n_layers=n_layers, device=device)
    return PredictionGainBonus(density, beta=beta)
