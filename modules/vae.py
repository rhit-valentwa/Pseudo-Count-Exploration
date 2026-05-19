"""
Variational Autoencoder novelty bonus for Atari frame stacks.

Drop this file in:
    modules/vae.py

Expected use from atari_dqn.py:
    from modules.vae import make_vae_bonus
    bonus_fn = make_vae_bonus((n_frames, h, w), beta=0.05)

The public API mirrors the other density modules:
    bonus, info = bonus_fn.bonus_and_update(obs)

`obs` may be:
    - uint8 image frame:        (H, W), values 0..255
    - uint8/float frame stack:  (C, H, W)
    - batched stack:            (B, C, H, W)

For your current DQN script, use it on the post-step frame stack, like PixelCNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariConvVAE(nn.Module):
    """Small convolutional VAE for 84x84 Atari frame stacks."""

    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int = 128) -> None:
        super().__init__()
        c, h, w = input_shape
        self.input_shape = (int(c), int(h), int(w))
        self.latent_dim = int(latent_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            encoded = self.encoder(dummy)
            self._encoded_shape = tuple(encoded.shape[1:])
            flat_dim = int(encoded.flatten(1).shape[1])

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, c, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(z.shape[0], *self._encoded_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class VAEBonusConfig:
    input_shape: Tuple[int, int, int]
    beta: float = 0.05
    latent_dim: int = 128
    lr: float = 1e-4
    kl_weight: float = 1e-3
    grad_clip: float = 10.0
    bonus_clip: float = 5.0
    normalize_bonus: bool = True
    ema_decay: float = 0.99
    device: str | None = None


class VAEBonus:
    """
    Online VAE novelty model.

    Intrinsic reward is based on reconstruction error. By default it is
    normalized with an exponential moving mean/variance so the reward scale is
    less brittle during training.
    """

    def __init__(self, config: VAEBonusConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = AtariConvVAE(config.input_shape, config.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.steps = 0
        self.err_mean = 0.0
        self.err_sq_mean = 0.0

    def _to_batch(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs)
        elif torch.is_tensor(obs):
            x = obs.detach().cpu()
        else:
            raise TypeError(f"obs must be a numpy array or torch tensor, got {type(obs)!r}")

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"obs must have shape (H,W), (C,H,W), or (B,C,H,W); got {tuple(x.shape)}")

        x = x.to(self.device, dtype=torch.float32, non_blocking=True)

        if float(x.max().item()) > 1.5:
            x = x / 255.0

        return x.clamp_(0.0, 1.0)

    def _normalize_error(self, error: float) -> float:
        cfg = self.config
        self.steps += 1

        if self.steps == 1:
            self.err_mean = error
            self.err_sq_mean = error * error
            return 1.0

        d = cfg.ema_decay
        self.err_mean = d * self.err_mean + (1.0 - d) * error
        self.err_sq_mean = d * self.err_sq_mean + (1.0 - d) * error * error

        variance = max(self.err_sq_mean - self.err_mean * self.err_mean, 1e-8)
        std = variance ** 0.5
        z = (error - self.err_mean) / std
        return max(0.0, z)

    def bonus_and_update(self, obs: np.ndarray | torch.Tensor) -> Tuple[float, Dict[str, float]]:
        x = self._to_batch(obs)

        self.model.train()
        recon, mu, logvar = self.model(x)

        recon_error_per = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
        recon_error = recon_error_per.mean()

        clipped_logvar = torch.clamp(logvar, -10.0, 10.0)
        kl_per = -0.5 * torch.sum(
            1.0 + clipped_logvar - mu.pow(2) - torch.exp(clipped_logvar),
            dim=1,
        )
        kl = kl_per.mean() / float(np.prod(self.config.input_shape))

        loss = recon_error + self.config.kl_weight * kl

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        raw_error = float(recon_error.detach().item())
        novelty = self._normalize_error(raw_error) if self.config.normalize_bonus else raw_error
        novelty = min(novelty, self.config.bonus_clip)
        bonus = float(self.config.beta * novelty)

        info = {
            "vae_bonus": bonus,
            "vae_recon_error": raw_error,
            "vae_kl": float(kl.detach().item()),
            "vae_loss": float(loss.detach().item()),
            "vae_steps": float(self.steps),
        }
        return bonus, info

    def state_dict(self) -> Dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "err_mean": self.err_mean,
            "err_sq_mean": self.err_sq_mean,
            "config": self.config,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state.get("steps", 0))
        self.err_mean = float(state.get("err_mean", 0.0))
        self.err_sq_mean = float(state.get("err_sq_mean", 0.0))


def make_vae_bonus(
    input_shape: Tuple[int, int, int],
    beta: float = 0.05,
    latent_dim: int = 128,
    lr: float = 1e-4,
    kl_weight: float = 1e-3,
    device: str | None = None,
) -> VAEBonus:
    """Factory matching the style of the CTS/PixelCNN bonus modules."""
    config = VAEBonusConfig(
        input_shape=tuple(int(v) for v in input_shape),
        beta=float(beta),
        latent_dim=int(latent_dim),
        lr=float(lr),
        kl_weight=float(kl_weight),
        device=device,
    )
    return VAEBonus(config)


class VAEPredictionGainBonus:
    """
    Prediction-gain wrapper for the online VAE.

    Computes log-prob proxy = -loss, performs one training step, and
    returns bonus = beta * sqrt(max(lp_after - lp_before, 0)).
    """

    def __init__(self, vae: VAEBonus, beta: float = 0.05):
        self.vae = vae
        self.beta = float(beta)

    def bonus_and_update(self, obs: np.ndarray | torch.Tensor):
        # Compute loss before update
        x = self.vae._to_batch(obs)

        self.vae.model.eval()
        with torch.no_grad():
            recon, mu, logvar = self.vae.model(x)
            recon_error_per = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
            recon_error = recon_error_per.mean()
            clipped_logvar = torch.clamp(logvar, -10.0, 10.0)
            kl_per = -0.5 * torch.sum(
                1.0 + clipped_logvar - mu.pow(2) - torch.exp(clipped_logvar),
                dim=1,
            )
            kl = kl_per.mean() / float(np.prod(self.vae.config.input_shape))
            loss_before = float((recon_error + self.vae.config.kl_weight * kl).detach().item())

        # One training step (in-place)
        self.vae.model.train()
        recon, mu, logvar = self.vae.model(x)
        recon_error_per = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
        recon_error = recon_error_per.mean()
        clipped_logvar = torch.clamp(logvar, -10.0, 10.0)
        kl_per = -0.5 * torch.sum(
            1.0 + clipped_logvar - mu.pow(2) - torch.exp(clipped_logvar),
            dim=1,
        )
        kl = kl_per.mean() / float(np.prod(self.vae.config.input_shape))

        loss = recon_error + self.vae.config.kl_weight * kl
        self.vae.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.vae.config.grad_clip and self.vae.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.vae.model.parameters(), self.vae.config.grad_clip)
        self.vae.optimizer.step()

        # Measure after-step loss
        self.vae.model.eval()
        with torch.no_grad():
            recon2, mu2, logvar2 = self.vae.model(x)
            recon_error_per2 = F.mse_loss(recon2, x, reduction="none").flatten(1).mean(dim=1)
            recon_error2 = recon_error_per2.mean()
            clipped_logvar2 = torch.clamp(logvar2, -10.0, 10.0)
            kl_per2 = -0.5 * torch.sum(
                1.0 + clipped_logvar2 - mu2.pow(2) - torch.exp(clipped_logvar2),
                dim=1,
            )
            kl2 = kl_per2.mean() / float(np.prod(self.vae.config.input_shape))
            loss_after = float((recon_error2 + self.vae.config.kl_weight * kl2).detach().item())

        # Prediction gain in log-prob proxy (-loss)
        pg = max(0.0, -loss_after - (-loss_before))
        bonus = float(self.beta * (pg ** 0.5))

        info = {
            "vae_loss": loss_after,
            "vae_recon_error": float(recon_error2.detach().item()),
            "vae_kl": float(kl2.detach().item()),
            "prediction_gain": pg,
        }
        return bonus, info


def make_vae_pg_bonus(
    input_shape: Tuple[int, int, int],
    beta: float = 0.05,
    latent_dim: int = 128,
    lr: float = 1e-4,
    kl_weight: float = 1e-3,
    device: str | None = None,
) -> VAEPredictionGainBonus:
    cfg = VAEBonusConfig(
        input_shape=tuple(int(v) for v in input_shape),
        beta=float(beta),
        latent_dim=int(latent_dim),
        lr=float(lr),
        kl_weight=float(kl_weight),
        device=device,
    )
    vae = VAEBonus(cfg)
    return VAEPredictionGainBonus(vae, beta=beta)