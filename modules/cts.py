from __future__ import annotations

from typing import Tuple


def make_cts_bonus(obs_shape: Tuple[int, ...], beta: float = 0.05):
    """
    Compatibility fallback for `make_cts_bonus` when the original
    Cython-backed CTS implementation is not available.

    This returns a VAE-based novelty bonus configured to accept single
    frames (shape (1, H, W)). The returned object exposes the same
    `bonus_and_update(obs)` API used by the rest of the codebase.
    """
    from modules.vae import make_vae_bonus

    h, w = obs_shape
    return make_vae_bonus((1, int(h), int(w)), beta=beta)
