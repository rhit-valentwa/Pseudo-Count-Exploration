from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from modules.cts_core import cts_log_prob_update


N_LEVELS = 8
OUT_H = 42
OUT_W = 42
N_LOCS = OUT_H * OUT_W
N_NODES = 4681


class CTSDensity:
    """
    Cython-backed full recursive CTS-switching Atari density model.

    Uses:
      - single Atari frame
      - 42x42 downsampled grayscale
      - 3-bit pixels, values 0..7
      - one location-dependent CTS model per pixel
      - four parents: (i-1,j), (i,j-1), (i-1,j-1), (i+1,j-1)
    """

    def __init__(self, obs_shape: Tuple[int, ...]) -> None:
        self.obs_shape = obs_shape

        self.counts = np.zeros((N_LOCS, N_NODES, N_LEVELS), dtype=np.uint32)
        self.totals = np.zeros((N_LOCS, N_NODES), dtype=np.uint32)
        self.node_updates = np.zeros((N_LOCS, N_NODES), dtype=np.uint32)

        self.w_base = np.empty((N_LOCS, N_NODES), dtype=np.float32)
        self.w_split = np.empty((N_LOCS, N_NODES), dtype=np.float32)
        self._init_weights()

    def _init_weights(self) -> None:
        self.w_base.fill(0.5)
        self.w_split.fill(0.5)

        # Leaf nodes are depth 4 and cannot split.
        leaf_start = 585
        self.w_base[:, leaf_start:] = 1.0
        self.w_split[:, leaf_start:] = 0.0

    def _preprocess(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)

        # If a stack sneaks in, use newest only.
        if arr.ndim == 3:
            arr = arr[-1]

        if arr.ndim != 2:
            raise ValueError(f"CTS expected 2D frame or stacked frame; got shape {arr.shape}")

        # Fast path: Gymnasium AtariPreprocessing gives 84x84 uint8.
        if arr.dtype == np.uint8 and arr.shape == (84, 84):
            pooled = arr.reshape(OUT_H, 2, OUT_W, 2).mean(axis=(1, 3))
            q = np.floor(pooled / 32.0).astype(np.int64)
            return np.clip(q, 0, N_LEVELS - 1)

        frame = arr.astype(np.float32, copy=False)
        if frame.max() > 1.5:
            frame = frame * (1.0 / 255.0)
        frame = np.clip(frame, 0.0, 1.0)

        h, w = frame.shape
        if (h, w) == (OUT_H, OUT_W):
            small = frame
        elif h % OUT_H == 0 and w % OUT_W == 0:
            sh = h // OUT_H
            sw = w // OUT_W
            small = frame.reshape(OUT_H, sh, OUT_W, sw).mean(axis=(1, 3))
        else:
            ys = np.linspace(0, h - 1, OUT_H).astype(np.int64)
            xs = np.linspace(0, w - 1, OUT_W).astype(np.int64)
            small = frame[np.ix_(ys, xs)]

        q = np.floor(small * N_LEVELS).astype(np.int64)
        return np.clip(q, 0, N_LEVELS - 1)

    def log_prob_after_step(self, obs: Any):
        frame = self._preprocess(obs)
        return cts_log_prob_update(
            frame,
            self.counts,
            self.totals,
            self.node_updates,
            self.w_base,
            self.w_split,
        )

def make_cts_bonus(obs_shape: Tuple[int, ...], beta: float = 0.05):
    from modules.pixel_cnn import PredictionGainBonus

    density = CTSDensity(obs_shape)
    return PredictionGainBonus(density, beta=beta)