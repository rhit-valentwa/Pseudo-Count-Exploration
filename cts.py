"""
Pixel-level Context Tree Switching (CTS) density model for pseudo-count exploration.

Based on the model described in Bellemare et al. 2016. Processes each image as a
raster-scan sequence of quantized pixels, modelling each pixel with a mixture of
Krichevsky-Trofimov (KT) estimators conditioned on contexts of increasing depth.

Key advantage over PixelCNN: rho(x) and rho'(x) have closed-form expressions from
the KT counts, so no gradient step is needed to compute the prediction gain.
Both values are computed analytically in a single sequential pass.

Usage in atari_dqn.py:
    python atari_dqn.py --density cts
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


N_LEVELS  = 8   # quantization bins per pixel
MAX_DEPTH = 4   # maximum context length (in preceding pixels)


# ---------------------------------------------------------------------------
# KT estimator node
# ---------------------------------------------------------------------------

class _KTNode:
    """
    Krichevsky-Trofimov estimator for one (depth, context) pair.

    P(symbol | history) = (count(symbol) + 0.5) / (total + n_levels * 0.5)

    The +0.5 prior gives good minimax regret over all symbol sequences.
    """

    __slots__ = ("counts", "total", "children")

    def __init__(self, n_levels: int) -> None:
        self.counts: np.ndarray = np.full(n_levels, 0.5)
        self.total: float = n_levels * 0.5
        self.children: Dict[int, "_KTNode"] = {}

    def prob(self, s: int) -> float:
        return float(self.counts[s] / self.total)

    def prob_after(self, s: int) -> float:
        """Probability that would result after observing s once more."""
        return float((self.counts[s] + 1.0) / (self.total + 1.0))

    def update(self, s: int) -> None:
        self.counts[s] += 1.0
        self.total      += 1.0


# ---------------------------------------------------------------------------
# CTS model
# ---------------------------------------------------------------------------

class _CTSModel:
    """
    Context Tree Switching mixture model for a discrete sequence.

    For each symbol x_t, blends KT estimators at depths 0 … max_depth:
        P_mix(x_t) = (1 / (max_depth+1)) * sum_d P_KT_d(x_t | ctx_d)

    Equal-weight mixing is simpler than full CTW but retains the key property:
    shallower contexts generalise across unseen deep contexts, while deeper
    contexts specialise when enough data is available.
    """

    def __init__(self, n_levels: int = N_LEVELS, max_depth: int = MAX_DEPTH) -> None:
        self.n_levels  = n_levels
        self.max_depth = max_depth
        self.root      = _KTNode(n_levels)

    def _node(self, ctx: Tuple[int, ...], create: bool) -> Optional[_KTNode]:
        node = self.root
        for c in ctx:
            if c not in node.children:
                if not create:
                    return None
                node.children[c] = _KTNode(self.n_levels)
            node = node.children[c]
        return node

    def _mixture(self, s: int, ctx: Tuple[int, ...], after: bool) -> float:
        """Mixture probability (equal weights) over depths 0 … min(max_depth, len(ctx))."""
        depths = min(self.max_depth, len(ctx))
        total  = 0.0
        for d in range(depths + 1):
            node = self._node(ctx[:d], create=False)
            if node is None:
                p = 1.0 / self.n_levels  # uniform for unseen context
            else:
                p = node.prob_after(s) if after else node.prob(s)
            total += p
        return total / (depths + 1)

    def step(self, s: int, ctx: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Return (log_prob_before, log_prob_after) then update all context nodes.
        Both values are computed from current counts — no rollback needed.
        """
        lp_before = np.log(max(self._mixture(s, ctx, after=False), 1e-30))
        lp_after  = np.log(max(self._mixture(s, ctx, after=True),  1e-30))
        depths = min(self.max_depth, len(ctx))
        for d in range(depths + 1):
            node = self._node(ctx[:d], create=True)
            node.update(s)
        return float(lp_before), float(lp_after)


# ---------------------------------------------------------------------------
# Density model (matching PixelCNNDensity's interface)
# ---------------------------------------------------------------------------

class CTSDensity:
    """
    CTS density model with the same interface as PixelCNNDensity:
    call log_prob_after_step(obs) to get (lp_before, lp_after) and update.

    obs_shape: stacked-frame shape, e.g. (4, 84, 84). Uses the last frame only.
    n_levels:  quantization bins. 8 keeps context tables small.
    max_depth: context length in pixels. 4 is a good default.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_levels:  int = N_LEVELS,
        max_depth: int = MAX_DEPTH,
    ) -> None:
        self.n_levels  = n_levels
        self.max_depth = max_depth
        self.h         = obs_shape[-2]
        self.w         = obs_shape[-1]
        self.model     = _CTSModel(n_levels, max_depth)

    def _quantize(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[-1]  # most recent frame from stack
        return (arr * (self.n_levels - 1)).astype(np.int32).ravel()

    def log_prob_after_step(self, obs: Any) -> Tuple[float, float]:
        """
        Process obs pixel by pixel in raster-scan order.
        Returns (sum_log_prob_before, sum_log_prob_after) and updates the model.
        No gradient computation — purely arithmetic on count tables.
        """
        pixels  = self._quantize(obs)
        ctx_buf: List[int] = []
        total_before = 0.0
        total_after  = 0.0

        for px in pixels:
            ctx = tuple(ctx_buf[-self.max_depth:])
            lb, la = self.model.step(int(px), ctx)
            total_before += lb
            total_after  += la
            ctx_buf.append(int(px))

        return total_before, total_after


# ---------------------------------------------------------------------------
# Factory (mirrors make_pixelcnn_bonus API)
# ---------------------------------------------------------------------------

def make_cts_bonus(
    obs_shape: Tuple[int, ...],
    beta:      float = 0.05,
    n_levels:  int   = N_LEVELS,
    max_depth: int   = MAX_DEPTH,
) -> "PredictionGainBonus":
    """
    Drop-in replacement for make_pixelcnn_bonus() from pixel_cnn.py.

    Usage in atari_dqn.py:
        from cts import make_cts_bonus
        bonus_fn = make_cts_bonus(obs_shape, beta=beta)
    """
    from pixel_cnn import PredictionGainBonus
    density = CTSDensity(obs_shape, n_levels=n_levels, max_depth=max_depth)
    return PredictionGainBonus(density, beta=beta)
