"""
squish/knapspec.py

KnapSpec — Training-Free Self-Speculative Decoding via Knapsack-Optimal
Layer Selection.

Based on:
  "KnapSpec: Efficient Self-Speculative Decoding via Knapsack-Optimal Layer
   Selection" — Wu et al. (2026), arXiv:2602.20217

Key insight
-----------
Attention layers and MLP layers have different latency profiles that vary
with context length:

  • Short context (<4K):  attention is cheap (quadratic but small n),
                          MLP is the bottleneck.
  • Long context (>32K):  attention is expensive, MLP relatively cheap.

KnapSpec decouples Attention and MLP blocks and models their
hardware-specific latencies as functions of context length.  It then
solves a 0/1 knapsack problem to find the subset of blocks that maximises
draft quality (acceptance-rate proxy) while staying within a latency budget.

Quality proxy (from paper)
--------------------------
Cosine similarity between hidden states at the draft exit and the final
layer.  Here we model this additively: each included block contributes a
fixed quality increment (uniform weights by default; override to taste).

Provides
--------
  KnapSpecConfig      — latency-model parameters + budget setting.
  KnapSpecSelector    — 0/1 knapsack DP for optimal block selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "KnapSpecConfig",
    "KnapSpecSelector",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KnapSpecConfig:
    """Configuration for KnapSpec layer-selection.

    Parameters
    ----------
    num_layers : int
        Total transformer layers (N).  Produces 2*N blocks
        (attention_0, mlp_0, attention_1, mlp_1, …).
    attn_base_latency : float
        Base latency of each attention block in arbitrary time units (≥ 0).
    attn_context_coeff : float
        Additional latency added per context token to each attention block
        (models the O(n) KV-read cost; ≥ 0).
    mlp_latency : float
        Latency of each MLP block (constant w.r.t. context length; ≥ 0).
    budget_fraction : float
        Draft latency budget as a fraction of the full-model latency.
        Must be in (0, 1].  Full model = budget_fraction 1.0.
    dp_resolution : int
        Number of discrete budget bins used by the DP solver.  Higher →
        more accurate but slower.  Must be ≥ 1.
    """

    num_layers:         int   = 32
    attn_base_latency:  float = 1.0
    attn_context_coeff: float = 0.001
    mlp_latency:        float = 1.5
    budget_fraction:    float = 0.5
    dp_resolution:      int   = 200

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")
        if self.attn_base_latency < 0:
            raise ValueError("attn_base_latency must be ≥ 0")
        if self.attn_context_coeff < 0:
            raise ValueError("attn_context_coeff must be ≥ 0")
        if self.mlp_latency < 0:
            raise ValueError("mlp_latency must be ≥ 0")
        if not (0 < self.budget_fraction <= 1.0):
            raise ValueError("budget_fraction must be in (0, 1]")
        if self.dp_resolution < 1:
            raise ValueError("dp_resolution must be ≥ 1")


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class KnapSpecSelector:
    """Finds the optimal draft-model block configuration via knapsack DP.

    Each block is either an attention block or an MLP block for a given
    transformer layer.  Blocks are indexed as::

        block 2*i   → attention of layer i
        block 2*i+1 → MLP of layer i

    The selector returns two lists of layer indices:

        attn_keep — layers whose attention block should be run in the draft.
        mlp_keep  — layers whose MLP block should be run in the draft.

    Layers absent from a list are *skipped* (identity / residual pass-through).
    """

    def __init__(self, config: KnapSpecConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def full_model_latency(self, context_len: int) -> float:
        """Total latency of the full model at *context_len* tokens."""
        cfg = self._cfg
        attn_cost = cfg.attn_base_latency + cfg.attn_context_coeff * context_len
        return cfg.num_layers * (attn_cost + cfg.mlp_latency)

    def block_costs(self, context_len: int) -> np.ndarray:
        """Return per-block latency costs as a ``(2*num_layers,)`` array.

        Layout: ``[attn_0, mlp_0, attn_1, mlp_1, …]``.
        """
        cfg = self._cfg
        attn_cost = cfg.attn_base_latency + cfg.attn_context_coeff * context_len
        costs = np.empty(2 * cfg.num_layers, dtype=np.float64)
        costs[0::2] = attn_cost      # attention blocks
        costs[1::2] = cfg.mlp_latency  # MLP blocks
        return costs

    # ------------------------------------------------------------------
    # Main selection
    # ------------------------------------------------------------------

    def select(
        self,
        context_len: int,
        quality_weights: Optional[np.ndarray] = None,
    ) -> tuple[list[int], list[int]]:
        """Select which blocks to keep in the draft model.

        Parameters
        ----------
        context_len : int
            Current context length (influences attention block latency).
        quality_weights : (2*num_layers,) float array, optional
            Per-block quality contribution.  Defaults to uniform 1.0.

        Returns
        -------
        (attn_keep, mlp_keep) where each is a sorted list of layer indices.
        The complement (layers NOT in the list) should be skipped.
        """
        cfg = self._cfg
        n = cfg.num_layers

        costs = self.block_costs(context_len)
        quality: np.ndarray = (
            np.ones(2 * n, dtype=np.float64)
            if quality_weights is None
            else np.asarray(quality_weights, dtype=np.float64)
        )

        total_cost = costs.sum()

        # ── Fast paths ────────────────────────────────────────────────────────
        if total_cost <= 0:
            # Degenerate: zero-cost model — keep everything
            return list(range(n)), list(range(n))

        budget = total_cost * cfg.budget_fraction

        if budget >= total_cost - 1e-9:
            # Budget covers full model
            return list(range(n)), list(range(n))

        if budget <= 0:
            # No budget — skip everything
            return [], []

        # ── 0/1 Knapsack DP ──────────────────────────────────────────────────
        res = cfg.dp_resolution
        scale = res / total_cost
        int_costs = np.maximum(1, np.round(costs * scale).astype(int))
        budget_bins = max(1, int(round(budget * scale)))

        n_items = 2 * n
        dp = np.zeros(budget_bins + 1, dtype=np.float64)
        chose = np.zeros((n_items, budget_bins + 1), dtype=bool)

        for i in range(n_items):
            c = int(int_costs[i])
            q = float(quality[i])
            for b in range(budget_bins, c - 1, -1):
                candidate = dp[b - c] + q
                if candidate > dp[b]:
                    dp[b] = candidate
                    chose[i, b] = True

        # ── Traceback ─────────────────────────────────────────────────────────
        selected: set[int] = set()
        b = budget_bins
        for i in range(n_items - 1, -1, -1):
            c = int(int_costs[i])
            if b >= c and chose[i, b]:
                selected.add(i)
                b -= c

        attn_keep = sorted(idx // 2 for idx in selected if idx % 2 == 0)
        mlp_keep  = sorted((idx - 1) // 2 for idx in selected if idx % 2 == 1)
        return attn_keep, mlp_keep
