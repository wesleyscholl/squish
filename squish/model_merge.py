"""ModelMerge — SLERP / DARE / TIES model weight merging.

Three merging algorithms for combining specialist models without retraining:

* **SLERP** (Spherical Linear Interpolation) — interpolates between two weight
  tensors along the great circle of the unit hypersphere.  Preserves vector
  norms and is better than linear interpolation for high-dimensional weights.

* **DARE** (Drop And REscale) — randomly drops a fraction of each model's
  delta (fine-tuned - base) before merging, then rescales to compensate.
  Reduces interference between models at high merge density (Xiao et al., 2024).

* **TIES** (TrIm, Elect Sign and Merge) — trims small deltas, elects a sign
  per-parameter by majority vote across models, then merges only same-sign
  contributors (Yadav et al., NeurIPS 2023).

References:
    Yadav et al., "TIES-Merging: Resolving Interference When Merging Models",
    NeurIPS 2023.  https://arxiv.org/abs/2306.01708

    Xiao et al., "DARE: Language Model Weight Pruning via Layer-wise Drop and
    Rescale", arXiv:2311.03099, 2024.

    Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985.

Usage::

    from squish.model_merge import ModelMerger, MergeConfig

    cfg    = MergeConfig(method="slerp", t=0.5)
    merger = ModelMerger(cfg)
    merged = merger.merge({"w": w_a}, {"w": w_b})
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "MergeConfig",
    "MergeStats",
    "ModelMerger",
    "slerp",
    "dare_merge",
    "ties_merge",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for model weight merging.

    Parameters
    ----------
    method : str
        Merge algorithm: one of ``"slerp"``, ``"dare"``, ``"ties"``.
    t : float
        Interpolation parameter in ``[0, 1]``.  ``0`` → model_a only;
        ``1`` → model_b only.
    dare_density : float
        Fraction of delta parameters to *keep* in DARE merging, in ``(0, 1]``.
        Dropped parameters are rescaled by ``1/dare_density`` to preserve the
        expected delta magnitude.
    ties_k : float
        Top-k fraction of parameters to retain after magnitude trimming in
        TIES merging, in ``(0, 1]``.
    base_weights : dict, optional
        Reference base model weights for DARE and TIES methods, keyed by
        parameter name.  When ``None``, zero tensors are used as the base.
    """

    method: str = "slerp"
    t: float = 0.5
    dare_density: float = 0.5
    ties_k: float = 0.2
    base_weights: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.method not in ("slerp", "dare", "ties"):
            raise ValueError(
                f"method must be one of 'slerp', 'dare', 'ties'; got {self.method!r}"
            )
        if not (0.0 <= self.t <= 1.0):
            raise ValueError(f"t must be in [0, 1]; got {self.t}")
        if not (0.0 < self.dare_density <= 1.0):
            raise ValueError(
                f"dare_density must be in (0, 1]; got {self.dare_density}"
            )
        if not (0.0 < self.ties_k <= 1.0):
            raise ValueError(f"ties_k must be in (0, 1]; got {self.ties_k}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class MergeStats:
    """Aggregate statistics across all ``ModelMerger.merge()`` calls.

    Parameters
    ----------
    n_merges : int
        Total number of ``merge()`` calls performed.
    total_keys : int
        Total number of weight keys merged (summed across all calls).
    method : str
        The merge algorithm used.
    """

    n_merges: int
    total_keys: int
    method: str

    @property
    def avg_keys_per_merge(self) -> float:
        """Average number of keys merged per call."""
        if self.n_merges == 0:
            return 0.0
        return self.total_keys / self.n_merges


# ---------------------------------------------------------------------------
# Module-level merge functions
# ---------------------------------------------------------------------------


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation (SLERP) between two weight tensors.

    Interpolates along the great-circle arc between the direction of ``a`` and
    the direction of ``b`` on the unit hypersphere, scaled to preserve the
    interpolated magnitude.  Falls back to linear interpolation when the two
    vectors are nearly parallel (cosine similarity > 0.9995) to avoid numerical
    instability near the singularity at :math:`\\theta = 0`.

    Parameters
    ----------
    a, b : np.ndarray
        Weight tensors of identical shape.
    t : float
        Interpolation factor in ``[0, 1]``.  ``t=0`` returns a value aligned
        with ``a``; ``t=1`` returns a value aligned with ``b``.

    Returns
    -------
    np.ndarray
        Interpolated tensor with the same shape and dtype as ``a``.
    """
    shape = a.shape
    a_f = a.astype(np.float32).ravel()
    b_f = b.astype(np.float32).ravel()

    norm_a = float(np.linalg.norm(a_f))
    norm_b = float(np.linalg.norm(b_f))

    if norm_a < 1e-8 or norm_b < 1e-8:
        # At least one vector is essentially zero — fall back to lerp.
        result = (1.0 - t) * a_f + t * b_f
        return result.reshape(shape).astype(a.dtype)

    a_unit = a_f / norm_a
    b_unit = b_f / norm_b

    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))

    if dot > 0.9995:
        # Nearly parallel — lerp is numerically safe and effectively equivalent.
        result = (1.0 - t) * a_f + t * b_f
        return result.reshape(shape).astype(a.dtype)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    coeff_a = math.sin((1.0 - t) * theta) / sin_theta
    coeff_b = math.sin(t * theta) / sin_theta

    result = coeff_a * a_f + coeff_b * b_f
    return result.reshape(shape).astype(a.dtype)


def dare_merge(
    base: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    density: float,
    t: float,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """DARE (Drop And REscale) merge of two fine-tuned delta tensors.

    Independently drops ``(1 - density)`` fraction of each delta at random,
    then rescales the surviving parameters by ``1 / density`` to preserve their
    expected magnitude.  The two rescaled deltas are linearly interpolated at
    ratio ``t`` and added to ``base``.

    Parameters
    ----------
    base : np.ndarray
        Base model weights (e.g. the pre-trained checkpoint before fine-tuning).
    delta_a, delta_b : np.ndarray
        Per-parameter deltas: ``finetuned - base`` for each specialist model.
        Must have the same shape as ``base``.
    density : float
        Fraction of delta parameters to retain, in ``(0, 1]``.  Lower density
        reduces interference between the two models.
    t : float
        Linear interpolation weight; ``t=0`` → only ``delta_a`` contributes,
        ``t=1`` → only ``delta_b`` contributes.
    seed : int, optional
        Random seed for reproducible dropout masks.

    Returns
    -------
    np.ndarray
        Merged weights with the same shape and dtype as ``base``.
    """
    rng = np.random.default_rng(seed)

    mask_a = (rng.random(delta_a.shape) < density).astype(np.float32)
    mask_b = (rng.random(delta_b.shape) < density).astype(np.float32)

    delta_a_scaled = delta_a.astype(np.float32) * mask_a / density
    delta_b_scaled = delta_b.astype(np.float32) * mask_b / density

    merged_delta = (1.0 - t) * delta_a_scaled + t * delta_b_scaled
    return (base.astype(np.float32) + merged_delta).astype(base.dtype)


def ties_merge(
    base: np.ndarray,
    deltas: List[Tuple[np.ndarray, np.ndarray]],
    k: float,
) -> np.ndarray:
    """TIES (TrIm, Elect Sign and Merge) weight merging.

    Executes three steps:

    1. **Trim** — for each model delta, zero out the bottom ``(1 - k)``
       fraction by absolute magnitude, keeping only the top-k parameters.
    2. **Elect sign** — for each parameter position, choose the sign held by
       the majority of the (trimmed) contributing deltas.  Ties break to
       positive.
    3. **Merge** — average only the deltas whose sign at each position matches
       the elected sign, then add the merged delta to ``base``.

    Parameters
    ----------
    base : np.ndarray
        Reference base model weights shared by all fine-tuned models.
    deltas : list of (base_weight, finetuned_weight) pairs
        Each element is a ``(W_base, W_finetuned)`` pair.  The delta
        ``W_finetuned - W_base`` is computed internally.
    k : float
        Top-k fraction of parameters to retain after trimming, in ``(0, 1]``.

    Returns
    -------
    np.ndarray
        Merged weights with the same shape and dtype as ``base``.
    """
    if not deltas:
        return base.copy()

    base_f = base.astype(np.float32)

    # Step 0: compute per-model deltas.
    raw_deltas: List[np.ndarray] = [
        ft.astype(np.float32) - bw.astype(np.float32) for bw, ft in deltas
    ]

    # Step 1: Trim each delta to the top-k fraction by absolute magnitude.
    trimmed: List[np.ndarray] = []
    for d in raw_deltas:
        flat_abs = np.abs(d).ravel()
        n_total = len(flat_abs)
        n_keep = max(1, int(math.ceil(n_total * k)))
        if n_keep >= n_total:
            trimmed.append(d.copy())
            continue
        threshold = np.partition(flat_abs, -n_keep)[-n_keep]
        mask = (np.abs(d) >= threshold).astype(np.float32)
        trimmed.append(d * mask)

    # Step 2: Elect sign per-parameter via majority vote.
    sign_sum = sum(np.sign(d) for d in trimmed)
    elected_sign = np.sign(sign_sum)
    # Tie (sign_sum == 0) → positive by convention.
    elected_sign[elected_sign == 0.0] = 1.0

    # Step 3: Average same-sign contributors.
    merged_delta = np.zeros_like(base_f)
    contributor_count = np.zeros_like(base_f)
    for d in trimmed:
        same_sign = ((np.sign(d) == elected_sign) & (d != 0.0)).astype(np.float32)
        merged_delta += d * same_sign
        contributor_count += same_sign

    # Guard against positions with no contributor (leave base unchanged there).
    safe_count = np.where(contributor_count == 0.0, 1.0, contributor_count)
    merged_delta = merged_delta / safe_count

    return (base_f + merged_delta).astype(base.dtype)


# ---------------------------------------------------------------------------
# ModelMerger
# ---------------------------------------------------------------------------


class ModelMerger:
    """Orchestrates weight-level model merging using SLERP, DARE, or TIES.

    Parameters
    ----------
    config : MergeConfig
        Algorithm and hyperparameter configuration for all merge calls.

    Examples
    --------
    >>> cfg    = MergeConfig(method="slerp", t=0.5)
    >>> merger = ModelMerger(cfg)
    >>> merged = merger.merge({"fc.weight": w_a}, {"fc.weight": w_b})
    """

    def __init__(self, config: MergeConfig) -> None:
        self._config = config
        self._n_merged: int = 0
        self._total_keys: int = 0
        self._last_stats: Optional[Dict] = None

    def merge(
        self,
        weights_a: Dict[str, np.ndarray],
        weights_b: Dict[str, np.ndarray],
        base_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Merge two weight dictionaries.

        Keys present in *both* dicts are merged with the configured algorithm.
        Keys present in only one dict are copied through unchanged.

        Parameters
        ----------
        weights_a, weights_b : dict[str, np.ndarray]
            Weight dictionaries for the two models to merge.
        base_weights : dict, optional
            Override for ``config.base_weights``.  Used by DARE and TIES
            methods as the reference base.

        Returns
        -------
        dict[str, np.ndarray]
            Merged weight dictionary.
        """
        cfg = self._config
        base = (
            base_weights
            if base_weights is not None
            else (cfg.base_weights if cfg.base_weights is not None else {})
        )

        common_keys = sorted(set(weights_a.keys()) & set(weights_b.keys()))
        merged: Dict[str, np.ndarray] = {}

        for key in common_keys:
            a = weights_a[key]
            b = weights_b[key]

            if cfg.method == "slerp":
                merged[key] = slerp(a, b, cfg.t)

            elif cfg.method == "dare":
                base_w = base.get(key, np.zeros_like(a, dtype=np.float32))
                delta_a = a.astype(np.float32) - base_w.astype(np.float32)
                delta_b = b.astype(np.float32) - base_w.astype(np.float32)
                merged[key] = dare_merge(
                    base_w, delta_a, delta_b, cfg.dare_density, cfg.t
                )

            else:  # ties
                base_w = base.get(key, np.zeros_like(a, dtype=np.float32))
                merged[key] = ties_merge(
                    base_w, [(base_w, a), (base_w, b)], cfg.ties_k
                )

        # Pass through keys present in only one model unchanged.
        for key, val in weights_a.items():
            if key not in merged:
                merged[key] = val.copy()
        for key, val in weights_b.items():
            if key not in merged:
                merged[key] = val.copy()

        n_keys = len(common_keys)
        self._n_merged += 1
        self._total_keys += n_keys
        self._last_stats = {
            "keys_merged": n_keys,
            "method": cfg.method,
            "t": cfg.t,
        }
        return merged

    @property
    def n_merged(self) -> int:
        """Number of ``merge()`` calls performed so far."""
        return self._n_merged

    @property
    def last_merge_stats(self) -> Optional[Dict]:
        """Stats dict for the most recent merge, or ``None`` before first call.

        Keys: ``keys_merged`` (int), ``method`` (str), ``t`` (float).
        """
        return self._last_stats

    def stats(self) -> MergeStats:
        """Return cumulative merge statistics across all ``merge()`` calls."""
        return MergeStats(
            n_merges=self._n_merged,
            total_keys=self._total_keys,
            method=self._config.method,
        )
