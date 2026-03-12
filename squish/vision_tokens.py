"""VisionTokens — Visual token pruning for multi-modal LLM efficiency.

Vision-language models encode images as sequences of visual tokens (196 for
ViT-B/16 on 224×224 images).  Most visual tokens are redundant — simple token
pruning retains the top-K most informative tokens while discarding the rest,
saving 50–80% of visual token FLOPs with minimal accuracy loss.

Methods:
  * **attention** — keep tokens with highest mean attention weight (from CLS token)
  * **magnitude** — keep tokens with highest L2 norm (feature magnitude)
  * **clustering** — k-means cluster tokens; keep cluster centroids

Reference:
    Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
    https://arxiv.org/abs/2210.09461

Usage::

    from squish.vision_tokens import VisionTokenCompressor, VTConfig

    cfg  = VTConfig(method="attention", keep_ratio=0.25)
    comp = VisionTokenCompressor(cfg)
    kept = comp.compress(visual_tokens, attention_weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "VTConfig",
    "VisionTokenCompressor",
    "VTStats",
]

# Maximum number of k-means iterations.
_KMEANS_MAX_ITER: int = 100

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VTConfig:
    """Configuration for visual token compression.

    Parameters
    ----------
    method : str
        Pruning method: ``"attention"``, ``"magnitude"``, or ``"clustering"``.
    keep_ratio : float
        Fraction of tokens to retain in (0, 1].
    min_tokens : int
        Minimum number of tokens to retain regardless of ``keep_ratio``.
    """

    method: str = "attention"
    keep_ratio: float = 0.25
    min_tokens: int = 16

    def __post_init__(self) -> None:
        valid_methods = ("attention", "magnitude", "clustering")
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}; got '{self.method}'."
            )
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(
                f"keep_ratio must be in (0, 1]; got {self.keep_ratio}."
            )
        if self.min_tokens < 1:
            raise ValueError(
                f"min_tokens must be >= 1; got {self.min_tokens}."
            )


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class VTStats:
    """Aggregate statistics for a :class:`VisionTokenCompressor`.

    Parameters
    ----------
    total_tokens_input : int
        Sum of input token counts across all :meth:`~VisionTokenCompressor.compress` calls.
    total_tokens_kept : int
        Sum of kept token counts across all calls.
    n_calls : int
        Number of :meth:`~VisionTokenCompressor.compress` calls made.
    """

    total_tokens_input: int = 0
    total_tokens_kept: int = 0
    n_calls: int = 0

    @property
    def mean_compression_ratio(self) -> float:
        """Mean ratio of kept tokens to input tokens across all calls."""
        if self.total_tokens_input == 0:
            return 0.0
        return self.total_tokens_kept / self.total_tokens_input


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class VisionTokenCompressor:
    """Prunes visual tokens using the strategy specified in :class:`VTConfig`.

    Parameters
    ----------
    config : VTConfig
        Compression configuration.
    """

    def __init__(self, config: VTConfig) -> None:
        self._cfg = config
        self._total_input: int = 0
        self._total_kept: int = 0
        self._n_compressions: int = 0
        self._kept_ratio_sum: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        tokens: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Prune *tokens* to the top-K most informative subset.

        Parameters
        ----------
        tokens : np.ndarray
            Shape ``(n_tokens, d)`` float32 visual token features.
        attention_weights : np.ndarray, optional
            Shape ``(n_tokens,)`` attention weights from the CLS token.
            Required when ``config.method == "attention"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_keep, d)`` pruned token features.

        Raises
        ------
        ValueError
            If *tokens* has an unexpected shape or *attention_weights* is
            required but not provided.
        """
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be 2-D (n_tokens, d); got shape {tokens.shape}."
            )
        n_tokens, d = tokens.shape
        n_keep = max(self._cfg.min_tokens, round(n_tokens * self._cfg.keep_ratio))
        n_keep = min(n_keep, n_tokens)  # cannot keep more than we have

        if n_keep == n_tokens:
            # No pruning needed.
            indices = np.arange(n_tokens)
        elif self._cfg.method == "attention":
            if attention_weights is None:
                raise ValueError(
                    "attention_weights must be provided for method='attention'."
                )
            indices = self._select_attention(tokens, attention_weights, n_keep)
        elif self._cfg.method == "magnitude":
            indices = self._select_magnitude(tokens, n_keep)
        elif self._cfg.method == "clustering":
            indices = self._select_clustering(tokens, n_keep)
        else:
            raise RuntimeError(f"Unknown method '{self._cfg.method}'.")

        kept = tokens[indices]
        self._total_input += n_tokens
        self._total_kept += n_keep
        self._n_compressions += 1
        self._kept_ratio_sum += n_keep / n_tokens
        return kept

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _select_attention(
        self,
        tokens: np.ndarray,
        attn_weights: np.ndarray,
        n_keep: int,
    ) -> np.ndarray:
        """Return the ``n_keep`` token indices with the highest attention weight.

        Parameters
        ----------
        tokens : np.ndarray
            Shape ``(n_tokens, d)``.
        attn_weights : np.ndarray
            Shape ``(n_tokens,)`` non-negative attention scores.
        n_keep : int
            Number of indices to return.

        Returns
        -------
        np.ndarray
            Sorted integer indices of shape ``(n_keep,)``.
        """
        attn_weights = np.asarray(attn_weights, dtype=np.float32).ravel()
        n_tokens = tokens.shape[0]
        if attn_weights.shape[0] != n_tokens:
            raise ValueError(
                f"attention_weights length {attn_weights.shape[0]} does not "
                f"match n_tokens {n_tokens}."
            )
        # Partial sort: argsort descending, take top n_keep.
        if n_keep >= n_tokens:
            return np.arange(n_tokens)
        top_indices = np.argpartition(-attn_weights, n_keep - 1)[:n_keep]
        return np.sort(top_indices)

    def _select_magnitude(
        self,
        tokens: np.ndarray,
        n_keep: int,
    ) -> np.ndarray:
        """Return the ``n_keep`` token indices with the highest L2 norm.

        Parameters
        ----------
        tokens : np.ndarray
            Shape ``(n_tokens, d)``.
        n_keep : int
            Number of indices to return.

        Returns
        -------
        np.ndarray
            Sorted integer indices of shape ``(n_keep,)``.
        """
        n_tokens = tokens.shape[0]
        if n_keep >= n_tokens:
            return np.arange(n_tokens)
        norms = np.linalg.norm(tokens, axis=1)  # (n_tokens,)
        top_indices = np.argpartition(-norms, n_keep - 1)[:n_keep]
        return np.sort(top_indices)

    def _select_clustering(
        self,
        tokens: np.ndarray,
        n_keep: int,
    ) -> np.ndarray:
        """Return indices of the token nearest each k-means cluster centroid.

        Runs Lloyd's algorithm for up to ``_KMEANS_MAX_ITER`` iterations with
        k-means++ initialisation.  After convergence, for each cluster the
        token with the smallest Euclidean distance to the centroid is selected.

        Parameters
        ----------
        tokens : np.ndarray
            Shape ``(n_tokens, d)``.
        n_keep : int
            Number of clusters (= number of tokens to keep).

        Returns
        -------
        np.ndarray
            Sorted integer indices of shape ``(n_keep,)``.
        """
        n_tokens, d = tokens.shape
        if n_keep >= n_tokens:
            return np.arange(n_tokens)

        rng = np.random.default_rng(seed=42)

        # k-means++ initialisation.
        centroids = np.empty((n_keep, d), dtype=np.float32)
        first_idx = rng.integers(0, n_tokens)
        centroids[0] = tokens[first_idx]
        for k in range(1, n_keep):
            # Squared distance to nearest centroid already chosen.
            dists = np.min(
                np.sum((tokens[:, None, :] - centroids[None, :k, :]) ** 2, axis=2),
                axis=1,
            )  # (n_tokens,)
            probs = dists / (dists.sum() + 1e-10)
            chosen = rng.choice(n_tokens, p=probs)
            centroids[k] = tokens[chosen]

        # Lloyd's iterations.
        labels = np.zeros(n_tokens, dtype=np.int64)
        for _ in range(_KMEANS_MAX_ITER):
            # Assignment step: (n_tokens, n_keep) distance matrix.
            dists = np.sum(
                (tokens[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )  # (n_tokens, n_keep)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update step.
            for k in range(n_keep):
                members = tokens[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)

        # For each cluster, find the member nearest to the centroid.
        selected_indices: list[int] = []
        for k in range(n_keep):
            member_mask = labels == k
            if not member_mask.any():
                # Empty cluster: pick the globally nearest token.
                global_dists = np.sum((tokens - centroids[k]) ** 2, axis=1)
                selected_indices.append(int(global_dists.argmin()))
            else:
                member_indices = np.where(member_mask)[0]
                member_tokens = tokens[member_indices]
                local_dists = np.sum(
                    (member_tokens - centroids[k]) ** 2, axis=1
                )
                selected_indices.append(int(member_indices[local_dists.argmin()]))

        # Deduplicate and sort.
        unique_indices = np.array(sorted(set(selected_indices)), dtype=np.int64)
        # If deduplication removed entries, pad by adding the next-highest
        # magnitude tokens not already selected.
        if len(unique_indices) < n_keep:
            norms = np.linalg.norm(tokens, axis=1)
            norms[unique_indices] = -1.0  # exclude already selected
            extra_needed = n_keep - len(unique_indices)
            extra = np.argpartition(-norms, extra_needed - 1)[:extra_needed]
            unique_indices = np.sort(
                np.concatenate([unique_indices, extra]).astype(np.int64)
            )

        return unique_indices[:n_keep]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def compression_ratio(n_original: int, n_kept: int) -> float:
        """Return the fraction of tokens retained: ``n_kept / n_original``.

        Parameters
        ----------
        n_original : int
            Original token count.
        n_kept : int
            Kept token count.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n_original == 0:
            return 0.0
        return n_kept / n_original

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_compressions(self) -> int:
        """Total number of :meth:`compress` calls."""
        return self._n_compressions

    @property
    def mean_kept_ratio(self) -> float:
        """Mean of ``n_kept / n_original`` across all :meth:`compress` calls."""
        if self._n_compressions == 0:
            return 0.0
        return self._kept_ratio_sum / self._n_compressions

    def stats(self) -> VTStats:
        """Return aggregate compression statistics."""
        return VTStats(
            total_tokens_input=self._total_input,
            total_tokens_kept=self._total_kept,
            n_calls=self._n_compressions,
        )
