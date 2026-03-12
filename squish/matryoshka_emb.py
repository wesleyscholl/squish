"""MatryoshkaEmb — Matryoshka Representation Learning (MRL) embedding adapter.

MRL (Kusupati et al., NeurIPS 2022) trains nested embeddings such that any
prefix of the full embedding vector is itself a valid, lower-dimensional
representation.  This allows a single model forward pass to produce embeddings
at 64 / 128 / 256 / 512 / 1024 dimensions — the caller truncates to the
desired size based on latency / quality tradeoff.

Reference:
    Kusupati et al., "Matryoshka Representation Learning",
    NeurIPS 2022.  https://arxiv.org/abs/2205.13147

Usage::

    from squish.matryoshka_emb import MatryoshkaEmbedding, MRLConfig

    cfg  = MRLConfig(full_dim=1536, nested_dims=[64, 128, 256, 512, 1536])
    emb  = MatryoshkaEmbedding(cfg)
    out  = emb.embed(raw_embedding, target_dim=256)  # truncate + L2-norm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "MRLConfig",
    "MatryoshkaEmbedding",
    "MRLStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_NESTED_DIMS = [64, 128, 256, 512]


@dataclass(frozen=True)
class MRLConfig:
    """Configuration for Matryoshka Representation Learning embeddings.

    Parameters
    ----------
    full_dim : int
        Dimensionality of the full embedding produced by the backbone model.
        Must be > 0.
    nested_dims : list[int], optional
        Sorted ascending list of valid truncation dimensions.  Every element
        must be in ``(0, full_dim]``.  Defaults to
        ``[64, 128, 256, 512, full_dim]``.
    normalize : bool
        Whether to L2-normalise the embedding after truncation.  Default
        ``True``.
    """

    full_dim: int = 1536
    nested_dims: Optional[List[int]] = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.full_dim <= 0:
            raise ValueError(f"full_dim must be > 0; got {self.full_dim}")

        # Resolve default nested_dims.
        resolved: List[int] = (
            list(self.nested_dims)
            if self.nested_dims is not None
            else [d for d in _DEFAULT_NESTED_DIMS if d <= self.full_dim]
            + ([] if self.full_dim in _DEFAULT_NESTED_DIMS else [self.full_dim])
        )
        # Ensure full_dim is always a valid target.
        if self.full_dim not in resolved:
            resolved.append(self.full_dim)
        resolved.sort()

        for d in resolved:
            if d <= 0:
                raise ValueError(
                    f"Every nested_dim must be > 0; got {d}"
                )
            if d > self.full_dim:
                raise ValueError(
                    f"Every nested_dim must be <= full_dim ({self.full_dim}); got {d}"
                )

        object.__setattr__(self, "nested_dims", resolved)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class MRLStats:
    """Usage statistics for a :class:`MatryoshkaEmbedding` instance.

    Parameters
    ----------
    n_embeds : int
        Total number of embedding calls.
    dims_used : dict[int, int]
        Map from target dimensionality to the number of times it was used.
    """

    n_embeds: int
    dims_used: Dict[int, int] = field(default_factory=dict)

    @property
    def most_used_dim(self) -> Optional[int]:
        """The target dimension requested most often, or ``None`` if no calls."""
        if not self.dims_used:
            return None
        return max(self.dims_used, key=lambda d: self.dims_used[d])


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding
# ---------------------------------------------------------------------------


class MatryoshkaEmbedding:
    """Inference adapter for Matryoshka Representation Learning.

    Takes full-dimensional backbone embeddings and truncates (plus optionally
    L2-normalises) them to any nested dimensionality listed in the config.
    No learnable parameters are introduced: the nested structure comes from
    the backbone that was fine-tuned with MRL training.

    Parameters
    ----------
    config : MRLConfig
        Configuration specifying ``full_dim``, ``nested_dims``, and
        ``normalize``.

    Examples
    --------
    >>> cfg = MRLConfig(full_dim=1536)
    >>> emb = MatryoshkaEmbedding(cfg)
    >>> v   = np.random.randn(1536).astype(np.float32)
    >>> out = emb.embed(v, target_dim=256)   # shape (256,), unit norm
    """

    def __init__(self, config: MRLConfig) -> None:
        self._config = config
        self._n_embeds: int = 0
        self._dims_used: Dict[int, int] = {d: 0 for d in config.nested_dims}

    # ── Core embedding ────────────────────────────────────────────────────────

    def embed(
        self,
        x: np.ndarray,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Truncate (and optionally L2-normalise) an embedding.

        Parameters
        ----------
        x : np.ndarray
            Full embedding of shape ``(full_dim,)`` or
            ``(batch, full_dim)``.
        target_dim : int, optional
            Target dimensionality.  Must be in ``config.nested_dims``.
            Defaults to ``config.full_dim`` (no truncation).

        Returns
        -------
        np.ndarray
            Embedding of shape ``(target_dim,)`` or ``(batch, target_dim)``,
            L2-normalised if ``config.normalize`` is ``True``.

        Raises
        ------
        ValueError
            If ``target_dim`` is not in ``config.nested_dims`` or if ``x``
            has an unexpected shape.
        """
        cfg = self._config
        if target_dim is None:
            target_dim = cfg.full_dim

        if target_dim not in cfg.nested_dims:
            raise ValueError(
                f"target_dim {target_dim} is not in nested_dims {cfg.nested_dims}."
            )

        x_f = x.astype(np.float32)

        if x_f.ndim == 1:
            if x_f.shape[0] != cfg.full_dim:
                raise ValueError(
                    f"Expected embedding of length {cfg.full_dim}; got {x_f.shape[0]}."
                )
            truncated = x_f[:target_dim]
            result = self._maybe_normalize(truncated)
        elif x_f.ndim == 2:
            if x_f.shape[1] != cfg.full_dim:
                raise ValueError(
                    f"Expected embedding width {cfg.full_dim}; got {x_f.shape[1]}."
                )
            truncated = x_f[:, :target_dim]
            result = self._maybe_normalize_batch(truncated)
        else:
            raise ValueError(
                f"x must be 1-D or 2-D; got shape {x_f.shape}."
            )

        self._n_embeds += 1
        self._dims_used[target_dim] = self._dims_used.get(target_dim, 0) + 1
        return result

    def batch_embed(
        self,
        xs: np.ndarray,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a batch of vectors.

        Equivalent to ``embed(xs, target_dim)`` for 2-D input.

        Parameters
        ----------
        xs : np.ndarray
            Shape ``(batch, full_dim)``.
        target_dim : int, optional
            Target dimensionality; defaults to ``config.full_dim``.

        Returns
        -------
        np.ndarray
            Shape ``(batch, target_dim)``.
        """
        if xs.ndim != 2:
            raise ValueError(f"batch_embed expects 2-D input; got shape {xs.shape}.")
        return self.embed(xs, target_dim=target_dim)

    # ── Similarity ────────────────────────────────────────────────────────────

    def similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
        dim: Optional[int] = None,
    ) -> float:
        """Cosine similarity between two raw embeddings at a given dimension.

        Parameters
        ----------
        a, b : np.ndarray
            Full-dimensional embeddings of shape ``(full_dim,)``.
        dim : int, optional
            Dimension at which to compare.  Defaults to ``config.full_dim``.

        Returns
        -------
        float
            Cosine similarity in ``[-1, 1]``.
        """
        a_emb = self.embed(a.ravel()[: self._config.full_dim], target_dim=dim)
        b_emb = self.embed(b.ravel()[: self._config.full_dim], target_dim=dim)

        dot = float(np.dot(a_emb, b_emb))
        if not self._config.normalize:
            # Normalise now for the cosine computation.
            na = float(np.linalg.norm(a_emb))
            nb = float(np.linalg.norm(b_emb))
            if na < 1e-10 or nb < 1e-10:
                return 0.0
            dot = dot / (na * nb)
        return float(np.clip(dot, -1.0, 1.0))

    # ── Nearest dim ──────────────────────────────────────────────────────────

    def nearest_dim(self, desired_dim: int) -> int:
        """Return the smallest configured nested dim >= ``desired_dim``.

        If ``desired_dim`` exceeds all configured dims, returns the largest
        available dim.

        Parameters
        ----------
        desired_dim : int
            The minimum dimensionality the caller needs.

        Returns
        -------
        int
            Smallest element of ``config.nested_dims`` that is >=
            ``desired_dim``, or the largest element if none qualifies.
        """
        candidates = [d for d in self._config.nested_dims if d >= desired_dim]
        if candidates:
            return min(candidates)
        return max(self._config.nested_dims)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> MRLStats:
        """Return accumulated embedding usage statistics."""
        return MRLStats(
            n_embeds=self._n_embeds,
            dims_used=dict(self._dims_used),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_normalize(self, v: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D vector if ``config.normalize`` is True."""
        if not self._config.normalize:
            return v
        norm = float(np.linalg.norm(v))
        if norm < 1e-10:
            return v
        return v / norm

    def _maybe_normalize_batch(self, mat: np.ndarray) -> np.ndarray:
        """Row-wise L2-normalise a 2-D matrix if ``config.normalize`` is True."""
        if not self._config.normalize:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return mat / norms
