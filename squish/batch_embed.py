"""BatchEmbed — Batched embedding with dynamic pooling strategies.

Computes document embeddings from token-level hidden states using one of four
pooling strategies:

* **mean** — average of all non-padding token embeddings
* **max** — element-wise max over token embeddings
* **cls** — first token ([CLS]) embedding directly
* **weighted** — attention-weighted average using a learned or supplied weight vector

Reference:
    Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
    BERT-Networks", EMNLP 2019.  https://arxiv.org/abs/1908.10084

Usage::

    from squish.batch_embed import BatchEmbedder, PoolingConfig

    cfg  = PoolingConfig(strategy="mean", hidden_dim=768, normalize=True)
    emb  = BatchEmbedder(cfg)
    out  = emb.pool(hidden_states, attention_mask)  # (batch, hidden_dim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PoolingConfig",
    "BatchEmbedder",
    "EmbeddingStats",
]

# Numerical stability epsilon for L2 normalisation.
_NORM_EPS: float = 1e-12

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PoolingConfig:
    """Configuration for a :class:`BatchEmbedder`.

    Parameters
    ----------
    strategy : str
        Pooling strategy: ``"mean"``, ``"max"``, ``"cls"``, or ``"weighted"``.
    hidden_dim : int
        Dimensionality of the hidden states (> 0).
    normalize : bool
        When True, L2-normalise the output embeddings.
    attention_temperature : float
        Softmax temperature used by the ``"weighted"`` strategy (> 0).
    """

    strategy: str = "mean"
    hidden_dim: int = 768
    normalize: bool = True
    attention_temperature: float = 1.0

    def __post_init__(self) -> None:
        valid_strategies = ("mean", "max", "cls", "weighted")
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}; "
                f"got '{self.strategy}'."
            )
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0; got {self.hidden_dim}."
            )
        if self.attention_temperature <= 0.0:
            raise ValueError(
                f"attention_temperature must be > 0; "
                f"got {self.attention_temperature}."
            )


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStats:
    """Aggregate statistics for a :class:`BatchEmbedder`.

    Parameters
    ----------
    n_batches : int
        Total number of :meth:`~BatchEmbedder.pool` calls.
    total_embeddings : int
        Total number of embeddings produced (sum of batch sizes).
    total_seq_tokens : int
        Total number of token positions processed (sum of batch × seq_len).
    strategy : str
        Pooling strategy in use.
    """

    n_batches: int = 0
    total_embeddings: int = 0
    total_seq_tokens: int = 0
    strategy: str = "mean"

    @property
    def avg_seq_len(self) -> float:
        """Average sequence length across all embeddings processed."""
        if self.total_embeddings == 0:
            return 0.0
        return self.total_seq_tokens / self.total_embeddings


# ---------------------------------------------------------------------------
# BatchEmbedder
# ---------------------------------------------------------------------------


class BatchEmbedder:
    """Pools token-level hidden states into fixed-size document embeddings.

    For the ``"weighted"`` strategy, a query vector ``q`` of shape
    ``(hidden_dim,)`` is initialised from a standard normal distribution scaled
    by ``1 / sqrt(hidden_dim)``.  Per-token attention weights are then computed
    as ``softmax(hidden_states @ q / temperature)`` and used to take a weighted
    average of the token embeddings.

    Parameters
    ----------
    config : PoolingConfig
        Pooling configuration.
    """

    def __init__(self, config: PoolingConfig) -> None:
        self._cfg = config
        # Initialise attention query vector for weighted pooling.
        rng = np.random.default_rng(seed=0)
        self._query: np.ndarray = (
            rng.standard_normal(config.hidden_dim).astype(np.float32)
            / (config.hidden_dim ** 0.5)
        )
        self._n_embeddings: int = 0
        self._n_batches: int = 0
        self._total_seq_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pool(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool token-level hidden states into batch embeddings.

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape ``(batch, seq_len, hidden_dim)`` float32.
        attention_mask : np.ndarray, optional
            Shape ``(batch, seq_len)`` int or bool; ``1``/``True`` = keep,
            ``0``/``False`` = padding.  Defaults to all-ones (no padding).

        Returns
        -------
        np.ndarray
            Shape ``(batch, hidden_dim)`` float32 embeddings.

        Raises
        ------
        ValueError
            If *hidden_states* is not 3-D or its last dimension does not match
            ``config.hidden_dim``.
        """
        hs = np.asarray(hidden_states, dtype=np.float32)
        if hs.ndim != 3:
            raise ValueError(
                f"hidden_states must be 3-D (batch, seq_len, hidden_dim); "
                f"got shape {hs.shape}."
            )
        batch, seq_len, h_dim = hs.shape
        if h_dim != self._cfg.hidden_dim:
            raise ValueError(
                f"hidden_states last dim {h_dim} does not match "
                f"config.hidden_dim {self._cfg.hidden_dim}."
            )

        # Build float mask (batch, seq_len, 1) for broadcasting.
        if attention_mask is None:
            mask = np.ones((batch, seq_len, 1), dtype=np.float32)
        else:
            m = np.asarray(attention_mask, dtype=np.float32)
            if m.shape != (batch, seq_len):
                raise ValueError(
                    f"attention_mask shape {m.shape} does not match "
                    f"(batch={batch}, seq_len={seq_len})."
                )
            mask = m[:, :, None]  # (batch, seq_len, 1)

        strategy = self._cfg.strategy
        if strategy == "mean":
            out = self._pool_mean(hs, mask)
        elif strategy == "max":
            out = self._pool_max(hs, mask)
        elif strategy == "cls":
            out = self._pool_cls(hs)
        elif strategy == "weighted":
            out = self._pool_weighted(hs, mask)
        else:
            raise RuntimeError(f"Unknown strategy '{strategy}'.")

        if self._cfg.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / (norms + _NORM_EPS)

        self._n_embeddings += batch
        self._n_batches += 1
        self._total_seq_tokens += batch * seq_len
        return out.astype(np.float32)

    def pool_single(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool a single sequence (no batch dimension).

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape ``(seq_len, hidden_dim)`` float32.
        attention_mask : np.ndarray, optional
            Shape ``(seq_len,)`` int or bool.

        Returns
        -------
        np.ndarray
            Shape ``(hidden_dim,)`` float32 embedding.
        """
        hs = np.asarray(hidden_states, dtype=np.float32)
        if hs.ndim != 2:
            raise ValueError(
                f"pool_single expects 2-D (seq_len, hidden_dim); "
                f"got shape {hs.shape}."
            )
        hs_batched = hs[None, :, :]  # (1, seq_len, hidden_dim)
        if attention_mask is not None:
            m = np.asarray(attention_mask)[None, :]  # (1, seq_len)
        else:
            m = None
        out = self.pool(hs_batched, m)  # (1, hidden_dim)
        return out[0]

    # ------------------------------------------------------------------
    # Pooling strategies
    # ------------------------------------------------------------------

    def _pool_mean(
        self,
        hs: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Masked mean pooling.

        Parameters
        ----------
        hs : np.ndarray
            Shape ``(B, S, H)``.
        mask : np.ndarray
            Shape ``(B, S, 1)`` float, 1 = include, 0 = exclude.

        Returns
        -------
        np.ndarray
            Shape ``(B, H)``.
        """
        masked = hs * mask  # (B, S, H)
        token_counts = mask.sum(axis=1).clip(min=1.0)  # (B, 1)
        return masked.sum(axis=1) / token_counts  # (B, H)

    def _pool_max(
        self,
        hs: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Masked max pooling.

        Padding tokens are assigned a large negative value before taking the
        element-wise max, ensuring they do not influence the result.

        Parameters
        ----------
        hs : np.ndarray
            Shape ``(B, S, H)``.
        mask : np.ndarray
            Shape ``(B, S, 1)`` float.

        Returns
        -------
        np.ndarray
            Shape ``(B, H)``.
        """
        neg_inf = np.finfo(np.float32).min
        masked = np.where(mask > 0.0, hs, neg_inf)  # (B, S, H)
        return masked.max(axis=1)  # (B, H)

    def _pool_cls(self, hs: np.ndarray) -> np.ndarray:
        """CLS token pooling (first token of each sequence).

        Parameters
        ----------
        hs : np.ndarray
            Shape ``(B, S, H)``.

        Returns
        -------
        np.ndarray
            Shape ``(B, H)``.
        """
        return hs[:, 0, :]  # (B, H)

    def _pool_weighted(
        self,
        hs: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Attention-weighted pooling using the stored query vector.

        Computes per-token scores as ``hs @ query / temperature``, applies
        masked softmax, then returns the weighted sum of token embeddings.

        Parameters
        ----------
        hs : np.ndarray
            Shape ``(B, S, H)``.
        mask : np.ndarray
            Shape ``(B, S, 1)`` float.

        Returns
        -------
        np.ndarray
            Shape ``(B, H)``.
        """
        T = self._cfg.attention_temperature
        # (B, S) raw scores.
        scores = (hs @ self._query) / T  # (B, S)
        # Mask: set padding positions to very large negative before softmax.
        mask_2d = mask[:, :, 0]  # (B, S)
        neg_inf = np.finfo(np.float32).min
        scores = np.where(mask_2d > 0.0, scores, neg_inf)
        # Numerically stable softmax over sequence dimension.
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        exp_scores = np.where(mask_2d > 0.0, exp_scores, 0.0)
        weights = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + _NORM_EPS)
        # Weighted sum: (B, S, 1) * (B, S, H) → (B, H).
        out = (weights[:, :, None] * hs).sum(axis=1)
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_embeddings(self) -> int:
        """Total number of individual embeddings produced across all calls."""
        return self._n_embeddings

    def stats(self) -> EmbeddingStats:
        """Return aggregate embedding statistics."""
        return EmbeddingStats(
            n_batches=self._n_batches,
            total_embeddings=self._n_embeddings,
            total_seq_tokens=self._total_seq_tokens,
            strategy=self._cfg.strategy,
        )
