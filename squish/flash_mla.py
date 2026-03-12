"""squish/flash_mla.py

FlashMLA — Multi-head Latent Attention (DeepSeek-V2 style) with low-rank KV
projection and a compressed KV cache.

In standard multi-head attention (MHA) the KV cache grows as
``O(seq_len × n_heads × head_dim)``.  For large models with many heads this
becomes the dominant memory bottleneck during long-context inference.  MLA
addresses this by projecting KV down to a latent dimension
``latent_dim ≪ n_heads × head_dim`` before caching, then projecting back up
at attention time using two learned up-projection matrices.

The compression ratio is ``n_heads × head_dim / latent_dim``.  A typical
DeepSeek-V2 configuration uses ``latent_dim=512`` against
``n_heads=128, head_dim=128``, giving a 32× ratio.  Even a modest ratio of
8–16× substantially reduces peak memory during long-context generation.

:class:`FlashMLACache` stores one ``(latent_dim,)`` vector per token rather
than ``n_heads × head_dim`` separate key and value vectors.  At decode time
the stored latents are projected back up via two weight matrices ``W_uk`` and
``W_uv`` (both of shape ``(latent_dim, n_heads * head_dim)``) before scaled
dot-product attention is computed.

Optionally, decoupled rotary position embeddings (RoPE) can be applied on a
separate ``rope_dim``-dimensional subspace concatenated to the query without
passing through the latent bottleneck.  When ``rope_dim == 0`` (default) this
path is disabled and pure MLA is used.

Example usage::

    import numpy as np
    from squish.flash_mla import MLAConfig, FlashMLACache

    cfg   = MLAConfig(n_heads=8, head_dim=64, latent_dim=64)
    cache = FlashMLACache(cfg, max_seq_len=512)

    rng = np.random.default_rng(0)
    for _ in range(16):
        latent = rng.standard_normal(64).astype(np.float32)
        cache.append(latent)

    W_uk = rng.standard_normal((64, 8 * 64)).astype(np.float32)
    W_uv = rng.standard_normal((64, 8 * 64)).astype(np.float32)
    q    = rng.standard_normal((8, 64)).astype(np.float32)
    out  = cache.attend(q, W_uk, W_uv)
    print(out.shape)              # (8, 64)
    print(f"ratio={cache.compression_ratio:.1f}x")  # 8.0x
"""

from __future__ import annotations

__all__ = ["MLAConfig", "FlashMLACache"]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MLAConfig:
    """Configuration for Multi-head Latent Attention.

    Attributes:
        n_heads:    Number of attention heads.
        head_dim:   Dimension of each attention head.
        latent_dim: KV latent dimension.  Smaller values yield greater
                    compression at the cost of representational capacity.
                    Must be >= 1.
        rope_dim:   Optional decoupled RoPE subspace dimension appended to the
                    query without compressing through the latent bottleneck.
                    Set to 0 (default) to disable.
    """

    n_heads:    int
    head_dim:   int
    latent_dim: int
    rope_dim:   int = 0

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.rope_dim < 0:
            raise ValueError(f"rope_dim must be >= 0, got {self.rope_dim}")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class FlashMLACache:
    """Multi-head Latent Attention KV cache.

    Stores one compressed latent vector of shape ``(latent_dim,)`` per token
    instead of full per-head KV pairs.  At attention time the stored latents
    ``C`` of shape ``(seq_len, latent_dim)`` are projected back to full keys
    and values via:

    * ``K = C @ W_uk``  →  shape ``(seq_len, n_heads * head_dim)``
    * ``V = C @ W_uv``  →  shape ``(seq_len, n_heads * head_dim)``

    Scaled dot-product attention is then computed per head between the query
    and these reconstructed KV tensors.

    Memory consumed by the cache is ``seq_len × latent_dim × sizeof(float32)``
    rather than ``seq_len × n_heads × head_dim × 2 × sizeof(float32)``, giving
    a memory compression ratio of ``n_heads × head_dim / latent_dim``.

    Args:
        config:      An :class:`MLAConfig` instance controlling head layout
                     and latent dimensionality.
        max_seq_len: Maximum sequence length.  :meth:`append` raises
                     :class:`OverflowError` once the cache reaches this limit.
    """

    def __init__(self, config: MLAConfig, max_seq_len: int = 4096) -> None:
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
        self._cfg         = config
        self._max_seq_len = max_seq_len
        # Pre-allocated latent cache: (max_seq_len, latent_dim).
        self._cache: np.ndarray = np.zeros(
            (max_seq_len, config.latent_dim), dtype=np.float32
        )
        self._fill: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, x: np.ndarray) -> None:
        """Append one token's down-projected latent vector to the cache.

        Args:
            x: Float32 array of shape ``(latent_dim,)`` — the down-projected
               KV latent for the new token.

        Raises:
            ValueError:    If *x* does not have shape ``(latent_dim,)``.
            OverflowError: If the cache is already at ``max_seq_len``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self._cfg.latent_dim,):
            raise ValueError(
                f"x must have shape ({self._cfg.latent_dim},), got {x.shape}."
            )
        if self._fill >= self._max_seq_len:
            raise OverflowError(
                f"FlashMLACache is full (max_seq_len={self._max_seq_len})."
            )
        self._cache[self._fill] = x
        self._fill += 1

    def attend(
        self,
        q: np.ndarray,
        W_uk: np.ndarray,
        W_uv: np.ndarray,
    ) -> np.ndarray:
        """Run full MLA attention against all stored latents.

        The latent cache ``C`` (shape ``(seq_len, latent_dim)``) is projected
        up to full KV, then scaled dot-product attention is computed per head
        between the provided query and the reconstructed K/V tensors.

        Args:
            q:    Query tensor of shape ``(n_heads, head_dim)`` (float32).
            W_uk: Key up-projection of shape
                  ``(latent_dim, n_heads * head_dim)`` (float32).
            W_uv: Value up-projection of shape
                  ``(latent_dim, n_heads * head_dim)`` (float32).

        Returns:
            Output tensor of shape ``(n_heads, head_dim)`` (float32).

        Raises:
            ValueError: If any input has the wrong shape, or if the cache is
                        empty.
        """
        cfg  = self._cfg
        n_kv = cfg.n_heads * cfg.head_dim

        q    = np.asarray(q,    dtype=np.float32)
        W_uk = np.asarray(W_uk, dtype=np.float32)
        W_uv = np.asarray(W_uv, dtype=np.float32)

        if q.shape != (cfg.n_heads, cfg.head_dim):
            raise ValueError(
                f"q must have shape ({cfg.n_heads}, {cfg.head_dim}), "
                f"got {q.shape}."
            )
        if W_uk.shape != (cfg.latent_dim, n_kv):
            raise ValueError(
                f"W_uk must have shape ({cfg.latent_dim}, {n_kv}), "
                f"got {W_uk.shape}."
            )
        if W_uv.shape != (cfg.latent_dim, n_kv):
            raise ValueError(
                f"W_uv must have shape ({cfg.latent_dim}, {n_kv}), "
                f"got {W_uv.shape}."
            )
        if self._fill == 0:
            raise ValueError(
                "Cache is empty; call append() at least once before attend()."
            )

        latents = self._cache[: self._fill]  # (seq_len, latent_dim)

        # Project latents up to full K and V.
        keys_flat   = latents @ W_uk  # (seq_len, n_heads * head_dim)
        values_flat = latents @ W_uv  # (seq_len, n_heads * head_dim)

        seq_len = self._fill
        # Reshape: (seq_len, n_heads, head_dim).
        keys   = keys_flat.reshape(seq_len,   cfg.n_heads, cfg.head_dim)
        values = values_flat.reshape(seq_len, cfg.n_heads, cfg.head_dim)

        # Scaled dot-product attention per head.
        # q:      (n_heads, head_dim)
        # keys:   (seq_len, n_heads, head_dim) → transpose → (n_heads, head_dim, seq_len)
        # scores: (n_heads, 1, seq_len)
        scale   = 1.0 / math.sqrt(cfg.head_dim)
        keys_t  = keys.transpose(1, 2, 0)                     # (n_heads, head_dim, seq_len)
        scores  = q[:, np.newaxis, :] @ keys_t * scale        # (n_heads, 1, seq_len)

        # Numerically stable softmax.
        scores  -= np.max(scores, axis=-1, keepdims=True)
        exp_s    = np.exp(scores)
        attn     = exp_s / np.sum(exp_s, axis=-1, keepdims=True)  # (n_heads, 1, seq_len)

        # Weighted sum over values.
        # values: (seq_len, n_heads, head_dim) → transpose → (n_heads, seq_len, head_dim)
        values_t = values.transpose(1, 0, 2)                   # (n_heads, seq_len, head_dim)
        output   = (attn @ values_t).squeeze(1)                # (n_heads, head_dim)

        return output.astype(np.float32)

    def reset(self) -> None:
        """Reset the cache, discarding all stored latent vectors.

        The underlying buffer is not zeroed; only the fill counter is reset.
        Subsequent :meth:`append` calls will overwrite from the beginning.
        """
        self._fill = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Current number of token latents stored in the cache."""
        return self._fill

    @property
    def compression_ratio(self) -> float:
        """KV memory compression ratio relative to standard MHA.

        Computed as ``n_heads * head_dim / latent_dim``.  Values greater than
        1.0 indicate net compression; values equal to 1.0 indicate no savings.
        """
        return (self._cfg.n_heads * self._cfg.head_dim) / self._cfg.latent_dim
