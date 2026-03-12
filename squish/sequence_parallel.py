"""squish/sequence_parallel.py

SequenceParallelScatter — DeepSpeed Ulysses-style sequence dimension
parallelism for multi-head attention in long-context LLM inference.

Standard data parallelism replicates the entire model; tensor parallelism
shards weight matrices.  Sequence parallelism instead shards the *token
sequence* across devices, enabling sub-linear memory growth with context
length when combined with attention.

This module implements the Ulysses design (Jacobs et al., 2023):

* ``scatter``: splits a QKV tensor of shape ``(n_heads, seq, head_dim)`` along
  the sequence dimension so each device holds a contiguous block of tokens for
  all heads.
* ``gather``: reconstructs the full-sequence tensor from per-device chunks.
* ``all_to_all``: simulates the key all-to-all communication primitive that
  converts from the sequence-sharded layout (every device holds all heads for
  its token slice) to the head-sharded layout (every device holds all tokens
  for its head subset).  The result has shape
  ``(n_devices, n_heads // n_devices, seq, head_dim)``.

The ``communication_bytes`` property estimates the data volume of a single
all-to-all for the most recently seen tensor shape, following the standard
model where each element leaves its owner device exactly once.

Reference:
    Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling
    Training of Extreme Long Sequence Transformer Models", 2023.
    https://arxiv.org/abs/2309.14509

Example usage::

    import numpy as np
    from squish.sequence_parallel import SPConfig, SequenceParallelScatter

    cfg    = SPConfig(n_devices=4, n_heads=16, head_dim=64)
    sp     = SequenceParallelScatter(cfg)

    x      = np.random.randn(16, 2048, 64).astype(np.float32)
    chunks = sp.scatter(x)             # list of 4 arrays (16, 512, 64)
    out    = sp.gather(chunks)         # (16, 2048, 64)
    a2a    = sp.all_to_all(x)          # (4, 4, 2048, 64)
    print(f"communication bytes: {sp.communication_bytes}")
"""

from __future__ import annotations

__all__ = ["SPConfig", "SequenceParallelScatter"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SPConfig:
    """Configuration for Ulysses-style sequence parallelism.

    Attributes:
        n_devices: Number of devices to scatter the sequence across.  Must be
                   >= 1.  ``all_to_all`` additionally requires that ``n_heads``
                   is evenly divisible by ``n_devices``.
        n_heads:   Total number of attention heads in the model.  Must be >= 1.
        head_dim:  Dimension of each attention head.  Must be >= 1.
    """

    n_devices: int
    n_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {self.n_devices}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")


# ---------------------------------------------------------------------------
# SequenceParallelScatter
# ---------------------------------------------------------------------------


class SequenceParallelScatter:
    """Ulysses-style sequence parallelism over multi-head attention tensors.

    Splits the sequence dimension of a ``(n_heads, seq, head_dim)`` tensor
    across ``n_devices`` devices for parallel attention computation, then
    reassembles the result.

    The ``all_to_all`` method simulates the all-to-all collective that
    transitions from sequence-sharded to head-sharded layout, which is the
    key communication primitive in the Ulysses design.

    Args:
        config: A :class:`SPConfig` instance.
    """

    def __init__(self, config: SPConfig) -> None:
        self._cfg = config
        self._last_x_shape: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scatter(self, x: np.ndarray) -> list[np.ndarray]:
        """Split ``x`` along the sequence dimension across devices.

        Args:
            x: Tensor of shape ``(n_heads, seq, head_dim)``, float32.

        Returns:
            List of ``n_devices`` arrays each of shape
            ``(n_heads, seq_chunk, head_dim)`` where ``seq_chunk`` is
            approximately ``seq // n_devices``.  The last chunk absorbs any
            remainder tokens.

        Raises:
            ValueError: If ``x`` does not have 3 dimensions, or if its
                        ``n_heads`` or ``head_dim`` axes do not match the
                        config.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(
                f"x must be 3-D (n_heads, seq, head_dim), got shape {x.shape}"
            )
        n_heads, seq, head_dim = x.shape
        if n_heads != self._cfg.n_heads:
            raise ValueError(
                f"x n_heads={n_heads} does not match "
                f"config.n_heads={self._cfg.n_heads}"
            )
        if head_dim != self._cfg.head_dim:
            raise ValueError(
                f"x head_dim={head_dim} does not match "
                f"config.head_dim={self._cfg.head_dim}"
            )

        self._last_x_shape = x.shape
        # Split along the sequence axis (axis=1).
        return np.array_split(x, self._cfg.n_devices, axis=1)

    def gather(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Reconstruct the full-sequence tensor from per-device chunks.

        Args:
            chunks: List of arrays, each of shape
                    ``(n_heads, seq_chunk, head_dim)``, as produced by
                    :meth:`scatter`.

        Returns:
            Concatenated tensor of shape ``(n_heads, seq, head_dim)``, float32.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("chunks must not be empty")
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def all_to_all(self, x: np.ndarray) -> np.ndarray:
        """Simulate the all-to-all communication for Ulysses sequence parallelism.

        In the Ulysses model, each device starts with all heads for its token
        slice and must end with all tokens for its head slice.  This method
        performs the conceptual in-memory reshape equivalent to that exchange.

        Given ``x`` of shape ``(n_heads, seq, head_dim)``, the result has shape
        ``(n_devices, n_heads // n_devices, seq, head_dim)`` where entry
        ``result[i]`` represents the data device ``i`` would hold after the
        all-to-all: the full sequence for its assigned ``n_heads // n_devices``
        heads.

        Args:
            x: Tensor of shape ``(n_heads, seq, head_dim)``, float32.

        Returns:
            Array of shape ``(n_devices, n_heads // n_devices, seq, head_dim)``,
            float32.

        Raises:
            ValueError: If ``x`` is not 3-D or if ``n_heads`` is not evenly
                        divisible by ``n_devices``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(
                f"x must be 3-D (n_heads, seq, head_dim), got shape {x.shape}"
            )
        n_heads, seq, head_dim = x.shape
        n_dev = self._cfg.n_devices
        if n_heads % n_dev != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by "
                f"n_devices={n_dev} for all_to_all"
            )
        heads_per_device = n_heads // n_dev
        # Reshape: (n_heads, seq, head_dim) → (n_devices, heads_per_device, seq, head_dim)
        return x.reshape(n_dev, heads_per_device, seq, head_dim).astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def communication_bytes(self) -> int:
        """Estimated bytes transferred in a single all-to-all for the last seen shape.

        In an all-to-all collective every device sends a unique chunk to every
        other device.  The volume each device sends is
        ``total_elements / n_devices`` elements, and there are ``n_devices - 1``
        destination peers, giving a total of
        ``total_elements * (n_devices - 1) / n_devices`` elements moved per
        device.  We report the aggregate across all devices (i.e. the total
        bytes injected into the network), which equals
        ``total_elements * (n_devices - 1) * bytes_per_element``.

        Returns:
            Estimated bytes transferred, or 0 if no tensor has been seen yet.
        """
        if not self._last_x_shape:
            return 0
        total_elements = int(np.prod(self._last_x_shape))
        bytes_per_element = np.dtype(np.float32).itemsize
        return total_elements * bytes_per_element * max(0, self._cfg.n_devices - 1)
