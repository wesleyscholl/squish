"""squish/activation_offload.py

ActivationOffloader — Selective layer activation offloading to a simulated
CPU buffer during prefill.

During the prefill phase of a large language model, every layer computes and
stores intermediate activation tensors (e.g., hidden states, attention outputs)
that would ordinarily remain on the accelerator for the backward pass or for
pipeline-parallel stages.  In inference-only and KV-cache-based serving, these
activations are not all needed simultaneously; some layers' activations can be
transferred to CPU memory immediately after they are computed, freeing GPU
VRAM for the next layer's computation and the growing KV cache.

:class:`ActivationOffloader` simulates this pattern by maintaining a
``dict[int, np.ndarray]`` CPU buffer.  In a real system the buffer would be
host-pinned memory and transfers would be asynchronous DMA operations.  Here
all operations are synchronous in-process copies (backed by plain NumPy
arrays), making the class useful for algorithm development, profiling, and
unit testing without a GPU.

:class:`OffloadPolicy` specifies which layer ids are subject to offloading
and by how many layers ahead prefetching should be initiated.  The
``prefetch_ahead`` field is a scheduling hint for the caller: when the caller
is at layer ``L``, it should pre-trigger a fetch of layer ``L - prefetch_ahead``
so the transfer can overlap with the next layer's compute.  This class does
not implement asynchronous scheduling itself; ``prefetch_ahead`` is exposed
purely as metadata for callers.

Example usage::

    import numpy as np
    from squish.activation_offload import OffloadPolicy, ActivationOffloader

    policy    = OffloadPolicy(offload_layers=[0, 2, 4], prefetch_ahead=1)
    offloader = ActivationOffloader(policy)

    act = np.random.randn(4, 512).astype(np.float32)  # (batch, hidden)

    for layer_id in range(6):
        if offloader.should_offload(layer_id):
            offloader.offload(layer_id, act)

    # Fetch back layer 0.
    restored = offloader.fetch(0)
    print(restored.shape)   # (4, 512)
    print(offloader.stats)
"""

from __future__ import annotations

__all__ = ["OffloadPolicy", "OffloadStats", "ActivationOffloader"]

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Policy and stats
# ---------------------------------------------------------------------------


@dataclass
class OffloadPolicy:
    """Policy controlling which layers are offloaded and prefetch depth.

    Attributes:
        offload_layers: List of layer ids whose activations should be
                        transferred to the CPU buffer.  An empty list
                        effectively disables offloading.
        prefetch_ahead: How many layers ahead to initiate a prefetch.
                        ``0`` means synchronous (fetch exactly when needed).
                        This is a scheduling *hint* for the calling code;
                        :class:`ActivationOffloader` does not implement
                        async scheduling itself.
    """

    offload_layers: list[int]
    prefetch_ahead: int = 1

    def __post_init__(self) -> None:
        if self.prefetch_ahead < 0:
            raise ValueError(
                f"prefetch_ahead must be >= 0, got {self.prefetch_ahead}"
            )


@dataclass
class OffloadStats:
    """Aggregate statistics for an :class:`ActivationOffloader` session.

    Attributes:
        n_offloaded:    Total number of successful :meth:`~ActivationOffloader.offload`
                        calls.
        n_fetched:      Total number of successful :meth:`~ActivationOffloader.fetch`
                        calls.
        bytes_offloaded: Cumulative bytes transferred to the CPU buffer.
        bytes_fetched:  Cumulative bytes read back from the CPU buffer.
    """

    n_offloaded:    int = 0
    n_fetched:      int = 0
    bytes_offloaded: int = 0
    bytes_fetched:  int = 0


# ---------------------------------------------------------------------------
# Offloader
# ---------------------------------------------------------------------------


class ActivationOffloader:
    """Simulated selective activation offloader.

    Stores activation tensors in an in-process dict buffer (simulating pinned
    CPU memory) and tracks bandwidth and count statistics.  In a real
    deployment this class would wrap asynchronous DMA copy operations and
    expose a prefetch queue.

    Args:
        policy: An :class:`OffloadPolicy` specifying which layers to offload
                and the prefetch scheduling depth.
    """

    def __init__(self, policy: OffloadPolicy) -> None:
        self._policy        = policy
        # Fast membership test for the offload-layer set.
        self._offload_set:  frozenset[int] = frozenset(policy.offload_layers)
        # CPU buffer: layer_id → stored activation array.
        self._buffer:       dict[int, np.ndarray] = {}
        # Running statistics.
        self._n_offloaded:      int = 0
        self._n_fetched:        int = 0
        self._bytes_offloaded:  int = 0
        self._bytes_fetched:    int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def offload(self, layer_id: int, tensor: np.ndarray) -> None:
        """Store a copy of *tensor* in the CPU buffer for *layer_id*.

        If *layer_id* already has an entry it is overwritten.

        Args:
            layer_id: Integer layer index.
            tensor:   NumPy array to store.  A copy is taken so subsequent
                      in-place modifications to the caller's tensor do not
                      affect the stored copy.

        Raises:
            ValueError: If *layer_id* < 0 or *tensor* is not a NumPy array.
        """
        if layer_id < 0:
            raise ValueError(f"layer_id must be >= 0, got {layer_id}")
        tensor = np.asarray(tensor)
        stored = tensor.copy()
        self._buffer[layer_id] = stored
        nbytes                 = stored.nbytes
        self._n_offloaded      += 1
        self._bytes_offloaded  += nbytes

    def fetch(self, layer_id: int) -> np.ndarray:
        """Retrieve the activation tensor for *layer_id* from the CPU buffer.

        The tensor remains in the buffer after fetching; call :meth:`evict`
        to release it.

        Args:
            layer_id: Integer layer index.

        Returns:
            A copy of the stored activation array.

        Raises:
            KeyError:    If *layer_id* has not been offloaded or has already
                         been evicted.
            ValueError:  If *layer_id* < 0.
        """
        if layer_id < 0:
            raise ValueError(f"layer_id must be >= 0, got {layer_id}")
        if layer_id not in self._buffer:
            raise KeyError(
                f"No activation found in CPU buffer for layer_id={layer_id}. "
                f"Ensure offload() was called before fetch()."
            )
        tensor                = self._buffer[layer_id].copy()
        self._n_fetched       += 1
        self._bytes_fetched   += tensor.nbytes
        return tensor

    def evict(self, layer_id: int) -> None:
        """Delete the activation for *layer_id* from the CPU buffer.

        Args:
            layer_id: Integer layer index to evict.

        Raises:
            KeyError:   If *layer_id* is not present in the buffer.
            ValueError: If *layer_id* < 0.
        """
        if layer_id < 0:
            raise ValueError(f"layer_id must be >= 0, got {layer_id}")
        if layer_id not in self._buffer:
            raise KeyError(
                f"Cannot evict layer_id={layer_id}: not present in buffer."
            )
        del self._buffer[layer_id]

    def should_offload(self, layer_id: int) -> bool:
        """Return ``True`` if *layer_id* is subject to offloading per policy.

        Args:
            layer_id: Integer layer index to query.

        Returns:
            ``True`` when *layer_id* is in :attr:`OffloadPolicy.offload_layers`.
        """
        return layer_id in self._offload_set

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> OffloadStats:
        """Return a snapshot of cumulative offload and fetch statistics."""
        return OffloadStats(
            n_offloaded=self._n_offloaded,
            n_fetched=self._n_fetched,
            bytes_offloaded=self._bytes_offloaded,
            bytes_fetched=self._bytes_fetched,
        )

    @property
    def buffer_bytes(self) -> int:
        """Total bytes currently held in the CPU buffer across all layers."""
        return sum(arr.nbytes for arr in self._buffer.values())

    @property
    def policy(self) -> OffloadPolicy:
        """The :class:`OffloadPolicy` this offloader was constructed with."""
        return self._policy
