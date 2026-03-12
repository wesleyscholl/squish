"""squish/tensor_parallel.py

TensorParallelShard — Megatron-LM-style row/column tensor parallelism
with simulated all-reduce for LLM weight matrices.

Large language models are often too large to fit on a single device.  Tensor
parallelism (Shoeybi et al., "Megatron-LM") partitions weight matrices across
multiple devices so that each device stores and computes only a fraction of
the full matrix multiplication.

Two modes are supported:

Column parallel (default):
    The weight matrix W of shape ``(in_features, out_features)`` is split along
    the output dimension.  Device i holds ``W[:, start_i:end_i]``.  Each device
    independently computes ``x @ W_shard``, producing a partial output of shape
    ``(batch, out_features // n_devices)``.  The partial outputs are
    concatenated (all-reduce via concat) to reconstruct the full
    ``(batch, out_features)`` output.

Row parallel:
    W is split along the input dimension.  Device i holds
    ``W[start_i:end_i, :]``.  The input x is also split along the feature axis:
    ``x_shard = x[:, start_i:end_i]``.  Each device computes
    ``x_shard @ W_shard``, producing a partial output of shape
    ``(batch, out_features)``.  The partial outputs are summed (all-reduce via
    sum) to reconstruct the full output.

The simulated all-reduce byte cost uses a ring-allreduce model where each
element traverses ``n_devices - 1`` hops.

Reference:
    Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
    Models Using Model Parallelism", 2019.
    https://arxiv.org/abs/1909.08053

Example usage::

    import numpy as np
    from squish.tensor_parallel import TPConfig, TensorParallelShard

    cfg    = TPConfig(n_devices=4, mode="column")
    tp     = TensorParallelShard(cfg)

    W      = np.random.randn(512, 2048).astype(np.float32)
    x      = np.random.randn(8, 512).astype(np.float32)
    shards = tp.shard(W)
    out    = tp.forward(x, shards)   # shape (8, 2048)
    print(tp.stats)
"""

from __future__ import annotations

__all__ = ["TPConfig", "TPStats", "TensorParallelShard"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TPConfig:
    """Configuration for tensor-parallel weight sharding.

    Attributes:
        n_devices: Number of devices (shards) to split across.  Must be >= 1.
        mode:      Parallelism mode — ``"column"`` splits W along the output
                   dimension; ``"row"`` splits W along the input dimension.
    """

    n_devices: int
    mode: str = "column"

    def __post_init__(self) -> None:
        if self.n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {self.n_devices}")
        if self.mode not in ("column", "row"):
            raise ValueError(
                f"mode must be 'column' or 'row', got '{self.mode}'"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TPStats:
    """Statistics snapshot for the most recent sharding operation.

    Attributes:
        n_shards:         Number of shards created (equals ``n_devices``).
        shard_shape:      Shape of the first shard (representative shape).
        all_reduce_bytes: Estimated bytes exchanged in the all-reduce step.
        mode:             Sharding mode used (``"column"`` or ``"row"``).
    """

    n_shards: int
    shard_shape: tuple[int, ...]
    all_reduce_bytes: int
    mode: str


# ---------------------------------------------------------------------------
# TensorParallelShard
# ---------------------------------------------------------------------------


class TensorParallelShard:
    """Simulates Megatron-LM-style tensor parallelism for a single weight matrix.

    Column-parallel mode broadcasts the full input to every device and splits
    the output dimension; partial outputs are concatenated.

    Row-parallel mode splits both the input and the weight matrix along the
    input dimension; partial outputs are summed.

    All communication (sharding and all-reduce) is simulated in-process — no
    actual inter-device transfer takes place.

    Args:
        config: A :class:`TPConfig` instance controlling the number of devices
                and the parallelism mode.
    """

    def __init__(self, config: TPConfig) -> None:
        self._cfg = config
        self._last_shard_shape: tuple[int, ...] = ()
        self._last_all_reduce_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shard(self, W: np.ndarray) -> list[np.ndarray]:
        """Shard weight matrix W across ``n_devices`` devices.

        For column-parallel mode the split is along ``axis=1`` (output dim).
        For row-parallel mode the split is along ``axis=0`` (input dim).
        ``np.array_split`` is used so that any remainder rows/columns are
        absorbed by the last shard rather than raising an error.

        Args:
            W: 2-D float32 weight matrix of shape ``(in_features, out_features)``.

        Returns:
            A list of ``n_devices`` contiguous sub-arrays.  In column-parallel
            mode each shard has shape approximately
            ``(in_features, out_features // n_devices)``; in row-parallel mode
            approximately ``(in_features // n_devices, out_features)``.

        Raises:
            ValueError: If W is not 2-D.
        """
        W = np.asarray(W, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        n = self._cfg.n_devices
        if self._cfg.mode == "column":
            shards = np.array_split(W, n, axis=1)
        else:
            shards = np.array_split(W, n, axis=0)

        self._last_shard_shape = shards[0].shape
        return shards

    def forward(
        self,
        x: np.ndarray,
        shards: list[np.ndarray],
    ) -> np.ndarray:
        """Run the full tensor-parallel forward pass.

        Column-parallel: the full input ``x`` is sent to all devices; each
        device computes ``x @ W_shard``; partial outputs are concatenated along
        the feature dimension.

        Row-parallel: ``x`` is split along the feature axis to match the weight
        shards; each device computes ``x_shard @ W_shard``; partial outputs are
        summed element-wise.

        Args:
            x:      Input tensor of shape ``(batch, in_features)``, float32.
            shards: Weight shards as returned by :meth:`shard`.

        Returns:
            Output tensor of shape ``(batch, out_features)``, float32.

        Raises:
            ValueError: If ``x`` is not 2-D or the number of shards does not
                        match ``n_devices``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(
                f"x must be 2-D (batch, in_features), got shape {x.shape}"
            )
        if len(shards) != self._cfg.n_devices:
            raise ValueError(
                f"Expected {self._cfg.n_devices} shards, got {len(shards)}"
            )

        if self._cfg.mode == "column":
            # Each device sees the full input; partial outputs differ in width.
            partial_outputs = [x @ s for s in shards]
        else:
            # Split x along the input-feature axis to match row shards.
            x_splits = np.array_split(x, self._cfg.n_devices, axis=1)
            if len(x_splits) != len(shards):
                raise ValueError(
                    "Number of input splits does not match number of weight shards."
                )
            partial_outputs = [xp @ s for xp, s in zip(x_splits, shards)]

        return self.all_reduce(partial_outputs)

    def all_reduce(self, partial_outputs: list[np.ndarray]) -> np.ndarray:
        """Simulate the all-reduce communication step across all shards.

        Column-parallel: partial outputs are concatenated along the last axis
        (each shard contributed a different set of output columns).

        Row-parallel: partial outputs are summed element-wise (each shard
        contributed a partial dot-product over a row slice of W).

        The byte cost estimate follows a ring all-reduce model: each element
        traverses ``n_devices - 1`` hops.

        Args:
            partial_outputs: List of partial output arrays, one per shard.

        Returns:
            Fully reduced output tensor, float32.

        Raises:
            ValueError: If ``partial_outputs`` is empty.
        """
        if not partial_outputs:
            raise ValueError("partial_outputs must not be empty")

        if self._cfg.mode == "column":
            result = np.concatenate(partial_outputs, axis=-1)
        else:
            result = np.add.reduce(partial_outputs)

        # Estimate bytes moved: ring all-reduce sends each element to
        # (n_devices - 1) peers.
        element_bytes = partial_outputs[0].dtype.itemsize
        total_elements = sum(p.size for p in partial_outputs)
        self._last_all_reduce_bytes = (
            total_elements * element_bytes * max(1, self._cfg.n_devices - 1)
        )

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> TPStats:
        """Return a snapshot of statistics from the most recent forward pass."""
        return TPStats(
            n_shards=self._cfg.n_devices,
            shard_shape=self._last_shard_shape,
            all_reduce_bytes=self._last_all_reduce_bytes,
            mode=self._cfg.mode,
        )
