"""squish/kv_migrate.py

KVMigrator — Live KV state migration between inference workers.

In a distributed inference fleet, rebalancing load or recovering from a
failing worker requires migrating KV caches between nodes without interrupting
the generation of in-flight requests.  This module provides a byte-level
serialisation protocol for KV state that includes a lightweight integrity
checksum so the receiving worker can verify the payload arrived intact.

Pack format (little-endian binary)::

    [ n_heads  : int32 ]   4 bytes
    [ seq_len  : int32 ]   4 bytes
    [ head_dim : int32 ]   4 bytes
    [ keys     : float32 * n_heads * seq_len * head_dim ]
    [ values   : float32 * n_heads * seq_len * head_dim ]

Total header size: 12 bytes.

Checksum:
    ``int(keys.sum() + values.sum()) % (2 ** 31)``

The checksum is stored in :class:`MigrateStats` (not in the binary payload)
and validated by :meth:`KVMigrator.unpack`.

Example usage::

    import numpy as np
    from squish.kv_migrate import KVMigrator, MigrateStats

    migrator = KVMigrator(n_heads=8, head_dim=64)

    rng    = np.random.default_rng(0)
    keys   = rng.standard_normal((8, 128, 64)).astype(np.float32)
    values = rng.standard_normal((8, 128, 64)).astype(np.float32)

    payload, stats = migrator.pack("seq-001", keys, values)
    k2, v2         = migrator.unpack(payload, stats)
    assert np.allclose(keys, k2)
    print(f"Packed {stats.packed_bytes} bytes, checksum {stats.unpack_checksum}")
    print(f"Total migrations: {migrator.n_migrations}")
"""

from __future__ import annotations

__all__ = ["MigrateStats", "KVMigrator"]

import dataclasses
import struct
from typing import Optional

import numpy as np


# Header layout: three little-endian int32 values (n_heads, seq_len, head_dim).
_HEADER_FORMAT: str = "<3i"
_HEADER_SIZE: int = struct.calcsize(_HEADER_FORMAT)   # 12 bytes


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MigrateStats:
    """Metadata and integrity information for a single KV migration.

    Attributes:
        seq_id:          Identifier of the sequence whose KV was packed.
        n_tokens:        Number of sequence tokens (the ``seq_len`` dimension).
        n_heads:         Number of attention heads.
        head_dim:        Dimension per attention head.
        packed_bytes:    Total byte length of the packed payload (header +
                         both tensors).
        unpack_checksum: Integer checksum used to verify payload integrity on
                         unpack.  Computed as
                         ``int(keys.sum() + values.sum()) % (2 ** 31)``.
    """

    seq_id: str
    n_tokens: int
    n_heads: int
    head_dim: int
    packed_bytes: int
    unpack_checksum: int


# ---------------------------------------------------------------------------
# KVMigrator
# ---------------------------------------------------------------------------


class KVMigrator:
    """Serialises and deserialises KV cache tensors for live worker migration.

    Each call to :meth:`pack` produces a self-contained byte payload and
    increments the internal migration counter.  :meth:`unpack` validates the
    checksum and raises :class:`ValueError` on mismatch, providing a
    lightweight data-integrity guarantee against truncated or corrupted
    payloads.

    Args:
        n_heads:  Number of attention heads.  Must match the tensors passed to
                  :meth:`pack`.
        head_dim: Dimension per attention head.  Must match the tensors.
    """

    def __init__(self, n_heads: int, head_dim: int) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        self._n_heads: int = n_heads
        self._head_dim: int = head_dim
        self._n_migrations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(
        self,
        seq_id: str,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> tuple[bytes, MigrateStats]:
        """Serialise KV tensors to a compact byte payload.

        Args:
            seq_id: Unique identifier for the sequence being migrated.
            keys:   Shape ``(n_heads, seq_len, head_dim)``, float32.
            values: Shape ``(n_heads, seq_len, head_dim)``, float32.

        Returns:
            Tuple ``(payload_bytes, stats)`` where ``payload_bytes`` is the
            binary representation and ``stats`` carries the metadata required
            by :meth:`unpack` for integrity verification.

        Raises:
            ValueError: If ``keys`` or ``values`` do not conform to the
                        expected 3-D shape, or if their shapes differ.
        """
        keys   = np.asarray(keys,   dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        self._validate_kv(keys,   "keys")
        self._validate_kv(values, "values")

        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have identical shape; "
                f"got {keys.shape} vs {values.shape}"
            )

        n_heads, seq_len, head_dim = keys.shape

        header  = struct.pack(_HEADER_FORMAT, n_heads, seq_len, head_dim)
        payload = header + keys.tobytes() + values.tobytes()

        checksum = int(keys.sum() + values.sum()) % (2 ** 31)

        stats = MigrateStats(
            seq_id=seq_id,
            n_tokens=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            packed_bytes=len(payload),
            unpack_checksum=checksum,
        )

        self._n_migrations += 1
        return payload, stats

    def unpack(
        self,
        data: bytes,
        stats: MigrateStats,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deserialise a KV payload and validate its integrity checksum.

        Args:
            data:  Bytes produced by a prior :meth:`pack` call.
            stats: The :class:`MigrateStats` returned alongside ``data`` by
                   :meth:`pack`.

        Returns:
            Tuple ``(keys, values)`` each of shape
            ``(n_heads, seq_len, head_dim)``, float32.

        Raises:
            ValueError: If the payload length does not match
                        ``stats.packed_bytes``, if the binary header disagrees
                        with the stats dimensions, or if the checksum does not
                        match.
        """
        if len(data) != stats.packed_bytes:
            raise ValueError(
                f"Payload length {len(data)} does not match "
                f"stats.packed_bytes={stats.packed_bytes}"
            )

        n_heads, seq_len, head_dim = struct.unpack(
            _HEADER_FORMAT, data[:_HEADER_SIZE]
        )

        if (n_heads, seq_len, head_dim) != (
            stats.n_heads,
            stats.n_tokens,
            stats.head_dim,
        ):
            raise ValueError(
                f"Header mismatch: payload encodes "
                f"({n_heads}, {seq_len}, {head_dim}) but stats expect "
                f"({stats.n_heads}, {stats.n_tokens}, {stats.head_dim})"
            )

        n_elements   = n_heads * seq_len * head_dim
        tensor_bytes = n_elements * np.dtype(np.float32).itemsize

        keys_start   = _HEADER_SIZE
        vals_start   = keys_start   + tensor_bytes
        payload_end  = vals_start   + tensor_bytes

        if len(data) < payload_end:
            raise ValueError(
                f"Payload too short: expected at least {payload_end} bytes, "
                f"got {len(data)}"
            )

        keys   = np.frombuffer(data[keys_start:vals_start], dtype=np.float32)
        values = np.frombuffer(data[vals_start:payload_end], dtype=np.float32)

        # .copy() makes the arrays writable (frombuffer returns read-only views).
        keys   = keys.reshape(n_heads, seq_len, head_dim).copy()
        values = values.reshape(n_heads, seq_len, head_dim).copy()

        checksum = int(keys.sum() + values.sum()) % (2 ** 31)
        if checksum != stats.unpack_checksum:
            raise ValueError(
                f"Checksum mismatch after unpack: "
                f"computed {checksum}, expected {stats.unpack_checksum}"
            )

        return keys, values

    def migration_cost_bytes(self, n_tokens: int) -> int:
        """Estimate the total byte cost of migrating ``n_tokens`` KV entries.

        The estimate accounts for the binary header (12 bytes) plus two
        float32 tensors of shape ``(n_heads, n_tokens, head_dim)``.

        Args:
            n_tokens: Number of sequence tokens to estimate for.

        Returns:
            Estimated byte count (header + keys + values).

        Raises:
            ValueError: If ``n_tokens`` is negative.
        """
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
        tensor_elements = self._n_heads * n_tokens * self._head_dim
        tensor_bytes    = 2 * tensor_elements * np.dtype(np.float32).itemsize
        return _HEADER_SIZE + tensor_bytes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_migrations(self) -> int:
        """Total number of :meth:`pack` calls made on this instance."""
        return self._n_migrations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_kv(self, arr: np.ndarray, name: str) -> None:
        """Raise :class:`ValueError` if ``arr`` does not conform to the KV shape.

        A valid KV tensor must be 3-D with ``axis=0 == n_heads`` and
        ``axis=2 == head_dim``.
        """
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must be 3-D (n_heads, seq_len, head_dim), "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] != self._n_heads:
            raise ValueError(
                f"{name} n_heads={arr.shape[0]} does not match "
                f"configured n_heads={self._n_heads}"
            )
        if arr.shape[2] != self._head_dim:
            raise ValueError(
                f"{name} head_dim={arr.shape[2]} does not match "
                f"configured head_dim={self._head_dim}"
            )
