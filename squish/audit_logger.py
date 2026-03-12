"""squish/audit_logger.py

AuditLogger — Cryptographically chained inference audit log.

Regulatory compliance, SOC 2 auditability, and debugging all benefit from an
append-only log of every inference request.  A naive list of records can be
silently tampered with — entries deleted, modified, or reordered — without
detection.

AuditLogger constructs a *hash chain*: each entry includes the SHA-256 hash
of the previous entry as part of its own hash computation.  Any retroactive
modification to any entry in the chain — including reordering or deletion —
invalidates the hashes of all subsequent entries, making tampering immediately
detectable by :meth:`verify`.

Hash computation per entry::

    entry_hash = sha256(
        f"{entry_id}|{request_id}|{tokens_in}|{tokens_out}|"
        f"{model}|{timestamp_ns}|{prev_hash}"
    )

The first entry uses ``"genesis"`` as its ``prev_hash``.

Example usage::

    from squish.audit_logger import AuditLogger

    logger = AuditLogger()

    e1 = logger.log("req-001", tokens_in=512,  tokens_out=128, model="llama-3")
    e2 = logger.log("req-002", tokens_in=1024, tokens_out=256, model="llama-3")

    print(logger.chain_length)   # 2
    print(logger.verify())       # True

    # Tamper detection: mutate a copy and verify the external list.
    from dataclasses import replace
    entries = logger.export()
    entries[0] = replace(entries[0], tokens_out=999)
    print(logger.verify(entries))  # False
"""

from __future__ import annotations

__all__ = ["AuditEntry", "AuditLogger"]

import hashlib
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------

# Sentinel prev_hash value for the first entry in the chain.
_GENESIS_HASH: str = "genesis"


@dataclass
class AuditEntry:
    """A single record in the cryptographically chained audit log.

    Attributes:
        entry_id:     Monotonically increasing integer index (0-based).
        request_id:   Caller-supplied request identifier.
        tokens_in:    Number of prompt tokens consumed.
        tokens_out:   Number of completion tokens generated.
        model:        Model identifier string.
        timestamp_ns: Monotonic timestamp in nanoseconds at time of logging.
        prev_hash:    SHA-256 hex digest of the previous entry, or
                      ``"genesis"`` for the first entry.
        entry_hash:   SHA-256 hex digest of this entry's canonical fields.
    """

    entry_id: int
    request_id: str
    tokens_in: int
    tokens_out: int
    model: str
    timestamp_ns: int
    prev_hash: str
    entry_hash: str


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Cryptographically chained inference audit log.

    Each :meth:`log` call appends a new :class:`AuditEntry` whose hash
    depends on the previous entry's hash, forming a tamper-evident chain.
    :meth:`verify` recomputes every hash in the chain and returns ``False``
    if any mismatch is found.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        request_id: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
    ) -> AuditEntry:
        """Append a new audit entry to the chain.

        Args:
            request_id: Caller-supplied request identifier string.
            tokens_in:  Number of prompt tokens in this inference call.
                        Must be >= 0.
            tokens_out: Number of completion tokens generated.  Must be >= 0.
            model:      Model identifier string (e.g. ``"llama-3-70b"``).

        Returns:
            The completed :class:`AuditEntry` with all fields populated,
            including the computed ``entry_hash``.

        Raises:
            ValueError: If *tokens_in* or *tokens_out* is negative.
        """
        if tokens_in < 0:
            raise ValueError(f"tokens_in must be >= 0, got {tokens_in}")
        if tokens_out < 0:
            raise ValueError(f"tokens_out must be >= 0, got {tokens_out}")

        entry_id = len(self._entries)
        prev_hash = (
            _GENESIS_HASH if entry_id == 0 else self._entries[-1].entry_hash
        )
        timestamp_ns = time.monotonic_ns()
        entry_hash = _compute_hash(
            entry_id=entry_id,
            request_id=request_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            timestamp_ns=timestamp_ns,
            prev_hash=prev_hash,
        )
        entry = AuditEntry(
            entry_id=entry_id,
            request_id=request_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            timestamp_ns=timestamp_ns,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def verify(self, entries: Optional[list[AuditEntry]] = None) -> bool:
        """Verify the integrity of the hash chain.

        Recomputes the expected hash for every entry and compares it to the
        stored ``entry_hash``.  Also verifies that each entry's ``prev_hash``
        matches the preceding entry's ``entry_hash``.

        Args:
            entries: Optional external list of :class:`AuditEntry` objects to
                     verify.  When ``None``, the internal log is verified.

        Returns:
            ``True`` iff all hashes are valid and the chain is unbroken.
            ``True`` for an empty chain.
        """
        chain = entries if entries is not None else self._entries
        if not chain:
            return True

        for idx, entry in enumerate(chain):
            expected_prev = (
                _GENESIS_HASH if idx == 0 else chain[idx - 1].entry_hash
            )
            # Verify the prev_hash linkage.
            if entry.prev_hash != expected_prev:
                return False
            # Recompute and verify the entry's own hash.
            expected_hash = _compute_hash(
                entry_id=entry.entry_id,
                request_id=entry.request_id,
                tokens_in=entry.tokens_in,
                tokens_out=entry.tokens_out,
                model=entry.model,
                timestamp_ns=entry.timestamp_ns,
                prev_hash=entry.prev_hash,
            )
            if entry.entry_hash != expected_hash:
                return False

        return True

    def export(self) -> list[AuditEntry]:
        """Return a shallow copy of the internal entry list.

        Returns:
            List of :class:`AuditEntry` objects in append order.
        """
        return list(self._entries)

    @property
    def chain_length(self) -> int:
        """Number of entries in the audit chain."""
        return len(self._entries)

    @property
    def head_hash(self) -> str:
        """SHA-256 hash of the most recent entry.

        Raises:
            IndexError: If the chain is empty.
        """
        if not self._entries:
            raise IndexError("Audit chain is empty; no head hash available.")
        return self._entries[-1].entry_hash


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_hash(
    entry_id: int,
    request_id: str,
    tokens_in: int,
    tokens_out: int,
    model: str,
    timestamp_ns: int,
    prev_hash: str,
) -> str:
    """Compute the SHA-256 hash for an audit entry.

    The canonical representation is a pipe-delimited string of all fields,
    deterministically ordered so that the same inputs always produce the
    same digest.

    Args:
        entry_id:     Entry index.
        request_id:   Request identifier string.
        tokens_in:    Prompt token count.
        tokens_out:   Completion token count.
        model:        Model identifier string.
        timestamp_ns: Monotonic timestamp in nanoseconds.
        prev_hash:    Previous entry hash or ``"genesis"``.

    Returns:
        Lowercase hexadecimal SHA-256 digest string (64 characters).
    """
    canonical = (
        f"{entry_id}|{request_id}|{tokens_in}|{tokens_out}|"
        f"{model}|{timestamp_ns}|{prev_hash}"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
