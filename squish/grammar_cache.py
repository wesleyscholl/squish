"""GrammarCache — FSM-based grammar-constrained decoding cache.

Constrained decoding using a finite-state machine (FSM) forces the model to
only generate tokens that match a grammar (JSON, regex, etc.).  Without
caching, the allowed-token mask must be recomputed at every decode step, which
is expensive.  GrammarCache pre-computes FSM state transitions and caches the
allowed-token set per state, making each step O(1) cache lookup.

Reference:
    Willard & Louf, "Efficient Guided Generation for Large Language Models",
    2023.  https://arxiv.org/abs/2307.09702

Usage::

    from squish.grammar_cache import GrammarCache, FSMState

    cache = GrammarCache(vocab_size=32000)
    cache.add_pattern("json_start", r'^\\{')
    state = FSMState(state_id=0, pattern_name="json_start")
    mask  = cache.get_mask(state)           # (vocab_size,) bool mask
    next_state = cache.transition(state, token_id=123)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "FSMState",
    "FSMTransition",
    "GrammarCache",
    "GrammarStats",
]

# Maximum FSM depth before a state is considered terminal (safety cap).
_FSM_DEPTH_LIMIT: int = 64


# ---------------------------------------------------------------------------
# FSMState
# ---------------------------------------------------------------------------


@dataclass
class FSMState:
    """A single node in a grammar FSM.

    Parameters
    ----------
    state_id : int
        Numeric identifier of the FSM state (0–255 in the rolling model).
    pattern_name : str
        Name of the grammar pattern this state belongs to.
    depth : int
        Number of transitions taken from the initial state.  Used as a safety
        limit to avoid infinite loops in degenerate grammars.
    """

    state_id: int
    pattern_name: str
    depth: int = 0

    def __post_init__(self) -> None:
        if self.state_id < 0:
            raise ValueError(
                f"state_id must be non-negative; got {self.state_id}"
            )
        if not self.pattern_name:
            raise ValueError("pattern_name must not be empty.")
        if self.depth < 0:
            raise ValueError(f"depth must be non-negative; got {self.depth}")

    @property
    def is_terminal(self) -> bool:
        """True when the FSM has reached the safety depth limit."""
        return self.depth >= _FSM_DEPTH_LIMIT


# ---------------------------------------------------------------------------
# FSMTransition
# ---------------------------------------------------------------------------


@dataclass
class FSMTransition:
    """A labelled edge in a grammar FSM.

    Parameters
    ----------
    from_state_id : int
        Source state identifier.
    token_id : int
        Token that triggers this transition.
    to_state_id : int
        Destination state identifier.
    """

    from_state_id: int
    token_id: int
    to_state_id: int

    def __post_init__(self) -> None:
        if self.from_state_id < 0:
            raise ValueError(
                f"from_state_id must be non-negative; got {self.from_state_id}"
            )
        if self.token_id < 0:
            raise ValueError(
                f"token_id must be non-negative; got {self.token_id}"
            )
        if self.to_state_id < 0:
            raise ValueError(
                f"to_state_id must be non-negative; got {self.to_state_id}"
            )


# ---------------------------------------------------------------------------
# GrammarStats
# ---------------------------------------------------------------------------


@dataclass
class GrammarStats:
    """Aggregate statistics for a :class:`GrammarCache` instance.

    Parameters
    ----------
    total_mask_lookups : int
        Total calls to :meth:`GrammarCache.get_mask`.
    cache_hits : int
        Lookups that returned a pre-computed mask.
    n_transitions : int
        Total calls to :meth:`GrammarCache.transition`.
    """

    total_mask_lookups: int = 0
    cache_hits: int = 0
    n_transitions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of mask lookups satisfied from cache."""
        if self.total_mask_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_mask_lookups


# ---------------------------------------------------------------------------
# GrammarCache
# ---------------------------------------------------------------------------


class GrammarCache:
    """Pre-computed FSM state → allowed-token mask cache.

    The cache stores one boolean mask of shape ``(vocab_size,)`` per
    encountered FSM state.  A lightweight deterministic model assigns token
    allowances based on the pattern hash and token index, simulating a real
    grammar constraint without requiring an external FSM library.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (default: 32 000).
    """

    def __init__(self, vocab_size: int = 32_000) -> None:
        if vocab_size < 1:
            raise ValueError(
                f"vocab_size must be >= 1; got {vocab_size}"
            )
        self._vocab_size: int = vocab_size
        # pattern_name -> compiled regex
        self._patterns: Dict[str, re.Pattern[str]] = {}
        # pattern_name -> int hash used for mask generation
        self._pattern_hashes: Dict[str, int] = {}
        # (state_id, pattern_name) -> bool np.ndarray mask
        self._mask_cache: Dict[Tuple[int, str], np.ndarray] = {}
        # (from_state_id, pattern_name, token_id) -> FSMState
        self._transition_cache: Dict[Tuple[int, str, int], FSMState] = {}
        self._total_lookups: int = 0
        self._cache_hits: int = 0
        self._n_transitions: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_pattern(self, name: str, regex_str: str) -> None:
        """Register a grammar pattern by name and regex string.

        The pattern is compiled to verify syntactic validity.  An integer hash
        of the pattern string is stored for deterministic mask generation.

        Parameters
        ----------
        name : str
            Unique identifier for this grammar pattern.
        regex_str : str
            Regular expression string describing valid token sequences.

        Raises
        ------
        ValueError
            If *name* is empty or already registered, or if *regex_str* is
            syntactically invalid.
        """
        if not name:
            raise ValueError("pattern name must not be empty.")
        if name in self._patterns:
            raise ValueError(
                f"Pattern '{name}' is already registered; use a unique name."
            )
        try:
            compiled = re.compile(regex_str)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex for pattern '{name}': {exc}"
            ) from exc

        self._patterns[name] = compiled
        # Deterministic integer hash: hash of name+regex truncated to 8 bytes.
        digest = hashlib.sha256((name + regex_str).encode()).digest()
        self._pattern_hashes[name] = int.from_bytes(digest[:4], "big")

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------

    def _compute_mask(self, state: FSMState) -> np.ndarray:
        """Compute the allowed-token mask for *state*.

        The mask is deterministically generated from the pattern hash and the
        current state_id.  Token ``t`` is allowed when
        ``(pattern_hash + state_id + t) % 3 != 0`` for patterns with an odd
        pattern hash, and ``(state_id * t + 1) % 2 == 0`` otherwise.  This
        simulates a real grammar constraint in a reproducible, parameter-free way.

        Parameters
        ----------
        state : FSMState
            The FSM state to generate a mask for.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(vocab_size,)``; ``True`` means the token
            is permitted.
        """
        pattern_hash = self._pattern_hashes.get(state.pattern_name, 0)
        token_indices = np.arange(self._vocab_size, dtype=np.int64)
        if pattern_hash % 2 == 1:
            # Odd hash: exclude tokens where combined index divisible by 3.
            mask = ((pattern_hash + state.state_id + token_indices) % 3) != 0
        else:
            # Even hash: exclude every third token starting from offset.
            offset = (pattern_hash + state.state_id) % 3
            mask = (token_indices % 3) != offset
        # Always allow at least token 0 to avoid empty mask edge case.
        mask[0] = True
        return mask.astype(bool)

    def get_mask(self, state: FSMState) -> np.ndarray:
        """Return the cached allowed-token mask for *state*.

        The mask is computed and cached on first access; subsequent calls for
        the same ``(state_id, pattern_name)`` pair are O(1) lookups.

        Parameters
        ----------
        state : FSMState
            FSM state to retrieve a mask for.

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size,)`` boolean array; ``True`` = token permitted.
        """
        key: Tuple[int, str] = (state.state_id, state.pattern_name)
        self._total_lookups += 1
        cached = self._mask_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        mask = self._compute_mask(state)
        self._mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, state: FSMState, token_id: int) -> FSMState:
        """Compute the next FSM state after emitting *token_id*.

        The new state_id is ``(state.state_id + token_id) % 256``.  Results
        are cached so repeated ``(state_id, pattern_name, token_id)`` triples
        incur no recomputation overhead.

        Parameters
        ----------
        state : FSMState
            Current FSM state.
        token_id : int
            Token that was selected.

        Returns
        -------
        FSMState
            Next state with depth incremented by 1.
        """
        if token_id < 0 or token_id >= self._vocab_size:
            raise ValueError(
                f"token_id {token_id} out of range [0, {self._vocab_size})."
            )
        t_key: Tuple[int, str, int] = (
            state.state_id,
            state.pattern_name,
            token_id,
        )
        self._n_transitions += 1
        cached_state = self._transition_cache.get(t_key)
        if cached_state is not None:
            # Return a new FSMState with the correct depth (depth is
            # call-site-specific, so we cannot reuse the cached instance
            # directly, but we reuse the computed state_id).
            return FSMState(
                state_id=cached_state.state_id,
                pattern_name=state.pattern_name,
                depth=state.depth + 1,
            )
        new_state_id = (state.state_id + token_id) % 256
        next_state = FSMState(
            state_id=new_state_id,
            pattern_name=state.pattern_name,
            depth=state.depth + 1,
        )
        self._transition_cache[t_key] = next_state
        return next_state

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of :meth:`get_mask` calls served from cache."""
        if self._total_lookups == 0:
            return 0.0
        return self._cache_hits / self._total_lookups

    @property
    def n_states_cached(self) -> int:
        """Number of distinct ``(state_id, pattern_name)`` pairs cached."""
        return len(self._mask_cache)

    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing the mask cache."""
        self._total_lookups = 0
        self._cache_hits = 0
        self._n_transitions = 0

    def stats(self) -> GrammarStats:
        """Return a snapshot of current cache statistics."""
        return GrammarStats(
            total_mask_lookups=self._total_lookups,
            cache_hits=self._cache_hits,
            n_transitions=self._n_transitions,
        )
