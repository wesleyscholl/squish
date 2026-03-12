"""squish/schema_gen.py

SchemaGenEngine — FSM-accelerated constrained JSON schema generation.

Constrained decoding ensures that a language model can only emit tokens that
keep the partially-generated output consistent with a target JSON schema.
This module implements a finite-state machine (FSM) that tracks the JSON
parsing context and, at each decoding step, masks invalid vocabulary positions
by setting their logits to ``-inf`` before sampling.

FSM states
----------
The engine tracks the current generation context via a *stack* stored in
:class:`SchemaState`.  The stack's top element is the current FSM state; the
remaining elements are "return states" that record what should happen after
a nested object or array is closed.

State codes used on the stack:

* ``"S"``  — before the first token (start state)
* ``"OK"`` — inside an object; expecting a key string (``"``) or ``}``
* ``"OC"`` — after a key string; expecting ``:``
* ``"OV"`` — after ``:``;  expecting any JSON value
* ``"OS"`` — after an object value; expecting ``,`` or ``}``
* ``"AV"`` — inside an array; expecting a value or ``]``
* ``"AS"`` — after an array value; expecting ``,`` or ``]``
* ``"SS"`` — inside a string literal; accepting any char or close ``"``
* ``"D"``  — done; the top-level JSON value is complete

Token categories
----------------
Structural tokens are registered via the *special_tokens* dictionary mapping
category strings to their vocabulary IDs.  Default IDs::

    "{"    → 0      "}"  → 1   "["  → 2    "]"  → 3
    '"'    → 4      ","  → 5   ":"  → 6
    "true" → 7   "false" → 8   "null" → 9   "digit" → 10

Any token ID not found in the special-token mapping is treated as a
``"string_char"`` — valid content inside a string literal.

Example usage::

    import numpy as np
    from squish.schema_gen import SchemaGenEngine

    engine = SchemaGenEngine(vocab_size=128)
    state  = engine.reset()

    rng    = np.random.default_rng(0)
    logits = rng.standard_normal(128).astype(np.float32)

    # Constrain to only allow "{" at the start.
    constrained = engine.constrain(logits, state)
    # Advance FSM after generating the "{" token.
    state = engine.advance(0, state)  # token_id 0 == "{"
    print(state.stack)           # ['OK']
    print(state.expected_tokens) # {'"', '}'}
"""

from __future__ import annotations

__all__ = ["SchemaState", "SchemaGenEngine"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FSM state constants (stack element values)
# ---------------------------------------------------------------------------

_ST_S:  str = "S"    # start — before any token
_ST_OK: str = "OK"   # in object — waiting for key string or "}"
_ST_OC: str = "OC"   # in object — after key, waiting for ":"
_ST_OV: str = "OV"   # in object — after ":", waiting for value
_ST_OS: str = "OS"   # in object — after value, waiting for "," or "}"
_ST_AV: str = "AV"   # in array  — waiting for first value or "]"
_ST_AS: str = "AS"   # in array  — after value, waiting for "," or "]"
_ST_SS: str = "SS"   # inside a string literal
_ST_D:  str = "D"    # done — top-level JSON complete

# ---------------------------------------------------------------------------
# Token category constants
# ---------------------------------------------------------------------------

_CAT_LBRACE:  str = "{"
_CAT_RBRACE:  str = "}"
_CAT_LBRACK:  str = "["
_CAT_RBRACK:  str = "]"
_CAT_QUOTE:   str = '"'
_CAT_COLON:   str = ":"
_CAT_COMMA:   str = ","
_CAT_TRUE:    str = "true"
_CAT_FALSE:   str = "false"
_CAT_NULL:    str = "null"
_CAT_DIGIT:   str = "digit"
_CAT_STRCHAR: str = "string_char"

# Categories that introduce a JSON value (used in several states).
_VALUE_CATS: frozenset[str] = frozenset({
    _CAT_LBRACE, _CAT_LBRACK, _CAT_QUOTE,
    _CAT_TRUE, _CAT_FALSE, _CAT_NULL, _CAT_DIGIT,
})

# Default token ID for each structural category.
_DEFAULT_SPECIAL: dict[str, int] = {
    _CAT_LBRACE: 0,
    _CAT_RBRACE: 1,
    _CAT_LBRACK: 2,
    _CAT_RBRACK: 3,
    _CAT_QUOTE:  4,
    _CAT_COMMA:  5,
    _CAT_COLON:  6,
    _CAT_TRUE:   7,
    _CAT_FALSE:  8,
    _CAT_NULL:   9,
    _CAT_DIGIT:  10,
}

# Valid token categories for each FSM state.
_VALID_CATS: dict[str, frozenset[str]] = {
    _ST_S:  _VALUE_CATS,
    _ST_OK: frozenset({_CAT_QUOTE, _CAT_RBRACE}),
    _ST_OC: frozenset({_CAT_COLON}),
    _ST_OV: _VALUE_CATS,
    _ST_OS: frozenset({_CAT_COMMA, _CAT_RBRACE}),
    _ST_AV: _VALUE_CATS | frozenset({_CAT_RBRACK}),
    _ST_AS: frozenset({_CAT_COMMA, _CAT_RBRACK}),
    _ST_SS: frozenset({_CAT_QUOTE, _CAT_STRCHAR}),
    _ST_D:  frozenset(),
}


# ---------------------------------------------------------------------------
# SchemaState
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SchemaState:
    """Immutable snapshot of the JSON generation FSM.

    :class:`SchemaGenEngine` treats this as a value object — every call to
    :meth:`~SchemaGenEngine.advance` returns a *new* :class:`SchemaState`
    rather than mutating the input.

    Attributes:
        stack:           JSON nesting stack.  The last element is the current
                         FSM state; earlier elements are "return states"
                         restored when a nested container is closed.
        expected_tokens: Human-readable set of token categories currently
                         allowed (derived from ``stack[-1]``).
        is_complete:     ``True`` when the top-level JSON value has been
                         fully closed.
    """

    stack:           list[str]
    expected_tokens: set[str]
    is_complete:     bool


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SchemaGenEngine:
    """FSM-accelerated constrained JSON schema generation engine.

    Tracks the JSON parsing context across decoding steps and provides two
    key operations:

    * :meth:`constrain` — masks invalid vocabulary positions by setting
      their logits to ``-inf`` before sampling.
    * :meth:`advance` — transitions the FSM to the next state after a token
      has been sampled.

    Args:
        vocab_size:     Size of the model vocabulary.  Must be >= 1.
        special_tokens: Optional mapping from token-category strings to their
                        vocabulary IDs.  Entries override the defaults listed
                        in the module docstring.  Any key not present falls
                        back to the default ID.

    Raises:
        ValueError: If *vocab_size* is less than 1.
    """

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Optional[dict[str, int]] = None,
    ) -> None:
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")

        self._vocab_size = vocab_size

        # Build the category → token-ID mapping.
        merged = {**_DEFAULT_SPECIAL, **(special_tokens or {})}
        self._tok: dict[str, int] = {
            cat: tid
            for cat, tid in merged.items()
            if 0 <= tid < vocab_size
        }

        # Build the reverse map (token-ID → category).  First category wins
        # when multiple categories share the same ID.
        self._id_to_cat: dict[int, str] = {}
        for cat, tid in self._tok.items():
            if tid not in self._id_to_cat:
                self._id_to_cat[tid] = cat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> SchemaState:
        """Return the FSM to its start state (before any token has been
        generated).

        Returns:
            A fresh :class:`SchemaState` ready to accept the first JSON token.
        """
        return SchemaState(
            stack=[_ST_S],
            expected_tokens=set(_VALID_CATS[_ST_S]),
            is_complete=False,
        )

    def constrain(
        self,
        logits: np.ndarray,
        state: SchemaState,
    ) -> np.ndarray:
        """Mask vocabulary positions that are invalid in the current FSM state.

        Invalid positions are set to ``-inf``.  Valid positions are left
        unchanged.

        Args:
            logits: float32 array of shape ``(vocab_size,)``.
            state:  Current :class:`SchemaState` from :meth:`reset` or a
                    previous :meth:`advance` call.

        Returns:
            A new float32 array of shape ``(vocab_size,)`` with invalid
            positions set to ``-inf``.

        Raises:
            ValueError: If *logits* has the wrong shape or if the state stack
                        is empty.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (self._vocab_size,):
            raise ValueError(
                f"logits must have shape ({self._vocab_size},), "
                f"got {logits.shape}"
            )
        if not state.stack:
            raise ValueError("Cannot constrain from an empty stack state.")

        current    = state.stack[-1]
        valid_cats = _VALID_CATS.get(current, frozenset())
        result     = logits.copy()

        if current == _ST_SS:
            # Inside a string: all structural tokens except '"' are invalid;
            # every non-structural (string_char) token remains valid.
            for cat, tid in self._tok.items():
                if cat != _CAT_QUOTE:
                    result[tid] = -np.inf
        else:
            # Outside a string: only specific structural tokens are valid.
            # Set the entire vocab to -inf, then restore valid positions.
            result[:] = -np.inf
            for cat in valid_cats:
                if cat == _CAT_STRCHAR:
                    continue  # string_char tokens are not valid outside strings
                tid = self._tok.get(cat)
                if tid is not None:
                    result[tid] = logits[tid]

        return result

    def advance(self, token_id: int, state: SchemaState) -> SchemaState:
        """Advance the FSM after sampling *token_id*.

        Args:
            token_id: Vocabulary index of the token that was sampled.
            state:    Current :class:`SchemaState`.

        Returns:
            A new :class:`SchemaState` reflecting the post-token context.

        Raises:
            ValueError: If the token is not valid in the current FSM state,
                        or if the state stack is empty.
        """
        if not state.stack:
            raise ValueError("Cannot advance from an empty stack state.")

        cat     = self._token_to_category(token_id)
        stack   = list(state.stack)  # mutable copy
        current = stack[-1]

        stack = self._transition(stack, current, cat)

        is_complete = bool(stack and stack[-1] == _ST_D)
        expected    = set(_VALID_CATS.get(stack[-1], frozenset())) if stack else set()
        return SchemaState(
            stack=stack,
            expected_tokens=expected,
            is_complete=is_complete,
        )

    def valid_next_chars(self, state: SchemaState) -> list[str]:
        """Return the human-readable token categories valid in *state*.

        Args:
            state: Current :class:`SchemaState`.

        Returns:
            A list of category strings (e.g. ``['"', '}']``).  Returns an
            empty list when generation is complete or the stack is empty.
        """
        if not state.stack:
            return []
        return list(_VALID_CATS.get(state.stack[-1], frozenset()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_to_category(self, token_id: int) -> str:
        """Map a vocabulary index to its token category.

        Any token ID not registered in the special-token table is classified
        as ``"string_char"``.
        """
        return self._id_to_cat.get(token_id, _CAT_STRCHAR)

    def _transition(
        self,
        stack: list[str],
        current: str,
        cat: str,
    ) -> list[str]:
        """Apply a single FSM transition and return the updated stack.

        Raises:
            ValueError: If *cat* is not valid in state *current*.
        """
        # ── Start ──────────────────────────────────────────────────────
        if current == _ST_S:
            stack.pop()
            if cat == _CAT_LBRACE:
                stack.append(_ST_OK)
            elif cat == _CAT_LBRACK:
                stack.append(_ST_AV)
            elif cat == _CAT_QUOTE:
                # Top-level string: return to DONE after closing quote.
                stack.append(_ST_D)
                stack.append(_ST_SS)
            elif cat in {_CAT_DIGIT, _CAT_TRUE, _CAT_FALSE, _CAT_NULL}:
                stack.append(_ST_D)
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_S}'"
                )

        # ── Object: waiting for key ─────────────────────────────────────
        elif current == _ST_OK:
            if cat == _CAT_QUOTE:
                # Replace OK with OC (post-key state), then enter string.
                stack[-1] = _ST_OC
                stack.append(_ST_SS)
            elif cat == _CAT_RBRACE:
                # Empty object.
                self._close_container(stack)
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_OK}'"
                )

        # ── Object: waiting for colon ───────────────────────────────────
        elif current == _ST_OC:
            if cat == _CAT_COLON:
                stack[-1] = _ST_OV
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_OC}'"
                )

        # ── Object: waiting for value ───────────────────────────────────
        elif current == _ST_OV:
            stack[-1] = _ST_OS  # after this value we expect "," or "}"
            if cat == _CAT_LBRACE:
                stack.append(_ST_OK)
            elif cat == _CAT_LBRACK:
                stack.append(_ST_AV)
            elif cat == _CAT_QUOTE:
                stack.append(_ST_SS)
            elif cat in {_CAT_DIGIT, _CAT_TRUE, _CAT_FALSE, _CAT_NULL}:
                pass  # terminal value; stack top is already OS
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_OV}'"
                )

        # ── Object: waiting for comma or close ──────────────────────────
        elif current == _ST_OS:
            if cat == _CAT_COMMA:
                stack[-1] = _ST_OK  # another key–value pair follows
            elif cat == _CAT_RBRACE:
                self._close_container(stack)
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_OS}'"
                )

        # ── Array: waiting for value or close ───────────────────────────
        elif current == _ST_AV:
            if cat == _CAT_RBRACK:
                # Empty array.
                self._close_container(stack)
            else:
                stack[-1] = _ST_AS  # after this value we expect "," or "]"
                if cat == _CAT_LBRACE:
                    stack.append(_ST_OK)
                elif cat == _CAT_LBRACK:
                    stack.append(_ST_AV)
                elif cat == _CAT_QUOTE:
                    stack.append(_ST_SS)
                elif cat in {_CAT_DIGIT, _CAT_TRUE, _CAT_FALSE, _CAT_NULL}:
                    pass  # terminal value; top is already AS
                else:
                    raise ValueError(
                        f"Unexpected token category '{cat}' in state '{_ST_AV}'"
                    )

        # ── Array: waiting for comma or close ───────────────────────────
        elif current == _ST_AS:
            if cat == _CAT_COMMA:
                stack[-1] = _ST_AV  # another element follows
            elif cat == _CAT_RBRACK:
                self._close_container(stack)
            else:
                raise ValueError(
                    f"Unexpected token category '{cat}' in state '{_ST_AS}'"
                )

        # ── Inside a string ─────────────────────────────────────────────
        elif current == _ST_SS:
            if cat == _CAT_QUOTE:
                # Close the string; return to whatever state was below.
                stack.pop()
            # Any other category (including structural chars) is string
            # content in this simplified model — stay in SS.

        # ── Done ────────────────────────────────────────────────────────
        elif current == _ST_D:
            raise ValueError(
                "Cannot advance from DONE state; generation is complete."
            )

        else:
            raise ValueError(f"Unknown FSM state: '{current}'")

        return stack

    @staticmethod
    def _close_container(stack: list[str]) -> None:
        """Pop the current container state.  If the stack becomes empty,
        push the DONE sentinel.

        Modifies *stack* in place.
        """
        stack.pop()
        if not stack:
            stack.append(_ST_D)
        # else: the element now on top is the parent's "after-value" state,
        # which is exactly what we want.
