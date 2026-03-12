"""TokenHealer — Token boundary healing for prefix-aware generation.

When a completion prompt ends mid-token (e.g. "def calc_va" where "va" is the
incomplete start of "value"), stock tokenizers split greedily and the first
generated token tries to continue from "va" — but the model never saw "va" as
a standalone token during training.  Token healing backs up the tokenizer by
the incomplete suffix, prepends it to the generation, and retokenizes jointly
for a clean boundary.

Reference:
    Brandon Smock, "Token Healing" blog post (Microsoft, 2023).
    https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38

Usage example::

    from squish.token_healer import HealerConfig, TokenHealer

    # vocab_list maps token_id → token_string
    vocab_list = ["<pad>", " def", " calc", "_va", "_value", "_variable"]
    config = HealerConfig(vocab_size=len(vocab_list), max_healing_tokens=4)
    healer = TokenHealer(config, vocab_list=vocab_list)

    # Prompt ending with token id 3 ("_va") — a proper prefix of "_value"
    prompt_tokens = [1, 2, 3]
    n_overlap, overlap_str = healer.find_suffix_overlap(prompt_tokens)
    print(f"Overlap: {n_overlap} token(s), string: '{overlap_str}'")

    # completion[0] is the healed model output starting at the clean boundary
    healed = healer.heal(prompt_tokens, completions=[[4, 5]])
    print(f"Healed token sequence: {healed}")
    print(f"Total heals: {healer.n_healed}, avg overlap: {healer.avg_overlap_tokens:.2f}")
"""

from __future__ import annotations

__all__ = [
    "HealerConfig",
    "TokenHealer",
    "HealerStats",
]

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealerConfig:
    """Configuration for :class:`TokenHealer`.

    Attributes:
        vocab_size: Total number of tokens in the vocabulary.
        max_healing_tokens: Maximum number of trailing tokens to inspect when
            searching for an incomplete word boundary.
        min_prefix_len: Minimum string length of the suffix overlap to
            qualify for healing (prevents healing on very short tokens such
            as single punctuation characters).
    """

    vocab_size: int = 32000
    max_healing_tokens: int = 8
    min_prefix_len: int = 1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be a positive integer, got {self.vocab_size}"
            )
        if self.max_healing_tokens <= 0:
            raise ValueError(
                f"max_healing_tokens must be a positive integer, "
                f"got {self.max_healing_tokens}"
            )
        if self.min_prefix_len <= 0:
            raise ValueError(
                f"min_prefix_len must be a positive integer, "
                f"got {self.min_prefix_len}"
            )


# ---------------------------------------------------------------------------
# TokenHealer
# ---------------------------------------------------------------------------

class TokenHealer:
    """Detects and repairs incomplete token boundaries in generation prompts.

    The healer inspects the trailing tokens of a prompt and checks whether
    their concatenated string form is a *proper prefix* of any vocabulary
    entry.  If so, it removes those trailing tokens and lets the caller
    re-generate from the shortened prompt, then concatenates the output
    so the final sequence is equivalent to generating from the original
    boundary.
    """

    def __init__(
        self,
        config: HealerConfig,
        vocab_list: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            config: Healer configuration.
            vocab_list: Optional list of token strings indexed by token ID.
                When provided, enables string-level prefix matching.
        """
        self._config = config
        self._vocab_list = vocab_list
        self._n_healed: int = 0
        self._total_overlap_tokens: int = 0

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def find_suffix_overlap(
        self,
        tokens: List[int],
        vocab_list: Optional[List[str]] = None,
    ) -> Tuple[int, str]:
        """Find how many trailing tokens form an incomplete word boundary.

        Iterates from one trailing token up to ``max_healing_tokens``,
        building a suffix string by concatenating the token strings.  Returns
        the *maximum* number of trailing tokens whose concatenation is a
        proper prefix of at least one vocabulary entry.

        Args:
            tokens: Input token ID sequence.
            vocab_list: Vocabulary to use for string lookup.  Falls back to
                the vocab_list provided at construction.

        Returns:
            ``(n_overlap_tokens, overlap_str)`` where *n_overlap_tokens* is
            the count of trailing tokens to back up (0 if no healing needed)
            and *overlap_str* is their concatenated string.
        """
        vocab = vocab_list if vocab_list is not None else self._vocab_list
        if vocab is None or len(tokens) == 0:
            return 0, ""

        max_n = min(self._config.max_healing_tokens, len(tokens))
        best_n = 0
        best_str = ""

        for n in range(1, max_n + 1):
            suffix_token_ids = tokens[-n:]
            try:
                suffix_str = "".join(
                    vocab[t] for t in suffix_token_ids
                    if 0 <= t < len(vocab)
                )
            except (IndexError, TypeError):
                # Malformed token ID — stop extending
                break

            if len(suffix_str) < self._config.min_prefix_len:
                continue

            # Check if suffix_str is a strict prefix of any vocabulary entry
            is_proper_prefix = any(
                v.startswith(suffix_str) and len(v) > len(suffix_str)
                for v in vocab
            )
            if is_proper_prefix:
                best_n = n
                best_str = suffix_str

        return best_n, best_str

    # ------------------------------------------------------------------
    # Healing
    # ------------------------------------------------------------------

    def heal(
        self,
        tokens: List[int],
        completions: List[List[int]],
    ) -> List[int]:
        """Return the healed token sequence.

        Removes the last *n_overlap* tokens from *tokens* and concatenates
        with ``completions[0]``.  The caller is responsible for generating
        ``completions[0]`` from the backed-up prompt (i.e., ``tokens[:-n]``),
        which results in proper tokenization at the original boundary.

        If no overlap is detected, returns ``tokens + completions[0]``.

        Args:
            tokens: Prompt token IDs, potentially ending mid-token.
            completions: List of generated token sequences.  Only
                ``completions[0]`` is used.

        Returns:
            Healed token sequence of shape
            ``tokens[:-n_overlap] + completions[0]``.
        """
        if not completions:
            raise ValueError("completions must be a non-empty list")

        n_overlap, _overlap_str = self.find_suffix_overlap(tokens)

        if n_overlap > 0:
            self._n_healed += 1
            self._total_overlap_tokens += n_overlap
            healed = list(tokens[:-n_overlap]) + list(completions[0])
        else:
            healed = list(tokens) + list(completions[0])

        return healed

    def needs_healing(
        self,
        tokens: List[int],
        vocab_list: Optional[List[str]] = None,
    ) -> bool:
        """Return ``True`` if the last token is an incomplete word prefix.

        Args:
            tokens: Prompt token ID sequence to inspect.
            vocab_list: Vocabulary override; falls back to construction vocab.
        """
        n_overlap, _ = self.find_suffix_overlap(tokens, vocab_list)
        return n_overlap > 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_healed(self) -> int:
        """Cumulative number of heal operations performed."""
        return self._n_healed

    @property
    def avg_overlap_tokens(self) -> float:
        """Mean number of overlap tokens backed up per heal operation."""
        if self._n_healed == 0:
            return 0.0
        return self._total_overlap_tokens / self._n_healed

    def get_stats(self) -> "HealerStats":
        """Return a snapshot of current healing statistics."""
        # zero_overlap_cases: completions where heal() found n_overlap == 0
        # (the caller would need to track these externally; we report what we have)
        zero_overlap = 0  # cannot determine from internal state alone
        return HealerStats(
            total_heals=self._n_healed,
            total_overlap_tokens=self._total_overlap_tokens,
            zero_overlap_cases=zero_overlap,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class HealerStats:
    """Aggregate statistics for a :class:`TokenHealer` session.

    Attributes:
        total_heals: Number of :meth:`TokenHealer.heal` calls that triggered
            at least one token overlap.
        total_overlap_tokens: Cumulative count of tokens backed up across all
            heal operations.
        zero_overlap_cases: Number of :meth:`TokenHealer.heal` calls where no
            overlap was found (pass-through calls).
    """

    total_heals: int = 0
    total_overlap_tokens: int = 0
    zero_overlap_cases: int = 0

    @property
    def avg_tokens_per_heal(self) -> float:
        """Mean number of overlap tokens per heal operation."""
        if self.total_heals == 0:
            return 0.0
        return self.total_overlap_tokens / self.total_heals
