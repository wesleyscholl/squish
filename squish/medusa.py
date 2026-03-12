"""MEDUSA — Multi-head parallel tree speculative decoding.

MEDUSA (Cai et al., ICML 2024) adds K additional draft heads to the LLM
that each predict the token k steps ahead.  All K heads run in a single
forward pass producing a draft tree which is verified against the target
model in one batch.  This enables 2–3× decode throughput.

Reference:
    Cai et al., "MEDUSA: Simple LLM Inference Acceleration Framework
    with Multiple Decoding Heads", ICML 2024.
    https://arxiv.org/abs/2401.10774

Usage example::

    import numpy as np
    from squish.medusa import MedusaConfig, MedusaDecoder

    config = MedusaConfig(n_heads=4, vocab_size=32000, hidden_dim=4096)
    decoder = MedusaDecoder(config)

    rng = np.random.default_rng(42)
    hidden = rng.standard_normal(4096).astype(np.float32)

    tree = decoder.draft(hidden)
    print(f"Draft paths: {len(tree.tokens)}, depth: {tree.depth}")

    # Simulate target logits for the last path (n_heads tokens)
    target_logits = [
        rng.standard_normal(32000).astype(np.float32)
        for _ in range(config.n_heads)
    ]
    draft_tokens = tree.tokens[-1]  # longest path
    accepted, n = decoder.verify(draft_tokens, target_logits)
    print(f"Accepted {n}/{config.n_heads} draft tokens")
    print(f"Throughput multiplier: {decoder.throughput_multiplier:.2f}x")
"""

from __future__ import annotations

__all__ = [
    "MedusaConfig",
    "MedusaHead",
    "MedusaDraftTree",
    "MedusaDecoder",
    "MedusaStats",
]

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MedusaConfig:
    """Configuration for MEDUSA multi-head speculative decoding.

    Attributes:
        n_heads: Number of MEDUSA draft heads (K).
        vocab_size: Token vocabulary size.
        hidden_dim: Hidden-state dimensionality fed to each draft head.
        tree_depth: Maximum depth of the draft candidate tree.  When
            ``tree_depth == n_heads`` the tree is a linear chain (one path
            per head prefix).
        top_k_per_head: Number of top-K candidates per head used to build
            the candidate tree.
        acceptance_threshold: Reserved for stochastic acceptance; greedy
            acceptance (argmax match) is used by :meth:`MedusaDecoder.verify`.
    """

    n_heads: int = 4
    vocab_size: int = 32000
    hidden_dim: int = 4096
    tree_depth: int = 4
    top_k_per_head: int = 10
    acceptance_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.n_heads <= 0:
            raise ValueError(
                f"n_heads must be a positive integer, got {self.n_heads}"
            )
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be a positive integer, got {self.vocab_size}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {self.hidden_dim}"
            )
        if self.tree_depth <= 0:
            raise ValueError(
                f"tree_depth must be a positive integer, got {self.tree_depth}"
            )
        if self.top_k_per_head <= 0:
            raise ValueError(
                f"top_k_per_head must be a positive integer, "
                f"got {self.top_k_per_head}"
            )
        if not (0.0 < self.acceptance_threshold <= 1.0):
            raise ValueError(
                f"acceptance_threshold must be in (0.0, 1.0], "
                f"got {self.acceptance_threshold}"
            )


# ---------------------------------------------------------------------------
# Draft head
# ---------------------------------------------------------------------------

class MedusaHead:
    """Single draft head: a linear layer from hidden_dim to vocab_size.

    Weights are initialised with a normal distribution scaled by
    ``1 / sqrt(hidden_dim)`` following the standard transformer convention.
    """

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}"
            )
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        rng = np.random.default_rng()
        scale = 1.0 / np.sqrt(hidden_dim)
        self._weight = rng.normal(
            0.0, scale, (hidden_dim, vocab_size)
        ).astype(np.float32)

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Compute unnormalized logits over the vocabulary.

        Args:
            hidden: Hidden-state vector of shape ``(hidden_dim,)``.

        Returns:
            Logit vector of shape ``(vocab_size,)``.
        """
        return hidden @ self._weight  # (vocab_size,)

    def top_k_tokens(self, hidden: np.ndarray, k: int) -> np.ndarray:
        """Return the top-k token indices sorted by descending logit.

        Args:
            hidden: Hidden-state vector of shape ``(hidden_dim,)``.
            k: Number of candidates to return.

        Returns:
            Integer array of shape ``(k,)`` sorted by logit (highest first).
        """
        logits = self.forward(hidden)
        k = min(k, self._vocab_size)
        # argpartition is O(V) then sort the top-k slice
        top_indices = np.argpartition(logits, -k)[-k:]
        top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
        return top_indices.astype(np.int64)


# ---------------------------------------------------------------------------
# Draft tree
# ---------------------------------------------------------------------------

@dataclass
class MedusaDraftTree:
    """Candidate token sequences produced by a MEDUSA draft pass.

    Attributes:
        tokens: List of candidate sequences.  ``tokens[i]`` is a list of
            ``i + 1`` token IDs representing a path from position 0 through
            position ``i``.
        probs: Joint probabilities for each path in *tokens*.
        depth: Number of draft heads (equals ``len(tokens)``).
    """

    tokens: List[List[int]]
    probs: List[float]
    depth: int


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MedusaDecoder:
    """MEDUSA speculative decoder that drafts with K heads and verifies in one
    target-model batch.
    """

    def __init__(self, config: MedusaConfig) -> None:
        self._config = config
        self._heads = [
            MedusaHead(config.hidden_dim, config.vocab_size)
            for _ in range(config.n_heads)
        ]
        self._total_drafts: int = 0   # draft tokens evaluated in verify()
        self._total_accepted: int = 0
        self._total_calls: int = 0

    # ------------------------------------------------------------------
    # Draft
    # ------------------------------------------------------------------

    def draft(self, hidden_states: np.ndarray) -> MedusaDraftTree:
        """Generate a draft token tree from a single hidden-state vector.

        Each head predicts the token at its respective lookahead position
        using the same *hidden_states*.  Top-1 per head is taken and the
        paths are built as cumulative prefixes:

        - path 0: ``[t0]``
        - path 1: ``[t0, t1]``
        - …
        - path K-1: ``[t0, t1, …, t_{K-1}]``

        Args:
            hidden_states: Float32 vector of shape ``(hidden_dim,)``.

        Returns:
            :class:`MedusaDraftTree` with ``n_heads`` candidate paths.
        """
        if hidden_states.ndim != 1 or hidden_states.shape[0] != self._config.hidden_dim:
            raise ValueError(
                f"hidden_states must have shape ({self._config.hidden_dim},), "
                f"got {hidden_states.shape}"
            )

        top_tokens: List[int] = []
        top_probs: List[float] = []

        for head in self._heads:
            logits = head.forward(hidden_states)
            probs = _softmax(logits)
            token = int(np.argmax(logits))
            top_tokens.append(token)
            top_probs.append(float(probs[token]))

        paths: List[List[int]] = []
        path_probs: List[float] = []
        for i in range(self._config.n_heads):
            path = list(top_tokens[: i + 1])
            # Joint probability is the product of individual head probabilities
            joint = 1.0
            for j in range(i + 1):
                joint *= top_probs[j]
            paths.append(path)
            path_probs.append(joint)

        return MedusaDraftTree(
            tokens=paths,
            probs=path_probs,
            depth=self._config.n_heads,
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(
        self,
        draft_tokens: List[int],
        target_logits: List[np.ndarray],
    ) -> Tuple[List[int], int]:
        """Greedily verify draft tokens against the target model's logits.

        Acceptance criterion: ``argmax(target_logits[i]) == draft_tokens[i]``.
        Verification stops at the first rejection; all tokens before the
        rejection position are returned as accepted.

        Args:
            draft_tokens: Sequence of draft token IDs to verify.
            target_logits: Per-position logit vectors from the target model
                (one per draft token).

        Returns:
            ``(accepted_tokens, n_accepted)`` where *accepted_tokens* is the
            prefix of *draft_tokens* that the target model agrees with.
        """
        if len(draft_tokens) != len(target_logits):
            raise ValueError(
                f"draft_tokens length ({len(draft_tokens)}) must match "
                f"target_logits length ({len(target_logits)})"
            )

        accepted: List[int] = []
        for token, logits in zip(draft_tokens, target_logits):
            if int(np.argmax(logits)) == token:
                accepted.append(token)
            else:
                break

        n_accepted = len(accepted)
        self._total_drafts += len(draft_tokens)
        self._total_accepted += n_accepted
        self._total_calls += 1
        return accepted, n_accepted

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Running fraction of draft tokens accepted across all verify calls."""
        if self._total_drafts == 0:
            return 0.0
        return self._total_accepted / self._total_drafts

    @property
    def throughput_multiplier(self) -> float:
        """Theoretical decode throughput multiplier.

        Computed as ``1 + acceptance_rate * n_heads``.  A value of 1.0
        corresponds to no speedup (all drafts rejected); the theoretical
        maximum is ``1 + n_heads``.
        """
        return 1.0 + self.acceptance_rate * self._config.n_heads

    def get_stats(self) -> "MedusaStats":
        """Return a snapshot of current decoding statistics."""
        return MedusaStats(
            total_drafts=self._total_drafts,
            total_accepted=self._total_accepted,
            total_calls=self._total_calls,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class MedusaStats:
    """Aggregate statistics for a MEDUSA decoding session.

    Attributes:
        total_drafts: Total number of draft tokens evaluated in verify calls.
        total_accepted: Total number of accepted draft tokens.
        total_calls: Number of :meth:`MedusaDecoder.verify` invocations.
    """

    total_drafts: int = 0
    total_accepted: int = 0
    total_calls: int = 0

    @property
    def mean_accepted_per_call(self) -> float:
        """Mean number of accepted tokens per verify call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_accepted / self.total_calls

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted over the entire session."""
        if self.total_drafts == 0:
            return 0.0
        return self.total_accepted / self.total_drafts
