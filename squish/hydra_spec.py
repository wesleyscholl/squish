"""squish/hydra_spec.py

HydraSpecDecoder — Multi-draft Hydra speculative decoding heads.

Standard speculative decoding uses a single small draft model to propose
``n_draft`` tokens per step.  *Hydra* speculation generalises this by
attaching ``n_heads`` independent lightweight draft heads to the shared
hidden state of the target model; each head proposes its own sequence of
``n_draft`` candidate tokens in a single forward pass over the hidden
representation.  The target model then verifies all ``n_heads × n_draft``
candidates in one batched forward pass, accepting or rejecting each according
to the rejection-sampling criterion.

Each draft head ``h`` owns a distinct weight matrix
``W_head[h]: (hidden_dim, vocab_size)``.  Given the current hidden state
``x: (hidden_dim,)`` the head computes logits via ``x @ W_head[h]`` and
greedily selects its top ``n_draft`` tokens by logit value.  Because every
head uses the *same* hidden state but has *different* weights, the heads
explore diverse areas of the vocabulary without requiring additional model
forward passes.

The verification step applies token-level rejection sampling independently
across heads and positions.  For head ``h`` and position ``i``, the draft
probability is ``softmax(draft_logits[h, i])[draft_tokens[h, i]]`` and the
target probability is ``softmax(target_logits[h, i])[draft_tokens[h, i]]``.
A draft token is accepted when ``u ~ Uniform(0, 1) < p_target / p_draft``.
For each head the longest accepted prefix is extracted; :meth:`verify` returns
the accepted sequence from the head that accepted the most tokens.  On ties
the lowest-indexed head is preferred.

Example usage::

    import numpy as np
    from squish.hydra_spec import HydraConfig, HydraSpecDecoder

    cfg     = HydraConfig(n_heads=4, n_draft=3, hidden_dim=64, vocab_size=512)
    decoder = HydraSpecDecoder(cfg)

    rng    = np.random.default_rng(0)
    hidden = rng.standard_normal(64).astype(np.float32)

    out = decoder.draft(hidden)
    print(out.draft_tokens.shape)  # (4, 3)
    print(out.draft_logits.shape)  # (4, 3, 512)

    # Simulate target logits (normally from the full target model).
    target_logits = rng.standard_normal((4, 3, 512)).astype(np.float32)
    accepted      = decoder.verify(out.draft_tokens, target_logits)
    print(accepted.dtype, accepted.ndim)  # int32, 1

    history = np.array([True, False, True, True, False], dtype=bool)
    print(f"accept_rate={decoder.acceptance_rate(history):.2f}")
"""

from __future__ import annotations

__all__ = ["HydraConfig", "HydraOutput", "HydraSpecDecoder"]

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration and output types
# ---------------------------------------------------------------------------


@dataclass
class HydraConfig:
    """Configuration for multi-draft Hydra speculative decoding.

    Attributes:
        n_heads:    Number of independent draft heads.  Each head has its own
                    projection matrix ``W: (hidden_dim, vocab_size)`` and
                    produces an independent draft sequence.  Must be >= 1.
        n_draft:    Number of candidate tokens each head proposes per step.
                    Must be >= 1.
        hidden_dim: Dimensionality of the shared hidden state passed to all
                    heads.  Must be >= 1.
        vocab_size: Vocabulary size.  Must be >= 1.
    """

    n_heads:    int
    n_draft:    int
    hidden_dim: int
    vocab_size: int

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.n_draft < 1:
            raise ValueError(f"n_draft must be >= 1, got {self.n_draft}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.n_draft > self.vocab_size:
            raise ValueError(
                f"n_draft ({self.n_draft}) cannot exceed vocab_size "
                f"({self.vocab_size})"
            )


@dataclass
class HydraOutput:
    """Output of a single :meth:`~HydraSpecDecoder.draft` call.

    Attributes:
        draft_tokens: Int32 array of shape ``(n_heads, n_draft)`` containing
                      the greedily selected draft token ids for each head.
        draft_logits: Float32 array of shape ``(n_heads, n_draft, vocab_size)``
                      containing the full logit vector for each draft position
                      (all positions within a head share the same logits since
                      they are derived from a single hidden-state projection).
    """

    draft_tokens: np.ndarray   # (n_heads, n_draft) int32
    draft_logits: np.ndarray   # (n_heads, n_draft, vocab_size) float32


# ---------------------------------------------------------------------------
# Helper — numerically stable softmax
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l   = np.exp(shifted)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class HydraSpecDecoder:
    """Multi-draft Hydra speculative decoder.

    At construction, ``n_heads`` weight matrices are initialised with Xavier
    uniform scaling.  Each matrix maps ``(hidden_dim,)`` to ``(vocab_size,)``
    logits, independently for every draft head.

    Args:
        config: A :class:`HydraConfig` instance.
    """

    def __init__(self, config: HydraConfig) -> None:
        self._cfg = config
        rng       = np.random.default_rng(0)
        # Xavier initialisation: scale = sqrt(2 / (fan_in + fan_out)).
        scale     = np.sqrt(2.0 / (config.hidden_dim + config.vocab_size))
        # W_heads[h]: (hidden_dim, vocab_size) — one weight matrix per head.
        self._W_heads: list[np.ndarray] = [
            (rng.standard_normal((config.hidden_dim, config.vocab_size)) * scale
             ).astype(np.float32)
            for _ in range(config.n_heads)
        ]
        # Internal RNG for rejection sampling.
        self._rng = np.random.default_rng(0)
        # Store last draft logits for use in verify().
        self._last_draft_logits: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft(self, hidden: np.ndarray) -> HydraOutput:
        """Produce draft token candidates from all heads.

        Each head computes ``logits = hidden @ W_head`` and greedily selects
        its top ``n_draft`` tokens by descending logit value.  All draft
        positions within a head share the same logits because the projection
        is from a single hidden state (not autoregressive).

        Args:
            hidden: Float32 array of shape ``(hidden_dim,)`` — the hidden
                    state of the current decode step from the target model.

        Returns:
            A :class:`HydraOutput` containing ``draft_tokens`` of shape
            ``(n_heads, n_draft)`` (int32) and ``draft_logits`` of shape
            ``(n_heads, n_draft, vocab_size)`` (float32).

        Raises:
            ValueError: If *hidden* does not have shape ``(hidden_dim,)``.
        """
        cfg    = self._cfg
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.shape != (cfg.hidden_dim,):
            raise ValueError(
                f"hidden must have shape ({cfg.hidden_dim},), "
                f"got {hidden.shape}."
            )

        draft_tokens = np.zeros((cfg.n_heads, cfg.n_draft), dtype=np.int32)
        # draft_logits: (n_heads, n_draft, vocab_size) — same logits for all
        # draft positions within a head.
        draft_logits = np.zeros(
            (cfg.n_heads, cfg.n_draft, cfg.vocab_size), dtype=np.float32
        )

        for h, W in enumerate(self._W_heads):
            logits = (hidden @ W).astype(np.float32)  # (vocab_size,)

            # Greedy top-n_draft selection.
            if cfg.n_draft < cfg.vocab_size:
                top_idx = np.argpartition(-logits, cfg.n_draft)[: cfg.n_draft]
                top_idx = top_idx[np.argsort(-logits[top_idx])]
            else:
                top_idx = np.argsort(-logits)

            draft_tokens[h] = top_idx.astype(np.int32)
            # Broadcast the same logit vector to every draft position.
            draft_logits[h] = logits[np.newaxis, :]  # (n_draft, vocab_size)

        self._last_draft_logits = draft_logits
        return HydraOutput(draft_tokens=draft_tokens, draft_logits=draft_logits)

    def verify(
        self,
        draft_tokens:  np.ndarray,
        target_logits: np.ndarray,
    ) -> np.ndarray:
        """Rejection-sampling verification across all heads.

        For each head ``h`` and draft position ``i``, the draft token is
        accepted if ``u ~ Uniform(0, 1) < p_target[draft_token] / p_draft[draft_token]``.
        Verification is sequential within a head: the first rejection
        terminates that head's accepted sequence.  The accepted sequences from
        all heads are compared and the longest one is returned.  On tie the
        lowest-indexed head wins.

        When no prior :meth:`draft` call has been made (i.e.,
        ``_last_draft_logits`` is ``None``), the draft probabilities are
        treated as uniform over the vocabulary so that rejection sampling
        degenerates to pure sampling from the target.

        Args:
            draft_tokens:  Int array of shape ``(n_heads, n_draft)``
                           containing the draft token ids to verify.
            target_logits: Float32 array of shape
                           ``(n_heads, n_draft, vocab_size)`` containing the
                           target model's logits for each candidate position.

        Returns:
            1-D int32 array of accepted token ids.  Length is in
            ``[0, n_draft]``; an empty array indicates every draft token on
            the best head was rejected.

        Raises:
            ValueError: If *draft_tokens* or *target_logits* have unexpected
                        shapes.
        """
        cfg           = self._cfg
        draft_tokens  = np.asarray(draft_tokens, dtype=np.int32)
        target_logits = np.asarray(target_logits, dtype=np.float32)

        if draft_tokens.shape != (cfg.n_heads, cfg.n_draft):
            raise ValueError(
                f"draft_tokens must have shape ({cfg.n_heads}, {cfg.n_draft}), "
                f"got {draft_tokens.shape}."
            )
        if target_logits.shape != (cfg.n_heads, cfg.n_draft, cfg.vocab_size):
            raise ValueError(
                f"target_logits must have shape "
                f"({cfg.n_heads}, {cfg.n_draft}, {cfg.vocab_size}), "
                f"got {target_logits.shape}."
            )

        # Retrieve draft logits from the last draft() call.
        if self._last_draft_logits is not None:
            draft_logits = self._last_draft_logits  # (n_heads, n_draft, vocab_size)
        else:
            # No prior draft call: treat draft as uniform.
            draft_logits = np.zeros(
                (cfg.n_heads, cfg.n_draft, cfg.vocab_size), dtype=np.float32
            )

        # Softmax probabilities.
        target_probs = _softmax(target_logits)  # (n_heads, n_draft, vocab_size)
        draft_probs  = _softmax(draft_logits)   # (n_heads, n_draft, vocab_size)

        # Determine accepted prefix length for each head; keep the longest.
        best_tokens: np.ndarray = np.empty(0, dtype=np.int32)
        best_len:    int        = -1

        for h in range(cfg.n_heads):
            accepted_tokens: list[int] = []
            for i in range(cfg.n_draft):
                token      = int(draft_tokens[h, i])
                p_target_i = float(target_probs[h, i, token])
                p_draft_i  = float(draft_probs[h,  i, token])

                # Guard against division by near-zero draft prob.
                if p_draft_i < 1e-12:
                    # Treat as always rejected when draft probability is zero.
                    break

                accept_ratio = p_target_i / p_draft_i
                u            = float(self._rng.random())
                if u < accept_ratio:
                    accepted_tokens.append(token)
                else:
                    break  # First rejection terminates this head's sequence.

            n_accepted = len(accepted_tokens)
            if n_accepted > best_len:
                best_len    = n_accepted
                best_tokens = np.array(accepted_tokens, dtype=np.int32)

        return best_tokens

    def acceptance_rate(self, history: np.ndarray) -> float:
        """Compute the mean acceptance rate from a boolean history array.

        Args:
            history: Boolean array of any shape where ``True`` denotes an
                     accepted token and ``False`` a rejection.

        Returns:
            Scalar float in ``[0, 1]``.  Returns ``0.0`` for an empty array.

        Raises:
            ValueError: If *history* is empty.
        """
        history = np.asarray(history, dtype=np.bool_)
        if history.size == 0:
            raise ValueError(
                "history must be non-empty to compute acceptance rate."
            )
        return float(np.mean(history))
