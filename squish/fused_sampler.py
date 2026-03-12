"""squish/fused_sampler.py

FusedSampler — Temperature, top-k, top-p, min-p, and repetition-penalty
sampling combined in a single in-place pass over the logit array.

Standard LLM sampling pipelines apply each filter as a separate transformation,
each producing an intermediate probability buffer.  For large vocabularies
(≥ 100 k tokens) these intermediate allocations add up to several megabytes of
heap pressure per decode step.  FusedSampler applies every filter
sequentially on the *same* working buffer, minimising total allocation.

The pipeline, applied in order to the same float64 working array:

1. **Repetition penalty** — logits for previously generated token ids are
   scaled to discourage repetition: positive logits are divided by the
   penalty factor; negative logits are multiplied by it (making them more
   negative and even less likely to be sampled).

2. **Temperature** — the working array is divided by ``temperature`` (must
   be > 0).  Values < 1 sharpen the distribution; values > 1 flatten it.

3. **Softmax** — convert logits to a valid probability distribution using
   the numerically stable ``exp(x - max(x)) / sum(...)`` identity.

4. **min-p filter** — tokens with probability below
   ``min_p * max_prob`` are zeroed in-place.

5. **top-k filter** — all but the ``top_k`` highest-probability tokens
   are zeroed in-place using ``np.argpartition`` for O(V) selection.

6. **top-p (nucleus) filter** — tokens are sorted by descending probability;
   the smallest nucleus whose cumulative probability >= ``top_p`` is retained;
   all remaining tokens are zeroed.

7. **Renormalise and sample** — surviving probabilities are normalised to
   sum to 1.0 and a token is drawn via ``np.random.Generator.choice``.

Example usage::

    import numpy as np
    from squish.fused_sampler import SamplerConfig, FusedSampler

    rng    = np.random.default_rng(42)
    logits = rng.standard_normal(32000).astype(np.float32)
    past   = np.array([1, 42, 100], dtype=np.int64)

    cfg     = SamplerConfig(temperature=0.8, top_k=50, top_p=0.95, seed=0)
    sampler = FusedSampler(cfg)
    token   = sampler.sample(logits, input_ids=past)
    print(f"sampled token={token}")

    batch   = rng.standard_normal((4, 32000)).astype(np.float32)
    tokens  = sampler.sample_batch(batch)
    print(tokens.shape)  # (4,)
"""

from __future__ import annotations

__all__ = ["SamplerConfig", "FusedSampler"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """Configuration for fused token sampling.

    Attributes:
        temperature:        Softmax temperature.  Must be > 0.  Values below 1
                            sharpen the distribution; values above 1 flatten it.
        top_k:              Keep only the *top_k* highest-probability tokens.
                            Set to 0 to disable.
        top_p:              Nucleus sampling probability threshold in (0, 1].
                            The smallest nucleus whose cumulative probability
                            is >= this value is retained.  Set to 1.0 to
                            disable.
        min_p:              Minimum per-token probability expressed as a
                            fraction of the peak probability.  Tokens with
                            ``prob < min_p * max_prob`` are masked.  Set to
                            0.0 to disable.
        repetition_penalty: Multiplicative penalty applied to logits of
                            previously seen tokens.  Values > 1 discourage
                            repetition.  1.0 disables the penalty.
        seed:               Optional integer seed for the internal RNG.  When
                            ``None`` the RNG is seeded non-deterministically.
    """

    temperature:        float         = 1.0
    top_k:              int           = 0
    top_p:              float         = 1.0
    min_p:              float         = 0.0
    repetition_penalty: float         = 1.0
    seed:               Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p must be in (0, 1], got {self.top_p}"
            )
        if not (0.0 <= self.min_p < 1.0):
            raise ValueError(
                f"min_p must be in [0, 1), got {self.min_p}"
            )
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class FusedSampler:
    """Fused token sampler: temperature, top-k, top-p, min-p, rep-penalty.

    All filter steps operate on the same float64 working copy of the input
    logits, avoiding intermediate full-vocabulary allocations between steps.

    Args:
        config: A :class:`SamplerConfig` controlling sampling behaviour.
    """

    def __init__(self, config: SamplerConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: np.ndarray,
        input_ids: Optional[np.ndarray] = None,
    ) -> int:
        """Sample a single token from the logit distribution.

        Args:
            logits:    Float32 array of shape ``(vocab_size,)``.
            input_ids: Optional integer array of previously generated token
                       ids used for repetition penalty.  Any shape is
                       accepted; values are flattened before use.

        Returns:
            A single integer token id in ``[0, vocab_size)``.

        Raises:
            ValueError: If *logits* is not 1-D.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError(
                f"logits must be 1-D (vocab_size,), got shape {logits.shape}."
            )
        probs = self._compute_probs(logits, input_ids)
        return int(self._rng.choice(len(probs), p=probs))

    def sample_batch(
        self,
        logits: np.ndarray,
        input_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample one token per row in a batch of logit vectors.

        Args:
            logits:    Float32 array of shape ``(batch, vocab_size)``.
            input_ids: Optional integer array for repetition penalty.  If 2-D
                       with shape ``(batch, context_len)`` each row applies to
                       the corresponding logit row.  If 1-D, it is shared
                       across the entire batch.

        Returns:
            Int64 array of shape ``(batch,)`` with sampled token ids.

        Raises:
            ValueError: If *logits* is not 2-D.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 2-D (batch, vocab_size), "
                f"got shape {logits.shape}."
            )
        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]
        output     = np.empty(batch_size, dtype=np.int64)

        for b in range(batch_size):
            row_ids: Optional[np.ndarray] = None
            if input_ids is not None:
                ids_arr = np.asarray(input_ids)
                row_ids = ids_arr[b] if ids_arr.ndim == 2 else ids_arr
            probs     = self._compute_probs(logits[b], row_ids)
            output[b] = int(self._rng.choice(vocab_size, p=probs))

        return output

    def reset_rng(self, seed: int) -> None:
        """Re-seed the internal random number generator.

        Args:
            seed: New non-negative integer seed.

        Raises:
            ValueError: If *seed* is negative.
        """
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_probs(
        self,
        logits: np.ndarray,
        input_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply all sampling filters and return a normalised probability vector.

        All transformations are applied in-place on a single float64 working
        copy of the input logits to avoid intermediate full-vocabulary
        allocations.

        Args:
            logits:    1-D float32 logit array of length ``vocab_size``.
            input_ids: Optional array of previously generated token ids.

        Returns:
            1-D float64 array of length ``vocab_size`` summing to 1.0.
        """
        cfg = self._cfg

        # Single working allocation (float64 for softmax numerical stability).
        work = logits.astype(np.float64)

        # Step 1 — repetition penalty (in-place).
        if cfg.repetition_penalty != 1.0 and input_ids is not None:
            ids = np.asarray(input_ids).ravel().astype(np.int64)
            # Discard out-of-range ids.
            ids = ids[(ids >= 0) & (ids < len(work))]
            if ids.size > 0:
                penalty    = cfg.repetition_penalty
                vals       = work[ids]
                work[ids]  = np.where(vals > 0, vals / penalty, vals * penalty)

        # Step 2 — temperature scaling (in-place).
        work /= cfg.temperature

        # Step 3 — softmax (in-place).
        work -= np.max(work)
        np.exp(work, out=work)
        total = work.sum()
        if total < 1e-30:
            # Extreme underflow: fall back to argmax greedy.
            work[:] = 0.0
            work[int(np.argmax(logits))] = 1.0
            return work
        work /= total

        # Step 4 — min-p filter (in-place: zero out below threshold).
        if cfg.min_p > 0.0:
            threshold        = cfg.min_p * float(work.max())
            work[work < threshold] = 0.0

        # Step 5 — top-k filter (in-place: zero out below top-k).
        if cfg.top_k > 0:
            k = min(cfg.top_k, len(work))
            if k < len(work):
                # argpartition: first k entries are the k largest probs.
                top_k_idx          = np.argpartition(-work, k)[:k]
                keep_mask          = np.zeros(len(work), dtype=np.bool_)
                keep_mask[top_k_idx] = True
                work[~keep_mask]   = 0.0

        # Step 6 — top-p (nucleus) filter (in-place).
        if cfg.top_p < 1.0:
            sorted_idx = np.argsort(-work)
            cumulative = np.cumsum(work[sorted_idx])
            # Find the first index where cumulative prob reaches top_p.
            at_or_over = cumulative >= cfg.top_p
            if at_or_over.any():
                first_idx = int(np.argmax(at_or_over))
                # Zero out everything beyond the nucleus boundary.
                work[sorted_idx[first_idx + 1 :]] = 0.0

        # Step 7 — renormalise.
        total = work.sum()
        if total < 1e-12:
            # Fallback: place all mass on the highest-logit token.
            work[:] = 0.0
            work[int(np.argmax(logits))] = 1.0
        else:
            work /= total

        return work
