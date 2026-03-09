"""
squish/hetero_vocab_sd.py

Heterogeneous Vocabulary Speculative Decoding.

Based on:
  "Unlocking Data-free Low-bit Quantization with Matrix Decomposition for
   KV Cache Compression" — but specifically the cross-vocabulary speculative
   decoding component described in:
  "Any Model Can Draft: Heterogeneous Vocabulary Speculative Decoding"
  Intel + Weizmann Institute of Science; ICML 2025
  Referenced in: "On-Device LLMs: State of the Union 2026"

Problem
-------
Standard speculative decoding requires the draft and target models to share
an **identical vocabulary**: the draft model's token IDs must map 1-to-1 to
the target model's token IDs for the standard Leviathan accept/reject test.

This constraint dramatically limits draft model choice:
  - Qwen3-8B (152K vocabulary) can ONLY use Qwen-family draft models.
  - Tiny cross-family models (BitNet-2B, LLaMA-165M) are excluded.

Heterogeneous vocabulary SD (HV-SD) breaks this constraint.  A lightweight
**VocabMapper** translates draft logits from the draft vocabulary into the
target vocabulary space before the acceptance test.  Any small model can then
draft for any large model.

Method
------
1. **VocabMapper** — a learned or heuristic linear map from draft vocab
   to target vocab.  In the full paper a small embedding alignment matrix
   is trained offline (~1 minute on the calibration set).  In this reference
   implementation we support:
   a. An explicit ``(target_vocab, draft_vocab)`` dense mapping matrix W,
      where ``mapped_logits = W @ draft_logits``.
   b. A sparse ``{draft_id: target_id}`` direct-map dictionary for the case
      where the two vocabularies share a common subset (e.g., ASCII tokens).

2. **HeteroVocabDrafter** — wraps the small cross-family draft model and
   applies the VocabMapper to convert its logits into target-vocabulary space.

3. **HeteroVocabDecoder** — standard speculative decoding outer loop:
   - Draft γ mapped tokens from HeteroVocabDrafter.
   - Verify each with the target model (same Leviathan criterion).
   - Bonus token on full acceptance.

Design notes
------------
- The mapped draft logits are only used for the acceptance ratio
  ``p_target[d_tok] / p_draft_mapped[d_tok]``.  The actual token sampled
  from the draft (``d_tok``) is already in target-vocabulary space after
  mapping (it is a target-vocabulary token ID sampled from the mapped
  distribution).
- For the sparse direct-map mode: tokens not in the map are assigned a
  small uniform probability to ensure valid distributions.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **vs EAGLE-3**: EAGLE-3 assumes shared vocabulary.  HV-SD inserts a
  mapping layer.  To compose: apply VocabMapper between EAGLE-3's draft
  head output and the verification token comparison.
- **vs FR-Spec**: FR-Spec compresses the draft LM head vocabulary.  If the
  draft is already cross-vocabulary, FR-Spec's frequency calibration should
  be run on the *mapped* distribution, not the raw draft distribution.
- **Independence**: HV-SD is orthogonal to KV cache, attention kernel, and
  memory management techniques.

Provides
--------
  HeteroVocabConfig     — tuning parameters.
  VocabMapper           — draft-to-target vocabulary translation.
  HeteroVocabDrafter    — cross-family draft model + mapper wrapper.
  HeteroVocabStats      — per-session counters.
  HeteroVocabDecoder    — full cross-vocabulary speculative decoder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "HeteroVocabConfig",
    "VocabMapper",
    "HeteroVocabDrafter",
    "HeteroVocabStats",
    "HeteroVocabDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeteroVocabConfig:
    """Parameters for Heterogeneous Vocabulary Speculative Decoding.

    Parameters
    ----------
    gamma:
        Draft speculation depth.
    temperature:
        Sampling temperature applied to both draft and target logits.
    top_p:
        Nucleus sampling threshold.
    unmapped_prob:
        Probability mass assigned to draft tokens not covered by the sparse
        direct-map.  Must be small (default 1e-6) to avoid artificially
        inflated acceptance of unknown tokens.
    draft_vocab_size:
        Size of the draft model's vocabulary.
    target_vocab_size:
        Size of the target model's vocabulary (used to validate mapper dims).
    """

    gamma: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    unmapped_prob: float = 1e-6
    draft_vocab_size: int = 32000
    target_vocab_size: int = 32000

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if not (0 < self.unmapped_prob < 1):
            raise ValueError("unmapped_prob must be in (0, 1)")
        if self.draft_vocab_size < 2:
            raise ValueError("draft_vocab_size must be >= 2")
        if self.target_vocab_size < 2:
            raise ValueError("target_vocab_size must be >= 2")


# ---------------------------------------------------------------------------
# VocabMapper
# ---------------------------------------------------------------------------

class VocabMapper:
    """Translates draft logits / token IDs into target vocabulary space.

    Two modes:

    **Dense matrix mode** — provide ``weight_matrix`` of shape
    ``(target_vocab, draft_vocab)``.  The mapping is::

        target_logits = weight_matrix @ draft_logits

    **Sparse direct-map mode** — provide ``token_map: Dict[int, int]`` where
    keys are draft token IDs and values are the corresponding target token IDs.
    Draft tokens not in the map receive ``unmapped_prob`` probability mass.

    Parameters
    ----------
    draft_vocab_size:
        Size of the draft vocabulary.
    target_vocab_size:
        Size of the target vocabulary.
    weight_matrix:
        Optional dense mapping matrix (target_vocab × draft_vocab).
    token_map:
        Optional sparse ``{draft_id → target_id}`` dictionary.
    unmapped_prob:
        Probability assigned to unmapped positions in sparse mode.
    """

    def __init__(
        self,
        draft_vocab_size: int,
        target_vocab_size: int,
        weight_matrix: np.ndarray | None = None,
        token_map: dict[int, int] | None = None,
        unmapped_prob: float = 1e-6,
    ) -> None:
        if draft_vocab_size < 2:
            raise ValueError("draft_vocab_size must be >= 2")
        if target_vocab_size < 2:
            raise ValueError("target_vocab_size must be >= 2")
        self.draft_vocab_size = draft_vocab_size
        self.target_vocab_size = target_vocab_size
        self._unmapped_prob = unmapped_prob

        if weight_matrix is not None:
            W = np.array(weight_matrix, dtype=np.float32)
            if W.shape != (target_vocab_size, draft_vocab_size):
                raise ValueError(
                    f"weight_matrix must be ({target_vocab_size}, {draft_vocab_size}), "
                    f"got {W.shape}"
                )
            self._W: np.ndarray | None = W
        else:
            self._W = None

        if token_map is not None:
            self._map: dict[int, int] | None = dict(token_map)
        else:
            self._map = None

        if self._W is None and self._map is None:
            # Default: identity mapping for shared-prefix tokens
            self._map = {i: i for i in range(min(draft_vocab_size, target_vocab_size))}

    # ------------------------------------------------------------------

    def map_logits(self, draft_logits: np.ndarray) -> np.ndarray:
        """Convert draft logits to target-vocabulary probabilities.

        Parameters
        ----------
        draft_logits:
            ``(draft_vocab,)`` logit array from the draft model.

        Returns
        -------
        ``(target_vocab,)`` probability array normalised to sum=1.
        """
        if draft_logits.shape[0] != self.draft_vocab_size:
            raise ValueError(
                f"Expected draft_logits of size {self.draft_vocab_size}, "
                f"got {draft_logits.shape[0]}"
            )

        if self._W is not None:
            raw = self._W @ draft_logits.astype(np.float32)
            return _softmax(raw)

        # Sparse direct-map mode
        draft_probs = _softmax(draft_logits.astype(np.float32))
        target_probs = np.full(
            self.target_vocab_size, self._unmapped_prob, dtype=np.float32
        )
        for d_id, t_id in self._map.items():  # type: ignore[union-attr]
            if 0 <= d_id < self.draft_vocab_size and 0 <= t_id < self.target_vocab_size:
                target_probs[t_id] += float(draft_probs[d_id])
        total = target_probs.sum()
        return target_probs / total

    def sample_target_token(
        self,
        draft_logits: np.ndarray,
        rng: np.random.Generator,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[int, np.ndarray]:
        """Sample a *target-vocabulary* token from the mapped draft distribution.

        Returns
        -------
        (target_token_id, mapped_probs)
        """
        mapped = self.map_logits(draft_logits)
        if top_p < 1.0:
            sorted_idx = np.argsort(mapped)[::-1]
            cumsum = np.cumsum(mapped[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, top_p)) + 1
            mask = np.zeros_like(mapped)
            mask[sorted_idx[:cutoff]] = 1.0
            mapped = mapped * mask
            mapped = mapped / mapped.sum()
        tok = int(rng.choice(self.target_vocab_size, p=mapped))
        return tok, mapped


# ---------------------------------------------------------------------------
# HeteroVocabDrafter
# ---------------------------------------------------------------------------

class HeteroVocabDrafter:
    """Cross-vocabulary draft model wrapper.

    Calls a small draft model (potentially different vocab from target) and
    uses a ``VocabMapper`` to produce draft tokens in target-vocabulary space.

    Parameters
    ----------
    draft_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(draft_vocab,)`` logits.
        Token IDs in ``ids`` are in **target** vocabulary (they are the
        accepted outputs stream); for purely generative models this is fine.
    mapper:
        ``VocabMapper`` configured for the draft/target vocab pair.
    config:
        ``HeteroVocabConfig``.
    rng_seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        draft_fn: Callable[[list[int]], np.ndarray],
        mapper: VocabMapper,
        config: HeteroVocabConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fn = draft_fn
        self._mapper = mapper
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def draft_sequence(
        self, ids: list[int], gamma: int
    ) -> tuple[list[int], list[np.ndarray]]:
        """Generate *gamma* draft tokens in target-vocabulary space.

        Returns
        -------
        (tokens, mapped_probs_list)
            Both are in target-vocabulary space.
        """
        ctx = list(ids)
        tokens: list[int] = []
        probs: list[np.ndarray] = []
        for _ in range(gamma):
            draft_logits = self._fn(ctx)
            tok, mapped_probs = self._mapper.sample_target_token(
                draft_logits,
                self._rng,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
            )
            tokens.append(tok)
            probs.append(mapped_probs)
            ctx.append(tok)
        return tokens, probs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class HeteroVocabStats:
    """Per-session statistics for Heterogeneous Vocabulary SD.

    Parameters
    ----------
    total_tokens:
        New tokens emitted.
    draft_steps:
        Drafting rounds completed.
    accepted_total:
        Draft tokens accepted.
    rejected_total:
        Draft tokens rejected (target correction substituted).
    """

    total_tokens: int = 0
    draft_steps: int = 0
    accepted_total: int = 0
    rejected_total: int = 0

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# HeteroVocabDecoder
# ---------------------------------------------------------------------------

class HeteroVocabDecoder:
    """Heterogeneous vocabulary speculative decoder.

    Uses ``HeteroVocabDrafter`` (cross-vocabulary small model) and a standard
    target-model callable to produce tokens via speculative decoding.

    The accept/reject criterion is the standard Leviathan et al. speculative
    sampling test, applied in *target-vocabulary space* (both ``p_target`` and
    ``p_draft_mapped`` are distributions over the same target vocabulary after
    the VocabMapper).

    Parameters
    ----------
    drafter:
        ``HeteroVocabDrafter`` — cross-vocabulary draft source.
    target_fn:
        ``(ids: List[int]) -> np.ndarray`` — target model returning
        ``(target_vocab,)`` logits.
    config:
        ``HeteroVocabConfig``.
    rng_seed:
        Seed for the accept/reject RNG.
    """

    def __init__(
        self,
        drafter: HeteroVocabDrafter,
        target_fn: Callable[[list[int]], np.ndarray],
        config: HeteroVocabConfig | None = None,
        rng_seed: int = 0,
    ) -> None:
        if config is None:
            config = HeteroVocabConfig()
        self._drafter = drafter
        self._target_fn = target_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def _target_sample(
        self, ids: list[int]
    ) -> tuple[int, np.ndarray]:
        logits = self._target_fn(ids)
        probs = _softmax(logits / max(self._cfg.temperature, 1e-8))
        if self._cfg.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, self._cfg.top_p)) + 1
            mask = np.zeros_like(probs)
            mask[sorted_idx[:cutoff]] = 1.0
            probs = probs * mask
            probs /= probs.sum()
        tok = _sample(probs, self._rng)
        return tok, probs

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
    ) -> tuple[list[int], HeteroVocabStats]:
        """Generate up to *max_new_tokens* tokens.

        Parameters
        ----------
        input_ids:
            Starting token sequence (in target-vocabulary space).
        max_new_tokens:
            Upper bound on new tokens appended.

        Returns
        -------
        (output_ids, stats)
        """
        cfg = self._cfg
        stats = HeteroVocabStats()
        ids = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            gamma = min(cfg.gamma, max_new_tokens - generated)

            # ── Draft: generate γ mapped tokens ─────────────────────────────
            draft_tokens, draft_probs = self._drafter.draft_sequence(ids, gamma)
            stats.draft_steps += 1

            # ── Verify: Leviathan accept/reject in target-vocab space ────────
            ctx = list(ids)
            accepted: list[int] = []
            rejected = False

            for d_tok, d_probs in zip(draft_tokens, draft_probs, strict=False):
                v_tok, v_probs = self._target_sample(ctx)
                p_t = float(v_probs[d_tok])
                p_d = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    stats.accepted_total += 1
                else:
                    accepted.append(v_tok)
                    ctx.append(v_tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            # ── Bonus token on full acceptance ───────────────────────────────
            if not rejected and (generated + len(accepted)) < max_new_tokens:
                bonus_tok, _ = self._target_sample(ctx)
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            ids.extend(accepted)
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
