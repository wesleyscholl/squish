#!/usr/bin/env python3
"""
squish/speculative.py

Speculative decoding for Squish — use a small draft model to propose K tokens
per step, then verify all K+1 positions with the large target model in a single
forward pass.

Algorithm: Leviathan et al., 2022 — "Fast Inference from Transformers via
Speculative Decoding", Algorithm 1.
https://arxiv.org/abs/2211.17192

Expected speedup vs auto-regressive decoding:
    7B  target + 0.5B draft  → 1.8–2.5×   (typical chat distributions)
    14B target + 1.5B draft  → 2.0–3.0×

Usage (from server.py, or standalone):
    from squish.speculative import SpeculativeGenerator, load_draft_model

    draft = load_draft_model(
        model_dir="~/models/Qwen2.5-0.5B-Instruct-bf16",
        compressed_dir="~/models/squish_0.5b",
    )
    gen = SpeculativeGenerator(target_model, target_tokenizer, draft)
    for token_text, finish_reason in gen.stream(prompt, max_tokens=512):
        ...

Standalone test:
    python3 -m squish.speculative \
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \
        --compressed-dir ~/models/squish_7b \
        --draft-model ~/models/Qwen2.5-0.5B-Instruct-bf16 \
        --draft-compressed ~/models/squish_0.5b \
        --prompt "Explain quantum entanglement in one paragraph."
"""

import sys
import time
import logging
from pathlib import Path
from typing import Callable, Generator, Iterator, List, Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as mx_nn

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────

_DEFAULT_K          = 4      # draft tokens to speculate per step
_DEFAULT_TEMP       = 0.7
_DEFAULT_TOP_P      = 0.9
_MAX_SPEC_TOKENS    = 8      # cap K at this regardless of setting

# ── Sampling helpers ──────────────────────────────────────────────────────────

def _softmax_np(logits_row: np.ndarray, temp: float) -> np.ndarray:
    """Compute softmax(logits / temp) in float64 for numerical stability."""
    logits64 = logits_row.astype(np.float64)
    logits64 /= max(temp, 1e-8)
    logits64 -= logits64.max()
    probs = np.exp(logits64)
    probs /= probs.sum()
    return probs.astype(np.float32)


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Zero out tokens outside the nucleus (top-p) and renormalise."""
    if top_p >= 1.0:
        return probs
    idx    = np.argsort(-probs)
    cumsum = np.cumsum(probs[idx])
    cutoff = int((cumsum <= top_p).sum()) + 1
    mask   = np.zeros_like(probs)
    mask[idx[:max(1, cutoff)]] = 1.0
    filtered = probs * mask
    total    = filtered.sum()
    return filtered / total if total > 0 else probs


def _sample(probs: np.ndarray) -> int:
    """Multinomial sample — returns a single token id."""
    try:
        return int(np.random.choice(len(probs), p=probs))
    except ValueError:
        # Numerical issues — fall back to argmax
        return int(np.argmax(probs))


def _greedy(logits_row: np.ndarray) -> int:
    return int(np.argmax(logits_row))


def _get_logits(model, ids: List[int]) -> np.ndarray:
    """
    Single synchronous forward pass.

    Returns the *last* row of logits as a float32 numpy array
    (shape: vocab_size).  This is the next-token prediction.
    """
    x      = mx.array(ids, dtype=mx.int32)[None]   # (1, seq_len)
    out    = model(x)                               # (1, seq_len, vocab)
    last   = out[0, -1]                             # (vocab,)
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _get_all_logits(model, ids: List[int], n_positions: int) -> np.ndarray:
    """
    Verification pass: run the model and return the last ``n_positions`` rows.

    Shape: (n_positions, vocab_size).
    Used to verify K draft tokens + produce one bonus token simultaneously.
    """
    x   = mx.array(ids, dtype=mx.int32)[None]      # (1, seq_len)
    out = model(x)                                  # (1, seq_len, vocab)
    # We want positions [-n_positions:] − the verification slice.
    rows = out[0, -n_positions:]                    # (n_positions, vocab)
    mx.eval(rows)
    return np.array(rows, dtype=np.float32)         # (n_positions, vocab)


# ── Draft model loader ────────────────────────────────────────────────────────

def load_draft_model(
    model_dir: str,
    compressed_dir: str = "",
    verbose: bool = False,
):
    """
    Load the small draft model through the same compressed_loader pipeline.

    Returns (model, tokenizer).  Vocabulary must be compatible with the target
    (same tokeniser family — e.g. both Qwen2.5, both Llama…).
    """
    from .compressed_loader import load_compressed_model

    model_dir_p = Path(model_dir).expanduser()
    comp_dir_p  = Path(compressed_dir).expanduser() if compressed_dir else \
                  Path(model_dir_p.parent / (model_dir_p.name + "-compressed"))

    model, tokenizer = load_compressed_model(
        model_dir  = str(model_dir_p),
        npz_path   = str(comp_dir_p),
        verbose    = verbose,
    )
    return model, tokenizer


# ── Speculative Generator ─────────────────────────────────────────────────────

class SpeculativeGenerator:
    """
    Speculative decoding wrapper.

    Wraps a (target_model, target_tokenizer) pair and an optional
    (draft_model, draft_tokenizer).  When the draft model is provided it uses
    the speculative algorithm; otherwise it falls back to standard greedy/sampling.

    Both models are assumed to share a common vocabulary (same tokeniser family).
    Token IDs from the draft tokeniser are used as draft candidates and verified
    against the target's distribution.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model             = None,
        draft_tokenizer         = None,
        k: int                  = _DEFAULT_K,
    ):
        self._target  = target_model
        self._ttok    = target_tokenizer
        self._draft   = draft_model
        self._dtok    = draft_tokenizer or target_tokenizer
        self._k       = min(max(1, k), _MAX_SPEC_TOKENS)

        # Acceptance stats (reset per stream() call)
        self.accepted_total  = 0
        self.proposed_total  = 0
        self.steps           = 0

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_total / self.proposed_total
                if self.proposed_total > 0 else 0.0)

    def _reset_stats(self) -> None:
        self.accepted_total = 0
        self.proposed_total = 0
        self.steps = 0

    # ── main streaming API ────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int       = 512,
        temperature: float    = _DEFAULT_TEMP,
        top_p: float          = _DEFAULT_TOP_P,
        stop_ids: Optional[List[List[int]]] = None,
        seed: Optional[int]   = None,
    ) -> Iterator[Tuple[str, Optional[str]]]:
        """
        Yield (token_text, finish_reason_or_None) tuples.
        finish_reason is 'stop' or 'length' on the final token; None otherwise.

        Identical external interface to server.py: _generate_tokens().
        """
        self._reset_stats()
        if seed is not None:
            np.random.seed(seed)
            try:
                mx.random.seed(seed)
            except Exception:
                pass

        eos_id    = getattr(self._ttok, "eos_token_id", None) or 151645
        input_ids = list(self._ttok.encode(prompt))
        stop_ids  = stop_ids or []

        if self._draft is None:
            # No draft model — plain auto-regressive
            yield from self._plain_stream(
                input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )
            return

        yield from self._speculative_stream(
            input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
        )

    # ── speculative inner loop ────────────────────────────────────────────────

    def _speculative_stream(
        self,
        ids: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: List[List[int]],
        eos_id: int,
    ) -> Iterator[Tuple[str, Optional[str]]]:
        """
        Core speculative decoding loop.

        Each step:
        1. Draft model proposes K tokens autoregressively.
        2. Target model verifies all K positions in one forward pass.
        3. Tokens are accepted/rejected per the speculative sampling rule.
        4. At least one bonus token (from target) is always appended.

        This gives the same distribution as sampling from the target alone.
        """
        generated     = 0
        stop_buf: List[int] = []
        context       = list(ids)

        while generated < max_tokens:
            # ── Step 1: draft K tokens ────────────────────────────────────
            draft_ids     : List[int]        = []
            draft_probs   : List[np.ndarray] = []

            for _ in range(self._k):
                logits = _get_logits(self._draft, context + draft_ids)
                probs  = _softmax_np(logits, temperature)
                probs  = _top_p_filter(probs, top_p)
                tok    = _sample(probs)
                draft_ids.append(tok)
                draft_probs.append(probs)
                if tok == eos_id:
                    break

            self.proposed_total += len(draft_ids)
            self.steps += 1

            # ── Step 2: target model verifies all K positions + 1 ────────
            # Feed context + all draft tokens; read last K+1 logit rows.
            full_seq     = context + draft_ids
            n_verify     = len(draft_ids) + 1   # K draft + 1 bonus
            target_rows  = _get_all_logits(self._target, full_seq, n_verify)
            # target_rows[i] is the logit for position (context_len - 1 + i)
            # which predicts token draft_ids[i].

            # ── Step 3: sequential accept/reject ─────────────────────────
            new_tokens: List[int] = []
            accepted   = 0
            for i, (d_tok, d_probs) in enumerate(zip(draft_ids, draft_probs)):
                t_probs = _softmax_np(target_rows[i], temperature)
                t_probs = _top_p_filter(t_probs, top_p)

                p_target = float(t_probs[d_tok])
                p_draft  = float(d_probs[d_tok])

                accept_prob = min(1.0, p_target / max(p_draft, 1e-12))
                if np.random.random() < accept_prob:
                    new_tokens.append(d_tok)
                    accepted += 1
                else:
                    # Rejection: sample from the adjusted distribution
                    adjusted = np.maximum(0.0, t_probs - d_probs)
                    s = adjusted.sum()
                    if s > 0:
                        adjusted /= s
                        fallback_tok = _sample(adjusted)
                    else:
                        fallback_tok = _greedy(target_rows[i])
                    new_tokens.append(fallback_tok)
                    break

            self.accepted_total += accepted

            # ── Step 4: bonus token from target ──────────────────────────
            if accepted == len(draft_ids):
                # All K accepted — greedily sample from the K+1 position
                bonus_probs = _softmax_np(target_rows[len(draft_ids)], temperature)
                bonus_probs = _top_p_filter(bonus_probs, top_p)
                new_tokens.append(_sample(bonus_probs))

            # ── Yield accepted tokens ─────────────────────────────────────
            for tok in new_tokens:
                if tok == eos_id:
                    yield self._tok_text(tok), "stop"
                    return

                tok_text = self._tok_text(tok)
                generated += 1
                stop_buf.append(tok)
                context.append(tok)

                # Check stop sequences
                for seq in stop_ids:
                    if stop_buf[-len(seq):] == seq:
                        yield tok_text, "stop"
                        return
                if len(stop_buf) > 64:
                    stop_buf = stop_buf[-64:]

                if generated >= max_tokens:
                    yield tok_text, "length"
                    return
                yield tok_text, None

        logger.debug(
            "spec: %d steps, %.1f%% acceptance, %d tokens",
            self.steps,
            self.acceptance_rate * 100,
            generated,
        )

    def _tok_text(self, tok_id: int) -> str:
        try:
            return self._ttok.decode([tok_id])
        except Exception:
            return ""

    # ── plain fallback ────────────────────────────────────────────────────────

    def _plain_stream(
        self,
        ids: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: List[List[int]],
        eos_id: int,
    ) -> Iterator[Tuple[str, Optional[str]]]:
        """Plain auto-regressive sampling — used when no draft model is available."""
        context    = list(ids)
        stop_buf   : List[int] = []
        generated  = 0

        for _ in range(max_tokens):
            logits = _get_logits(self._target, context)
            if temperature == 0.0:
                tok = _greedy(logits)
            else:
                probs = _softmax_np(logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)

            if tok == eos_id:
                yield self._tok_text(tok), "stop"
                return

            tok_text = self._tok_text(tok)
            context.append(tok)
            generated += 1
            stop_buf.append(tok)

            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    yield tok_text, "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

            if generated >= max_tokens:
                yield tok_text, "length"
                return
            yield tok_text, None

        yield "", "stop"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test speculative decoding")
    ap.add_argument("--model-dir",        required=True)
    ap.add_argument("--compressed-dir",   required=True)
    ap.add_argument("--draft-model",      default="")
    ap.add_argument("--draft-compressed", default="")
    ap.add_argument("--prompt",           default="What is the capital of France?")
    ap.add_argument("--max-tokens",       type=int, default=200)
    ap.add_argument("--k",                type=int, default=_DEFAULT_K)
    ap.add_argument("--temperature",      type=float, default=0.0)
    args = ap.parse_args()

    from .compressed_loader import load_compressed_model

    print("Loading target model …")
    t0 = time.perf_counter()
    target_model, target_tok = load_compressed_model(
        model_dir=args.model_dir, npz_path=args.compressed_dir, verbose=True,
    )
    print(f"Target loaded in {time.perf_counter() - t0:.1f}s\n")

    draft_model = draft_tok = None
    if args.draft_model:
        print("Loading draft model …")
        t0 = time.perf_counter()
        draft_model, draft_tok = load_draft_model(
            args.draft_model, args.draft_compressed
        )
        print(f"Draft loaded in {time.perf_counter() - t0:.1f}s\n")

    gen = SpeculativeGenerator(
        target_model, target_tok,
        draft_model=draft_model, draft_tokenizer=draft_tok,
        k=args.k,
    )

    print(f"Prompt: {args.prompt!r}\n")
    print("─" * 60)

    t0 = time.perf_counter()
    n_tokens = 0
    for tok_text, finish_reason in gen.stream(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        print(tok_text, end="", flush=True)
        n_tokens += 1
        if finish_reason:
            break

    elapsed = time.perf_counter() - t0
    print(f"\n\n─ {n_tokens} tokens in {elapsed:.2f}s  "
          f"({n_tokens / elapsed:.1f} tok/s)")
    if draft_model is not None:
        print(f"  acceptance rate: {gen.acceptance_rate * 100:.1f}%  "
              f"({gen.steps} steps)")
