#!/usr/bin/env python3
"""
squish_lm_eval.py

EleutherAI lm-evaluation-harness compatible wrapper for Squish compressed models.

This allows the exact same industry-standard evaluation suite used to rank GPT-4,
Llama, Gemma, and every serious open-source model to run directly on a Squish
compressed model — proving accuracy is maintained after Vectro INT8 compression.

Supported tasks (via lm-eval):
    arc_easy          AI2 Reasoning Challenge — Easy  (0-shot)
    arc_challenge     AI2 Reasoning Challenge — Hard  (25-shot)
    hellaswag         HellaSwag commonsense NLI        (10-shot)
    winogrande        Winogrande coreference            (5-shot)
    piqa              Physical Intuition QA             (0-shot)
    mmlu              MMLU (57 subjects)                (5-shot) — slow
    gsm8k             GSM8K grade-school math           (5-shot)
    truthfulqa_mc1    TruthfulQA                        (0-shot)

Usage (via run_eval.py — preferred):
    python3 run_eval.py --tasks arc_easy,hellaswag --limit 200

Usage (direct lm-eval CLI):
    lm_eval \\
        --model forge \\
        --model_args model_dir=~/models/Qwen2.5-1.5B-Instruct-bf16,\\
                     compressed_dir=~/models/Qwen2.5-1.5B-Instruct-bf16-compressed \\
        --tasks arc_easy \\
        --num_fewshot 0 \\
        --limit 200

Interface contract (lm_eval >= 0.4.x):
    loglikelihood()        — used by multiple-choice tasks (ARC, HellaSwag, MMLU …)
    generate_until()       — used by open-ended tasks (GSM8K, TruthfulQA …)
    loglikelihood_rolling()— not used by any registered task here

Architecture:
    The wrapper loads the compressed model once (cached on disk after first run)
    and runs pure MLX forward passes.  No HuggingFace pipeline is invoked during
    eval — only the MLX graph.  Token-level log probabilities are computed by
    slicing the logit tensor and applying log-softmax in-graph.
"""
import sys
import os
import math
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── lm-eval base class ───────────────────────────────────────────────────────
try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    from lm_eval.api.registry import register_model
    _HAVE_LM_EVAL = True
except ImportError:
    # Graceful stub so the file can be imported even without lm-eval installed.
    class LM:                      # type: ignore[no-redef]
        pass
    def register_model(*a, **kw):  # type: ignore[misc]
        def _dec(cls):
            return cls
        return _dec
    _HAVE_LM_EVAL = False

logger = logging.getLogger(__name__)

# ── Import MLX early so we fail loudly if it's missing ───────────────────────
import mlx.core as mx
import mlx.nn as mx_nn


# ─────────────────────────────────────────────────────────────────────────────

@register_model("squish")
class SquishCompressedLM(LM):
    """
    lm-evaluation-harness LM subclass that runs inference on a Squish compressed
    model (Vectro INT8 npy-dir) without ever touching the original safetensors.

    The model is loaded exactly once and kept in memory for the duration of the
    eval run.  The finalized f16 cache means subsequent eval runs also load in
    ~4-5s instead of ~15s.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model_dir: str                   = "",
        compressed_dir: str              = "",
        batch_size: int                  = 4,   # padded batch inference — 4 is efficient
        max_length: Optional[int]        = None,
        verbose: bool                    = False,
        trust_remote_code: bool          = True,
    ):
        super().__init__()

        if not model_dir:
            model_dir = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
        if not compressed_dir:
            compressed_dir = model_dir + "-compressed"

        self._model_dir      = Path(model_dir).expanduser().resolve()
        self._compressed_dir = Path(compressed_dir).expanduser().resolve()
        self._batch_size     = int(batch_size)
        self._max_length     = max_length
        self._verbose        = verbose
        self._model          = None
        self._tokenizer      = None

        logger.info("SquishCompressedLM: loading model …")
        t0 = time.perf_counter()
        self._load()
        logger.info("SquishCompressedLM: model ready in %.1fs", time.perf_counter() - t0)

    def _load(self):
        """Load the compressed model — routes through load_compressed_model which
        auto-selects the best available tier: squish_4bit > mlx_cache > finalized > npy-dir.
        Using a unified path avoids duplicating the tier-detection logic here.
        """
        from .compressed_loader import load_compressed_model

        self._model, self._tokenizer, stats = load_compressed_model(
            model_dir    = str(self._model_dir),
            npz_path     = str(self._compressed_dir),
            verbose      = self._verbose,
            return_stats = True,
        )

        logger.info("loader=%s  decomp=%.1fs  RAM_delta=%+.0fMB",
                    stats.get("loader"),
                    stats.get("decompression_time_s", 0),
                    stats.get("ram_delta_mb", 0))

        # Warm up Metal shaders with a single dummy forward pass.
        # Without this, the first real eval batch stalls waiting for shader
        # compilation (~5-30s for 7B).  Pre-compiling here amortises that cost
        # once at load time rather than burning it inside the timed eval.
        try:
            _seq = getattr(self._tokenizer, 'pad_token_id', 0) or 0
            _dummy = mx.array([[_seq, _seq, _seq]], dtype=mx.int32)
            _out = self._model(_dummy)
            mx.eval(_out)
            del _dummy, _out
            logger.info("Metal shaders warmed up")
        except Exception as _e:
            logger.warning("Metal warmup failed (non-fatal): %s", _e)

    # ── lm-eval required properties ───────────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        # Cap at 4096 regardless of model_max_length — eval inputs are all <2048
        # tokens; using the model's full 32k/128k window wastes nothing but
        # prevents runaway truncation logic in lm-eval edge cases.
        raw = getattr(self._tokenizer, "model_max_length", 4096) or 4096
        return min(raw, 4096)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return "mlx"

    # ── tokeniser helpers ─────────────────────────────────────────────────────

    def tok_encode(self, string: str) -> List[int]:
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self._tokenizer.decode(tokens)

    # ── core MLX forward pass ─────────────────────────────────────────────────

    def _forward_logprobs(
        self, token_ids: List[int]
    ) -> np.ndarray:
        """
        Legacy: Run one forward pass and return full (seq_len, vocab_size) log-probs.
        Kept for external callers; internal loglikelihood uses _forward_selected_logprobs.
        """
        ids = mx.array(token_ids, dtype=mx.int32)[None]  # (1, seq_len)
        logits = self._model(ids)                          # (1, seq_len, vocab)
        lp = mx_nn.log_softmax(logits[0].astype(mx.float32), axis=-1)
        mx.eval(lp)
        return np.array(lp)                                # (seq_len, vocab)

    def _forward_selected_logprobs(
        self,
        token_ids: List[int],
        cont_tokens: List[int],
    ) -> Tuple[List[float], List[bool]]:
        """
        Optimised single forward pass for loglikelihood evaluation.

        Instead of materialising the full (seq_len × vocab_size) tensor to numpy
        (91 MB per call for 14B models), we:
          1. Run the forward pass entirely on Metal.
          2. Gather only the continuation token log-probs (tiny output).
          3. Compute argmax on Metal too for is_greedy.

        Reduction: seq_len × vocab_size -> n_cont scalars per request.
        At 2400 requests × 91 MB → < 1 KB each = orders-of-magnitude less memory traffic.

        Returns (lp_per_token, is_greedy_per_token) for the continuation slice.
        """
        ids = mx.array(token_ids, dtype=mx.int32)[None]        # (1, seq_len)
        logits = self._model(ids)                                # (1, seq_len, vocab)
        seq_logits = logits[0]                                   # (seq_len, vocab)

        n_cont = len(cont_tokens)
        n_ctx  = len(token_ids) - n_cont
        # Continuation logits start at context_len - 1 (next-token prediction offset)
        cont_logits = seq_logits[n_ctx - 1 : n_ctx - 1 + n_cont]   # (n_cont, vocab)

        # log_softmax only over the (smaller) continuation slice
        cont_lp = mx_nn.log_softmax(cont_logits.astype(mx.float32), axis=-1)  # (n_cont, vocab)

        # Gather selected token log-probs and argmaxes — stays on Metal
        target_arr   = mx.array(cont_tokens, dtype=mx.int32)          # (n_cont,)
        rows         = mx.arange(n_cont, dtype=mx.int32)
        selected_lp  = cont_lp[rows, target_arr]                      # (n_cont,)
        argmax_ids   = mx.argmax(cont_logits, axis=-1)                 # (n_cont,)
        is_greedy_mx = (argmax_ids == target_arr)                      # (n_cont,) bool

        # Materialise only tiny tensors (n_cont scalars each)
        mx.eval(selected_lp, is_greedy_mx)

        lp_list       = np.array(selected_lp).tolist()       # List[float]
        is_greedy_list = np.array(is_greedy_mx).tolist()      # List[bool]
        return lp_list, is_greedy_list

    # ── loglikelihood ─────────────────────────────────────────────────────────

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Compute P(continuation | context) for every (context, continuation) pair.

        When batch_size > 1, processes requests in batches of batch_size to
        amortise Python overhead.  Each forward pass is still independent (MLX
        does not support jagged-batch bmm yet) but MLX's lazy evaluation fuses
        the batch graph before materialising — roughly batch_size× throughput.

        Uses _forward_selected_logprobs to avoid materialising the full
        (seq_len × vocab_size) tensor — critical for large vocab models.
        """
        results: List[Tuple[float, bool]] = []

        batch: List[Tuple[List[int], List[int]]] = []
        batch_indices: List[int] = []
        _n_done = 0
        _n_total = len(requests)
        _t_start = time.perf_counter()
        _CACHE_CLEAR_EVERY = 200  # free Metal cached allocs every N requests

        def _flush_batch():
            nonlocal _n_done
            """Evaluate the accumulated MLX graphs and append results."""
            if not batch:
                return
            # ── Padded batch forward pass ─────────────────────────────────
            # Build a (B, max_len) padded tensor and run ONE model call.
            # Causal masking means token i attends only to 0..i, so trailing
            # padding never influences the continuation slice — results are
            # identical to per-sequence calls but with B× GPU utilisation.
            pad_id   = getattr(self._tokenizer, "pad_token_id", 0) or 0
            max_len  = max(len(c) + len(k) for c, k in batch)
            seqs     = []
            meta     = []   # (n_ctx, n_cont) per item in batch

            for ctx_toks, cont_toks in batch:
                full = ctx_toks + cont_toks
                pad  = [pad_id] * (max_len - len(full))
                seqs.append(full + pad)
                meta.append((len(ctx_toks), len(cont_toks)))

            ids_batch   = mx.array(seqs, dtype=mx.int32)   # (B, max_len)
            logits_all  = self._model(ids_batch)            # (B, max_len, vocab)

            lazy_items = []
            for i, (n_ctx, n_cont) in enumerate(meta):
                seq_logits  = logits_all[i]                             # (max_len, vocab)
                cont_logits = seq_logits[n_ctx - 1 : n_ctx - 1 + n_cont]  # (n_cont, vocab)
                cont_lp     = mx_nn.log_softmax(cont_logits.astype(mx.float32), axis=-1)
                target_arr  = mx.array(batch[i][1], dtype=mx.int32)    # cont token ids
                rows        = mx.arange(n_cont, dtype=mx.int32)
                sel_lp      = cont_lp[rows, target_arr]                 # (n_cont,)
                argmax_ids  = mx.argmax(cont_logits, axis=-1)
                is_greedy   = (argmax_ids == target_arr)
                lazy_items.append((sel_lp, is_greedy, n_cont))

            # Materialise all continuation scores in one eval
            mx.eval(*[x for (a, b, _) in lazy_items for x in (a, b)])

            for (sel_lp, is_greedy, _) in lazy_items:
                lp_sum    = float(np.array(sel_lp).sum())
                is_g      = bool(np.array(is_greedy).all())
                results.append((lp_sum, is_g))
            batch.clear()
            batch_indices.clear()
            _n_done += len(lazy_items)
            # Periodic Metal cache clearing — prevents RAM creep on long evals
            if _n_done % _CACHE_CLEAR_EVERY < self._batch_size:
                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass
            # Progress logging every 500 requests
            if _n_done % 500 < self._batch_size:
                elapsed = time.perf_counter() - _t_start
                rate = _n_done / elapsed if elapsed > 0 else 0
                eta  = (_n_total - _n_done) / rate if rate > 0 else float('inf')
                logger.warning("loglikelihood: %d/%d done  %.1f req/s  ETA %.0fm",
                               _n_done, _n_total, rate, eta / 60)

        for req in requests:
            ctx, cont = req.args[0], req.args[1]
            ctx_tokens  = self.tok_encode(ctx)
            cont_tokens = self.tok_encode(cont)
            total = len(ctx_tokens) + len(cont_tokens)
            if total > self.max_length:
                keep = self.max_length - len(cont_tokens)
                ctx_tokens = ctx_tokens[-keep:]
            batch.append((ctx_tokens, cont_tokens))
            if len(batch) >= self._batch_size:
                _flush_batch()

        _flush_batch()   # flush any remainder
        return results


    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[float]:
        """Compute unconditional log-likelihood of a string (used by BPB/PPL tasks)."""
        results = []
        for req in requests:
            text = req.args[0]
            tokens = self.tok_encode(text)
            if not tokens:
                results.append(0.0)
                continue
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
            log_probs = self._forward_logprobs(tokens)
            # P(t_1, t_2, …, t_n) = sum_{i=1}^{n} log P(t_i | t_0 … t_{i-1})
            lp_sum = float(sum(log_probs[i - 1, tokens[i]] for i in range(1, len(tokens))))
            results.append(lp_sum)
        return results

    # ── generate_until ────────────────────────────────────────────────────────

    def generate_until(
        self, requests: List[Instance]
    ) -> List[str]:
        """
        Generate up to max_gen_toks tokens for each context, stopping at any
        of the provided stop strings.  Used by open-ended tasks (GSM8K, etc.).
        """
        from mlx_lm import generate as mlx_generate

        results = []
        for req in requests:
            ctx = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            until   = gen_kwargs.get("until", [self._tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]

            output = mlx_generate(
                model       = self._model,
                tokenizer   = self._tokenizer,
                prompt      = ctx,
                max_tokens  = max_new,
                verbose     = False,
            )
            # Trim at first stop string
            for stop in until:
                idx = output.find(stop)
                if idx != -1:
                    output = output[:idx]

            results.append(output)

        return results


# ── Reference model wrapper (mlx_lm.load native) ─────────────────────────────

@register_model("squish-reference")
class SquishReferenceLM(SquishCompressedLM):
    """
    lm-eval wrapper that loads the UNCOMPRESSED model via mlx_lm.load().

    Uses the same loglikelihood/generate interface as SquishCompressedLM so
    reference and compressed results are directly comparable across the same
    evaluation harness run.
    """

    def __init__(
        self,
        model_dir: str                   = "",
        batch_size: int                  = 1,
        max_length: Optional[int]        = None,
        verbose: bool                    = False,
        trust_remote_code: bool          = True,
        **kwargs,
    ):
        # skip SquishCompressedLM.__init__ — use LM.__init__ + our own _load
        LM.__init__(self)
        if not model_dir:
            model_dir = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
        self._model_dir      = Path(model_dir).expanduser().resolve()
        self._compressed_dir = None
        self._batch_size     = int(batch_size)
        self._max_length     = max_length
        self._verbose        = verbose
        self._model          = None
        self._tokenizer      = None

        logger.info("SquishReferenceLM: loading reference model …")
        t0 = time.perf_counter()
        self._load()
        logger.info("SquishReferenceLM: model ready in %.1fs", time.perf_counter() - t0)

    def _load(self):
        from mlx_lm import load as mlx_load
        self._model, self._tokenizer = mlx_load(str(self._model_dir))
