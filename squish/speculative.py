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

import logging
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────

_DEFAULT_K          = 4      # draft tokens to speculate per step
_DEFAULT_TEMP       = 0.7
_DEFAULT_TOP_P      = 0.9
_MAX_SPEC_TOKENS    = 8      # cap K at this regardless of setting

# ── FSM adaptive speculation-length controller ────────────────────────────────

class FSMGammaController:
    """Finite-State-Machine adaptive speculation-length controller.

    After each speculative step the caller reports how many tokens were
    proposed and how many were accepted.  The FSM adjusts the draft length
    ``gamma`` by a fixed delta:

    * **Full accept** (``n_accepted >= n_proposed``): ``gamma += 1``,
      clamped at ``max_gamma``.
    * **Any rejection** (``n_accepted < n_proposed``): ``gamma -= 1``,
      clamped at ``min_gamma``.

    Based on: FSM speculation control described in Liu et al. 2025 and the
    AdaEAGLE line of work.  Zero hyperparameter tuning required — the
    controller is self-adapting.

    Parameters
    ----------
    initial_gamma : int
        Starting draft length.
    min_gamma : int
        Floor for ``gamma`` (≥ 1).
    max_gamma : int
        Ceiling for ``gamma``.
    """

    def __init__(
        self,
        initial_gamma: int = 4,
        min_gamma:     int = 2,
        max_gamma:     int = 8,
    ) -> None:
        if min_gamma < 1:
            raise ValueError("min_gamma must be ≥ 1")
        if max_gamma < min_gamma:
            raise ValueError("max_gamma must be ≥ min_gamma")
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gamma     = max(min_gamma, min(max_gamma, initial_gamma))

    def step(self, n_accepted: int, n_proposed: int) -> int:
        """Update ``gamma`` based on the latest speculation result.

        Parameters
        ----------
        n_accepted : int
            Number of draft tokens accepted (not counting bonus token).
        n_proposed : int
            Total draft tokens proposed in this step.

        Returns
        -------
        Updated ``gamma`` value.
        """
        if n_accepted >= n_proposed:
            self.gamma = min(self.gamma + 1, self.max_gamma)
        else:
            self.gamma = max(self.gamma - 1, self.min_gamma)
        return self.gamma

    def reset(self, gamma: "int | None" = None) -> None:
        """Reset ``gamma`` to *gamma* (or midpoint if ``None``)."""
        if gamma is not None:
            self.gamma = max(self.min_gamma, min(self.max_gamma, gamma))
        else:
            self.gamma = (self.min_gamma + self.max_gamma) // 2


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


def _get_logits(model, ids: list[int]) -> np.ndarray:
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


def _get_all_logits(model, ids: list[int], n_positions: int) -> np.ndarray:
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


# ── Stateful (KV-cached) helpers ──────────────────────────────────────────────
# These replace the stateless helpers above when mlx_lm's KV cache is
# available.  The stateless path is kept as a fallback.

def _try_make_model_cache(model):
    """
    Create an mlx_lm KV prompt-cache for *model*.

    Tries the public API (mlx_lm >= 0.18) first, then an older internal path.
    Returns ``None`` when neither is available OR when the model uses
    ``RotatingKVCache`` (whose internal state cannot be safely truncated by
    simply setting ``.offset``).
    """
    cache = None
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except Exception:
        pass
    if cache is None:
        try:
            import mlx_lm.utils as _u
            cache = _u.make_kv_caches(model)
        except Exception:
            pass
    if cache is None:
        return None
    # RotatingKVCache wraps around — offset truncation would corrupt state.
    try:
        for c in cache:
            if "rotating" in type(c).__name__.lower():
                return None
    except Exception:
        return None
    return cache


def _cache_offset(cache) -> int:
    """Current token offset of the first cache entry (0 if unavailable)."""
    try:
        return cache[0].offset
    except Exception:
        return 0


def _cache_set_offset(cache, offset: int) -> None:
    """
    Roll back all cache entries to *offset* tokens.

    mlx_lm's ``KVCache`` stores K/V arrays and tracks position via
    ``.offset``.  Setting it back to a prior value causes the next forward
    pass to overwrite the rejected suffix — correct and allocation-free.
    """
    if cache is None:
        return
    try:
        for c in cache:
            c.offset = offset
    except Exception:
        pass


def _prefill_cached(model, cache, ids: list[int]) -> np.ndarray:
    """
    Prefill *model*'s KV cache with *ids*.

    Returns last-row logits (shape: ``vocab_size``) as float32 numpy —
    the prediction for the very next token.
    """
    x   = mx.array(ids, dtype=mx.int32)[None]   # (1, seq_len)
    out = model(x, cache=cache)                  # (1, seq_len, vocab)
    last = out[0, -1]
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _decode_step_cached(model, cache, token_id: int) -> np.ndarray:
    """Single-token incremental decode.  Returns next-token logits (vocab_size,)."""
    x   = mx.array([[token_id]], dtype=mx.int32)  # (1, 1)
    out = model(x, cache=cache)                   # (1, 1, vocab)
    last = out[0, -1]
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _decode_multi_cached(model, cache, token_ids: list[int]) -> np.ndarray:
    """
    Multi-token incremental decode with KV cache.

    Returns ALL output logit rows — shape ``(len(token_ids), vocab_size)``.
    Row ``j`` is the prediction for what follows ``token_ids[j]``.
    """
    x   = mx.array(token_ids, dtype=mx.int32)[None]  # (1, T)
    out = model(x, cache=cache)                       # (1, T, vocab)
    rows = out[0]
    mx.eval(rows)
    return np.array(rows, dtype=np.float32)


# ── Phase 1A: N-gram in-context draft table ───────────────────────────────────

class NgramTable:
    """
    In-context n-gram draft table for zero-cost speculative proposals.

    Built from prompt tokens at the start of each decode cycle and extended
    incrementally as new tokens are generated.  When a suffix of the live
    context matches a known n-gram prefix the continuation is used as a free
    draft proposal instead of running the neural draft model.

    Benefits vs. neural draft
    ─────────────────────────
    • No forward pass — effectively zero latency overhead.
    • ~100 % acceptance on exact repetitions (code blocks, lists, copied quotes).
    • Falls through to neural draft on misses — no regression.

    Complexity: build O(max_n × seq_len) · update O(max_n) · lookup O(max_n)
    Space: O(max_n × unique_ngrams) — bounded by context length.
    """

    def __init__(self, max_n: int = 8):
        self._max_n = max_n
        # prefix_tuple → {next_token_id: count}
        self._table: dict[tuple, dict[int, int]] = {}

    def build(self, ids: list[int]) -> None:
        """Build the full n-gram table from a token sequence (typically the prompt)."""
        self._table.clear()
        for n in range(2, min(self._max_n + 1, len(ids) + 1)):
            for i in range(len(ids) - n + 1):
                prefix   = tuple(ids[i : i + n - 1])
                next_tok = ids[i + n - 1]
                tbl = self._table.setdefault(prefix, {})
                tbl[next_tok] = tbl.get(next_tok, 0) + 1

    def update(self, new_tok: int, context_before: list[int]) -> None:
        """
        Extend the table after generating *new_tok*.

        *context_before* is the sequence immediately BEFORE *new_tok* was added;
        the longest prefix used is ``context_before[-(max_n-1):]``.
        """
        for prefix_len in range(1, min(self._max_n, len(context_before) + 1)):
            prefix = tuple(context_before[-prefix_len:])
            tbl = self._table.setdefault(prefix, {})
            tbl[new_tok] = tbl.get(new_tok, 0) + 1

    def lookup_k(self, context: list[int], k: int) -> list[int]:
        """
        Greedily extend *context* by up to *k* tokens using n-gram lookups.
        Returns a (possibly shorter) list of proposed token IDs.
        """
        result: list[int] = []
        ctx = list(context)
        for _ in range(k):
            tok = self._lookup_one(ctx)
            if tok is None:
                break
            result.append(tok)
            ctx.append(tok)
        return result

    def _lookup_one(self, context: list[int]) -> int | None:
        """Return the most-common next token for the longest matching prefix."""
        for n in range(min(self._max_n, len(context)), 0, -1):
            prefix     = tuple(context[-n:])
            candidates = self._table.get(prefix)
            if candidates:
                return max(candidates, key=candidates.__getitem__)
        return None


# ── Phase 1B: Hidden-state capture shim ──────────────────────────────────────

class HiddenStateCapture:
    """
    Thin wrapper around a mlx_lm model that captures the final hidden states
    (BEFORE the lm_head projection) after each forward pass.

    Required by the EAGLE-3 draft head, which conditions on the target
    model's internal representations rather than its output logits.

    Usage
    ─────
        capture = HiddenStateCapture(target_model)
        logits  = capture(x, cache=kv_cache)
        hidden  = capture.last_hidden  # (1, seq_len, hidden_dim) or None
    """

    def __init__(self, model):
        self._m           = model
        # Standard mlx_lm layout: model.model → inner transformer,
        #                          model.lm_head → final linear projection.
        self._can_capture = (
            hasattr(model, "model") and hasattr(model, "lm_head")
        )
        self.last_hidden: "mx.array | None" = None

    @property
    def can_capture(self) -> bool:
        return self._can_capture

    def __call__(self, x, cache=None):
        if self._can_capture:
            h = self._m.model(x, cache=cache)
            self.last_hidden = h
            return self._m.lm_head(h)
        self.last_hidden = None
        return self._m(x, cache=cache)

    def __getattr__(self, name: str):
        return getattr(self._m, name)


# ── Phase 1B: EAGLE-3 draft head ─────────────────────────────────────────────

class EagleDraftHead:
    """
    EAGLE-3 lightweight draft head.

    The head is a small transformer (1-2 decoder layers) that conditions on
    the *target* model's last hidden state and autoregressively produces k
    draft tokens using its own internal KV cache.  Because it sees the
    target's full internal representation it achieves much higher acceptance
    rates than a separate small model (typically 75-85 % vs 55-65 %).

    Architecture  (EAGLE-3 paper: Chen et al., 2025)
    ─────────────────────────────────────────────────
      fc        : Linear(2 × hidden_dim → hidden_dim)   [fuse hidden + embed]
      layers[i] : 1-2 decoder layers  (same config as target)
      norm      : RMSNorm(hidden_dim)
      lm_head   : Linear(hidden_dim → vocab_size)       [tied to target weights]

    Weight source
    ─────────────
    Download with:
        squish pull-head qwen3:8b          # auto-resolve
        squish pull-head --repo yuhuili/EAGLE3-Qwen3-Instruct-8B
    Weights are stored in ~/.squish/eagle-heads/<model-name>/ by default.
    """

    def __init__(
        self,
        eagle_model,
        lm_head_weight: "mx.array",
        embed_weight:   "mx.array",
    ):
        self._model      = eagle_model
        self._lm_head_w  = lm_head_weight   # (vocab_size, hidden_dim)
        self._embed_w    = embed_weight      # (vocab_size, hidden_dim)
        self._cache      = _try_make_model_cache(eagle_model)
        # Detect whether inner model has the fc fusion layer
        self._has_fc = (
            hasattr(eagle_model, "model")
            and hasattr(getattr(eagle_model, "model", None), "fc")
        )

    @classmethod
    def from_dir(
        cls,
        head_dir: str,
        target_model,
        verbose: bool = False,
    ) -> "EagleDraftHead":
        """
        Load an EAGLE-3 head from *head_dir*.

        lm_head and embed_tokens weights are shared with the target model
        (standard EAGLE protocol).  Target weights are preferred; if absent
        they are loaded from the head checkpoint.
        """
        from mlx_lm import load as _mlx_load

        head_dir_p = Path(head_dir).expanduser()
        if not head_dir_p.exists():
            raise FileNotFoundError(
                f"EAGLE head directory not found: {head_dir_p}\n"
                "  → Run: squish pull-head <model>"
            )
        if verbose:
            logger.info("Loading EAGLE-3 head from %s", head_dir_p)

        eagle_model, _ = _mlx_load(str(head_dir_p))

        # Shared weight resolution: target > eagle checkpoint
        lm_head_w = getattr(getattr(target_model, "lm_head", None), "weight", None)
        embed_w   = getattr(
            getattr(getattr(target_model, "model", None), "embed_tokens", None),
            "weight", None,
        )
        if lm_head_w is None:
            lm_head_w = getattr(
                getattr(eagle_model, "lm_head", None), "weight", None
            )
        if embed_w is None:
            embed_w = getattr(
                getattr(getattr(eagle_model, "model", None), "embed_tokens", None),
                "weight", None,
            )
        if lm_head_w is None or embed_w is None:
            raise RuntimeError(
                "Cannot locate lm_head / embed_tokens weights in target or EAGLE head. "
                "Ensure the EAGLE head was downloaded with `squish pull-head`."
            )

        if verbose:
            logger.info("EAGLE-3 head loaded (vocab=%d, hidden=%d)",
                        embed_w.shape[0], embed_w.shape[1])
        return cls(eagle_model, lm_head_w, embed_w)

    def reset_cache(self) -> None:
        """Roll back the EAGLE KV cache to position 0 (start of new request)."""
        _cache_set_offset(self._cache, 0)

    def draft_k(
        self,
        target_hidden: "mx.array",  # (1, T, hidden_dim) — from HiddenStateCapture
        k: int,
        prev_token_id: int,
        temperature: float,
        top_p: float,
        eos_id: int,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Produce up to *k* draft tokens conditioned on *target_hidden*.

        Step 0 — fuse the target's last hidden state with the previous token
        embedding via the ``fc`` projection and run through EAGLE's layers.
        Steps 1..k-1 — standard autoregressive decode with EAGLE KV cache.

        Returns ``(draft_ids, draft_probs)`` for the standard
        accept/reject verification loop.
        """
        self.reset_cache()
        draft_ids:   list[int]        = []
        draft_probs: list[np.ndarray] = []

        prev_embed = self._embed_w[prev_token_id]   # (hidden_dim,)
        h_last     = target_hidden[0, -1]            # (hidden_dim,)

        # ── Step 0: first draft token via fc fusion ───────────────────────────
        if self._has_fc:
            fused = mx.concatenate([h_last, prev_embed], axis=-1)[None, None]  # (1,1,2H)
            h = self._model.model.fc(fused)
            for i, layer in enumerate(self._model.model.layers):
                c = self._cache[i] if self._cache is not None else None
                h = layer(h, cache=c)
            h = self._model.model.norm(h)
            logit0 = (h[0, -1] @ self._lm_head_w.T)
        else:
            # Fallback: run eagle model with target hidden as plain input
            x_in   = mx.array([[prev_token_id]], dtype=mx.int32)
            logit0 = self._model(x_in, cache=self._cache)[0, -1]

        mx.eval(logit0)
        probs0 = _top_p_filter(_softmax_np(np.array(logit0, dtype=np.float32), temperature), top_p)
        tok0   = _sample(probs0)
        draft_ids.append(tok0)
        draft_probs.append(probs0)
        if tok0 == eos_id or k == 1:
            return draft_ids, draft_probs

        # ── Steps 1..k-1: autoregressive via EAGLE KV cache ─────────────────
        cur_tok = tok0
        for _ in range(k - 1):
            x_in = mx.array([[cur_tok]], dtype=mx.int32)
            logits = self._model(x_in, cache=self._cache)
            mx.eval(logits)
            logit_np = np.array(logits[0, -1], dtype=np.float32)
            probs = _top_p_filter(_softmax_np(logit_np, temperature), top_p)
            tok   = _sample(probs)
            draft_ids.append(tok)
            draft_probs.append(probs)
            if tok == eos_id:
                break
            cur_tok = tok

        return draft_ids, draft_probs


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
        eagle_head: "EagleDraftHead | None" = None,
        ngram_max_n: int        = 8,
        fsm_gamma:   bool       = False,
        fsm_min:     int        = 2,
        fsm_max:     int        = 8,
    ):
        self._target  = target_model
        self._ttok    = target_tokenizer
        self._draft   = draft_model
        self._dtok    = draft_tokenizer or target_tokenizer
        self._k       = min(max(1, k), _MAX_SPEC_TOKENS)

        # Phase 6W: FSM adaptive speculation length
        self._fsm: FSMGammaController | None = (
            FSMGammaController(self._k, fsm_min, fsm_max)
            if fsm_gamma else None
        )

        # Phase 1B: EAGLE-3 head + hidden-state capture
        self._eagle_head    = eagle_head
        self._target_capture: HiddenStateCapture | None = None
        if eagle_head is not None:
            self._target_capture = HiddenStateCapture(target_model)
            if not self._target_capture.can_capture:
                logger.warning(
                    "EAGLE-3: target model does not expose model.model/lm_head "
                    "— EAGLE disabled, falling back to standard spec decode"
                )
                self._eagle_head    = None
                self._target_capture = None

        # Phase 1A: n-gram table (one per stream() call, created in _reset_caches)
        self._ngram_max_n = ngram_max_n
        self._ngram: NgramTable | None = None   # built at stream() start

        # Acceptance stats (reset per stream() call)
        self.accepted_total  = 0
        self.proposed_total  = 0
        self.steps           = 0

        # ── Stateful KV caches ────────────────────────────────────────────────
        # When EAGLE is active we use target_capture as the model object so that
        # hidden states are captured automatically; otherwise use target directly.
        _target_for_cache = (
            self._target_capture if self._target_capture is not None
            else target_model
        )
        self._target_cache = _try_make_model_cache(_target_for_cache)
        self._draft_cache  = (
            _try_make_model_cache(draft_model) if draft_model is not None else None
        )
        _use_stateful = (
            self._target_cache is not None
            and (draft_model is None or self._draft_cache is not None)
        )
        logger.debug(
            "speculative: %s KV caches  eagle=%s  ngram_max_n=%d",
            "stateful" if _use_stateful else "stateless (fallback)",
            eagle_head is not None,
            ngram_max_n,
        )

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_total / self.proposed_total
                if self.proposed_total > 0 else 0.0)

    def _reset_stats(self) -> None:
        self.accepted_total = 0
        self.proposed_total = 0
        self.steps = 0

    def _reset_caches(self) -> None:
        """Roll both KV caches back to position 0 (start of new request)."""
        _cache_set_offset(self._target_cache, 0)
        _cache_set_offset(self._draft_cache, 0)

    def _update_fsm(self, n_accepted: int, n_proposed: int) -> None:
        """Update the FSM gamma controller and sync ``self._k``."""
        if self._fsm is not None:
            new_k = self._fsm.step(n_accepted, n_proposed)
            self._k = min(new_k, _MAX_SPEC_TOKENS)

    # ── main streaming API ────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int       = 512,
        temperature: float    = _DEFAULT_TEMP,
        top_p: float          = _DEFAULT_TOP_P,
        stop_ids: list[list[int]] | None = None,
        seed: int | None   = None,
    ) -> Iterator[tuple[str, str | None]]:
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

        # Phase 1A: build n-gram table from prompt tokens for this request
        self._ngram = NgramTable(self._ngram_max_n)
        self._ngram.build(input_ids)

        if self._draft is None and self._eagle_head is None:
            # No draft model — plain auto-regressive
            yield from self._plain_stream(
                input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )
            return

        yield from self._speculative_stream(
            input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
        )

    # ── speculative dispatch ──────────────────────────────────────────────────

    def _speculative_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Dispatch to stateful / EAGLE / stateless path per capability."""
        # Phase 1B: prefer EAGLE head when hidden-state capture is available
        if self._eagle_head is not None and self._target_capture is not None:
            if self._target_cache is not None:
                yield from self._eagle_spec_stream(
                    ids, max_tokens, temperature, top_p, stop_ids, eos_id
                )
                return
        # Standard speculative: prefer stateful (KV-cached) path
        if self._draft_cache is not None and self._target_cache is not None:
            yield from self._stateful_spec_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )
        else:
            yield from self._stateless_spec_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )

    # ── stateful speculative inner loop ──────────────────────────────────────

    def _stateful_spec_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """
        Speculative decoding with incremental KV caches.

        Each cycle costs:
          • Draft:  k × O(1) forward passes  (1 token each, cache grows)
          • Target: 1 × O(k) forward pass    (k tokens, cache rolled back then refilled)

        vs. the stateless path where both models re-scan the full growing context
        on every step — O(context_len²) total vs O(context_len) here.

        Cache consistency rule
        ----------------------
        After every accepted sequence of n tokens, both caches sit at
        ``base + n``.  A single forward of the ''final'' token (correction or
        bonus) advances both to ``base + n + 1``, ready for the next cycle.
        """
        self._reset_caches()

        # Prefill both models; keep their last-row logits as the starting
        # prediction for the first draft token.
        d_last = _prefill_cached(self._draft,  self._draft_cache,  ids)
        t_last = _prefill_cached(self._target, self._target_cache, ids)

        generated  = 0
        stop_buf: list[int] = []
        # Phase 1A: track live context for n-gram updates
        context = list(ids)

        while generated < max_tokens:
            base      = _cache_offset(self._target_cache)
            vocab_sz  = len(d_last)

            # ── Step 1: draft proposes k tokens ───────────────────────────────
            # Phase 1A: fill as many slots as possible from the n-gram table
            # (zero cost — no forward pass) then fall back to neural draft.
            draft_ids  : list[int]        = []
            draft_probs: list[np.ndarray] = []

            if self._ngram is not None:
                ngram_toks = self._ngram.lookup_k(context, self._k)
                for tok in ngram_toks:
                    # One-hot distribution: n-gram was certain about this token.
                    # Acceptance probability = min(1, p_target / 1.0) = p_target.
                    probs = np.zeros(vocab_sz, dtype=np.float32)
                    probs[tok] = 1.0
                    draft_ids.append(tok)
                    draft_probs.append(probs)
                    if tok == eos_id:
                        break

            # Neural draft fills remaining slots
            cur_d_logits = d_last
            n_neural     = self._k - len(draft_ids)
            for _ in range(n_neural):
                if draft_ids and draft_ids[-1] == eos_id:
                    break
                probs = _softmax_np(cur_d_logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)
                draft_ids.append(tok)
                draft_probs.append(probs)
                if tok == eos_id:
                    break
                cur_d_logits = _decode_step_cached(
                    self._draft, self._draft_cache, tok)

            self.proposed_total += len(draft_ids)
            self.steps += 1

            # ── Step 2: target verifies (1 pass, k tokens) ────────────────────
            # Roll target cache back to before draft tokens.
            _cache_set_offset(self._target_cache, base)

            # Forward all draft tokens through target.
            # target_fwd[j] = prediction AFTER draft_ids[j]
            #               = verification logit for draft_ids[j+1]  (j < k-1)
            #               = bonus token logit                       (j == k-1)
            target_fwd = _decode_multi_cached(
                self._target, self._target_cache, draft_ids)

            # Prepend t_last (prediction for draft_ids[0], from the prior round)
            # to form the complete verification window.
            # target_rows[i] predicts draft_ids[i]:
            #   target_rows[0] = t_last          (no extra forward needed)
            #   target_rows[1] = target_fwd[0]
            #   ...
            #   target_rows[k] = target_fwd[k-1]  (bonus)
            target_rows = np.concatenate(
                [t_last[np.newaxis], target_fwd], axis=0)  # (k+1, vocab)

            # ── Step 3: sequential accept / reject ────────────────────────────
            new_tokens: list[int] = []
            accepted = 0
            for i, (d_tok, d_probs) in enumerate(
                    zip(draft_ids, draft_probs, strict=False)):
                t_probs  = _softmax_np(target_rows[i], temperature)
                t_probs  = _top_p_filter(t_probs, top_p)
                p_target = float(t_probs[d_tok])
                p_draft  = float(d_probs[d_tok])

                if np.random.random() < min(1.0, p_target / max(p_draft, 1e-12)):
                    new_tokens.append(d_tok)
                    accepted += 1
                else:
                    adjusted = np.maximum(0.0, t_probs - d_probs)
                    s = adjusted.sum()
                    if s > 0:
                        adjusted /= s
                        fallback = _sample(adjusted)
                    else:
                        fallback = _greedy(target_rows[i])
                    new_tokens.append(fallback)
                    break

            self.accepted_total += accepted
            self._update_fsm(accepted, len(draft_ids))

            # ── Step 4: bonus token (all k accepted) ──────────────────────────
            if accepted == len(draft_ids):
                bonus_probs = _softmax_np(target_rows[len(draft_ids)], temperature)
                bonus_probs = _top_p_filter(bonus_probs, top_p)
                new_tokens.append(_sample(bonus_probs))

            # ── Step 5: advance caches to end of accepted sequence ────────────
            # n_acc tokens before the "final" token (correction or bonus).
            n_acc     = len(new_tokens) - 1
            final_tok = new_tokens[-1]
            # Trim both caches to base + n_acc, then run final_tok once
            # so both sit at base + n_acc + 1 for the next cycle.
            _cache_set_offset(self._draft_cache,  base + n_acc)
            _cache_set_offset(self._target_cache, base + n_acc)
            d_last = _decode_step_cached(self._draft,  self._draft_cache,  final_tok)
            t_last = _decode_step_cached(self._target, self._target_cache, final_tok)

            # ── Step 6: yield accepted + final token ──────────────────────────
            for tok in new_tokens:
                if tok == eos_id:
                    yield self._tok_text(tok), "stop"
                    return
                tok_text = self._tok_text(tok)
                generated += 1
                stop_buf.append(tok)
                # Phase 1A: update n-gram table and live context
                if self._ngram is not None:
                    self._ngram.update(tok, context)
                context.append(tok)
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
            "stateful spec: %d steps, %.1f%% acceptance, %d tokens",
            self.steps, self.acceptance_rate * 100, generated,
        )

    # ── Phase 1B: EAGLE-3 speculative inner loop ──────────────────────────────

    def _eagle_spec_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """
        EAGLE-3 speculative decode.

        Uses the target model's last hidden state (captured via
        ``HiddenStateCapture``) to drive the EAGLE head for draft proposals
        instead of a separate small model.  Typically achieves 75-85 %
        acceptance rates vs. 55-65 % for a separate draft model.

        Each cycle:
        • Target capture: 1 forward pass (prefill or single token) → hidden + logit
        • EAGLE head: k autoregressive steps (light, 1-2 layers each)
        • Target verify: 1 forward pass over k positions
        """
        assert self._eagle_head is not None
        assert self._target_capture is not None
        assert self._target_cache is not None

        self._reset_caches()
        self._eagle_head.reset_cache()

        # Prefill target (via capture shim so we get the hidden state)
        t_capture = self._target_capture
        t_last    = _prefill_cached(t_capture, self._target_cache, ids)
        target_hidden = t_capture.last_hidden   # (1, T, hidden_dim)

        generated  = 0
        stop_buf: list[int] = []
        context    = list(ids)
        prev_tok   = ids[-1] if ids else 0

        while generated < max_tokens:
            base = _cache_offset(self._target_cache)

            # ── Step 1: EAGLE head proposes k draft tokens ────────────────────
            if target_hidden is not None:
                draft_ids, draft_probs = self._eagle_head.draft_k(
                    target_hidden, self._k, prev_tok,
                    temperature, top_p, eos_id,
                )
            else:
                # No hidden states — fall back to using target logits for greedy
                probs  = _top_p_filter(_softmax_np(t_last, temperature), top_p)
                draft_ids   = [_sample(probs)]
                draft_probs = [probs]

            # Phase 1A: supplement with n-gram proposals if EAGLE proposed fewer
            if self._ngram is not None and len(draft_ids) < self._k:
                ngram_extra = self._ngram.lookup_k(
                    context + draft_ids, self._k - len(draft_ids)
                )
                vocab_sz = len(t_last)
                for tok in ngram_extra:
                    probs = np.zeros(vocab_sz, dtype=np.float32)
                    probs[tok] = 1.0
                    draft_ids.append(tok)
                    draft_probs.append(probs)
                    if tok == eos_id:
                        break

            self.proposed_total += len(draft_ids)
            self.steps          += 1

            # ── Step 2: target verifies k draft tokens ────────────────────────
            _cache_set_offset(self._target_cache, base)
            target_fwd = _decode_multi_cached(
                t_capture, self._target_cache, draft_ids
            )
            # Capture hidden states for next EAGLE step
            target_hidden = t_capture.last_hidden

            target_rows = np.concatenate(
                [t_last[np.newaxis], target_fwd], axis=0  # (k+1, vocab)
            )

            # ── Step 3: sequential accept / reject ───────────────────────────
            new_tokens: list[int] = []
            accepted = 0
            for i, (d_tok, d_probs) in enumerate(
                    zip(draft_ids, draft_probs, strict=False)):
                t_probs  = _softmax_np(target_rows[i], temperature)
                t_probs  = _top_p_filter(t_probs, top_p)
                p_target = float(t_probs[d_tok])
                p_draft  = float(d_probs[d_tok])
                if np.random.random() < min(1.0, p_target / max(p_draft, 1e-12)):
                    new_tokens.append(d_tok)
                    accepted += 1
                else:
                    adjusted = np.maximum(0.0, t_probs - d_probs)
                    s = adjusted.sum()
                    new_tokens.append(
                        _sample(adjusted / s) if s > 0 else _greedy(target_rows[i])
                    )
                    break

            self.accepted_total += accepted
            self._update_fsm(accepted, len(draft_ids))

            # ── Step 4: bonus token ───────────────────────────────────────────
            if accepted == len(draft_ids):
                bonus_probs = _top_p_filter(
                    _softmax_np(target_rows[len(draft_ids)], temperature), top_p
                )
                new_tokens.append(_sample(bonus_probs))

            # ── Step 5: advance target cache ─────────────────────────────────
            n_acc     = len(new_tokens) - 1
            final_tok = new_tokens[-1]
            _cache_set_offset(self._target_cache, base + n_acc)
            t_last = _decode_step_cached(t_capture, self._target_cache, final_tok)
            target_hidden = t_capture.last_hidden
            prev_tok = final_tok

            # ── Step 6: yield ─────────────────────────────────────────────────
            for tok in new_tokens:
                if tok == eos_id:
                    yield self._tok_text(tok), "stop"
                    return
                tok_text = self._tok_text(tok)
                generated += 1
                stop_buf.append(tok)
                if self._ngram is not None:
                    self._ngram.update(tok, context)
                context.append(tok)
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
            "eagle spec: %d steps, %.1f%% acceptance, %d tokens",
            self.steps, self.acceptance_rate * 100, generated,
        )

    # ── stateless speculative inner loop (fallback) ───────────────────────────

    def _stateless_spec_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
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
        stop_buf: list[int] = []
        context       = list(ids)

        while generated < max_tokens:
            # ── Step 1: draft K tokens ────────────────────────────────────
            draft_ids     : list[int]        = []
            draft_probs   : list[np.ndarray] = []

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
            new_tokens: list[int] = []
            accepted   = 0
            for i, (d_tok, d_probs) in enumerate(zip(draft_ids, draft_probs, strict=False)):
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
            self._update_fsm(accepted, len(draft_ids))

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

    # ── plain stream (no draft model) ────────────────────────────────────────

    def _plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Dispatch to stateful (KV-cached) or stateless auto-regressive path."""
        if self._target_cache is not None:
            yield from self._stateful_plain_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id)
        else:
            yield from self._stateless_plain_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id)

    def _stateful_plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Plain auto-regressive with stateful KV cache — O(1) per token."""
        _cache_set_offset(self._target_cache, 0)
        last_logits = _prefill_cached(self._target, self._target_cache, ids)
        stop_buf: list[int] = []
        generated = 0

        for _ in range(max_tokens):
            if temperature == 0.0:
                tok = _greedy(last_logits)
            else:
                probs = _softmax_np(last_logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)

            if tok == eos_id:
                yield self._tok_text(tok), "stop"
                return

            tok_text = self._tok_text(tok)
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
            last_logits = _decode_step_cached(self._target, self._target_cache, tok)

        yield "", "stop"

    def _stateless_plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Plain auto-regressive sampling — stateless fallback (O(n²) in context)."""
        context    = list(ids)
        stop_buf   : list[int] = []
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


# ── Medusa: multi-head speculative decoding ───────────────────────────────────
#
# Based on:
#   "Medusa: Simple LLM Inference Acceleration Framework with Multiple
#    Decoding Heads" — Cai et al., Together AI 2024  (arXiv:2401.10774)
#
# Key idea: attach K lightweight prediction heads to the final hidden state of
# the target model.  Each head independently predicts one future token
# (head k predicts token at offset k+1).  The K-head proposals are arranged
# into a tree of candidates and verified by the target model in a single
# batched forward pass, accepting the longest valid prefix.
#
# This module provides a numpy-only simulation layer:
#   MedusaConfig  — number of heads, top-k per head, acceptance threshold.
#   MedusaHead    — lightweight linear head that maps hidden_dim → vocab_size.
#   MedusaTreeDraft — assembles per-head top-k candidates into candidate chains.
#   MedusaGenerator — drives the tree-speculative decode loop.

from dataclasses import dataclass as _mdc


@_mdc
class MedusaConfig:
    """Configuration for Medusa multi-head speculative decoding.

    Parameters
    ----------
    num_heads : int
        Number of Medusa heads (speculative look-ahead distance).
    top_k : int
        Number of candidate tokens considered per head per step.
    hidden_dim : int
        Dimensionality of the hidden state fed into each head.
    vocab_size : int
        Vocabulary size (output dimension of each head).
    acceptance_threshold : float
        Minimum probability ratio for accepting a speculative token
        (if the target model assigns probability ≥ threshold × draft_prob,
        the token is accepted).  Set to 0.0 for greedy acceptance.
    """

    num_heads:            int   = 3
    top_k:                int   = 5
    hidden_dim:           int   = 4096
    vocab_size:           int   = 32000
    acceptance_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")


class MedusaHead:
    """Single Medusa prediction head: a linear projection hidden → vocab.

    Parameters
    ----------
    hidden_dim : int
    vocab_size : int
    rng : np.random.Generator | None
        If provided, weights are initialised randomly (for tests/simulation).
        Pass ``None`` to initialise to zeros and set weights manually.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        rng: "np.random.Generator | None" = None,
    ) -> None:
        if hidden_dim < 1 or vocab_size < 1:
            raise ValueError("hidden_dim and vocab_size must be >= 1")
        if rng is not None:
            scale       = (2.0 / hidden_dim) ** 0.5
            self.weight = rng.standard_normal((vocab_size, hidden_dim)).astype(np.float32) * scale
            self.bias   = np.zeros(vocab_size, dtype=np.float32)
        else:
            self.weight = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
            self.bias   = np.zeros(vocab_size, dtype=np.float32)

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        hidden : (hidden_dim,) float array — last hidden state from the model.

        Returns
        -------
        (vocab_size,) float array — raw logits.
        """
        h = np.asarray(hidden, dtype=np.float32)
        if h.shape != (self.weight.shape[1],):
            raise ValueError(
                f"hidden shape {h.shape} does not match hidden_dim {self.weight.shape[1]}"
            )
        return self.weight @ h + self.bias

    def top_k_tokens(self, hidden: np.ndarray, k: int) -> np.ndarray:
        """Return the top-``k`` token ids (highest logit) as a 1-D array."""
        raw = self.logits(hidden)
        return np.argpartition(raw, -k)[-k:].astype(np.int32)


class MedusaTreeDraft:
    """Build a tree of speculative candidates from K Medusa heads.

    Each head proposes ``top_k`` candidates for its offset position.
    The tree is the Cartesian product limited by ``max_candidates`` paths.

    Parameters
    ----------
    config : MedusaConfig
    heads  : list[MedusaHead] — must have len == config.num_heads.
    """

    def __init__(self, config: MedusaConfig, heads: "list[MedusaHead]") -> None:
        if len(heads) != config.num_heads:
            raise ValueError(
                f"Expected {config.num_heads} heads, got {len(heads)}"
            )
        self._cfg   = config
        self._heads = heads

    def draft(
        self,
        hidden: np.ndarray,
        max_candidates: int = 16,
    ) -> "list[list[int]]":
        """Generate speculative candidate sequences.

        Parameters
        ----------
        hidden : (hidden_dim,) float — current last hidden state.
        max_candidates : int — cap the returned candidate list.

        Returns
        -------
        List of candidate token sequences (each of length num_heads),
        sorted by descending joint logit sum.
        """
        h   = np.asarray(hidden, dtype=np.float32)
        k   = self._cfg.top_k
        per_head: list[np.ndarray] = [head.top_k_tokens(h, k) for head in self._heads]

        # Build candidate paths: start from head-0 candidates
        candidates: list[list[int]] = [[int(t)] for t in per_head[0]]
        for head_toks in per_head[1:]:
            new_cands: list[list[int]] = []
            for path in candidates:
                for t in head_toks:
                    new_cands.append(path + [int(t)])
            # Prune early to avoid combinatorial explosion
            if len(new_cands) > max_candidates * 4:
                new_cands = new_cands[:max_candidates * 4]
            candidates = new_cands

        return candidates[:max_candidates]


class MedusaGenerator:
    """Drive inference with Medusa tree-speculative decoding.

    Parameters
    ----------
    hidden_forward : callable
        ``hidden_forward(ids) -> (np.ndarray, np.ndarray)``
        Returns ``(hidden_state, logits)`` where ``hidden_state`` is
        ``(hidden_dim,)`` and ``logits`` is ``(vocab_size,)``.
    verify_forward : callable
        ``verify_forward(ids) -> np.ndarray`` of shape ``(vocab_size,)``
        — used to verify a single speculative token.
    config : MedusaConfig
    heads : list[MedusaHead]
    """

    def __init__(
        self,
        hidden_forward: "callable",
        verify_forward: "callable",
        config: MedusaConfig,
        heads: "list[MedusaHead]",
    ) -> None:
        self._hfwd   = hidden_forward
        self._vfwd   = verify_forward
        self._cfg    = config
        self._tree   = MedusaTreeDraft(config, heads)
        self._n_acc  = 0
        self._n_rej  = 0

    @property
    def acceptance_rate(self) -> float:
        total = self._n_acc + self._n_rej
        return self._n_acc / total if total > 0 else 0.0

    def generate(
        self,
        input_ids: "list[int]",
        max_new_tokens: int = 64,
    ) -> "list[int]":
        """Generate tokens using tree-speculative Medusa decoding.

        Parameters
        ----------
        input_ids : list[int] — prompt token ids.
        max_new_tokens : int.

        Returns
        -------
        input_ids + generated token ids.
        """
        ids       = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # -- Draft phase --
            hidden, base_logits = self._hfwd(ids)
            candidates = self._tree.draft(hidden, max_candidates=16)

            if not candidates:
                # Fallback: greedy from base logits
                tok = int(np.argmax(base_logits))
                ids.append(tok)
                generated += 1
                continue

            # -- Verify phase: accept longest valid prefix from first candidate --
            best = candidates[0]
            accepted: list[int] = []
            ctx = list(ids)
            for d_tok in best:
                v_logits = self._vfwd(ctx)
                v_tok    = int(np.argmax(v_logits))
                if v_tok == d_tok:
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    self._n_acc += 1
                else:
                    accepted.append(v_tok)
                    self._n_rej += 1
                    break

            ids.extend(accepted)
            generated += len(accepted)

        return ids


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
