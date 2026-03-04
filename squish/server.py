#!/usr/bin/env python3
"""
squish_server.py

OpenAI-compatible HTTP API server for Squish compressed models.

Exposes endpoints:
    GET  /v1/models                    — list loaded model
    GET  /v1/models/{model_id}         — model detail
    POST /v1/chat/completions          — chat (streaming + non-streaming)
    POST /v1/completions               — legacy text completion
    POST /v1/embeddings                — mean-pooled token embeddings
    POST /v1/tokenize                  — tokenize text (non-standard, useful for debugging)
    GET  /v1/metrics                   — Prometheus-compatible metrics
    GET  /health                       — health check with real-time stats

Drop-in replacement for cloud APIs:
    export OPENAI_BASE_URL=http://localhost:11435/v1
    export OPENAI_API_KEY=squish        # or your --api-key value
    # Any OpenAI client now routes to local Squish inference

Usage:
    python3 squish_server.py \\
        --model-dir   ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --compressed-dir ~/models/Qwen2.5-7B-Instruct-bf16-compressed \\
        --port 11435 [--api-key mysecret]

Dependencies:
    pip install fastapi "uvicorn[standard]"
"""
import argparse
import json
import time
import uuid
import sys
import os
import hashlib
import hmac
import threading
import collections
from pathlib import Path
from typing import AsyncIterator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

# ── Ensure the squish package root is importable when run as a script ────────
# cli.py launches this file directly with `python3 .../squish/server.py`, so
# the package parent directory must be on sys.path for `from squish.*` imports.
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# ── Validate dependencies ────────────────────────────────────────────────────

def _require(pkg: str, install: str | None = None) -> None:
    try:
        __import__(pkg)
    except ImportError:
        hint = install or pkg
        print(f"Missing dependency: {pkg}.  Install with:  pip install {hint}")
        sys.exit(1)

_require("fastapi")
_require("uvicorn", "uvicorn[standard]")

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
try:
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    _STATIC_FILES_AVAILABLE = True
except ImportError:
    _STATIC_FILES_AVAILABLE = False

# ── KV cache (Phase 1.3 — lazily imported to keep startup fast) ──────────────
_kv_cache = None   # QuantizedKVCache | None — set in main() after model load

# ── Batch scheduler (Phase 2.1 — continuous batching) ───────────────────────
_scheduler       = None  # BatchScheduler | None — set in main() when --batch-scheduler given
_QueueFullError  = None  # QueueFullError class — imported alongside BatchScheduler

# ── Tool calling + Ollama compat (Phase 2.2) ─────────────────────────────────
# Imported lazily in endpoints — no startup cost when unused
import uvicorn

# ── Model state ──────────────────────────────────────────────────────────────

class _ModelState:
    model        = None
    tokenizer    = None
    model_name   = ""
    loaded_at    = 0.0
    load_time_s  = 0.0
    loader_tag   = "squish"
    requests     = 0
    tokens_gen   = 0
    # Real-time performance tracking
    inflight     = 0          # concurrent requests in flight
    _lock        = threading.Lock()
    # Rolling window: last 20 (tps, ttft_s) samples
    _tps_window: collections.deque = None

    def __init__(self):
        self._tps_window = collections.deque(maxlen=20)

    def record_completion(self, n_tokens: int, duration_s: float, ttft_s: float) -> None:
        tps = n_tokens / max(duration_s, 1e-6)
        with self._lock:
            self._tps_window.append((tps, ttft_s))
            self.tokens_gen += n_tokens
            self.requests   += 1

    @property
    def avg_tps(self) -> float:
        with self._lock:
            items = list(self._tps_window)
        return sum(t for t, _ in items) / len(items) if items else 0.0

    @property
    def avg_ttft(self) -> float:
        with self._lock:
            items = list(self._tps_window)
        return sum(f for _, f in items) / len(items) if items else 0.0

_state = _ModelState()
_API_KEY: str | None = None          # set from --api-key at startup
_bearer  = HTTPBearer(auto_error=False)

# ── Draft model state (speculative decoding) ─────────────────────────────────

class _DraftState:
    model      = None
    tokenizer  = None
    model_dir  = ""
    generator  = None   # SpeculativeGenerator instance (created after both models load)

_draft = _DraftState()

# ── Prefix cache (Phase 1.4) ─────────────────────────────────────────────────
# Maps SHA-256(prompt) → (full_response_text, finish_reason).
# Exact-match cache with a fixed capacity — avoids re-running the model for
# repeated identical prompts (e.g. benchmark harnesses, agent tool loops).

import hashlib as _hashlib

class _PrefixCache:
    """Thread-safe O(1) LRU cache of (prompt → response) for exact prompt matches.

    Backed by ``collections.OrderedDict`` — ``move_to_end`` and
    ``popitem(last=False)`` are both O(1) vs the O(n) ``list.remove`` +
    ``list.pop(0)`` approach.
    """

    def __init__(self, maxsize: int = 256):
        self._cache: collections.OrderedDict[str, Tuple[str, str]] = (
            collections.OrderedDict()
        )
        self._maxsize = maxsize
        self._lock    = threading.Lock()
        self.hits     = 0
        self.misses   = 0

    def _key(self, prompt: str) -> str:
        # blake2b (128-bit) is ~3× faster than sha256 and sufficient for a
        # non-security-critical local cache with ≤1024 slots.
        return _hashlib.blake2b(prompt.encode(), digest_size=16).hexdigest()

    def get(self, prompt: str) -> Optional[Tuple[str, str]]:
        k = self._key(prompt)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)   # O(1) — promote to MRU end
                self.hits += 1
                return self._cache[k]
            self.misses += 1
            return None

    def put(self, prompt: str, response: str, finish: str) -> None:
        k = self._key(prompt)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)   # O(1) update in-place
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # O(1) evict LRU (first item)
            self._cache[k] = (response, finish)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

_prefix_cache = _PrefixCache(maxsize=512)


def _sample_mx(logits_row, temperature: float, top_p: float) -> int:
    """
    Sample a single token id from an MLX logits vector.

    Parameters
    ----------
    logits_row  : mx.array  shape (vocab_size,)
    temperature : float — <= 0 means greedy argmax
    top_p       : float — nucleus sampling probability mass (1.0 = disabled)

    Returns
    -------
    int token id
    """
    import mlx.core as mx
    import numpy as np
    if temperature <= 0.0 or temperature < 1e-5:
        return int(mx.argmax(logits_row).item())
    probs_np = np.array(mx.softmax(logits_row.astype(mx.float32) / temperature, axis=-1))
    if top_p < 1.0:
        idx    = np.argsort(-probs_np)
        cumsum = np.cumsum(probs_np[idx])
        cutoff = min(int((cumsum <= top_p).sum()) + 1, len(idx))
        mask   = np.zeros_like(probs_np)
        mask[idx[:max(1, cutoff)]] = 1.0
        probs_np = probs_np * mask
        probs_np /= probs_np.sum() + 1e-9
    return int(np.random.choice(len(probs_np), p=probs_np))


def _check_auth(creds: HTTPAuthorizationCredentials | None) -> None:
    """Raise 401 if an API key is configured and the request doesn't match.

    Uses hmac.compare_digest to prevent timing-oracle attacks.
    """
    if _API_KEY is None:
        return
    if creds is None or not hmac.compare_digest(
        creds.credentials.encode(), _API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _system_fingerprint() -> str:
    """Stable fingerprint derived from model name + load timestamp."""
    return "sq-" + hashlib.md5(
        f"{_state.model_name}{_state.loaded_at}".encode()
    ).hexdigest()[:8]


def load_model(model_dir: str, compressed_dir: str, verbose: bool = True) -> None:
    """Load the Squish compressed model into global state."""
    from .compressed_loader import load_compressed_model as _load_compressed_model
    # Keep backward-compat shim
    load_from_npy_dir = _load_compressed_model

    t0 = time.perf_counter()
    if verbose:
        print(f"  Loading model: {compressed_dir}")

    model, tokenizer, stats = load_from_npy_dir(
        model_dir  = model_dir,
        npz_path   = compressed_dir,
        verbose    = verbose,
        return_stats = True,
    )
    elapsed = time.perf_counter() - t0

    _state.model      = model
    _state.tokenizer  = tokenizer
    _state.model_name = Path(compressed_dir).name
    _state.loaded_at  = time.time()

    _state.load_time_s = elapsed
    _state.loader_tag  = stats.get("loader", "squish")
    if verbose:
        print(f"  ✓ Model ready  ({elapsed:.2f}s, loader={_state.loader_tag})")


def load_draft_model(draft_model_dir: str, draft_compressed_dir: str = "",
                     verbose: bool = True) -> None:
    """Load the small draft model used for speculative decoding."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from squish.speculative import load_draft_model as _load_draft
    if verbose:
        print(f"  Loading draft model: {draft_model_dir}")
    draft_m, draft_tok = _load_draft(
        draft_model_dir,
        draft_compressed_dir or (draft_model_dir + "-compressed"),
        verbose=verbose,
    )
    _draft.model     = draft_m
    _draft.tokenizer = draft_tok
    _draft.model_dir = draft_model_dir
    if verbose:
        print("  ✓ Draft model ready")

    # Build the SpeculativeGenerator now that both models are loaded
    _rebuild_spec_gen()


def _rebuild_spec_gen() -> None:
    """(Re-)create the SpeculativeGenerator from current target + draft state."""
    if _state.model is None or _draft.model is None:
        _draft.generator = None
        return
    from squish.speculative import SpeculativeGenerator
    _draft.generator = SpeculativeGenerator(
        _state.model, _state.tokenizer,
        draft_model=_draft.model, draft_tokenizer=_draft.tokenizer,
    )


# ── Token generation ─────────────────────────────────────────────────────────

def _apply_chat_template(messages: List[Dict[str, str]], tokenizer) -> str:
    """Apply chat template if available, fall back to manual formatting."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize          = False,
                add_generation_prompt = True,
            )
            return token_ids
        except Exception:
            pass

    # Manual fallback: Qwen / ChatML format
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _count_tokens(text: str) -> int:
    """Count tokens using the loaded tokenizer. Falls back to word-split estimate."""
    tok = _state.tokenizer
    if tok is None:
        return len(text.split())
    try:
        return len(tok.encode(text))
    except Exception:
        return len(text.split())


def _get_stop_ids(stop: List[str] | str | None) -> List[List[int]]:
    """Convert stop string(s) to lists of token IDs."""
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    tok = _state.tokenizer
    result = []
    for s in stop:
        try:
            ids = tok.encode(s, add_special_tokens=False)
            if ids:
                result.append(ids)
        except Exception:
            pass
    return result


def _generate_tokens(
    prompt: str,
    max_tokens: int    = 512,
    temperature: float = 0.7,
    top_p: float       = 0.9,
    stop: List[str] | str | None = None,
    seed: int | None   = None,
    use_cache: bool    = True,
):
    """
    Stream (token_text, finish_reason_or_None) tuples from the MLX model.
    finish_reason is 'stop' (eos hit or stop sequence matched) or
    'length' (max_tokens exhausted).

    Dispatch priority:
      1. Prefix cache (exact-match, deterministic prompts only)
      2. Speculative decoding  (when draft model loaded + temp > 0)
      3. mlx_lm.stream_generate  (mlx_lm >= 0.12)
      4. Manual sampling loop  (fallback)
    """
    model     = _state.model
    tokenizer = _state.tokenizer
    stop_ids  = _get_stop_ids(stop)
    eos_id    = getattr(tokenizer, "eos_token_id", None) or 151645

    # ── Batch scheduler dispatch (Phase 2.1) ──────────────────────────────────
    # Route non-deterministic requests through the coalescing batch scheduler.
    # submit_sync() is a plain blocking generator — compatible with this sync
    # generator function without any async bridge required.
    is_deterministic = (temperature == 0.0 or seed is not None)
    if _scheduler is not None and not is_deterministic:
        try:
            yield from _scheduler.submit_sync(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                top_p       = top_p,
                stop_ids    = _get_stop_ids(stop),
                seed        = seed,
            )
        except _QueueFullError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        return

    # ── Prefix cache lookup (Phase 1.4) ──────────────────────────────────────
    # Only cache deterministic outputs (temp==0 or seed fixed) so non-
    # deterministic completions never return stale cached text.
    cache_eligible = use_cache and (temperature == 0.0 or seed is not None)
    if cache_eligible:
        cached = _prefix_cache.get(prompt)
        if cached is not None:
            full_text, finish_reason = cached
            for char in full_text:
                yield char, None
            yield "", finish_reason
            return

    # Collect full output so we can populate the cache after generation
    _cache_buf: List[str] = [] if cache_eligible else []
    _last_finish = "stop"

    # Apply optional seed for reproducible generation
    if seed is not None:
        try:
            import mlx.core as mx
            mx.random.seed(seed)
        except Exception:
            pass

    # ── Speculative decoding (Phase 0.2) ─────────────────────────────────────
    # Use when a draft model is loaded AND temperature > 0 (greedy draft on
    # temp==0 benchmarks offers less benefit and adds overhead).
    if _draft.generator is not None and temperature > 0.0:
        try:
            gen = _draft.generator.stream(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                top_p       = top_p,
                stop_ids    = stop_ids,
                seed        = seed,
            )
            for tok_text, finish in gen:
                if cache_eligible:
                    _cache_buf.append(tok_text)
                    _last_finish = finish or _last_finish
                yield tok_text, finish
                if finish is not None:
                    break
            if cache_eligible and _cache_buf:
                _prefix_cache.put(prompt, "".join(_cache_buf), _last_finish)
            return
        except Exception as _spec_err:
            import logging as _log
            _log.getLogger(__name__).warning("Speculative decoding failed (%s); "
                                             "falling back to standard generation", _spec_err)

    # ── Quantized KV cache generation path ─────────────────────────────────────
    if _kv_cache is not None:
        _kv_cache.reset()
        try:
            import mlx.core as mx
            input_ids = (
                tokenizer.encode(prompt)
                if hasattr(tokenizer, "encode")
                else tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            layer_caches = _kv_cache._layers
            x = mx.array(input_ids, dtype=mx.int32)[None]
            logits = model(x, cache=layer_caches)
            mx.eval(logits)
            next_id = _sample_mx(logits[0, -1], temperature, top_p)
            stop_buf = [next_id]
            for step in range(max_tokens):
                tok_text = (
                    tokenizer.decode([next_id])
                    if hasattr(tokenizer, "decode")
                    else str(next_id)
                )
                if next_id == eos_id:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
                    yield tok_text, "stop"
                    return
                if stop_ids:
                    for seq in stop_ids:
                        if stop_buf[-len(seq):] == seq:
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
                            yield tok_text, "stop"
                            return
                    if len(stop_buf) > 64:
                        stop_buf = stop_buf[-64:]
                if step == max_tokens - 1:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(prompt, "".join(_cache_buf), "length")
                    yield tok_text, "length"
                    return
                if cache_eligible:
                    _cache_buf.append(tok_text)
                yield tok_text, None
                x = mx.array([[next_id]], dtype=mx.int32)
                logits = model(x, cache=layer_caches)
                mx.eval(logits)
                next_id = _sample_mx(logits[0, -1], temperature, top_p)
                stop_buf.append(next_id)
            if cache_eligible and _cache_buf:
                _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
            yield "", "stop"
            return
        except Exception as _kv_err:
            import logging as _kv_log
            _kv_log.getLogger(__name__).warning(
                "Quantized KV cache path failed (%s); falling back to stream_generate",
                _kv_err,
            )
            _kv_cache.reset()

    # ── mlx_lm.stream_generate (preferred, available mlx_lm >= 0.12) ────────
    try:
        import mlx_lm
        gen = mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt     = prompt,
            max_tokens = max_tokens,
            temp       = temperature,
            top_p      = top_p,
        )
        emitted = 0
        stop_buf: List[int] = []
        for item in gen:
            # mlx_lm >= 0.19 yields GenerationResult objects; older yields strings
            if hasattr(item, "text"):
                tok_text = item.text
            else:
                tok_text = str(item)
            emitted += 1

            # Check stop sequences against a rolling token-id buffer
            if stop_ids and hasattr(tokenizer, "encode"):
                new_ids = tokenizer.encode(tok_text, add_special_tokens=False)
                stop_buf.extend(new_ids)
                hit = False
                for seq in stop_ids:
                    if stop_buf[-len(seq):] == seq:
                        hit = True
                        break
                if hit:
                    if cache_eligible:
                        _cache_buf.append(tok_text)
                        _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
                    yield tok_text, "stop"
                    return
                if len(stop_buf) > 64:
                    stop_buf = stop_buf[-64:]

            if emitted >= max_tokens:
                if cache_eligible:
                    _cache_buf.append(tok_text)
                    _prefix_cache.put(prompt, "".join(_cache_buf), "length")
                yield tok_text, "length"
                return
            if cache_eligible:
                _cache_buf.append(tok_text)
            yield tok_text, None
        if cache_eligible and _cache_buf:
            _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
        yield "", "stop"
        return
    except (AttributeError, TypeError):
        pass

    # ── Fallback: manual sampling loop ───────────────────────────────────────
    import mlx.core as mx
    import numpy as np

    input_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else \
                tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()

    def _sample(logits_row, temp: float, top_p: float) -> int:
        if temp == 0.0:
            return int(mx.argmax(logits_row).item())
        logits_f = logits_row.astype(mx.float32)
        probs_np = np.array(mx.softmax(logits_f / temp, axis=-1))
        if top_p < 1.0:
            idx      = np.argsort(-probs_np)
            cumsum   = np.cumsum(probs_np[idx])
            cutoff   = min(int((cumsum <= top_p).sum()) + 1, len(idx))
            mask     = np.zeros_like(probs_np)
            mask[idx[:max(1, cutoff)]] = 1.0
            probs_np = probs_np * mask
            probs_np /= probs_np.sum()
        return int(np.random.choice(len(probs_np), p=probs_np))

    ids      = list(input_ids)
    stop_buf = []
    for step in range(max_tokens):
        x       = mx.array(ids, dtype=mx.int32)[None]
        logits  = model(x)
        next_id = _sample(logits[0, -1], temperature, top_p)
        if next_id == eos_id:
            yield "", "stop"
            return
        ids.append(next_id)
        tok_text = tokenizer.decode([next_id])

        if stop_ids:
            stop_buf.append(next_id)
            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    yield tok_text, "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

        if step == max_tokens - 1:
            if cache_eligible and _cache_buf:
                _prefix_cache.put(prompt, "".join(_cache_buf), "length")
            yield tok_text, "length"
            return
        if cache_eligible:
            _cache_buf.append(tok_text)
        yield tok_text, None

    if cache_eligible and _cache_buf:
        _prefix_cache.put(prompt, "".join(_cache_buf), "stop")
    yield "", "stop"


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Squish OpenAI-compatible API",
    description = "Local LLM inference via Squish compressed models",
    version     = "1.0.0",
)

# Allow browser clients (e.g. Open WebUI) to call without CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Ollama compatibility layer (POST /api/chat etc.) ────────────────────────
try:
    from .ollama_compat import mount_ollama as _mount_ollama          # package import
except ImportError:
    from ollama_compat import mount_ollama as _mount_ollama            # direct script run
_mount_ollama(
    app,
    get_state     = lambda: _state,
    get_generate  = lambda: _generate_tokens,
    get_tokenizer = lambda: _state.tokenizer,
)

# ── Web chat UI (/chat) ────────────────────────────────────────────────
if _STATIC_FILES_AVAILABLE:
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():
        app.mount("/static", _StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/chat")
async def web_chat_ui():
    """Serve the single-page web chat interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"error": "Web UI not found. Is squish/static/index.html present?"}, status_code=404)


@app.get("/v1/models")
async def list_models(creds: HTTPAuthorizationCredentials | None = Security(_bearer)):
    _check_auth(creds)
    if _state.model is None:
        return {"object": "list", "data": []}
    return {"object": "list", "data": [_model_card()]}


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    _check_auth(creds)
    if _state.model is None or model_id not in (_state.model_name, "squish"):
        raise HTTPException(404, f"Model '{model_id}' not found")
    return _model_card()


def _model_card() -> dict:
    return {
        "id":         _state.model_name,
        "object":     "model",
        "created":    int(_state.loaded_at),
        "owned_by":   "squish",
        "permission": [],
        "root":       _state.model_name,
        "parent":     None,
        "squish": {
            "loader":      _state.loader_tag,
            "load_time_s": round(_state.load_time_s, 2),
            "requests":    _state.requests,
            "tokens_gen":  _state.tokens_gen,
        },
    }


def _make_chunk(content: str, model: str, cid: str, finish_reason=None) -> str:
    """Build an SSE data line in OpenAI streaming format."""
    chunk = {
        "id":                cid,
        "object":            "chat.completion.chunk",
        "created":           int(time.time()),
        "model":             model,
        "system_fingerprint": _system_fingerprint(),
        "choices": [{
            "index":         0,
            "delta":         {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/chat/completions

    Accepts standard OpenAI ChatCompletion request body.
    Returns streaming (stream=true) or non-streaming response.
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body: Dict[str, Any] = await request.json()
    messages    = body.get("messages", [])
    max_tokens  = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))
    stream      = bool(body.get("stream", False))
    stop        = body.get("stop", None)
    seed        = body.get("seed", None)
    model_id    = body.get("model", _state.model_name)
    tools       = body.get("tools", [])

    if not messages:
        raise HTTPException(400, "'messages' must be a non-empty list")

    # ── Tool calling: inject schema into system prompt ────────────────────
    if tools:
        from squish.tool_calling import format_tools_prompt
        messages = format_tools_prompt(messages, tools)
        # When tools are requested, force non-streaming so we can inspect
        # the full output before deciding between text and tool_calls.
        stream = False

    prompt         = _apply_chat_template(messages, _state.tokenizer)
    prompt_tokens  = _count_tokens(prompt)
    cid            = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    req_start      = time.perf_counter()
    _state.inflight += 1

    if stream:
        # ── Streaming response ────────────────────────────────────────────
        async def event_stream() -> AsyncIterator[str]:
            # Opening chunk (role delta)
            role_chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_id,
                "system_fingerprint": _system_fingerprint(),
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            gen = _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed)
            n_comp   = 0
            ttft_s   = 0.0
            last_finish = "stop"
            try:
                for tok_text, finish in gen:
                    if tok_text:
                        if n_comp == 0:
                            ttft_s = time.perf_counter() - req_start
                        n_comp += 1
                        yield _make_chunk(tok_text, model_id, cid)
                    if finish is not None:
                        last_finish = finish
                        break
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            finally:
                _state.inflight -= 1
                dur = time.perf_counter() - req_start
                _state.record_completion(n_comp, dur, ttft_s)
            yield _make_chunk("", model_id, cid, finish_reason=last_finish)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type = "text/event-stream",
            headers    = {
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id":     cid,
            },
        )
    else:
        # ── Non-streaming response ────────────────────────────────────────
        full_text    = ""
        last_finish  = "stop"
        ttft_s       = 0.0
        n_comp       = 0
        try:
            for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                if tok_text:
                    if n_comp == 0:
                        ttft_s = time.perf_counter() - req_start
                    n_comp   += 1
                    full_text += tok_text
                if finish is not None:
                    last_finish = finish
                    break
        finally:
            _state.inflight -= 1
            dur = time.perf_counter() - req_start
            _state.record_completion(n_comp, dur, ttft_s)

        comp_tokens = _count_tokens(full_text)

        # ── Tool calling: detect function call in output ──────────────────────
        if tools:
            from squish.tool_calling import parse_tool_calls, build_tool_calls_response
            raw_calls = parse_tool_calls(full_text)
            if raw_calls is not None:
                return JSONResponse({
                    "id":                 cid,
                    "object":             "chat.completion",
                    "created":            int(time.time()),
                    "model":              model_id,
                    "system_fingerprint": _system_fingerprint(),
                    "choices": [{
                        "index":   0,
                        "message": {
                            "role":       "assistant",
                            "content":    None,
                            "tool_calls": build_tool_calls_response(raw_calls),
                        },
                        "finish_reason": "tool_calls",
                        "logprobs":      None,
                    }],
                    "usage": {
                        "prompt_tokens":     prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens":      prompt_tokens + comp_tokens,
                    },
                })

        return JSONResponse({
            "id":                 cid,
            "object":             "chat.completion",
            "created":            int(time.time()),
            "model":              model_id,
            "system_fingerprint": _system_fingerprint(),
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": full_text},
                "finish_reason": last_finish,
                "logprobs":      None,
            }],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens":      prompt_tokens + comp_tokens,
            },
        })


@app.post("/v1/completions")
async def completions(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/completions — legacy text completion endpoint.
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body: Dict[str, Any] = await request.json()
    prompt      = body.get("prompt", "")
    max_tokens  = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))
    stream      = bool(body.get("stream", False))
    stop        = body.get("stop", None)
    seed        = body.get("seed", None)
    model_id    = body.get("model", _state.model_name)
    cid         = f"cmpl-{uuid.uuid4().hex[:12]}"
    req_start   = time.perf_counter()
    _state.inflight += 1

    if not prompt:
        raise HTTPException(400, "'prompt' must be a non-empty string")

    if stream:
        def _comp_chunk(text: str, finish_reason=None) -> str:
            chunk = {
                "id": cid, "object": "text_completion",
                "created": int(time.time()), "model": model_id,
                "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        async def comp_stream() -> AsyncIterator[str]:
            last_finish = "stop"
            n_comp = 0
            ttft_s = 0.0
            try:
                for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                    if tok_text:
                        if n_comp == 0:
                            ttft_s = time.perf_counter() - req_start
                        n_comp += 1
                        yield _comp_chunk(tok_text)
                    if finish is not None:
                        last_finish = finish
                        break
            finally:
                _state.inflight -= 1
                _state.record_completion(n_comp, time.perf_counter() - req_start, ttft_s)
            yield _comp_chunk("", finish_reason=last_finish)
            yield "data: [DONE]\n\n"

        return StreamingResponse(comp_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Request-Id": cid})
    else:
        full_text   = ""
        last_finish = "stop"
        n_comp      = 0
        ttft_s      = 0.0
        try:
            for tok_text, finish in _generate_tokens(prompt, max_tokens, temperature, top_p, stop, seed):
                if tok_text:
                    if n_comp == 0:
                        ttft_s = time.perf_counter() - req_start
                    n_comp   += 1
                    full_text += tok_text
                if finish is not None:
                    last_finish = finish
                    break
        finally:
            _state.inflight -= 1
            _state.record_completion(n_comp, time.perf_counter() - req_start, ttft_s)

        prompt_tokens = _count_tokens(prompt)
        comp_tokens   = _count_tokens(full_text)

        return JSONResponse({
            "id": cid, "object": "text_completion",
            "created": int(time.time()), "model": model_id,
            "choices": [{"text": full_text, "index": 0, "finish_reason": last_finish}],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens":      prompt_tokens + comp_tokens,
            },
        })


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/embeddings — mean-pooled last-hidden-state embeddings.

    Compatible with OpenAI embeddings API.
    Input: {'input': str | list[str], 'model': '...'}
    Output: {'object':'list', 'data':[{'object':'embedding','embedding':[...],'index':0}]}
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    import mlx.core as mx
    import numpy as np

    body: Dict[str, Any] = await request.json()
    inputs   = body.get("input", "")
    model_id = body.get("model", _state.model_name)
    if isinstance(inputs, str):
        inputs = [inputs]

    model     = _state.model
    tokenizer = _state.tokenizer
    results   = []
    total_tokens = 0

    for i, text in enumerate(inputs):
        ids = tokenizer.encode(text) if hasattr(tokenizer, "encode") else \
              tokenizer(text, return_tensors="np")["input_ids"][0].tolist()
        total_tokens += len(ids)

        x = mx.array(ids, dtype=mx.int32)[None]       # (1, seq)
        try:
            # Preferred path: last hidden state (proper semantic embeddings)
            hidden = model.model(x)                           # (1, seq, hidden_dim)
            emb_np = np.array(mx.mean(hidden, axis=1)[0])    # (hidden_dim,)
        except (AttributeError, TypeError):
            try:
                # Second-best: input token embeddings (less useful but available)
                tok_emb = model.model.embed_tokens(x)        # (1, seq, D)
                emb_np  = np.array(mx.mean(tok_emb, axis=1)[0])
            except AttributeError:
                # Last-resort: mean-pool logits (not suitable for similarity tasks)
                logits = model(x)                            # (1, seq, vocab)
                emb_np = np.array(mx.mean(logits[0], axis=0))

        # L2-normalize
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        results.append({
            "object":    "embedding",
            "embedding": emb_np.tolist(),
            "index":     i,
        })

    return JSONResponse({
        "object": "list",
        "model":  model_id,
        "data":   results,
        "usage":  {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


@app.get("/health")
async def health():
    return {
        "status":       "ok" if _state.model is not None else "no_model",
        "model":        _state.model_name,
        "loaded":       _state.model is not None,
        "loader":       _state.loader_tag,
        "load_time_s":  round(_state.load_time_s, 2),
        "requests":     _state.requests,
        "tokens_gen":   _state.tokens_gen,
        "inflight":     _state.inflight,
        "avg_tps":      round(_state.avg_tps, 1),
        "avg_ttft_s":   round(_state.avg_ttft, 3),
        "uptime_s":     round(time.time() - _state.loaded_at, 1) if _state.loaded_at else 0,
    }


@app.get("/v1/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics."""
    now = time.time()
    uptime = round(now - _state.loaded_at, 1) if _state.loaded_at else 0
    lines = [
        "# HELP squish_requests_total Total inference requests served",
        "# TYPE squish_requests_total counter",
        f"squish_requests_total {_state.requests}",
        "# HELP squish_tokens_generated_total Total tokens generated",
        "# TYPE squish_tokens_generated_total counter",
        f"squish_tokens_generated_total {_state.tokens_gen}",
        "# HELP squish_inflight_requests Current in-flight requests",
        "# TYPE squish_inflight_requests gauge",
        f"squish_inflight_requests {_state.inflight}",
        "# HELP squish_avg_tokens_per_second Rolling average tokens/sec (last 20 requests)",
        "# TYPE squish_avg_tokens_per_second gauge",
        f"squish_avg_tokens_per_second {_state.avg_tps:.2f}",
        "# HELP squish_avg_ttft_seconds Rolling average time-to-first-token (last 20 requests)",
        "# TYPE squish_avg_ttft_seconds gauge",
        f"squish_avg_ttft_seconds {_state.avg_ttft:.4f}",
        "# HELP squish_uptime_seconds Server uptime",
        "# TYPE squish_uptime_seconds counter",
        f"squish_uptime_seconds {uptime}",
        "# HELP squish_model_load_seconds Time taken to load the model",
        "# TYPE squish_model_load_seconds gauge",
        f"squish_model_load_seconds {_state.load_time_s:.3f}",
        "# HELP squish_prefix_cache_hits_total Prefix cache exact-match hits",
        "# TYPE squish_prefix_cache_hits_total counter",
        f"squish_prefix_cache_hits_total {_prefix_cache.hits}",
        "# HELP squish_prefix_cache_size Current entries in prefix cache",
        "# TYPE squish_prefix_cache_size gauge",
        f"squish_prefix_cache_size {_prefix_cache.size}",
        "# HELP squish_spec_draft_loaded Whether a draft model is loaded",
        "# TYPE squish_spec_draft_loaded gauge",
        f"squish_spec_draft_loaded {1 if _draft.generator is not None else 0}",
        "# HELP squish_kv_cache_tokens Current KV cache token count",
        "# TYPE squish_kv_cache_tokens gauge",
        f"squish_kv_cache_tokens {_kv_cache.n_tokens if _kv_cache is not None else 0}",
        "# HELP squish_kv_cache_memory_mb KV cache memory in MB",
        "# TYPE squish_kv_cache_memory_mb gauge",
        f"squish_kv_cache_memory_mb {_kv_cache.memory_mb if _kv_cache is not None else 0:.2f}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/tokenize")
async def tokenize(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/tokenize — tokenize text and return token IDs + count.
    Non-standard endpoint, useful for prompt engineering / debugging.

    Body: {"text": "..."}  or  {"messages": [{"role":"user","content":"..."}]}
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body = await request.json()
    if "messages" in body:
        text = _apply_chat_template(body["messages"], _state.tokenizer)
    elif "text" in body:
        text = body["text"]
    else:
        raise HTTPException(400, "Provide 'text' or 'messages' in request body")

    tok = _state.tokenizer
    try:
        ids = tok.encode(text) if hasattr(tok, "encode") else \
              tok(text, return_tensors="np")["input_ids"][0].tolist()
    except Exception as e:
        raise HTTPException(500, f"Tokenization failed: {e}")

    return JSONResponse({
        "token_ids":   ids,
        "token_count": len(ids),
        "model":       _state.model_name,
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description = "Squish OpenAI-compatible inference server",
        formatter_class = argparse.RawTextHelpFormatter,
        epilog = """
Examples:
  # Start server with 7B model
  python3 squish_server.py \\
    --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
    --compressed-dir ~/models/Qwen2.5-7B-Instruct-bf16-compressed

  # Use from any OpenAI client
  export OPENAI_BASE_URL=http://localhost:11435/v1
  export OPENAI_API_KEY=squish
  python3 -c "from openai import OpenAI; c=OpenAI(); print(c.chat.completions.create(model='squish', messages=[{'role':'user','content':'hello'}]).choices[0].message.content)"
"""
    )
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16"))
    ap.add_argument("--compressed-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16-compressed"))
    ap.add_argument("--port",    type=int, default=11435)
    ap.add_argument("--host",    default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--api-key", default=None,
                    help="Optional bearer token required on all requests. "
                         "Also readable from the SQUISH_API_KEY environment variable "
                         "(env var preferred — avoids key appearing in ps aux). "
                         "If omitted, no auth is enforced.")
    ap.add_argument("--draft-model", default="",
                    help="Path to small draft model dir for speculative decoding. "
                         "Should share tokeniser family with target (e.g. Qwen2.5-0.5B "
                         "with Qwen2.5-7B). Enables 1.8-2.5× throughput.")
    ap.add_argument("--draft-compressed", default="",
                    help="Compressed dir for the draft model (default: <draft-model>-compressed)")
    ap.add_argument("--no-prefix-cache", action="store_true", default=False,
                    help="Disable the prefix (exact-match) response cache")
    ap.add_argument("--prefix-cache-size", type=int, default=512,
                    help="LRU prefix cache capacity (default 512 entries)")
    # ── Phase 1.3: KV cache quantization ─────────────────────────────────────
    ap.add_argument("--kv-cache-mode",
                    choices=["fp16", "int8", "snap"],
                    default="fp16",
                    help="KV cache compression mode:\n"
                         "  fp16  — standard / no compression (default)\n"
                         "  int8  — KIVI: INT8 older tokens, FP16 recent window\n"
                         "  snap  — KIVI+SnapKV: INT8 + importance-based eviction")
    ap.add_argument("--kv-cache-window", type=int, default=64,
                    help="Recent-token FP16 window for int8/snap modes (default 64)")
    ap.add_argument("--kv-cache-budget", type=int, default=4096,
                    help="Max K/V positions in snap mode (default 4096)")
    ap.add_argument("--log-level",
                    choices=["critical", "error", "warning", "info", "debug", "trace"],
                    default="warning",
                    help="Uvicorn log verbosity (default: warning)")
    # ── Phase 2.1: Batch scheduler ────────────────────────────────────────────
    ap.add_argument("--batch-scheduler", action="store_true", default=False,
                    help="Enable continuous batching scheduler: collects concurrent\n"
                         "requests within --batch-window-ms and runs them in one\n"
                         "padded forward pass.  Improves throughput ~N× at moderate load.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Max concurrent requests per batch (default 8)")
    ap.add_argument("--batch-window-ms", type=float, default=20.0,
                    help="Collect window in ms before starting a batch (default 20)")
    args = ap.parse_args()

    global _API_KEY
    # Prefer explicit CLI flag; fall back to SQUISH_API_KEY env var.
    # Reading from env var prevents the secret appearing in `ps aux`.
    _API_KEY = args.api_key or os.environ.get("SQUISH_API_KEY")

    if args.no_prefix_cache:
        _prefix_cache._maxsize = 0
    elif args.prefix_cache_size != 512:
        _prefix_cache._maxsize = args.prefix_cache_size

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║       Squish OpenAI-compatible Server        ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  Model dir     : {args.model_dir}")
    print(f"  Compressed dir: {args.compressed_dir}")
    if args.draft_model:
        print(f"  Draft model   : {args.draft_model}")
    print(f"  Prefix cache  : {'disabled' if args.no_prefix_cache else args.prefix_cache_size}")
    if args.kv_cache_mode != "fp16":
        print(f"  KV cache mode : {args.kv_cache_mode}  "
              f"window={args.kv_cache_window}  budget={args.kv_cache_budget}")
    print(f"  Listen        : http://{args.host}:{args.port}")
    print()

    load_model(args.model_dir, args.compressed_dir, verbose=args.verbose)

    # ── Phase 2.1C: CPU/GPU split loading (auto-activates if model > Metal budget) ──
    if _state.model is not None:
        try:
            from squish.split_loader import SplitLayerLoader
            _split_info = SplitLayerLoader.auto_split(_state.model, verbose=True)
            if _split_info:
                print(
                    f"  CPU/GPU split   : {_split_info.cpu_count} layers offloaded  "
                    f"GPU={_split_info.gpu_gb:.2f}GB  CPU={_split_info.cpu_gb:.2f}GB"
                )
        except Exception as e:
            if args.verbose:
                print(f"  [split_loader] Skipped: {e}")

    # ── Phase 2.3: Flash Attention status check ──────────────────────────────
    if _state.model is not None:
        try:
            from squish.flash_attention import patch_model_attention
            patch_model_attention(_state.model, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"  [flash_attention] Skipped: {e}")

    # ── Phase 1.3: attach quantized KV cache if requested ─────────────
    global _kv_cache
    if args.kv_cache_mode != "fp16" and _state.model is not None:
        try:
            from squish.kv_cache import patch_model_kv_cache
            _kv_cache = patch_model_kv_cache(
                _state.model,
                mode=args.kv_cache_mode,
                window=args.kv_cache_window,
                budget=args.kv_cache_budget,
                verbose=True,
            )
            print(f"  KV cache ready ({args.kv_cache_mode})")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[KV cache] could not attach (%s) — running without KV quantisation", e
            )

    # ── Phase 2.1: start batch scheduler if requested ────────────────────────
    global _scheduler
    if args.batch_scheduler and _state.model is not None:
        try:
            from squish.scheduler import BatchScheduler, QueueFullError as _QFE
            global _QueueFullError
            _QueueFullError = _QFE
            _scheduler = BatchScheduler(
                _state.model, _state.tokenizer,
                max_batch_size  = args.batch_size,
                batch_window_ms = args.batch_window_ms,
            )
            _scheduler.start()
            print(f"  Batch scheduler : enabled  "
                  f"(max_batch={args.batch_size}  window={args.batch_window_ms:.0f}ms)")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[Scheduler] could not start (%s) — falling back to sequential mode", e
            )
            _scheduler = None

    if args.draft_model:
        print()
        load_draft_model(args.draft_model, args.draft_compressed, verbose=args.verbose)

    print()
    print(f"  Server ready → http://{args.host}:{args.port}/v1")
    print(f"  Web chat UI  → http://{args.host}:{args.port}/chat")
    print(f"  Ollama compat→ http://{args.host}:{args.port}/api/chat")
    print(f"  Set in clients:")
    print(f"    OPENAI_BASE_URL=http://{args.host}:{args.port}/v1")
    print(f"    OPENAI_API_KEY=squish")
    print()

    uvicorn.run(
        app,
        host      = args.host,
        port      = args.port,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
