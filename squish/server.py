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
import collections
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# ── Ensure the squish package root is importable when run as a script ────────
# cli.py launches this file directly with `python3 .../squish/server.py`, so
# the package parent directory must be on sys.path for `from squish.*` imports.
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:  # pragma: no cover
    sys.path.insert(0, _pkg_root)

# ── Validate dependencies ────────────────────────────────────────────────────

def _require(pkg: str, install: str | None = None) -> None:
    try:
        __import__(pkg)
    except ImportError:  # pragma: no cover
        hint = install or pkg
        print(f"  {_C.PK}✗  Missing dependency:{_C.R}  {_C.W}{pkg}{_C.R}  {_C.DIM}→  pip install {hint}{_C.R}")
        sys.exit(1)

_require("fastapi")
_require("uvicorn", "uvicorn[standard]")

from fastapi import FastAPI, HTTPException, Request, Security  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402

try:
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    _STATIC_FILES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STATIC_FILES_AVAILABLE = False

# ── KV cache (Phase 1.3 — lazily imported to keep startup fast) ──────────────
_kv_cache = None         # QuantizedKVCache | None — set in main() after model load
_paged_kv_cache = None   # PagedKVCache | None — set in main() when --paged-attention
_disk_prompt_cache = None  # DiskKVCache | None — set in main() when --disk-prompt-cache given
_lazy_llm_state = None  # _PruneState | None — set in main() when --lazy-llm given

# ── Wave optimization module state (lazily instantiated) ─────────────────────
_prompt_lookup_decoder  = None  # PromptLookupDecoder    — --prompt-lookup
_seq_packer             = None  # SequencePacker         — --seq-packing
_ada_serve_scheduler    = None  # AdaServeScheduler      — --ada-serve
_conf_spec_verifier     = None  # ConfSpecVerifier        — --conf-spec
_kvsharer_map           = None  # KVShareMap             — --kv-share
_kv_slab_allocator      = None  # KVSlabAllocator        — --kv-slab
_paris_kv_codebook      = None  # ParisKVCodebook        — --paris-kv
_streaming_sink_cache   = None  # SinkKVCache            — --streaming-sink
_diffkv_policy_mgr      = None  # DiffKVPolicyManager    — --diff-kv
_smallkv_cache          = None  # SmallKVCache           — --small-kv
_lookahead_engine       = None  # LookaheadReasoningEngine — --lookahead
_spec_reason_orch       = None  # SpecReasonOrchestrator — --spec-reason
_sage_attn_kernel       = None  # SageAttentionKernel     — --sage-attention
_sage_attn2_kernel      = None  # SageAttention2Kernel    — --sage-attention2
_sparge_engine          = None  # SpargeAttnEngine        — --sparge-attention
_squeeze_cache          = None  # SqueezeKVCache          — --squeeze-attention
_yoco_config            = None  # YOCOConfig              — --yoco-kv
_cla_config             = None  # CLAConfig               — --cla
_kvtuner_config         = None  # KVTunerConfig           — --kvtuner
_robust_sched           = None  # AMaxScheduler           — --robust-scheduler
_gemfilter_config       = None  # GemFilterConfig         — --gemfilter
_svdq_config            = None  # SVDqConfig              — --svdq
_sparse_spec_config     = None  # SparseSpecConfig        — --sparse-spec
_sparse_verify_config   = None  # SparseVerifyConfig      — --sparse-verify
_trail_config           = None  # TrailConfig             — --trail
_specontext_config      = None  # SpeContextConfig        — --specontext
_forelen_config         = None  # ForelenConfig           — --forelen
_ipw_config             = None  # IPWConfig               — --ipw
_layer_skip_config      = None  # EarlyExitConfig         — --layer-skip
_long_spec_config       = None  # LongSpecConfig          — --long-spec
_fr_spec_config         = None  # FRSpecConfig            — --fr-spec
_diffusion_draft_model  = None  # DiffusionDraftModel     — --diffusion-draft
# ── Wave 12: Reasoning-aware KV + Async I/O + MoE compression ──────────────
_pm_kvq_scheduler       = None  # PMKVQScheduler          — --pm-kvq
_mix_kvq_quantizer      = None  # MixKVQQuantizer         — --mix-kvq
_cocktail_kv_store      = None  # CocktailKVStore         — --cocktail-kv
_agile_io_manager       = None  # AgileIOManager          — --agile-io
_milo_quantizer         = None  # MiLoQuantizer           — --milo
_block_expert_archive   = None  # BlockExpertArchive      — --block-expert
# ── Wave 13: Vector Quantization + Adaptive Speculative Decoding ─────────────
_commvq_codebook        = None  # MultiCodebookVQ         — --commvq
_vptq_quantizer         = None  # VPTQQuantizer           — --vptq
_online_sd_updater      = None  # OnlineDraftUpdater      — --online-sd
_rasd_batcher           = None  # RASDBatcher             — --rasd
_dovetail_config        = None  # DovetailConfig          — --dovetail
_pipo_scheduler         = None  # PIPOScheduler           — --pipo
_disc_router            = None  # DISCRouter              — --disc-router
_mobile_moe_router      = None  # MoBiLERouter            — --mobile-moe
_meta_reasoner          = None  # MetaReasoner            — --meta-reasoner
# ── Wave 13b: Ultra-Long Context + Adaptive Speculative Decoding ─────────────
_duo_attn_manager       = None  # DuoKVManager            — --duo-attention
_shadow_kv_cache        = None  # ShadowKVCache           — --shadow-kv
_pq_cache_index         = None  # PQKeyIndex              — --pq-cache
_spe_cache_prefetcher   = None  # SpeCachePrefetcher      — --spe-cache
_duo_decoding_decoder   = None  # DuoDecodingDecoder      — --duo-decoding
_knapspec_selector      = None  # KnapSpecSelector        — --knapspec
_token_merging_cfg      = None  # TokenMergingConfig      — --token-merging
_token_swift_decoder    = None  # TokenSwiftDecoder       — --token-swift
_c2t_tree_builder       = None  # AdaptiveTreeBuilder     — --c2t
_clasp_decoder          = None  # CLaSPDecoder            — --clasp
# ── Wave 14: Quantization + Vocabulary-Adaptive Spec-Decode + Expert Mixing ──
_soup_experts_mixer     = None  # SoupOfExperts           — --soup-experts
_vision_prefix_cache    = None  # VisionPrefixCache       — --vision-cache
_vector_index           = None  # MRLIndex                — --vector-index
_sub_spec_decoder       = None  # SubSpecDecoder          — --sub-spec
_del_decoder_inst       = None  # DELDecoder              — --del-decoder
_dfloat11_cfg           = None  # DFloat11Config          — --dfloat11
_rans_codec_inst        = None  # RANSCodec               — --rans-codec
_qspec_decoder          = None  # QSpecDecoder            — --qspec
_quant_spec_decoder     = None  # QuantSpecDecoder        — --quant-spec
_copy_spec_drafter      = None  # CopySpecDrafter         — --copy-spec
_squeeze_llm_quant      = None  # SqueezeLLMQuantizer     — --squeeze-llm
_hetero_vocab_decoder   = None  # HeteroVocabDecoder      — --hetero-vocab-sd
_head_aware_kv_store    = None  # HeadAwareKVStore        — --head-infer
# Phase 3: cross-session persistent KV cache
_session_kv_cache    = None   # SessionKVCache | None — set in main() when --session-cache-dir given
# Phase 4: prompt compression settings (active when --compress-prompt is set)
_compress_enabled         = False
_compress_ratio           = 0.5
_compress_min_tokens      = 512
_compress_preserve_tokens = 0   # protect first N words from compression (RadixAttention synergy)

# ── Phase E1: Babbling Suppression (February 2026) ───────────────────────────
# Qwen3 architecture is a confirmed "babbler" — emits filler content after the
# task is complete, wasting 44–89% of decode energy.  Three complementary guards:
#   1. EOS probability monitoring: stop when model "wants" to stop (P(eos) > threshold)
#   2. Grammar terminal state: stop when XGrammar FSM accepts (schema is complete)
#   3. Hard token caps: per-task-type maximum output length
_babbling_suppression: bool    = True   # on by default; --no-babbling-suppression to disable
_babbling_eos_threshold: float = 0.30   # EOS softmax probability threshold
_babbling_min_tokens: int      = 10     # never trigger before this many decode steps

# Per-task-type hard token caps (0 = uncapped for that type).
# Tuned from real Squish output distributions.
_TASK_TOKEN_CAPS: dict = {
    "git_commit":  100,
    "devops_plan": 500,
    "code_review": 200,
    "email_draft": 300,
}

# ── Phase E2: Polynomial GELU approximation ──────────────────────────────────
# For GELU-based models, replace erf-based GELU with x * sigmoid(1.702x) —
# a single fused Metal op that the ANE handles at peak throughput.
# No-op for Qwen3 (already uses SiLU = x * sigmoid(x), already ANE-optimal).
_fast_gelu_enabled: bool = True  # on by default; --no-fast-gelu to disable

# ── Phase E3: Semantic response cache ────────────────────────────────────────
# Bypass the model entirely for semantically repeated queries.
# Per-task-type cosine similarity thresholds and response TTLs.
_semantic_cache = None   # SquishSemanticCache | None — set in main()
_SEMANTIC_CACHE_CONFIG: dict = {
    "git_commit":  {"threshold": 0.95, "ttl_hours": 24},
    "devops_plan": {"threshold": 0.88, "ttl_hours": 168},
    "code_review": {"threshold": 0.92, "ttl_hours": 72},
    "email_draft": {"threshold": 0.85, "ttl_hours": 48},
    "default":     {"threshold": 0.92, "ttl_hours": 48},
}

# ── Phase 3A: Chunked prefill (COMPRESS_PATH long sequences) ─────────────────
_chunk_prefill_enabled   = False  # set in main() via --chunk-prefill
_chunk_prefill_threshold = 512    # min token count to trigger chunking (default 512)
_chunk_prefill_size      = 512    # tokens per chunk (default 512)

# ── Phase 3C: MInference sparse attention ─────────────────────────────────────
_minference_enabled      = False  # set in main() via --minference
_minference_threshold    = 1024   # min seq_len to apply sparse attention (default 1024)

# ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────────
_thinking_budget: int = -1            # -1=unlimited, 0=disable thinking, >0=token limit
_think_close_token_id: int | None = None  # ID of </think> token, resolved at model load
# ── Phase A2: explicit MLX rotating KV cache size ────────────────────────────
_max_kv_size: int | None = None       # None = mlx_lm default (4K); set to extend context
# ── Phase A3: concise output mode ────────────────────────────────────────────
_concise_responses: bool = False      # prepend concision prefix + EOS bias
_CONCISION_PREFIX = (
    "Respond with only the requested output. "
    "No preamble, no explanation, no apologies.\n\n"
)
# ── Phase B: Structured output (XGrammar) ────────────────────────────────────
_grammar_engine: "Any | None" = None       # GrammarEngine instance, set at startup
_structured_output_mode: str = "none"      # "none" | "json" | "json-schema"
_structured_output_schema: "dict | None" = None  # parsed JSON schema (json-schema mode)
# ── Phase C: Power & Energy Modes ────────────────────────────────────────────
_power_monitor: "Any | None" = None        # PowerMonitor instance (auto mode only)
_power_mode: str = "performance"           # current effective mode name

# ── Conflict-Resolution Routing (Phase 0) ────────────────────────────────────
# Two exclusive request paths prevent incompatible optimizations firing together:
#
#   COMPRESS_PATH  — word count > _compress_threshold AND compress enabled
#       Uses: LLMLingua → chunked prefill → LazyLLM → EAGLE-3/N-gram draft
#       Skips: exact-match prefix cache (compressed text never matches cache)
#       Cache key: pre-compression token hash (future identical calls still hit)
#
#   PREFIX_PATH    — short or previously-cached prompts (default path)
#       Uses: RadixAttention → EAGLE-3/N-gram → LazyLLM (prefill-only mode)
#       Skips: LLMLingua (would invalidate cache keys)
#
# _inference_backend controls Phase 4 hardware dispatch (mutually exclusive):
#   'mlx-eager'    — standard MLX path (default)
#   'mlx-compiled' — mx.compile fused draft+verify decode kernel (Phase 4A)
#   'ane-disagg'   — Core ML ANE prefill + MLX decode (Phase 4B)
_compress_threshold  = 512          # word-count proxy above which COMPRESS_PATH fires
_inference_backend   = "mlx-eager"  # overridden by --inference-backend in main()

# ── Phase F: Inference Backend Abstraction ───────────────────────────────────

class _InferenceBackend:
    """Base shim for inference backend dispatch.

    Concrete subclasses override ``generate_stream``.  All generation paths
    are hardware-bound and marked ``# pragma: no cover``.  The ``__init__``
    constructors are testable (no hardware required).
    """

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _MLXEagerBackend(_InferenceBackend):
    """Standard MLX Metal eager execution path.

    Stores a reference to the loaded model and tokenizer so the dispatch
    layer can route ``generate_stream`` calls without global lookups.
    """

    def __init__(self, model: "Any", tokenizer: "Any") -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Route through _generate_tokens instead")


class _MLCBackend(_InferenceBackend):
    """MLC-LLM engine path for large-context requests.

    Probes for ``mlc_llm`` at construction time and sets
    :meth:`is_available` accordingly so callers can gate on its presence.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        try:
            import mlc_llm as _mlc  # noqa: F401,PLC0415
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        """Return ``True`` when ``mlc_llm`` was importable at construction time."""
        return self._available

    def generate_stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("MLC backend not yet wired")


_active_backend: "_InferenceBackend | None" = None  # set in main() when dispatching

# ── Batch scheduler (Phase 2.1 — continuous batching) ───────────────────────
_scheduler       = None  # BatchScheduler | None — set in main() when --batch-scheduler given
_QueueFullError  = None  # QueueFullError class — imported alongside BatchScheduler

# ── Terminal colours & ASCII art ─────────────────────────────────────────────
from squish._term import LOGO_GRAD as _LOGO_GRAD
from squish._term import C as _C  # noqa: E402
from squish._term import gradient as _term_gradient
from squish._term import has_truecolor as _has_truecolor  # noqa: E402

_TTY:           bool = sys.stdout.isatty()
_TTY_ERR:       bool = sys.stderr.isatty()
_TRUE_COLOR:     bool = _has_truecolor(sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else 1)
_TRUE_COLOR_ERR: bool = _has_truecolor(sys.stderr.fileno() if hasattr(sys.stderr, "fileno") else 2)


def _gradient(text: str, stops: list[tuple[int, int, int]]) -> str:
    """Thin wrapper so tests can monkeypatch _TRUE_COLOR to control output."""
    return _term_gradient(text, stops, force_color=_TRUE_COLOR)


def _cprint(color: str, label: str, value: str = "", end: str = "\n") -> None:
    """Print a coloured label + plain value line."""
    R = _C.R
    if value:
        print(f"  {color}{label}{R}  {_C.W}{value}{R}", end=end)
    else:
        print(f"  {color}{label}{R}", end=end)


def _ok(msg: str) -> None:
    """Print a success tick line."""
    print(f"  {_C.G}✓{_C.R}  {_C.W}{msg}{_C.R}")


def _info(label: str, value: str) -> None:
    """Print a key → value config line."""
    print(f"  {_C.L}◈{_C.R}  {_C.DIM}{label:<18}{_C.R}{_C.W}{value}{_C.R}")


def _warn(msg: str) -> None:
    """Print a yellow-ish warning line."""
    print(f"  {_C.PK}⚠{_C.R}  {_C.LPK}{msg}{_C.R}")


def _section(title: str) -> None:
    """Print a dimmed section divider."""
    print(f"  {_C.DIM}{'─' * 52}{_C.R}")
    if title:
        print(f"  {_C.MG}{title}{_C.R}")


def _print_banner() -> None:
    """Print the full ASCII-art startup banner."""
    R  = _C.R
    V  = _C.V;  L  = _C.L;  MG = _C.MG
    T  = _C.T;  PK = _C.PK
    W  = _C.W;  SIL = _C.SIL; DIM = _C.DIM

    print()

    if _TTY:
        # ── Squished character (clamp pressing cube flat — 1-row body = max squish) ──
        # Left connector bars = teal (inputs), right = pink (outputs), body = violet
        print(f"        {SIL}           ╤           {R}")
        print(f"        {SIL}   ╔═══════╧═══════╗   {R}")
        print(f"       {T}════{R}{V}╫{R}{W}   ◕  {R}{MG}˶‿˶{R}{W}  ◕   {R}{V}╫{R}{PK}════{R}")
        print(f"        {V}   ╚═══════════════╝{R}")
        print(f"            {DIM}═══════════════{R}")
        print(f"              {L}✦{R}    {PK}✦{R}    {L}✦{R}")
        print()

        # ── SQUISH gradient logo (box-drawing block font) ─────────────────────
        logo_lines = [
            " ██████╗   ██████╗  ██╗   ██╗  ██╗   ██████╗  ██╗  ██╗",
            "██╔════╝  ██╔═══██╗ ██║   ██║  ██║  ██╔════╝  ██║  ██║",
            "╚█████╗   ██║   ██║ ██║   ██║  ██║  ╚█████╗   ███████║",
            " ╚═══██╗  ██║▄▄ ██║ ██║   ██║  ██║   ╚═══██╗  ██╔══██║",
            "██████╔╝  ╚██████╔╝ ╚██████╔╝  ██║  ██████╔╝  ██║  ██║",
            "╚═════╝    ╚══▀▀═╝   ╚═════╝   ╚═╝  ╚═════╝   ╚═╝  ╚═╝",
        ]
        for line in logo_lines:
            print(f"  {_gradient(line, _LOGO_GRAD)}{R}")
        print()

        sub = "✦  Squish it. Run it. Go. &   ✦"
        print(f"            {_gradient(sub, _LOGO_GRAD)}{R}")
        print(f"  {DIM}{'─' * 56}{R}")
    else:
        # Plain-text fallback for non-TTY environments
        print("*** SQUISH — Squish it. Run it. Go.   ***")
        print("-" * 48)

    print()


# ── Verbose inference tracing ─────────────────────────────────────────────────
_trace: bool       = False   # set True by --trace in main()
_trace_tokens: bool = False  # set True by --trace-tokens in main()
_trace_file = None           # IO | None — file handle opened by --trace-file


def _tlog(msg: str) -> None:
    """Write a timestamped trace line to stderr (and _trace_file when set)."""
    _ke = lambda s: s if _TRUE_COLOR_ERR else ""  # noqa: E731
    ts  = f"{_ke(_C.MG)}[{time.strftime('%H:%M:%S')}]{_ke(_C.R)}"
    tag = f"{_ke(_C.V)}SQUISH{_ke(_C.R)}"
    line_color = f"{ts} {tag}  {_ke(_C.W)}{msg}{_ke(_C.R)}"
    line_plain = f"[SQUISH {time.strftime('%H:%M:%S')}] {msg}"
    print(line_color, file=sys.stderr, flush=True)
    if _trace_file is not None:
        try:
            _trace_file.write(line_plain + "\n")
            _trace_file.flush()
        except Exception:
            pass

# ── Tool calling + Ollama compat (Phase 2.2) ─────────────────────────────────
# Imported lazily in endpoints — no startup cost when unused
import uvicorn  # noqa: E402

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
    eagle_head = None   # EagleDraftHead instance (Phase 1B)

_draft = _DraftState()

# ── Prefix cache + RadixTree (Phase 1.4 / Phase 2B) ─────────────────────────
# Exact-match text response cache backed by RadixTree.
# RadixTree is a drop-in replacement for the old _PrefixCache:
#   • get() / put() / hits / size / _maxsize / clear() — same interface
#   • find_prefix(token_ids) / insert_prefix(token_ids, block_refs) — new (Phase 2B)
# When --paged-attention is enabled the server also records KV block refs so
# future requests with matching token prefixes can skip prefill entirely.
from squish.radix_cache import RadixTree as _RadixTree  # noqa: E402

_PrefixCache = _RadixTree   # backward-compat alias used by tests
_prefix_cache = _RadixTree(maxsize=512)


def _sample_mx(logits_row, temperature: float, top_p: float) -> int:  # pragma: no cover
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


# ── Phase E1: Task-type detection ─────────────────────────────────────────────
# Match on the first 200 chars of the prompt to classify the task.
# Only used to select the right token cap and semantic cache threshold.
_TASK_TYPE_KEYWORDS: dict = {
    "git_commit":  ("write a commit", "commit message", "git commit",
                    "summarize this diff", "write commit", "generate a commit"),
    "devops_plan": ("devops", "kubernetes", "deploy", "infrastructure",
                    "k8s", "argo ", "helm ", "kubectl", "ci/cd"),
    "code_review": ("review this code", "code review", "review the following",
                    "what's wrong with", "find bugs in", "critique this"),
    "email_draft": ("write an email", "draft an email", "email draft",
                    "compose an email", "write a message to"),
}


def _detect_task_type(prompt: str) -> str:
    """Return a task-type key by scanning the first 200 chars of *prompt*."""
    lower = prompt[:200].lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return task_type
    return "default"


# ── Phase E2: Polynomial GELU activation patch ────────────────────────────────


def _apply_fast_gelu(model_dir: str) -> None:  # pragma: no cover
    """
    Replace erf-based GELU activations with *x·sigmoid(1.702x)* — a single
    fused Metal op that the ANE executes at peak throughput.

    Skipped automatically for SiLU/SwiGLU models (Qwen3, LLaMA) because
    their activation is already ``x·sigmoid(x)``, which IS ANE-optimal.
    Only applied when the model config reports a GELU-family ``hidden_act``.
    """
    import json
    try:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            return
        cfg = json.loads(config_path.read_text())
        hidden_act = cfg.get("hidden_act", cfg.get("hidden_activation", "")).lower()
        # SiLU / SwiGLU: already x*sigmoid(x) → no-op
        if not hidden_act or hidden_act in ("silu", "swish", "swiglu"):
            return
        # Only patch GELU-family activations
        if "gelu" not in hidden_act:
            return
        import mlx.core as mx
        import mlx.nn as nn

        def _fast_gelu_fn(x: "mx.array") -> "mx.array":
            """x · σ(1.702x)  — single fused Metal multiply+sigmoid."""
            return x * mx.sigmoid(1.702 * x)

        patched = 0
        for layer in getattr(_state.model, "layers", []):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            for attr in ("act", "act_fn", "activation_fn", "activation"):
                current = getattr(mlp, attr, None)
                if current is nn.gelu or current is getattr(nn, "gelu_approx", None):
                    setattr(mlp, attr, _fast_gelu_fn)
                    patched += 1
        if patched > 0:
            _info("fast-gelu",
                  f"patched {patched} FFN activation layers  "
                  f"({hidden_act} → x·sigmoid(1.702x))")
    except Exception:
        pass   # never block startup on activation patching


def load_model(model_dir: str, compressed_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """Load the Squish compressed model into global state."""
    try:
        from .compressed_loader import load_compressed_model as _load_compressed_model
    except ImportError:
        # server.py launched directly (not as package) — use absolute import
        from squish.compressed_loader import load_compressed_model as _load_compressed_model
    # Keep backward-compat shim
    load_from_npy_dir = _load_compressed_model

    t0 = time.perf_counter()
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading model:{_C.R}  {_C.W}{compressed_dir}{_C.R}")

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
        _ok(f"Model ready  ({elapsed:.2f}s  loader={_state.loader_tag})")

    _cap_metal_cache(verbose=verbose)


def load_mlx_model(mlx_model_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """
    Load a native mlx_lm model directory directly via ``mlx_lm.load()``.

    This is the memory-efficient path: INT4/INT8 quantized mlx_lm models
    keep weights quantized in Metal (≈4-5 GB for 8B INT4) rather than
    dequantizing to BF16 at load time (≈15 GB).

    Use after converting with::

        python3 -m mlx_lm.convert \\
            --hf-path  <bf16-model-dir> \\
            --mlx-path <mlx-int4-model-dir> \\
            -q --q-bits 4

    Parameters
    ----------
    mlx_model_dir : path to the mlx_lm-format quantized model directory
    """
    import mlx_lm
    t0 = time.perf_counter()
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading mlx_lm model:{_C.R}  {_C.W}{mlx_model_dir}{_C.R}")

    model, tokenizer = mlx_lm.load(mlx_model_dir)
    elapsed = time.perf_counter() - t0

    _state.model      = model
    _state.tokenizer  = tokenizer
    _state.model_name = Path(mlx_model_dir).name
    _state.loaded_at  = time.time()
    _state.load_time_s = elapsed
    _state.loader_tag  = "mlx_lm"
    if verbose:
        _ok(f"Model ready  ({elapsed:.2f}s  loader=mlx_lm)")

    _cap_metal_cache(verbose=verbose)


def _cap_metal_cache(verbose: bool = False, limit_mb: int = 256) -> None:  # pragma: no cover
    """
    Cap the MLX Metal allocator's buffer pool after model load.

    By default MLX keeps an unbounded Metal buffer cache for reuse.  After
    the model is fully loaded and eval'd, this cache can hold gigabytes of
    stale buffers.  Capping it to ``limit_mb`` MB frees that memory back to
    the OS without affecting inference performance (the cache is only used
    for *new* allocations, not existing model weights).
    """
    try:
        import gc

        import mlx.core as mx
        gc.collect()
        # eval outstanding lazy ops so nothing is unexpectedly freed
        mx.eval(())
        limit_bytes = limit_mb * 1024 * 1024
        if hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(limit_bytes)
            if verbose:
                print(f"  {_C.DIM}◈  Metal buffer cache capped at {limit_mb} MB{_C.R}")
        gc.collect()
    except Exception:
        pass


def load_draft_model(draft_model_dir: str, draft_compressed_dir: str = "",  # pragma: no cover
                     verbose: bool = True) -> None:
    """Load the small draft model used for speculative decoding."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from squish.speculative import load_draft_model as _load_draft
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading draft model:{_C.R}  {_C.W}{draft_model_dir}{_C.R}")
    draft_m, draft_tok = _load_draft(
        draft_model_dir,
        draft_compressed_dir or (draft_model_dir + "-compressed"),
        verbose=verbose,
    )
    _draft.model     = draft_m
    _draft.tokenizer = draft_tok
    _draft.model_dir = draft_model_dir
    if verbose:
        _ok("Draft model ready")

    # Build the SpeculativeGenerator now that both models are loaded
    _rebuild_spec_gen()


def load_eagle_head(head_dir: str, verbose: bool = True) -> None:  # pragma: no cover
    """Load an EAGLE-3 draft head and wire it into the SpeculativeGenerator."""
    from squish.speculative import EagleDraftHead
    if verbose:
        print(f"  {_C.L}⟳{_C.R}  {_C.DIM}Loading EAGLE-3 head:{_C.R}  {_C.W}{head_dir}{_C.R}")
    _draft.eagle_head = EagleDraftHead.from_dir(head_dir, _state.model, verbose=verbose)
    if verbose:
        _ok("EAGLE-3 head ready")
    _rebuild_spec_gen()


def _rebuild_spec_gen() -> None:  # pragma: no cover
    """(Re-)create the SpeculativeGenerator from current target + draft state."""
    if _state.model is None:
        _draft.generator = None
        return
    # Require at least one draft source (neural draft model OR EAGLE head)
    if _draft.model is None and _draft.eagle_head is None:
        _draft.generator = None
        return
    from squish.speculative import SpeculativeGenerator
    _draft.generator = SpeculativeGenerator(
        _state.model, _state.tokenizer,
        draft_model=_draft.model, draft_tokenizer=_draft.tokenizer,
        eagle_head=_draft.eagle_head,
    )


# ── Token generation ─────────────────────────────────────────────────────────

def _apply_chat_template(messages: list[dict[str, str]], tokenizer) -> str:
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


def _get_stop_ids(stop: list[str] | str | None) -> list[list[int]]:
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


def _generate_tokens(  # pragma: no cover
    prompt: str,
    max_tokens: int    = 512,
    temperature: float = 0.7,
    top_p: float       = 0.9,
    stop: list[str] | str | None = None,
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

    # ── Phase E: task-type classification ────────────────────────────────────
    # Detect once per request; used for babbling suppression caps and
    # semantic cache threshold selection.
    _task_type = _detect_task_type(prompt)

    # ── Phase E3: Semantic response cache lookup ──────────────────────────────
    # Check BEFORE any model work.  A warm cache hit returns in <20 ms.
    if _semantic_cache is not None:
        try:
            _cached_response = _semantic_cache.lookup(prompt, _task_type)
            if _cached_response is not None:
                for _ch in _cached_response:
                    yield _ch, None
                yield "", "stop"
                return
        except Exception:
            pass  # never block generation on cache lookup failure

    # ── Phase 4: prompt compression ───────────────────────────────────────────
    # Compress long prompts before tokenization to reduce prefill cost.
    # Only applied when --compress-prompt is set and the prompt meets the
    # minimum length threshold.
    #
    # CONFLICT RESOLUTION (LLMLingua ↔ DiskKVCache / prefix cache):
    # Cache keys must use the *original* (pre-compression) prompt so that a
    # future identical request hits the cache even when compression was applied.
    # We capture _orig_prompt NOW, then route based on prompt length.
    _orig_prompt = prompt         # pre-compression canonical text for all cache keys
    _on_compress_path = False     # True → COMPRESS_PATH; False → PREFIX_PATH

    if _compress_enabled:
        _word_count = len(prompt.split())
        if _word_count >= _compress_min_tokens:
            _on_compress_path = True
            try:
                from squish.prompt_compressor import compress as _compress_fn
                prompt = _compress_fn(
                    prompt,
                    ratio=_compress_ratio,
                    # preserve_tokens protects the fixed system-prompt prefix from
                    # compression so that RadixAttention still hits on that prefix
                    # for PREFIX_PATH requests (LLMLingua ↔ RadixAttention synergy).
                    # Controlled by --compress-preserve-tokens (default 0 = disabled).
                    preserve_tokens=_compress_preserve_tokens,
                )
            except Exception:
                pass  # never block generation on compression failure

    # ── Trace: log request entry ───────────────────────────────────────────────
    _rid = uuid.uuid4().hex[:8]          # short per-request ID for log correlation
    if _trace:
        _prompt_tokens_approx = len(prompt.split())
        _prompt_preview = prompt[:400].replace("\n", "↵") + ("…" if len(prompt) > 400 else "")
        _tlog(f"REQ {_rid}  max_tokens={max_tokens}  temp={temperature}  "
              f"top_p={top_p}  seed={seed}  prompt_words≈{_prompt_tokens_approx}")
        _tlog(f"REQ {_rid}  prompt: {_prompt_preview}")

    # Reset LazyLLM pruning state for this request (Item 3)
    if _lazy_llm_state is not None:
        _lazy_llm_state.active_mask = None

    # ── Batch scheduler dispatch (Phase 2.1) ──────────────────────────────────
    # Route non-deterministic requests through the coalescing batch scheduler.
    # submit_sync() is a plain blocking generator — compatible with this sync
    # generator function without any async bridge required.
    is_deterministic = (temperature == 0.0 or seed is not None)
    if _scheduler is not None and not is_deterministic:
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → batch-scheduler")
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
    #
    # CONFLICT RESOLUTION (LLMLingua ↔ prefix cache):
    # Requests on COMPRESS_PATH have a stochastically-compressed prompt whose
    # token sequence differs on every call — prefix caching would never hit.
    # Skip the prefix cache entirely for COMPRESS_PATH requests.
    # Keys always use _orig_prompt so a future identical *uncompressed* request
    # still matches a response that was generated after compression.
    cache_eligible = (use_cache
                      and (temperature == 0.0 or seed is not None)
                      and not _on_compress_path)
    if cache_eligible:
        cached = _prefix_cache.get(_orig_prompt)
        if cached is not None:
            full_text, finish_reason = cached
            if _trace:
                _tlog(f"REQ {_rid}  dispatch → prefix-cache HIT  "
                      f"({len(full_text)} chars, finish={finish_reason})")
            for char in full_text:
                yield char, None
            yield "", finish_reason
            return

    # Collect full output so we can populate the cache after generation
    _cache_buf: list[str] = [] if cache_eligible else []
    _sc_buf:    list[str] = []  # Phase E3: full response text for semantic cache
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
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → speculative-decoding")
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
                if _trace_tokens and tok_text:
                    _tlog(f"REQ {_rid}  tok={tok_text!r}")
                yield tok_text, finish
                if finish is not None:
                    if _trace:
                        _n_spec = len(_cache_buf) if _cache_buf else 0
                        _tlog(f"REQ {_rid}  DONE  path=speculative  "
                              f"tokens={_n_spec}  finish={finish}")
                    break
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), _last_finish)
            return
        except Exception as _spec_err:
            import logging as _log
            _log.getLogger(__name__).warning("Speculative decoding failed (%s); "
                                             "falling back to standard generation", _spec_err)

    # ── Quantized KV cache generation path ─────────────────────────────────────
    if _kv_cache is not None:
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → kv-cache ({_kv_cache.__class__.__name__})")
        _kv_cache.reset()
        try:
            import mlx.core as mx
            import numpy as np
            # Tokenize the *original* (pre-compression) prompt for KV/disk cache
            # key derivation, then re-tokenize the (possibly compressed) prompt for
            # the actual model forward pass.  This ensures the disk cache key is
            # stable even when LLMLingua produces a different compressed form.
            _orig_input_ids = (
                tokenizer.encode(_orig_prompt)
                if hasattr(tokenizer, "encode")
                else tokenizer(_orig_prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            input_ids = (
                tokenizer.encode(prompt)
                if hasattr(tokenizer, "encode")
                else tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()
            )
            layer_caches = _kv_cache._layers
            # ── Phase 3: session KV cache lookup ───────────────────────────────
            # Restore KV state from a prior conversation if a matching session
            # exists.  Key is SHA-256 of the first 2 KB of the ORIGINAL prompt.
            _session_key = None
            if _session_kv_cache is not None:
                try:
                    import hashlib as _hl
                    _session_key = _hl.sha256(_orig_prompt[:2048].encode()).hexdigest()[:32]
                    _sess_result = _session_kv_cache.load_session(_session_key)
                    if _sess_result is not None:
                        _kv_cache.restore_from(_sess_result)
                        if _trace:
                            _tlog(f"REQ {_rid}  session-cache HIT  key={_session_key}")
                    elif _trace:
                        _tlog(f"REQ {_rid}  session-cache MISS  key={_session_key}")
                except Exception:
                    _session_key = None  # never block generation on session error
            # ── Disk prompt-cache lookup (Item 2) ──────────────────────────────
            # On a hit, restore KV state from NVMe and skip prefill (O(n) → O(1))
            _disk_hit_logit = None
            if _disk_prompt_cache is not None:
                try:
                    # Key by the original (pre-compression) token IDs so that
                    # different LLMLingua compressions of the same prompt still hit.
                    _disk_result = _disk_prompt_cache.lookup(_orig_input_ids)
                    if _disk_result is not None:
                        _disk_qkv, _disk_last_logit = _disk_result
                        _kv_cache.restore_from(_disk_qkv)
                        _disk_hit_logit = _disk_last_logit
                        if _trace:
                            _tlog(f"REQ {_rid}  disk-prompt-cache HIT  "
                                  f"orig_tokens={len(_orig_input_ids)}  → skipped prefill")
                    elif _trace:
                        _tlog(f"REQ {_rid}  disk-prompt-cache MISS  orig_tokens={len(_orig_input_ids)}")
                except Exception:
                    pass  # disk lookup error — fall through to normal prefill

            if _disk_hit_logit is not None:
                # Cache hit: use stored logit to sample first token; no prefill needed
                last_logit_mlx = mx.array(_disk_hit_logit, dtype=mx.float32)
                next_id = _sample_mx(last_logit_mlx, temperature, top_p)
            else:
                # Cache miss: run full prefill
                # ── Phase 3C: patch sparse attention for long sequences ────────
                # Applied BEFORE prefill; must be unpatched after regardless of
                # the prefill path taken (standard or chunked).
                # Guard: only when NOT using ane-disagg backend (Core ML graphs
                # are pre-compiled and cannot accept Python-level mask injection).
                _minf_restore = None
                if (_minference_enabled
                        and len(input_ids) > _minference_threshold
                        and _inference_backend != "ane-disagg"):
                    try:
                        from squish.minference_patch import (
                            patch_model_minference as _patch_minf,
                        )
                        from squish.minference_patch import (
                            select_pattern_for_sequence as _minf_pattern,
                        )
                        _pattern = _minf_pattern(len(input_ids))
                        _minf_restore = _patch_minf(
                            model,
                            seq_len_threshold=0,   # already gated above
                            pattern=_pattern,
                        )
                        if _trace:
                            _tlog(f"REQ {_rid}  minference PATCHED  "
                                  f"pattern={_pattern}  seq_len={len(input_ids)}")
                    except Exception as _minf_err:
                        import logging as _mlog
                        _mlog.getLogger(__name__).debug(
                            "[minference] patch failed (%s) — dense fallback", _minf_err
                        )
                        _minf_restore = None

                # ── Phase 3A: chunked prefill (COMPRESS_PATH, long prompts) ────
                # CRITICAL: spec decode starts only after is_final_chunk=True.
                # Interleaved greedy tokens emitted on non-final chunks DO count
                # toward the output but bypass the speculative decode path.
                _last_logit_vec = None   # [vocab_size] mlx array
                if (_on_compress_path
                        and _chunk_prefill_enabled
                        and len(input_ids) > _chunk_prefill_threshold):
                    try:
                        from squish.chunked_prefill import (
                            ChunkedPrefillConfig as _CPFConfig,
                        )
                        from squish.chunked_prefill import (
                            chunk_prefill as _chunk_prefill_fn,
                        )
                        _cpf_cfg = _CPFConfig(chunk_size=_chunk_prefill_size)
                        if _trace:
                            _tlog(f"REQ {_rid}  chunked-prefill START  "
                                  f"tokens={len(input_ids)}  "
                                  f"chunk={_chunk_prefill_size}")
                        for _clogit, _is_fin in _chunk_prefill_fn(
                                model, input_ids, layer_caches, _cpf_cfg):
                            if _is_fin:
                                _last_logit_vec = _clogit
                            elif _cpf_cfg.interleave_decode:
                                # Yield one greedy token between chunks for TTFT.
                                # CRITICAL: spec decode MUST NOT start here.
                                _il_id = _sample_mx(_clogit, temperature, top_p)
                                _il_tok = (
                                    tokenizer.decode([_il_id])
                                    if hasattr(tokenizer, "decode") else str(_il_id)
                                )
                                if cache_eligible:
                                    _cache_buf.append(_il_tok)
                                yield _il_tok, None
                        if _trace:
                            _tlog(f"REQ {_rid}  chunked-prefill DONE")
                    except Exception as _cpf_err:
                        import logging as _cpflog
                        _cpflog.getLogger(__name__).warning(
                            "[chunk-prefill] failed (%s) — standard prefill", _cpf_err
                        )
                        _last_logit_vec = None  # fall through below

                if _last_logit_vec is None:
                    # Standard single-shot prefill (non-compress path or fallback)
                    x = mx.array(input_ids, dtype=mx.int32)[None]
                    logits_full = model(x, cache=layer_caches)
                    mx.eval(logits_full)
                    _last_logit_vec = logits_full[0, -1]

                # ── Phase 3C: restore dense attention after prefill ────────────
                if _minf_restore is not None:
                    try:
                        from squish.minference_patch import (
                            unpatch_model_minference as _unpatch_minf,
                        )
                        _unpatch_minf(model, _minf_restore)
                        if _trace:
                            _tlog(f"REQ {_rid}  minference UNPATCHED")
                    except Exception:
                        pass  # never block generation on unpatch failure
                    _minf_restore = None

                next_id = _sample_mx(_last_logit_vec, temperature, top_p)
                # Persist for future requests in background
                if _disk_prompt_cache is not None:
                    try:
                        _last_logit_np = np.array(_last_logit_vec.astype(mx.float32))
                        # Store under original token IDs for stable cache keys
                        _disk_prompt_cache.store(_orig_input_ids, _kv_cache, _last_logit_np)
                    except Exception:
                        pass
            stop_buf = [next_id]
            # Compile the single-token decode step for faster subsequent calls.
            # layer_caches is captured as a constant closure; the list reference
            # never changes, so mx.compile reuses the compiled graph every step.
            _decode_fn = None
            if not getattr(_state, "_no_compile", False):
                try:
                    _decode_fn = mx.compile(
                        lambda tok_x: model(tok_x, cache=layer_caches)
                    )
                except Exception:
                    pass  # mx.compile unavailable or incompatible — use plain call
            # Phase A1: thinking budget tracking state
            _in_think_block = False
            _think_step_count = 0
            # Phase B: initialise grammar FSM state for this request
            _grammar_state = None
            if _grammar_engine is not None:
                if _structured_output_mode == "json":
                    _grammar_state = _grammar_engine.json_object_grammar()
                elif _structured_output_mode == "json-schema" and _structured_output_schema is not None:
                    _grammar_state = _grammar_engine.json_schema_grammar(_structured_output_schema)
            for step in range(max_tokens):
                # ── Phase E1: Hard token cap (babbling suppression) ──────────────
                if _babbling_suppression:
                    _bs_cap = _TASK_TOKEN_CAPS.get(_task_type, 0)
                    if _bs_cap > 0 and step >= _bs_cap:
                        if cache_eligible and _cache_buf:
                            _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                        if _trace:
                            _tlog(f"REQ {_rid}  babbling-cap  step={step}  task={_task_type}  cap={_bs_cap}")
                        yield "", "stop"
                        return
                tok_text = (
                    tokenizer.decode([next_id])
                    if hasattr(tokenizer, "decode")
                    else str(next_id)
                )
                # Phase A1: track thinking block boundaries
                if _thinking_budget >= 0:
                    if "<think>" in tok_text:
                        _in_think_block = True
                        _think_step_count = 0
                    elif "</think>" in tok_text:
                        _in_think_block = False
                    elif _in_think_block:
                        _think_step_count += 1
                if next_id == eos_id:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                    # Phase E3: persist clean EOS completion to semantic cache
                    if _semantic_cache is not None and _sc_buf:
                        try:
                            _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                        except Exception:
                            pass
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=kv-cache  tokens={step}  finish=stop(eos)")
                    yield tok_text, "stop"
                    return
                if stop_ids:
                    for seq in stop_ids:
                        if stop_buf[-len(seq):] == seq:
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                            # Phase E3: persist stop-sequence completion to semantic cache
                            if _semantic_cache is not None and _sc_buf:
                                try:
                                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                except Exception:
                                    pass
                            if _trace:
                                _tlog(f"REQ {_rid}  DONE  path=kv-cache  "
                                      f"tokens={step}  finish=stop(stop-seq)")
                            yield tok_text, "stop"
                            return
                    if len(stop_buf) > 64:
                        stop_buf = stop_buf[-64:]
                if step == max_tokens - 1:
                    if cache_eligible and _cache_buf:
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=kv-cache  tokens={step + 1}  finish=length")
                    yield tok_text, "length"
                    return
                if cache_eligible:
                    _cache_buf.append(tok_text)
                if _semantic_cache is not None:
                    _sc_buf.append(tok_text)
                if _trace_tokens:
                    _tlog(f"REQ {_rid}  tok={tok_text!r}")
                yield tok_text, None
                x = mx.array([[next_id]], dtype=mx.int32)
                logits = _decode_fn(x) if _decode_fn is not None else model(x, cache=layer_caches)
                mx.eval(logits)
                # Phase A1/A3: apply logit biases before sampling
                _logit_vec = logits[0, -1]
                if (_thinking_budget > 0
                        and _in_think_block
                        and _think_step_count >= _thinking_budget
                        and _think_close_token_id is not None):
                    _lg_np = np.array(_logit_vec.astype(mx.float32))
                    _lg_np[_think_close_token_id] += 100.0
                    _logit_vec = mx.array(_lg_np)
                if _concise_responses and step >= 20:
                    _lg_np = np.array(_logit_vec.astype(mx.float32))
                    _lg_np[eos_id] += 8.0
                    _logit_vec = mx.array(_lg_np)
                # ── Phase E1: EOS probability monitoring (babbling suppression) ──
                if _babbling_suppression and step >= _babbling_min_tokens:
                    _eos_logit_val = float(_logit_vec[eos_id].item())
                    _max_logit_val = float(mx.max(_logit_vec).item())
                    if _eos_logit_val > _max_logit_val - 1.5:  # pre-filter: EOS is near-top
                        _bs_np = np.array(_logit_vec.astype(mx.float32))
                        _bs_shifted = _bs_np - _bs_np.max()
                        _bs_exp = np.exp(np.clip(_bs_shifted, -30, 0))
                        _eos_prob = _bs_exp[eos_id] / (_bs_exp.sum() + 1e-9)
                        if _eos_prob > _babbling_eos_threshold:
                            if cache_eligible and _cache_buf:
                                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                            # Phase E3: model-chosen stop — cache it
                            if _semantic_cache is not None and _sc_buf:
                                try:
                                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                except Exception:
                                    pass
                            if _trace:
                                _tlog(f"REQ {_rid}  babbling-eos  step={step}  p={_eos_prob:.3f}  task={_task_type}")
                            yield "", "stop"
                            return
                # Phase B: grammar-constrained logits
                if _grammar_engine is not None and _grammar_state is not None:
                    _logit_vec = _grammar_engine.constrain_logits(_logit_vec, _grammar_state)
                next_id = _sample_mx(_logit_vec, temperature, top_p)
                # Phase B: advance grammar FSM after sampling
                if _grammar_engine is not None and _grammar_state is not None:
                    _grammar_state = _grammar_engine.advance(_grammar_state, next_id)
                    # ── Phase E1: Grammar terminal state (babbling suppression) ──
                    if _babbling_suppression and _grammar_state is not None:
                        try:
                            if _grammar_state.is_terminated():
                                if cache_eligible and _cache_buf:
                                    _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                                # Phase E3: FSM-complete response — worth caching
                                if _semantic_cache is not None and _sc_buf:
                                    try:
                                        _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                                    except Exception:
                                        pass
                                if _trace:
                                    _tlog(f"REQ {_rid}  babbling-grammar-terminal  step={step}")
                                yield "", "stop"
                                return
                        except AttributeError:
                            pass  # xgrammar version without is_terminated()
                stop_buf.append(next_id)
                # Wave 12: advance PM-KVQ scheduler each decode step
                if _pm_kvq_scheduler is not None:
                    try:
                        _pm_kvq_scheduler.advance()
                    except Exception:
                        pass
                # Phase 0C: fire async CPU dequant for next step while we set up
                # the token embedding — hides O(n_old_tokens) numpy cost behind
                # the model's token-embedding + layernorm overhead.
                for _lc in layer_caches:
                    if hasattr(_lc, "start_prefetch"):
                        _lc.start_prefetch()
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
            # Phase E3: end-of-loop clean completion — store in semantic cache
            if _semantic_cache is not None and _sc_buf:
                try:
                    _semantic_cache.store(_orig_prompt, "".join(_sc_buf), _task_type)
                except Exception:
                    pass
            # Phase 3: persist KV state for future sessions (background thread)
            if _session_kv_cache is not None and _session_key is not None:
                try:
                    _session_kv_cache.save_session(_session_key, _kv_cache)
                except Exception:
                    pass
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
        if _trace:
            _tlog(f"REQ {_rid}  dispatch → mlx_lm.stream_generate")
        _sg_kwargs = {}
        if _max_kv_size is not None:
            _sg_kwargs["max_kv_size"] = _max_kv_size
        gen = mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt     = prompt,
            max_tokens = max_tokens,
            temp       = temperature,
            top_p      = top_p,
            **_sg_kwargs,
        )
        emitted = 0
        stop_buf: list[int] = []
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
                        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  "
                              f"finish=stop(stop-seq)")
                    yield tok_text, "stop"
                    return
                if len(stop_buf) > 64:
                    stop_buf = stop_buf[-64:]

            if emitted >= max_tokens:
                if cache_eligible:
                    _cache_buf.append(tok_text)
                    _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
                if _trace:
                    _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  finish=length")
                yield tok_text, "length"
                return
            if cache_eligible:
                _cache_buf.append(tok_text)
            if _trace_tokens:
                _tlog(f"REQ {_rid}  tok={tok_text!r}")
            yield tok_text, None
        if cache_eligible and _cache_buf:
            _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
        if _trace:
            _tlog(f"REQ {_rid}  DONE  path=mlx_lm  tokens={emitted}  finish=stop(eos)")
        yield "", "stop"
        return
    except (AttributeError, TypeError):
        pass

    # ── Fallback: manual sampling loop ───────────────────────────────────────
    import mlx.core as mx
    import numpy as np

    if _trace:
        _tlog(f"REQ {_rid}  dispatch → manual-sampling-loop (fallback)")
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
            if _trace:
                _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step}  finish=stop(eos)")
            yield "", "stop"
            return
        ids.append(next_id)
        tok_text = tokenizer.decode([next_id])

        if stop_ids:
            stop_buf.append(next_id)
            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    if _trace:
                        _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step}  "
                              f"finish=stop(stop-seq)")
                    yield tok_text, "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

        if step == max_tokens - 1:
            if cache_eligible and _cache_buf:
                _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "length")
            if _trace:
                _tlog(f"REQ {_rid}  DONE  path=manual  tokens={step + 1}  finish=length")
            yield tok_text, "length"
            return
        if cache_eligible:
            _cache_buf.append(tok_text)
        if _trace_tokens:
            _tlog(f"REQ {_rid}  tok={tok_text!r}")
        yield tok_text, None

    if cache_eligible and _cache_buf:
        _prefix_cache.put(_orig_prompt, "".join(_cache_buf), "stop")
    if _trace:
        _tlog(f"REQ {_rid}  DONE  path=manual  tokens={max_tokens}  finish=stop")
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
    from .ollama_compat import mount_ollama as _mount_ollama  # package import
except ImportError:  # pragma: no cover
    from ollama_compat import mount_ollama as _mount_ollama  # direct script run
_mount_ollama(
    app,
    get_state     = lambda: _state,
    get_generate  = lambda: _generate_tokens,
    get_tokenizer = lambda: _state.tokenizer,
)

# ── Web chat UI (/chat) ────────────────────────────────────────────────
if _STATIC_FILES_AVAILABLE:  # pragma: no branch
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():  # pragma: no branch
        app.mount("/static", _StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/chat")
async def web_chat_ui():
    """Serve the single-page web chat interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"error": "Web UI not found. Is squish/static/index.html present?"}, status_code=404)  # pragma: no cover


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
    return _model_card()  # pragma: no cover


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
async def chat_completions(  # pragma: no cover
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

    body: dict[str, Any] = await request.json()
    messages    = body.get("messages", [])
    max_tokens  = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))
    stream      = bool(body.get("stream", False))
    stop        = body.get("stop", None)
    seed        = body.get("seed", None)
    model_id    = body.get("model", _state.model_name)
    tools       = body.get("tools", [])

    # ── Phase A1: /no_think mode (thinking_budget == 0) ──────────────────────
    if _thinking_budget == 0:
        _msgs_copy = []
        _found_sys = False
        for _m in messages:
            if _m.get("role") == "system" and not _found_sys:
                _msgs_copy.append({**_m, "content": (_m.get("content", "") + " /no_think").strip()})
                _found_sys = True
            else:
                _msgs_copy.append(_m)
        if not _found_sys:
            _msgs_copy = [{"role": "system", "content": "/no_think"}] + list(messages)
        messages = _msgs_copy

    # ── Phase A3: concision prefix ────────────────────────────────────────────
    if _concise_responses:
        _msgs_copy = []
        _found_sys = False
        for _m in messages:
            if _m.get("role") == "system" and not _found_sys:
                _msgs_copy.append({**_m, "content": _CONCISION_PREFIX + _m.get("content", "")})
                _found_sys = True
            else:
                _msgs_copy.append(_m)
        if not _found_sys:
            _msgs_copy = [{"role": "system", "content": _CONCISION_PREFIX}] + list(messages)
        messages = _msgs_copy

    if not messages:
        raise HTTPException(400, "'messages' must be a non-empty list")

    # ── Trace: log incoming messages ────────────────────────────────────────
    if _trace:
        for _mi, _m in enumerate(messages):
            _role    = _m.get("role", "?")
            _content = str(_m.get("content", ""))
            _preview = _content[:300].replace("\n", "↵") + ("…" if len(_content) > 300 else "")
            _tlog(f"CHAT [{_role}] msg[{_mi}]: {_preview}")

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
                if _trace:
                    _tps = n_comp / dur if dur > 0 else 0.0
                    _tlog(f"CHAT stream DONE  id={cid}  tokens={n_comp}  "
                          f"ttft={ttft_s:.3f}s  total={dur:.3f}s  tps={_tps:.1f}  "
                          f"finish={last_finish}")
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
            if _trace:
                _tps = n_comp / dur if dur > 0 else 0.0
                _tlog(f"CHAT  DONE  id={cid}  tokens={n_comp}  "
                      f"ttft={ttft_s:.3f}s  total={dur:.3f}s  tps={_tps:.1f}  "
                      f"finish={last_finish}")
                _resp_preview = full_text[:400].replace("\n", "↵") + (
                    "…" if len(full_text) > 400 else "")
                _tlog(f"CHAT  resp: {_resp_preview}")

        comp_tokens = _count_tokens(full_text)

        # ── Tool calling: detect function call in output ──────────────────────
        if tools:
            from squish.tool_calling import build_tool_calls_response, parse_tool_calls
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
async def completions(  # pragma: no cover
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/completions — legacy text completion endpoint.
    """
    _check_auth(creds)
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")

    body: dict[str, Any] = await request.json()
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
                _dur = time.perf_counter() - req_start
                _state.inflight -= 1
                _state.record_completion(n_comp, _dur, ttft_s)
                if _trace:
                    _tps = n_comp / _dur if _dur > 0 else 0.0
                    _tlog(f"CMPL stream DONE  id={cid}  tokens={n_comp}  "
                          f"ttft={ttft_s:.3f}s  total={_dur:.3f}s  tps={_tps:.1f}  "
                          f"finish={last_finish}")
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
            _dur = time.perf_counter() - req_start
            _state.inflight -= 1
            _state.record_completion(n_comp, _dur, ttft_s)
            if _trace:
                _tps = n_comp / _dur if _dur > 0 else 0.0
                _tlog(f"CMPL  DONE  id={cid}  tokens={n_comp}  "
                      f"ttft={ttft_s:.3f}s  total={_dur:.3f}s  tps={_tps:.1f}  "
                      f"finish={last_finish}")

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

    body: dict[str, Any] = await request.json()
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
        except (AttributeError, TypeError):  # pragma: no cover
            try:
                # Second-best: input token embeddings (less useful but available)
                tok_emb = model.model.embed_tokens(x)        # (1, seq, D)
                emb_np  = np.array(mx.mean(tok_emb, axis=1)[0])
            except AttributeError:  # pragma: no cover
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
    _battery_level: float | None = None
    if _power_monitor is not None:
        _battery_level = round(_power_monitor.get_battery_level(), 2)
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
        "power_mode":   _power_mode,
        "battery_level": _battery_level,
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
        "# HELP squish_radix_prefix_hits_total RadixTree token-prefix KV reuse hits",
        "# TYPE squish_radix_prefix_hits_total counter",
        f"squish_radix_prefix_hits_total {_prefix_cache.prefix_hits}",
        "# HELP squish_paged_kv_free_blocks Paged KV cache free block count",
        "# TYPE squish_paged_kv_free_blocks gauge",
        f"squish_paged_kv_free_blocks {_paged_kv_cache.stats()['free_blocks'] if _paged_kv_cache is not None else -1}",
        "# HELP squish_paged_kv_used_blocks Paged KV cache used block count",
        "# TYPE squish_paged_kv_used_blocks gauge",
        f"squish_paged_kv_used_blocks {_paged_kv_cache.stats()['used_blocks'] if _paged_kv_cache is not None else -1}",
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
        raise HTTPException(500, f"Tokenization failed: {e}") from e

    return JSONResponse({
        "token_ids":   ids,
        "token_count": len(ids),
        "model":       _state.model_name,
    })


@app.post("/v1/learn")
async def learn(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """
    POST /v1/learn — absorb training examples into the block-expert archive.

    Requires ``--block-expert <archive-dir>`` to be set at server start.

    Body:
        {
          "examples": [{"input": "...", "output": "..."}],
          "domain":   "legal",
          "steps":    50
        }

    Returns a JSON summary of the learning operation.
    """
    _check_auth(creds)
    if _block_expert_archive is None:
        raise HTTPException(
            501,
            "Block-expert archive not loaded — start the server with --block-expert <archive-dir>",
        )

    from squish.self_learning import LearnConfig, LearnExample, SelfLearner

    body = await request.json()
    raw_examples = body.get("examples", [])
    if not raw_examples:
        raise HTTPException(400, "'examples' must be a non-empty list")

    domain  = str(body.get("domain", "general"))[:64]
    steps   = int(body.get("steps", 50))
    lr      = float(body.get("lr", 1e-4))
    epsilon = float(body.get("epsilon", 1e-3))
    max_rank = int(body.get("max_rank", 8))

    examples = [
        LearnExample(
            input=str(ex.get("input", "")),
            output=str(ex.get("output", "")),
        )
        for ex in raw_examples
        if isinstance(ex, dict)
    ]
    if not examples:
        raise HTTPException(400, "No valid examples found in request body")

    # Build base weights dict from archive (use zero-ish proxies if unavailable)
    n_blocks = _block_expert_archive.num_blocks() or 1
    import numpy as _np
    hidden = 256  # lightweight proxy dimension — real models pass via the archive
    base_weights = {
        bi: _np.zeros((hidden, hidden), dtype=_np.float32)
        for bi in range(n_blocks)
    }

    cfg = LearnConfig(
        steps=max(1, min(steps, 500)),
        lr=lr,
        epsilon=epsilon,
        max_rank=max_rank,
        domain=domain,
    )
    learner = SelfLearner(base_weights, cfg)
    result = learner.learn_from_examples(examples, cfg)
    learner.apply_result_to_archive(result, _block_expert_archive)

    try:
        _block_expert_archive.save()
    except Exception as _save_err:
        _warn(f"[block-expert] archive save failed: {_save_err}")

    return JSONResponse({
        "status":        "ok",
        "domain":        result.domain,
        "steps_run":     result.steps_run,
        "examples_used": result.examples_used,
        "snr_db":        round(result.snr_db, 2),
        "elapsed_s":     round(result.elapsed_s, 3),
        "archive":       _block_expert_archive.summary(),
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def main():  # pragma: no cover
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
    ap.add_argument("--mlx-model-dir", default="",
                    metavar="DIR",
                    help="Load a native mlx_lm model directory directly (INT4/INT8 quantized).\n"
                         "Keeps weights quantized in Metal (~4-5 GB for 8B INT4) instead of\n"
                         "dequantizing to BF16 (~15 GB via --compressed-dir).\n"
                         "Create with: python3 -m mlx_lm.convert --hf-path <bf16-dir> \\\n"
                         "  --mlx-path <output-dir> -q --q-bits 4\n"
                         "When set, --model-dir and --compressed-dir are ignored.")
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
    ap.add_argument("--eagle-head-dir", default="",
                    help="Path to EAGLE-3 draft head directory (from `squish pull-head`). "
                         "Enables EAGLE-3 speculative decoding (~75-85%% acceptance rate). "
                         "Incompatible with --draft-model.")
    ap.add_argument("--no-prefix-cache", action="store_true", default=False,
                    help="Disable the prefix (exact-match) response cache")
    ap.add_argument("--prefix-cache-size", type=int, default=512,
                    help="LRU prefix cache capacity (default 512 entries)")
    ap.add_argument("--paged-attention", action="store_true", default=False,
                    help="Enable PagedAttention block table for KV prefix reuse. "
                         "Pre-allocates a fixed KV block pool from unified memory.")
    ap.add_argument("--paged-attention-fraction", type=float, default=0.25,
                    help="Fraction of total RAM to allocate for paged KV blocks "
                         "(default 0.25 = 25%%).  Ignored when --paged-attention "
                         "is not set.")
    # ── Phase 3A: Chunked prefill ─────────────────────────────────────────────
    ap.add_argument("--chunk-prefill", action="store_true", default=False,
                    help="Enable chunked prefill for long COMPRESS_PATH requests.\n"
                         "Splits the prompt into chunks and interleaves one greedy\n"
                         "decode token between chunks to minimise TTFT.\n"
                         "Only activates on the COMPRESS_PATH (--compress-prompt).")
    ap.add_argument("--chunk-prefill-threshold", type=int, default=512,
                    metavar="N",
                    help="Minimum prompt token count to trigger chunked prefill\n"
                         "(default 512).  Requests shorter than N use standard\n"
                         "single-shot prefill regardless of --chunk-prefill.")
    ap.add_argument("--chunk-prefill-size", type=int, default=512,
                    metavar="N",
                    help="Tokens per prefill chunk (default 512).")
    # ── Phase 3C: MInference sparse attention ─────────────────────────────────
    ap.add_argument("--minference", action="store_true", default=False,
                    help="Enable MInference-style sparse attention during prefill.\n"
                         "Reduces attention cost from O(n²) to O(n·k) for prompts\n"
                         "longer than --minference-threshold.\n"
                         "Automatically selects the best sparsity pattern.\n"
                         "Incompatible with --inference-backend ane-disagg.")
    ap.add_argument("--minference-threshold", type=int, default=1024,
                    metavar="N",
                    help="Minimum sequence length to activate sparse attention\n"
                         "(default 1024 tokens).")
    # ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────
    ap.add_argument("--thinking-budget", type=int, default=-1, metavar="N",
                    help="Qwen3 thinking token budget (-1=unlimited, 0=disable thinking mode).\n"
                         "0 appends /no_think to system messages (non-thinking mode).\n"
                         ">0 forces </think> after N thinking tokens via logit bias (+100).")
    # ── Phase A2: explicit KV cache size ─────────────────────────────────────
    ap.add_argument("--max-kv-size", type=int, default=None, metavar="N",
                    help="MLX rotating KV cache size in tokens.\n"
                         "MLX defaults to 4096, silently truncating contexts longer than 4K.\n"
                         "Set to 131072 for 128K context. Passed directly to mlx_lm.stream_generate.")
    # ── Phase A3: concise responses ───────────────────────────────────────────
    ap.add_argument("--concise-responses", action="store_true", default=False,
                    help="Prepend a concision directive to every system message and apply\n"
                         "+8.0 EOS logit bias after 20 tokens to reduce verbosity.")
    # ── Phase B: Structured output (XGrammar) ─────────────────────────────────
    ap.add_argument("--structured-output",
                    choices=["none", "json", "json-schema"],
                    default="none",
                    metavar="MODE",
                    help="Constrain model output to structured formats via XGrammar:\n"
                         "  none        — unconstrained (default)\n"
                         "  json        — constrain to any valid JSON object\n"
                         "  json-schema — constrain to the schema given by --structured-output-schema\n"
                         "Requires: pip install 'squish[grammar]'")
    ap.add_argument("--structured-output-schema", type=str, default=None,
                    metavar="PATH",
                    help="Path to a JSON file containing the JSON-schema used when\n"
                         "--structured-output json-schema is set.")
    # ── Phase C: Power & Energy Modes ─────────────────────────────────────────
    ap.add_argument("--power-mode",
                    choices=["performance", "balanced", "battery", "auto"],
                    default="performance",
                    metavar="MODE",
                    help="Inference resource profile:\n"
                         "  performance — maximum throughput (default)\n"
                         "  balanced    — moderate resource use\n"
                         "  battery     — minimal resource use\n"
                         "  auto        — poll pmset every 30 s and switch automatically")
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
    # Phase 1 SVD compression
    ap.add_argument("--kv-cache-svd-rank", type=int, default=0,
                    metavar="N",
                    help="SVD rank for KV compression: project head_dim → N before INT8.\n"
                         "0 = off (default).  Recommended: 64 for head_dim=128 models.\n"
                         "Requires --kv-cache-mode int8 or snap.")
    ap.add_argument("--kv-commvq-bits", type=int, default=0,
                    metavar="N",
                    choices=[0, 2, 4],
                    help="CommVQ vector-quantization for old KV tokens (arXiv:2506.18879).\n"
                         "0 = off (default); 2 = 2-bit (8× vs FP16); 4 = 4-bit (4× vs FP16).\n"
                         "Replaces INT8 for old tokens; recent window stays FP16.\n"
                         "Fits codebook online from the first 64 evicted tokens per layer.")
    # Phase 2 retrieval attention
    ap.add_argument("--retrieval-attention", action="store_true", default=False,
                    help="Enable retrieval attention: fetch only the top-k most relevant\n"
                         "disk-tier tokens via HNSW ANNS search instead of scanning all\n"
                         "disk tokens.  Requires --disk-prompt-cache.  Needs hnswlib.")
    ap.add_argument("--retrieval-top-k", type=int, default=32,
                    metavar="N",
                    help="ANNS top-k tokens to retrieve from disk tier (default 32)")
    ap.add_argument("--retrieval-hot-window", type=int, default=256,
                    metavar="N",
                    help="Number of most-recent RAM INT8 tokens always returned\n"
                         "(hot window guarantee, default 256)")
    ap.add_argument("--log-level",
                    choices=["critical", "error", "warning", "info", "debug", "trace"],
                    default="warning",
                    help="Uvicorn log verbosity (default: warning)")
    # ── Phase 2.1: Batch scheduler ────────────────────────────────────────────
    ap.add_argument("--batch-scheduler", action="store_true", default=False,
                    help="Enable continuous batching scheduler: collects concurrent\n"
                         "requests within --batch-window-ms and runs them in one\n"
                         "padded forward pass.  Improves throughput ~N× at moderate load.")
    ap.add_argument("--scheduler", choices=["nested-wait", "legacy"],
                    default="nested-wait",
                    help="Scheduler algorithm when --batch-scheduler is enabled:\n"
                         "  nested-wait — Nested WAIT continuous batcher: merges newly-"
                         "prefilled\n"
                         "                requests between decode steps, eliminating inter-"
                         "batch GPU idle\n"
                         "                time.  Lower TTFT under load.  (default)\n"
                         "  legacy      — Original static coalescing-window batcher.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Max concurrent requests per batch (default 8)")
    ap.add_argument("--batch-window-ms", type=float, default=20.0,
                    help="Collect window in ms before starting a batch (default 20)")
    ap.add_argument("--no-compile", action="store_true", default=False,
                    help="Disable mx.compile for the single-token decode step\n"
                         "(useful for debugging or models incompatible with tracing)")
    ap.add_argument("--disk-prompt-cache", default="",
                    metavar="DIR",
                    help="Enable persistent cross-request KV-state prompt cache stored\n"
                         "as compressed .npz files under DIR (on SSD/NVMe).  Repeated\n"
                         "identical prompts skip prefill entirely.  64-entry LRU default.")
    ap.add_argument("--disk-prompt-cache-size", type=int, default=64,
                    metavar="N",
                    help="Max entries in the disk prompt cache (default 64)")
    # Phase 3: persistent cross-session KV cache
    ap.add_argument("--session-cache-dir", default="",
                    metavar="DIR",
                    help="Enable persistent cross-session KV state cache under DIR.\\n"
                         "The session key is auto-derived from the last 8 message\\n"
                         "contents (SHA-256), so no client changes are needed.\\n"
                         "Surviving a server restart resumes generation from the\\n"
                         "cached KV state.")
    # Phase 4: prompt compression
    ap.add_argument("--compress-prompt", action="store_true", default=False,
                    help="Enable prompt compression before prefill.\\n"
                         "Uses TF-IDF sentence scoring by default; delegates to\\n"
                         "LLMLingua if installed (pip install squish[llmlingua]).")
    ap.add_argument("--compress-ratio", type=float, default=0.5,
                    metavar="F",
                    help="Target compression fraction: 0.5 = compress to half the\\n"
                         "token count (default 0.5).  Range: (0, 1).")
    ap.add_argument("--compress-min-tokens", type=int, default=512,
                    metavar="N",
                    help="Only compress prompts longer than N tokens (default 512).")
    ap.add_argument("--compress-preserve-tokens", type=int, default=0,
                    metavar="N",
                    help="Protect the first N words of each prompt from compression.\n"
                         "Set to the typical system-prompt length to keep the prefix\n"
                         "identical across requests for RadixAttention cache hits.")
    # ── Phase E1: Babbling suppression ─────────────────────────────────────────
    ap.add_argument("--babbling-suppression", action="store_true", default=False,
                    help="Stop generation early when the model strongly prefers EOS "
                         "(EOS probability > 30%%), a grammar FSM reaches a terminal "
                         "state, or a per-task token cap is exceeded.\n"
                         "Reduces average energy cost by 44-89%% on short-output tasks.")
    ap.add_argument("--no-babbling-suppression", dest="babbling_suppression",
                    action="store_false",
                    help="Disable babbling suppression (keep generating until max_tokens).")
    ap.add_argument("--babbling-eos-threshold", type=float, default=0.30,
                    metavar="P",
                    help="EOS probability threshold for babbling suppression (default 0.30).")
    ap.add_argument("--babbling-min-tokens", type=int, default=10,
                    metavar="N",
                    help="Never stop early before emitting N tokens (default 10).")
    # ── Phase E2: Polynomial GELU approximation ───────────────────────────────
    ap.add_argument("--fast-gelu", action="store_true", default=False,
                    help="Replace erf-GELU with x·sigmoid(1.702x) for GELU-based models.\n"
                         "No-op for SiLU/SwiGLU models (Qwen3, LLaMA). "
                         "Provides ~3-5%% speedup on GPU, larger on ANE.")
    ap.add_argument("--no-fast-gelu", dest="fast_gelu", action="store_false",
                    help="Disable polynomial GELU approximation.")
    # ── Phase E3: Semantic response cache ─────────────────────────────────────
    ap.add_argument("--semantic-cache", action="store_true", default=False,
                    help="Enable semantic response caching. Semantically similar prompts "
                         "(cosine distance < task threshold) return a cached response, "
                         "delivering 25-250× latency reduction for warm repeat patterns.")
    ap.add_argument("--no-semantic-cache", dest="semantic_cache", action="store_false",
                    help="Disable semantic response cache.")
    ap.add_argument("--semantic-cache-db", default="",
                    metavar="PATH",
                    help="Path to the sqlite-vec semantic cache database "
                         "(default: ~/.squish/response_cache.db).")
    # ── Phase 4: hardware inference backend ──────────────────────────────────
    ap.add_argument("--inference-backend",
                    choices=["mlx-eager", "mlx-compiled", "ane-disagg", "mlc"],
                    default="mlx-eager",
                    metavar="BACKEND",
                    help="Hardware dispatch strategy (default: mlx-eager):\n"
                         "  mlx-eager    — standard MLX Metal execution (safest)\n"
                         "  mlx-compiled — mx.compile fused decode (lower GPU overhead)\n"
                         "  ane-disagg   — Apple Neural Engine prefill + GPU decode\n"
                         "  mlc          — MLC-LLM engine (large-context requests)\n"
                         "mlx-compiled and ane-disagg are mutually exclusive.")
    # ── Item 3: LazyLLM token pruning ─────────────────────────────────────────
    ap.add_argument("--lazy-llm", action="store_true", default=False,
                    help="Enable LazyLLM dynamic token pruning during prefill.\n"
                         "Skips low-importance positions in later transformer layers,\n"
                         "reducing TTFT by ~20-35%% on long prompts.")
    ap.add_argument("--lazy-llm-keep-ratio", type=float, default=0.70,
                    metavar="F",
                    help="Fraction of tokens to keep per layer (default 0.70)")
    ap.add_argument("--lazy-llm-start-layer", type=int, default=2,
                    metavar="N",
                    help="First layer index where pruning is applied (default 2)")
    ap.add_argument("--lazy-llm-revive-window", type=int, default=4,
                    metavar="N",
                    help="Always keep the N most recent tokens active (default 4)")
    # ── Verbose inference tracing ─────────────────────────────────────────────
    ap.add_argument("--trace", action="store_true", default=False,
                    help="Log full per-request detail to stderr: prompt, dispatch path, "
                         "finish reason, TTFT, TPS, and cache hit/miss status.")
    ap.add_argument("--trace-tokens", action="store_true", default=False,
                    help="Also log every generated token text (implies --trace; "
                         "very verbose — useful for debugging output corruption).")
    ap.add_argument("--trace-file", default="",
                    metavar="FILE",
                    help="Append trace output to FILE in addition to stderr. "
                         "Useful when the server stdout/stderr is not visible "
                         "(e.g. when launched by _run_all.py).")

    # ── Wave optimization flags ───────────────────────────────────────────────
    ap.add_argument("--prompt-lookup", action="store_true", default=False,
                    help="Enable n-gram prompt lookup speculative decoding.")
    ap.add_argument("--prompt-lookup-n", type=int, default=3, metavar="N",
                    help="N-gram size for prompt lookup (default: 3).")
    ap.add_argument("--prompt-lookup-k", type=int, default=4, metavar="K",
                    help="Max draft tokens per lookup step (default: 4).")
    ap.add_argument("--seq-packing", action="store_true", default=False,
                    help="Enable sequence packing for higher batch GPU utilisation.")
    ap.add_argument("--seq-packing-budget", type=int, default=2048, metavar="N",
                    help="Token budget per packed batch (default: 2048).")
    ap.add_argument("--ada-serve", action="store_true", default=False,
                    help="Enable SLO-adaptive gamma scheduling for speculative decoding.")
    ap.add_argument("--ada-serve-slo", default="general",
                    choices=["git_commit", "devops_plan", "general", "code_review"],
                    help="Default SLO profile for AdaServe (default: general).")
    ap.add_argument("--conf-spec", action="store_true", default=False,
                    help="Enable confidence-gated speculative step verification.")
    ap.add_argument("--conf-spec-high-gate", type=float, default=0.90, metavar="F",
                    help="Confidence above which steps are auto-accepted (default: 0.90).")
    ap.add_argument("--conf-spec-low-gate", type=float, default=0.50, metavar="F",
                    help="Confidence below which full target verify is used (default: 0.50).")
    ap.add_argument("--kv-share", action="store_true", default=False,
                    help="Enable cross-layer KV sharing (KVSharer).")
    ap.add_argument("--kv-share-every", type=int, default=2, metavar="N",
                    help="Share KV every N layers (default: 2).")
    ap.add_argument("--kv-slab", action="store_true", default=False,
                    help="Enable slab-based KV memory allocator for reduced fragmentation.")
    ap.add_argument("--kv-slab-pages", type=int, default=256, metavar="N",
                    help="Number of slab pages (default: 256).")
    ap.add_argument("--paris-kv", action="store_true", default=False,
                    help="Enable PARIS KV codebook compression.")
    ap.add_argument("--paris-kv-centroids", type=int, default=64, metavar="N",
                    help="PARIS codebook centroid count (default: 64).")
    ap.add_argument("--streaming-sink", action="store_true", default=False,
                    help="Enable StreamingLLM-style sink KV cache.")
    ap.add_argument("--streaming-sink-size", type=int, default=2048, metavar="N",
                    help="Sink KV cache token budget (default: 2048).")
    ap.add_argument("--diff-kv", action="store_true", default=False,
                    help="Enable DiffKV 3-axis differentiated KV precision.")
    ap.add_argument("--small-kv", action="store_true", default=False,
                    help="Enable SmallKV saliency-shift compensation.")
    ap.add_argument("--sage-attention", action="store_true", default=False,
                    help="Enable SageAttention INT8 quantized QK^T computation.")
    ap.add_argument("--sage-attention2", action="store_true", default=False,
                    help="Enable SageAttention2 INT4/FP8 quantized attention.")
    ap.add_argument("--sparge-attention", action="store_true", default=False,
                    help="Enable SpargeAttn sparse+quantized attention.")
    ap.add_argument("--squeeze-attention", action="store_true", default=False,
                    help="Enable SqueezeAttention adaptive KV budget allocation.")
    ap.add_argument("--yoco-kv", action="store_true", default=False,
                    help="Enable YOCO cross-layer KV reuse (you-only-cache-once).")
    ap.add_argument("--cla", action="store_true", default=False,
                    help="Enable Cross-Layer Attention KV sharing.")
    ap.add_argument("--kvtuner", action="store_true", default=False,
                    help="Enable KVTuner adaptive per-layer KV budget.")
    ap.add_argument("--robust-scheduler", action="store_true", default=False,
                    help="Use the robust A-max/A-balanced batch scheduler.")
    ap.add_argument("--gemfilter", action="store_true", default=False,
                    help="Enable GemFilter attention head filtering.")
    ap.add_argument("--svdq", action="store_true", default=False,
                    help="Enable SVD-based KV quantization (SVDQ).")
    ap.add_argument("--sparse-spec", action="store_true", default=False,
                    help="Enable sparse speculative decoding.")
    ap.add_argument("--sparse-verify", action="store_true", default=False,
                    help="Enable sparse draft verification.")
    ap.add_argument("--trail", action="store_true", default=False,
                    help="Enable TRAIL token-importance-aware layer skipping.")
    ap.add_argument("--specontext", action="store_true", default=False,
                    help="Enable SpecContext speculative context extension.")
    ap.add_argument("--forelen", action="store_true", default=False,
                    help="Enable ForeLen forward-looking token length prediction.")
    ap.add_argument("--ipw", action="store_true", default=False,
                    help="Enable IPW importance-weighted prefill compression.")
    ap.add_argument("--layer-skip", action="store_true", default=False,
                    help="Enable LayerSkip early-exit adaptive layer skipping.")
    ap.add_argument("--lookahead", action="store_true", default=False,
                    help="Enable LookaheadReasoning parallel step verification.")
    ap.add_argument("--lookahead-k", type=int, default=4, metavar="K",
                    help="Lookahead window size (default: 4).")
    ap.add_argument("--spec-reason", action="store_true", default=False,
                    help="Enable SpecReason step-level speculative reasoning.")
    ap.add_argument("--long-spec", action="store_true", default=False,
                    help="Enable LongSpec extended speculative decoding.")
    ap.add_argument("--fr-spec", action="store_true", default=False,
                    help="Enable FR-Spec frequency-based token speculative decoding.")
    ap.add_argument("--lora-adapter", default="", metavar="PATH",
                    help="Path to LoRA adapter directory to load via LoRAManager.")
    ap.add_argument("--diffusion-draft", default="", metavar="PATH",
                    help="Path to a diffusion-based draft model directory for speculative decoding.")
    ap.add_argument("--block-expert", default="", metavar="PATH",
                    help="Path to a block-expert archive directory. "
                         "Creates a new archive at PATH if the directory does not yet exist. "
                         "Enables the POST /v1/learn endpoint for on-device self-learning.")
    ap.add_argument("--block-expert-clusters", type=int, default=4, metavar="K",
                    help="Number of expert clusters per Transformer block when creating a new archive (default: 4).")
    # ── Wave 13b: Ultra-Long Context + Adaptive Speculative Decoding ─────────
    ap.add_argument("--duo-attention", action="store_true", default=False,
                    help="Enable DuoAttention retrieval/streaming head separation for long-context inference.")
    ap.add_argument("--duo-attention-layers", type=int, default=32, metavar="N",
                    help="Number of Transformer layers (default: 32).")
    ap.add_argument("--duo-attention-heads", type=int, default=32, metavar="N",
                    help="Number of attention heads (default: 32).")
    ap.add_argument("--duo-attention-head-dim", type=int, default=128, metavar="D",
                    help="Per-head dimension (default: 128).")
    ap.add_argument("--duo-attention-window", type=int, default=512, metavar="W",
                    help="Streaming-head local window size in tokens (default: 512).")
    ap.add_argument("--shadow-kv", action="store_true", default=False,
                    help="Enable ShadowKV low-rank pre-RoPE key cache + CPU value shadow for 128K+ contexts.")
    ap.add_argument("--shadow-kv-rank", type=int, default=128, metavar="R",
                    help="SVD rank for low-rank key projection (default: 128).")
    ap.add_argument("--shadow-kv-landmarks", type=int, default=64, metavar="M",
                    help="Number of landmark tokens for sparse key retrieval (default: 64).")
    ap.add_argument("--pq-cache", action="store_true", default=False,
                    help="Enable PQCache product-quantization KV ANN retrieval for retrieval heads.")
    ap.add_argument("--pq-cache-subvectors", type=int, default=8, metavar="M",
                    help="Number of PQ sub-vectors (default: 8).")
    ap.add_argument("--pq-cache-codes", type=int, default=256, metavar="K",
                    help="Number of PQ codes per sub-vector (default: 256).")
    ap.add_argument("--spe-cache", action="store_true", default=False,
                    help="Enable SpeCache speculative KV-cache prefetching for multi-turn dialogue.")
    ap.add_argument("--spe-cache-block-size", type=int, default=16, metavar="B",
                    help="KV block granularity for prefetching (default: 16).")
    ap.add_argument("--spe-cache-budget", type=int, default=8, metavar="N",
                    help="Number of blocks to prefetch speculatively (default: 8).")
    ap.add_argument("--duo-decoding", action="store_true", default=False,
                    help="Enable DuoDecoding hardware-aware dynamic multi-sequence speculative decoding.")
    ap.add_argument("--duo-decoding-gamma", type=int, default=4, metavar="G",
                    help="Base number of draft tokens per step (default: 4).")
    ap.add_argument("--duo-decoding-kmax", type=int, default=8, metavar="K",
                    help="Maximum draft sequences in parallel (default: 8).")
    ap.add_argument("--knapspec", action="store_true", default=False,
                    help="Enable KnapSpec training-free self-speculative decoding via knapsack layer selection.")
    ap.add_argument("--knapspec-layers", type=int, default=32, metavar="N",
                    help="Total number of model layers (default: 32).")
    ap.add_argument("--knapspec-budget", type=float, default=0.7, metavar="F",
                    help="Fraction of total layer latency to use as draft budget (default: 0.7).")
    ap.add_argument("--token-merging", action="store_true", default=False,
                    help="Enable Token Merging (ToMe) to reduce sequence length during prefill.")
    ap.add_argument("--token-merging-r", type=int, default=8, metavar="R",
                    help="Tokens to merge per layer (default: 8).")
    ap.add_argument("--token-merging-start", type=int, default=4, metavar="L",
                    help="First layer to apply ToMe (default: 4).")
    ap.add_argument("--token-merging-end", type=int, default=-1, metavar="L",
                    help="Last layer for ToMe, -1 = all remaining (default: -1).")
    ap.add_argument("--token-swift", action="store_true", default=False,
                    help="Enable TokenSwift multi-token draft heads + partial KV reuse for ultra-long generation.")
    ap.add_argument("--token-swift-heads", type=int, default=4, metavar="N",
                    help="Number of TokenSwift draft heads (default: 4).")
    ap.add_argument("--token-swift-window", type=int, default=512, metavar="W",
                    help="KV reuse window size in tokens (default: 512).")
    ap.add_argument("--token-swift-vocab", type=int, default=151936, metavar="V",
                    help="Vocabulary size (default: 151936 for Qwen).")
    ap.add_argument("--c2t", action="store_true", default=False,
                    help="Enable C2T classifier-based adaptive candidate tree for speculative decoding.")
    ap.add_argument("--c2t-depth", type=int, default=4, metavar="D",
                    help="Tree depth for speculative candidates (default: 4).")
    ap.add_argument("--c2t-wide", type=int, default=3, metavar="B",
                    help="Wide-branch fan-out at uncertain positions (default: 3).")
    ap.add_argument("--c2t-narrow", type=int, default=1, metavar="B",
                    help="Narrow-branch fan-out at confident positions (default: 1).")
    ap.add_argument("--clasp", action="store_true", default=False,
                    help="Enable CLaSp in-context layer-skip with adaptive DP feedback for spec-decode.")
    ap.add_argument("--clasp-layers", type=int, default=32, metavar="N",
                    help="Total number of model layers (default: 32).")
    ap.add_argument("--clasp-max-skip", type=int, default=8, metavar="K",
                    help="Maximum consecutive layers to skip in the draft pass (default: 8).")
    ap.add_argument("--clasp-gamma", type=int, default=4, metavar="G",
                    help="Speculative draft tokens per verification step (default: 4).")
    # ── Wave 14: Quantization + Vocabulary-Adaptive Spec-Decode + Expert Mixing ─
    ap.add_argument("--soup-experts", action="store_true", default=False,
                    help="Enable SoupOfExperts sparse LoRA-expert adapter blending.")
    ap.add_argument("--soup-experts-tolerance", type=float, default=0.01, metavar="T",
                    help="Coefficient tolerance for expert blending (default: 0.01).")
    ap.add_argument("--vision-cache", action="store_true", default=False,
                    help="Enable VisionPrefixCache SHA-256 dedup for vision encoder outputs.")
    ap.add_argument("--vision-cache-max-entries", type=int, default=64, metavar="N",
                    help="Maximum cached vision prefix entries (default: 64).")
    ap.add_argument("--vector-index", action="store_true", default=False,
                    help="Enable MRLIndex + HNSWIndex for semantic KV retrieval.")
    ap.add_argument("--vector-index-dim", type=int, default=512, metavar="D",
                    help="Full representation dimension for MRL index (default: 512).")
    ap.add_argument("--vector-index-coarse-dim", type=int, default=64, metavar="D",
                    help="Coarse representation dimension for MRL index (default: 64).")
    ap.add_argument("--sub-spec", action="store_true", default=False,
                    help="Enable SubSpecDecoder speculative decoding via quantized substitute layers.")
    ap.add_argument("--sub-spec-gamma", type=int, default=4, metavar="G",
                    help="SubSpec draft tokens per step (default: 4).")
    ap.add_argument("--sub-spec-gpu-layers", type=int, default=16, metavar="N",
                    help="SubSpec number of GPU layers for the substitute model (default: 16).")
    ap.add_argument("--del-decoder", action="store_true", default=False,
                    help="Enable DELDecoder dynamic early-layer exit speculative decoding.")
    ap.add_argument("--del-decoder-gamma", type=int, default=4, metavar="G",
                    help="DEL draft tokens per verification step (default: 4).")
    ap.add_argument("--del-decoder-min-exit", type=int, default=4, metavar="L",
                    help="Minimum exit layer for dynamic early-exit (default: 4).")
    ap.add_argument("--del-decoder-max-exit", type=int, default=8, metavar="L",
                    help="Maximum exit layer for dynamic early-exit (default: 8).")
    ap.add_argument("--dfloat11", action="store_true", default=False,
                    help="Enable DFloat11 block-float compression for model weights.")
    ap.add_argument("--dfloat11-block-size", type=int, default=256, metavar="B",
                    help="DFloat11 block size for entropy coding (default: 256).")
    ap.add_argument("--rans-codec", action="store_true", default=False,
                    help="Enable rANS entropy codec for KV/weight compression.")
    ap.add_argument("--rans-codec-mbits", type=int, default=14, metavar="M",
                    help="rANS codec precision in bits (default: 14).")
    ap.add_argument("--qspec", action="store_true", default=False,
                    help="Enable QSpecDecoder quantisation-aware speculative decoding.")
    ap.add_argument("--qspec-gamma", type=int, default=4, metavar="G",
                    help="QSpec draft tokens per step (default: 4).")
    ap.add_argument("--qspec-draft-bits", type=int, default=4, metavar="B",
                    help="Quantisation bits for QSpec draft activations (default: 4).")
    ap.add_argument("--qspec-verify-bits", type=int, default=8, metavar="B",
                    help="Quantisation bits for QSpec verify activations (default: 8).")
    ap.add_argument("--quant-spec", action="store_true", default=False,
                    help="Enable QuantSpecDecoder draft-quantised speculative decoding.")
    ap.add_argument("--quant-spec-gamma", type=int, default=4, metavar="G",
                    help="QuantSpec draft tokens per step (default: 4).")
    ap.add_argument("--quant-spec-bits", type=int, default=4, metavar="B",
                    help="QuantSpec draft quantisation bits (default: 4).")
    ap.add_argument("--copy-spec", action="store_true", default=False,
                    help="Enable CopySpecDrafter copy-based speculative decoding from history.")
    ap.add_argument("--copy-spec-max-draft", type=int, default=8, metavar="K",
                    help="CopySpec maximum draft length per step (default: 8).")
    ap.add_argument("--copy-spec-history-len", type=int, default=2048, metavar="N",
                    help="CopySpec token history window size (default: 2048).")
    ap.add_argument("--squeeze-llm", action="store_true", default=False,
                    help="Enable SqueezeLLM sparse + dense mixed-precision weight quantisation.")
    ap.add_argument("--squeeze-llm-bits", type=int, default=4, metavar="B",
                    help="SqueezeLLM quantisation bits (default: 4).")
    ap.add_argument("--squeeze-llm-sparsity", type=float, default=0.45, metavar="S",
                    help="SqueezeLLM sparsity ratio for sensitive weight extraction (default: 0.45).")
    ap.add_argument("--hetero-vocab-sd", action="store_true", default=False,
                    help="Enable HeteroVocabDecoder spec-decode with mismatched draft/target vocabularies.")
    ap.add_argument("--hetero-vocab-gamma", type=int, default=4, metavar="G",
                    help="HeteroVocab draft tokens per step (default: 4).")
    ap.add_argument("--hetero-vocab-draft-size", type=int, default=32000, metavar="V",
                    help="HeteroVocab draft model vocabulary size (default: 32000).")
    ap.add_argument("--head-infer", action="store_true", default=False,
                    help="Enable HeadAwareKVStore head-level inference KV separation.")
    ap.add_argument("--head-infer-layers", type=int, default=32, metavar="N",
                    help="Number of transformer layers for head-infer (default: 32).")
    ap.add_argument("--head-infer-heads", type=int, default=32, metavar="H",
                    help="Number of attention heads for head-infer (default: 32).")
    ap.add_argument("--nf4-quant", action="store_true", default=False,
                    help="Enable NF4 (Normal Float 4-bit) quantisation for weights.")
    ap.add_argument("--spin-quant", action="store_true", default=False,
                    help="Enable SpinQuant Hadamard rotation for quantisation-friendly weight layout.")
    ap.add_argument("--life-model", action="store_true", default=False,
                    help="Enable model lifecycle predictor for cache eviction guidance.")
    ap.add_argument(
        "--all-optimizations", action="store_true", default=False,
        help=(
            "Enable ALL built-in optimization modules at once. "
            "Activates every attention kernel, KV cache strategy, speculative "
            "decoding variant, and adaptive-layer technique. Equivalent to "
            "passing every --sage-attention, --sparge-attention, --yoco-kv, "
            "--squeeze-attention, --kvtuner, --robust-scheduler, --gemfilter, "
            "--svdq, --sparse-spec, --sparse-verify, --trail, --specontext, "
            "--forelen, --ipw, --layer-skip, --long-spec, --fr-spec, --cla, "
            "--prompt-lookup, --seq-packing, --conf-spec, --kv-share, --kv-slab, "
            "--paris-kv, --streaming-sink, --diff-kv, --small-kv, --lookahead, "
            "--spec-reason flags simultaneously. "
            "Useful for local testing. Modules that fail to init are skipped."
        ),
    )

    args = ap.parse_args()

    # ── Expand --all-optimizations into individual flags ─────────────────────
    if getattr(args, "all_optimizations", False):
        _bool_wave_flags = [
            "sage_attention", "sage_attention2", "sparge_attention",
            "squeeze_attention", "yoco_kv", "cla", "kvtuner",
            "robust_scheduler", "gemfilter", "svdq",
            "sparse_spec", "sparse_verify", "trail", "specontext",
            "forelen", "ipw", "layer_skip", "long_spec", "fr_spec",
            "prompt_lookup", "seq_packing", "ada_serve", "conf_spec",
            "kv_share", "kv_slab", "paris_kv", "streaming_sink",
            "diff_kv", "small_kv", "lookahead", "spec_reason",
            # Wave 13b
            "duo_attention", "shadow_kv", "pq_cache", "spe_cache",
            "duo_decoding", "knapspec", "token_merging",
            "token_swift", "c2t", "clasp",
            # Wave 14
            "soup_experts", "vision_cache", "vector_index", "sub_spec",
            "del_decoder", "dfloat11", "rans_codec", "qspec", "quant_spec",
            "copy_spec", "squeeze_llm", "hetero_vocab_sd", "head_infer",
            "nf4_quant", "spin_quant", "life_model",
        ]
        for _f in _bool_wave_flags:
            if not getattr(args, _f, False):
                setattr(args, _f, True)

    global _API_KEY
    # Prefer explicit CLI flag; fall back to SQUISH_API_KEY env var.
    # Reading from env var prevents the secret appearing in `ps aux`.
    _API_KEY = args.api_key or os.environ.get("SQUISH_API_KEY")

    # ── Tracing globals ───────────────────────────────────────────────────────
    global _trace, _trace_tokens, _trace_file
    _trace        = args.trace or args.trace_tokens
    _trace_tokens = args.trace_tokens
    if args.trace_file:
        try:
            _trace_file = open(args.trace_file, "a", buffering=1)  # noqa: WPS515
        except OSError as _tf_err:
            _warn(f"[trace] Could not open trace file {args.trace_file!r}: {_tf_err}")

    if args.no_prefix_cache:
        _prefix_cache._maxsize = 0
    elif args.prefix_cache_size != 512:
        _prefix_cache._maxsize = args.prefix_cache_size

    # ── Phase 2A/2B: PagedKVCache + RadixTree prefix trie ────────────────────
    global _paged_kv_cache
    if getattr(args, "paged_attention", False) and _state.model is not None:
        try:
            from squish.paged_attention import PagedKVCache as _PagedKVCache
            _paged_kv_cache = _PagedKVCache.from_model(
                _state.model,
                metal_fraction=getattr(args, "paged_attention_fraction", 0.25),
            )
            s = _paged_kv_cache.stats()
            _ok("Paged KV cache ready")
            _info("paged-kv-blocks",
                  f"{s['total_blocks']} blocks  "
                  f"({s['memory_mb']} MB  page={s['page_size']}tok  "
                  f"{s['n_layers']}L×{s['n_kv_heads']}H×{s['head_dim']}d)")
        except Exception as _paged_err:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[paged-attention] could not initialise (%s) — disabled", _paged_err
            )

    _print_banner()

    if getattr(args, "mlx_model_dir", ""):
        _info("model", f"{args.mlx_model_dir}  {_C.DIM}(mlx_lm INT4){_C.R}")
    else:
        _info("model-dir", args.model_dir)
        _info("compressed", args.compressed_dir)
    if args.draft_model:
        _info("draft-model", args.draft_model)
    if getattr(args, "eagle_head_dir", ""):
        _info("eagle-head", args.eagle_head_dir)
    _info("prefix-cache", "disabled" if args.no_prefix_cache else str(args.prefix_cache_size))
    if args.kv_cache_mode != "fp16":
        _info("kv-cache", f"{args.kv_cache_mode}  window={args.kv_cache_window}  budget={args.kv_cache_budget}")
    _info("listen", f"http://{args.host}:{args.port}")
    if _trace:
        _info("trace", f"ON  tokens={'yes' if _trace_tokens else 'no'}"
              f"{'  file=' + args.trace_file if args.trace_file else ''}")
    print()

    if getattr(args, "mlx_model_dir", ""):
        load_mlx_model(args.mlx_model_dir, verbose=args.verbose)
    else:
        load_model(args.model_dir, args.compressed_dir, verbose=args.verbose)
    _state._no_compile = args.no_compile  # propagate --no-compile flag

    # ── Disk prompt-cache init (Item 2) ──────────────────────────────────────
    global _disk_prompt_cache
    if getattr(args, "disk_prompt_cache", ""):
        try:
            from squish.kv_cache import DiskKVCache as _DiskKVCache
        except ImportError:
            from kv_cache import DiskKVCache as _DiskKVCache  # direct run
        _disk_prompt_cache = _DiskKVCache(
            cache_dir   = args.disk_prompt_cache,
            max_entries = args.disk_prompt_cache_size,
        )
        if args.verbose:
            _info("disk-cache", f"{args.disk_prompt_cache}  {_C.DIM}(max {args.disk_prompt_cache_size} entries){_C.R}")

    # ── LazyLLM token-pruning init (Item 3) ──────────────────────────────────
    global _lazy_llm_state
    if getattr(args, "lazy_llm", False) and _state.model is not None:
        try:
            try:
                from squish.lazy_llm import LazyLLMConfig
                from squish.lazy_llm import patch_model_lazy_llm as _patch_llm
            except ImportError:
                from lazy_llm import LazyLLMConfig
                from lazy_llm import patch_model_lazy_llm as _patch_llm
            _lazy_llm_cfg = LazyLLMConfig(
                keep_ratio    = args.lazy_llm_keep_ratio,
                start_layer   = args.lazy_llm_start_layer,
                revive_window = args.lazy_llm_revive_window,
                verbose       = _trace,   # tie to --trace flag
            )
            _lazy_llm_state = _patch_llm(_state.model, _lazy_llm_cfg)
            if args.verbose:
                _info("lazy-llm", f"keep={args.lazy_llm_keep_ratio}  "
                      f"start_layer={args.lazy_llm_start_layer}  "
                      f"revive={args.lazy_llm_revive_window}")
        except Exception as _llm_err:
            _warn(f"[lazy_llm] Skipped: {_llm_err}")

    if _state.model is not None:
        try:
            from squish.split_loader import SplitLayerLoader
            _split_info = SplitLayerLoader.auto_split(_state.model, verbose=True)
            if _split_info:
                _info("cpu/gpu split", f"{_split_info.cpu_count} layers offloaded  "
                      f"GPU={_split_info.gpu_gb:.2f}GB  CPU={_split_info.cpu_gb:.2f}GB")
        except Exception as e:
            if args.verbose:
                _warn(f"[split_loader] Skipped: {e}")

    # ── Phase 2.3: Flash Attention status check ──────────────────────────────
    if _state.model is not None:
        try:
            from squish.flash_attention import patch_model_attention
            patch_model_attention(_state.model, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                _warn(f"[flash_attention] Skipped: {e}")

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
                svd_rank=getattr(args, "kv_cache_svd_rank", 0),
                comm_vq_bits=getattr(args, "kv_commvq_bits", 0),
                verbose=True,
            )
            _info("kv-cache", f"ready ({args.kv_cache_mode})")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[KV cache] could not attach (%s) — running without KV quantisation", e
            )

    # ── Phase 3: persistent cross-session KV cache ────────────────────────────
    global _session_kv_cache
    _session_cache_dir = getattr(args, "session_cache_dir", "")
    if _session_cache_dir:
        try:
            from squish.kv_cache import SessionKVCache as _SessionKVCache
            _session_kv_cache = _SessionKVCache(cache_dir=_session_cache_dir)
            _info("session-cache", f"{_session_cache_dir}")
        except Exception as _e:
            _warn(f"[session-cache] Could not enable: {_e}")

    # ── Phase 4: prompt compression settings ─────────────────────────────────
    global _compress_enabled, _compress_ratio, _compress_min_tokens, _compress_preserve_tokens
    _compress_enabled        = getattr(args, "compress_prompt", False)
    _compress_ratio          = getattr(args, "compress_ratio", 0.5)
    _compress_min_tokens     = getattr(args, "compress_min_tokens", 512)
    _compress_preserve_tokens = getattr(args, "compress_preserve_tokens", 0)
    if _compress_enabled:
        _info("compress", f"ratio={_compress_ratio}  min_tokens={_compress_min_tokens}"
              + (f"  preserve_tokens={_compress_preserve_tokens}" if _compress_preserve_tokens else ""))

    # ── Phase E1: Babbling suppression settings ───────────────────────────────
    global _babbling_suppression, _babbling_eos_threshold, _babbling_min_tokens
    _babbling_suppression    = getattr(args, "babbling_suppression", False)
    _babbling_eos_threshold  = getattr(args, "babbling_eos_threshold", 0.30)
    _babbling_min_tokens     = getattr(args, "babbling_min_tokens", 10)
    if _babbling_suppression:
        _info("babbling-suppression",
              f"enabled  eos_threshold={_babbling_eos_threshold}  min_tokens={_babbling_min_tokens}")

    # ── Phase E2: Polynomial GELU ─────────────────────────────────────────────
    global _fast_gelu_enabled
    _fast_gelu_enabled = getattr(args, "fast_gelu", False)
    if _fast_gelu_enabled and _state.model is not None:
        _model_dir_for_gelu = getattr(args, "model_dir", "") or getattr(args, "mlx_model_dir", "")
        if _model_dir_for_gelu:
            _apply_fast_gelu(_model_dir_for_gelu)

    # ── Phase E3: Semantic response cache ─────────────────────────────────────
    global _semantic_cache
    if getattr(args, "semantic_cache", False):
        try:
            from squish.semantic_cache import SquishSemanticCache  # noqa: PLC0415
            _sc_db = getattr(args, "semantic_cache_db", "") or \
                     str(Path.home() / ".squish" / "response_cache.db")
            _semantic_cache = SquishSemanticCache(db_path=_sc_db)
            _info("semantic-cache", f"enabled  db={_sc_db}")
        except Exception as _sc_err:
            _warn(f"[semantic-cache] Could not enable: {_sc_err}\n"
                  "Install sqlite-vec: pip install 'squish[cache]'")

    # ── Phase 3A: chunked prefill settings ───────────────────────────────────
    global _chunk_prefill_enabled, _chunk_prefill_threshold, _chunk_prefill_size
    _chunk_prefill_enabled   = getattr(args, "chunk_prefill", False)
    _chunk_prefill_threshold = getattr(args, "chunk_prefill_threshold", 512)
    _chunk_prefill_size      = getattr(args, "chunk_prefill_size", 512)
    if _chunk_prefill_enabled:
        _info("chunk-prefill",
              f"threshold={_chunk_prefill_threshold}  chunk={_chunk_prefill_size}")

    # ── Phase 3C: MInference settings ────────────────────────────────────────
    global _minference_enabled, _minference_threshold, _inference_backend
    _minference_enabled   = getattr(args, "minference", False)
    _minference_threshold = getattr(args, "minference_threshold", 1024)
    if _minference_enabled:
        if _inference_backend == "ane-disagg":
            _warn("[minference] disabled — incompatible with --inference-backend ane-disagg")
            _minference_enabled = False
        else:
            _info("minference", f"sparse-attention  threshold={_minference_threshold}")

    # ── Phase A1: Qwen3 thinking budget ──────────────────────────────────────
    global _thinking_budget, _think_close_token_id
    _thinking_budget = getattr(args, "thinking_budget", -1)
    if _thinking_budget >= 0 and _state.tokenizer is not None:
        try:
            _think_close_token_id = _state.tokenizer.convert_tokens_to_ids("</think>")
        except Exception:
            _think_close_token_id = None
    if _thinking_budget == 0:
        _info("thinking-budget", "disabled (no_think mode)")
    elif _thinking_budget > 0:
        _info("thinking-budget", f"{_thinking_budget} tokens  close_id={_think_close_token_id}")

    # ── Phase A2: max KV size ─────────────────────────────────────────────────
    global _max_kv_size
    _max_kv_size = getattr(args, "max_kv_size", None)
    if _max_kv_size is not None:
        _info("max-kv-size", f"{_max_kv_size} tokens")

    # ── Phase A3: concise responses ───────────────────────────────────────────
    global _concise_responses
    _concise_responses = getattr(args, "concise_responses", False)
    if _concise_responses:
        _info("concise-responses", "enabled")

    # ── Phase B: Structured output (XGrammar) ─────────────────────────────────
    global _grammar_engine, _structured_output_mode, _structured_output_schema
    _structured_output_mode = getattr(args, "structured_output", "none")
    if _structured_output_mode != "none" and _state.tokenizer is not None:
        from squish.grammar_engine import GrammarEngine  # noqa: PLC0415
        if GrammarEngine.is_available():
            _grammar_engine = GrammarEngine(_state.tokenizer)
            if _structured_output_mode == "json-schema":
                _schema_path = getattr(args, "structured_output_schema", None)
                if _schema_path:
                    import json as _json  # noqa: PLC0415
                    with open(_schema_path) as _sf:
                        _structured_output_schema = _json.load(_sf)
            _info("structured-output", f"mode={_structured_output_mode}")
        else:
            _warn("[structured-output] xgrammar not installed; "
                  "falling back to unconstrained generation. "
                  "Install: pip install 'squish[grammar]'")

    # ── Phase C: Power & Energy Modes ─────────────────────────────────────────
    global _power_monitor, _power_mode
    _power_mode = getattr(args, "power_mode", "performance")
    if _power_mode == "auto":
        from squish.power_monitor import PowerMonitor, apply_mode  # noqa: PLC0415
        _power_monitor = PowerMonitor()
        _initial_mode = _power_monitor.get_recommended_mode()
        apply_mode(_initial_mode, globals())
        _power_mode = _initial_mode
        _info("power-mode", f"auto  initial={_initial_mode}")
        # Background timer: re-evaluate and apply every 30 s
        import threading as _threading  # noqa: PLC0415
        def _power_auto_tick() -> None:
            global _power_mode
            if _power_monitor is None:
                return
            _new_mode = _power_monitor.get_recommended_mode()
            if _new_mode != _power_mode:
                from squish.power_monitor import apply_mode as _am  # noqa: PLC0415
                _am(_new_mode, globals())
                _power_mode = _new_mode
                _info("power-mode", f"switched → {_new_mode}")
            _t = _threading.Timer(30.0, _power_auto_tick)
            _t.daemon = True
            _t.start()
        _pt = _threading.Timer(30.0, _power_auto_tick)
        _pt.daemon = True
        _pt.start()
    elif _power_mode != "performance":
        from squish.power_monitor import apply_mode  # noqa: PLC0415
        apply_mode(_power_mode, globals())
        _info("power-mode", _power_mode)

    # ── Phase 0C: hardware inference backend ─────────────────────────────────
    _inference_backend = getattr(args, "inference_backend", "mlx-eager")
    if _inference_backend != "mlx-eager":
        _info("inference-backend", _inference_backend)

    # ── Phase 2.1: start batch scheduler if requested ────────────────────────
    global _scheduler
    if args.batch_scheduler and _state.model is not None:
        try:
            from squish.scheduler import BatchScheduler, NestedWaitScheduler
            from squish.scheduler import QueueFullError as _QFE
            global _QueueFullError
            _QueueFullError = _QFE
            _sched_cls = (BatchScheduler
                          if getattr(args, "scheduler", "nested-wait") == "legacy"
                          else NestedWaitScheduler)
            _scheduler = _sched_cls(
                _state.model, _state.tokenizer,
                max_batch_size  = args.batch_size,
                batch_window_ms = args.batch_window_ms,
            )
            _scheduler.start()
            _info("batch-scheduler",
                  f"enabled  algo={getattr(args, 'scheduler', 'nested-wait')}  "
                  f"max_batch={args.batch_size}  window={args.batch_window_ms:.0f}ms")
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[Scheduler] could not start (%s) — falling back to sequential mode", e
            )
            _scheduler = None

    if args.draft_model:
        print()
        load_draft_model(args.draft_model, args.draft_compressed, verbose=args.verbose)

    if getattr(args, "eagle_head_dir", ""):
        print()
        load_eagle_head(args.eagle_head_dir, verbose=args.verbose)

    # ── Wave optimization module initialisation ───────────────────────────────
    global _prompt_lookup_decoder, _seq_packer, _ada_serve_scheduler
    global _conf_spec_verifier, _kvsharer_map, _kv_slab_allocator
    global _paris_kv_codebook, _streaming_sink_cache
    global _diffkv_policy_mgr, _smallkv_cache, _lookahead_engine, _spec_reason_orch
    global _sage_attn_kernel, _sage_attn2_kernel, _sparge_engine, _squeeze_cache
    global _yoco_config, _cla_config, _kvtuner_config, _robust_sched
    global _gemfilter_config, _svdq_config, _sparse_spec_config, _sparse_verify_config
    global _trail_config, _specontext_config, _forelen_config, _ipw_config
    global _layer_skip_config, _long_spec_config, _fr_spec_config, _diffusion_draft_model
    global _pm_kvq_scheduler, _mix_kvq_quantizer, _cocktail_kv_store
    global _agile_io_manager, _milo_quantizer
    global _block_expert_archive
    global _commvq_codebook, _vptq_quantizer, _online_sd_updater, _rasd_batcher
    global _dovetail_config, _pipo_scheduler, _disc_router, _mobile_moe_router
    global _meta_reasoner
    global _duo_attn_manager, _shadow_kv_cache, _pq_cache_index, _spe_cache_prefetcher
    global _duo_decoding_decoder, _knapspec_selector, _token_merging_cfg
    global _token_swift_decoder, _c2t_tree_builder, _clasp_decoder
    global _soup_experts_mixer, _vision_prefix_cache, _vector_index
    global _sub_spec_decoder, _del_decoder_inst, _dfloat11_cfg, _rans_codec_inst
    global _qspec_decoder, _quant_spec_decoder, _copy_spec_drafter
    global _squeeze_llm_quant, _hetero_vocab_decoder, _head_aware_kv_store

    if getattr(args, "prompt_lookup", False):
        try:
            from squish.prompt_lookup import PromptLookupConfig, PromptLookupDecoder
            _plcfg = PromptLookupConfig(
                ngram_min=2,
                ngram_max=getattr(args, "prompt_lookup_n", 3),
                max_speculative=getattr(args, "prompt_lookup_k", 4),
            )
            # PromptLookupDecoder needs the forward callable; defer full init to inference.
            # Store config now; decoder is instantiated on first generation call.
            _prompt_lookup_decoder = _plcfg  # type: ignore[assignment]
            _info("prompt-lookup", f"ngram_max={_plcfg.ngram_max}  max_speculative={_plcfg.max_speculative}")
        except Exception as _e:
            _warn(f"[prompt-lookup] Skipped: {_e}")

    if getattr(args, "seq_packing", False):
        try:
            from squish.seq_packing import PackingConfig, SequencePacker
            _spcfg = PackingConfig(max_packed_length=getattr(args, "seq_packing_budget", 2048))
            _seq_packer = SequencePacker(_spcfg)
            _info("seq-packing", f"max_packed_length={_spcfg.max_packed_length}")
        except Exception as _e:
            _warn(f"[seq-packing] Skipped: {_e}")

    if getattr(args, "ada_serve", False):
        try:
            from squish.ada_serve import BUILT_IN_SLOS, AdaServeConfig, AdaServeScheduler
            _slo_name = getattr(args, "ada_serve_slo", "general")
            _ada_slo = BUILT_IN_SLOS.get(_slo_name, BUILT_IN_SLOS["general"])
            _ada_cfg = AdaServeConfig()
            _ada_serve_scheduler = AdaServeScheduler(_ada_cfg)
            _ada_serve_scheduler.register_slo(_slo_name, _ada_slo)
            _info("ada-serve", f"slo={_slo_name}  min_γ={_ada_cfg.min_gamma}  max_γ={_ada_cfg.max_gamma}")
        except Exception as _e:
            _warn(f"[ada-serve] Skipped: {_e}")

    if getattr(args, "conf_spec", False):
        try:
            from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
            _cscfg = ConfSpecConfig(
                high_gate=getattr(args, "conf_spec_high_gate", 0.90),
                low_gate=getattr(args, "conf_spec_low_gate", 0.50),
            )
            _conf_spec_verifier = ConfSpecVerifier(_cscfg)
            _info("conf-spec", f"high_gate={_cscfg.high_gate}  low_gate={_cscfg.low_gate}")
        except Exception as _e:
            _warn(f"[conf-spec] Skipped: {_e}")

    if getattr(args, "kv_share", False):
        try:
            from squish.kvsharer import KVShareMap, KVSharerConfig
            _kvshr_cfg = KVSharerConfig(share_every_n_layers=getattr(args, "kv_share_every", 2))
            _kvsharer_map = KVShareMap(_kvshr_cfg)
            _info("kv-share", f"every={_kvshr_cfg.share_every_n_layers} layers")
        except Exception as _e:
            _warn(f"[kv-share] Skipped: {_e}")

    if getattr(args, "kv_slab", False):
        try:
            from squish.kv_slab import KVSlabAllocator
            _kv_slab_allocator = KVSlabAllocator(n_pages=getattr(args, "kv_slab_pages", 256))
            _info("kv-slab", f"pages={getattr(args, 'kv_slab_pages', 256)}")
        except Exception as _e:
            _warn(f"[kv-slab] Skipped: {_e}")

    if getattr(args, "paris_kv", False):
        try:
            from squish.paris_kv import ParisKVCodebook, ParisKVConfig
            _paris_cfg = ParisKVConfig(n_centroids=getattr(args, "paris_kv_centroids", 64))
            _paris_kv_codebook = ParisKVCodebook(_paris_cfg)
            _info("paris-kv", f"centroids={_paris_cfg.n_centroids}")
        except Exception as _e:
            _warn(f"[paris-kv] Skipped: {_e}")

    if getattr(args, "streaming_sink", False):
        try:
            from squish.streaming_sink import SinkConfig, SinkKVCache
            _sink_cfg = SinkConfig(max_tokens=getattr(args, "streaming_sink_size", 2048))
            _streaming_sink_cache = SinkKVCache(_sink_cfg)
            _info("streaming-sink", f"budget={_sink_cfg.max_tokens}")
        except Exception as _e:
            _warn(f"[streaming-sink] Skipped: {_e}")

    if getattr(args, "diff_kv", False):
        try:
            from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
            _diffkv_cfg = DiffKVConfig()
            _diffkv_policy_mgr = DiffKVPolicyManager(_diffkv_cfg)
            _info("diff-kv", f"critical={_diffkv_cfg.critical_fraction}  marginal={_diffkv_cfg.marginal_fraction}")
        except Exception as _e:
            _warn(f"[diff-kv] Skipped: {_e}")

    if getattr(args, "small_kv", False):
        try:
            from squish.smallkv import SmallKVCache, SmallKVConfig
            _smallkv_cfg = SmallKVConfig()
            _smallkv_cache = SmallKVCache(_smallkv_cfg)
            _info("small-kv", f"budget={_smallkv_cfg.kv_budget_fraction}  recall_k={_smallkv_cfg.recall_top_k}")
        except Exception as _e:
            _warn(f"[small-kv] Skipped: {_e}")

    if getattr(args, "lookahead", False):
        try:
            from squish.lookahead_reasoning import LookaheadConfig, LookaheadReasoningEngine
            _la_cfg = LookaheadConfig(lookahead_k=getattr(args, "lookahead_k", 4))
            # draft_fn is wired to the actual model at inference time; store config only
            _la_cfg._server_enabled = True  # marker checked during generation
            _lookahead_engine = _la_cfg  # type: ignore[assignment]  # full engine per-request
            _info("lookahead", f"k={_la_cfg.lookahead_k}  family={_la_cfg.model_family}")
        except Exception as _e:
            _warn(f"[lookahead] Skipped: {_e}")

    if getattr(args, "spec_reason", False):
        try:
            from squish.spec_reason import SpecReasonConfig
            _sr_cfg = SpecReasonConfig()
            _sr_cfg._server_enabled = True  # marker checked during generation
            _spec_reason_orch = _sr_cfg  # type: ignore[assignment]  # full orch per-request
            _info("spec-reason", f"min_score={_sr_cfg.min_acceptance_score}  max_draft={_sr_cfg.max_draft_steps}")
        except Exception as _e:
            _warn(f"[spec-reason] Skipped: {_e}")

    # ── Attention and KV kernels ─────────────────────────────────────────────
    if getattr(args, "sage_attention", False):
        try:
            from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
            _sage_attn_kernel = SageAttentionKernel(SageAttentionConfig())
            _info("sage-attention", "INT8 QK^T kernel ready  (~2.1× attention speedup)")
        except Exception as _e:
            _warn(f"[sage-attention] Skipped: {_e}")

    if getattr(args, "sage_attention2", False):
        try:
            from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
            _sage_attn2_kernel = SageAttention2Kernel(SageAttention2Config())
            _info("sage-attention2", "INT4/FP8 kernel ready  (~3.1× attention speedup)")
        except Exception as _e:
            _warn(f"[sage-attention2] Skipped: {_e}")

    if getattr(args, "sparge_attention", False):
        try:
            from squish.sparge_attn import SpargeAttnConfig, SpargeAttnEngine
            _sparge_engine = SpargeAttnEngine(SpargeAttnConfig())
            _info("sparge-attention", "sparse+quantized attention engine ready  (2.5–5× speedup)")
        except Exception as _e:
            _warn(f"[sparge-attention] Skipped: {_e}")

    if getattr(args, "squeeze_attention", False):
        try:
            from squish.squeeze_attention import LayerKVBudget, SqueezeConfig, SqueezeKVCache
            _sq_cfg = SqueezeConfig()
            _sq_budgets = [
                LayerKVBudget(layer_idx=i, token_budget=_sq_cfg.total_kv_budget // _sq_cfg.n_layers)
                for i in range(_sq_cfg.n_layers)
            ]
            _squeeze_cache = SqueezeKVCache(budgets=_sq_budgets, config=_sq_cfg)
            _info("squeeze-attention", f"adaptive KV budget: {_sq_cfg.total_kv_budget} total tokens across {_sq_cfg.n_layers} layers")
        except Exception as _e:
            _warn(f"[squeeze-attention] Skipped: {_e}")

    # ── KV cache strategies ──────────────────────────────────────────────────
    if getattr(args, "yoco_kv", False):
        try:
            from squish.yoco import YOCOConfig
            _yoco_config = YOCOConfig()
            _yoco_config._server_enabled = True
            _info("yoco-kv", f"cross-layer KV reuse enabled  (self_attn_layers={_yoco_config.n_self_attn_layers})")
        except Exception as _e:
            _warn(f"[yoco-kv] Skipped: {_e}")

    if getattr(args, "cla", False):
        try:
            from squish.cla import CLAConfig
            _cla_config = CLAConfig()
            _cla_config._server_enabled = True
            _info("cla", f"cross-layer attention enabled  (sharing_factor={_cla_config.sharing_factor})")
        except Exception as _e:
            _warn(f"[cla] Skipped: {_e}")

    if getattr(args, "kvtuner", False):
        try:
            from squish.kvtuner import KVTunerConfig
            _kvtuner_config = KVTunerConfig()
            _kvtuner_config._server_enabled = True
            _info("kvtuner", f"adaptive KV budget  (target_avg_bits={_kvtuner_config.target_avg_bits})")
        except Exception as _e:
            _warn(f"[kvtuner] Skipped: {_e}")

    if getattr(args, "robust_scheduler", False):
        try:
            from squish.robust_scheduler import AMaxScheduler, RobustSchedulerConfig
            _robust_sched = AMaxScheduler(RobustSchedulerConfig())
            _info("robust-scheduler", f"A-max scheduling enabled  (max_batch_tokens={_robust_sched.config.max_batch_tokens})")
        except Exception as _e:
            _warn(f"[robust-scheduler] Skipped: {_e}")

    if getattr(args, "gemfilter", False):
        try:
            from squish.gemfilter import GemFilterConfig
            _gemfilter_config = GemFilterConfig()
            _gemfilter_config._server_enabled = True
            _info("gemfilter", f"attention head filter  (top_k_tokens={_gemfilter_config.top_k_tokens})")
        except Exception as _e:
            _warn(f"[gemfilter] Skipped: {_e}")

    if getattr(args, "svdq", False):
        try:
            from squish.svdq import SVDqConfig
            _svdq_config = SVDqConfig()
            _svdq_config._server_enabled = True
            _info("svdq", f"SVD KV quantization  (target_avg_bits={_svdq_config.target_avg_bits})")
        except Exception as _e:
            _warn(f"[svdq] Skipped: {_e}")

    # ── Speculative decoding variants ─────────────────────────────────────────
    if getattr(args, "sparse_spec", False):
        try:
            from squish.sparse_spec import SparseSpecConfig
            _sparse_spec_config = SparseSpecConfig()
            _sparse_spec_config._server_enabled = True
            _info("sparse-spec", f"sparse speculative decoding  (gamma={_sparse_spec_config.gamma}  top_k_ratio={_sparse_spec_config.top_k_ratio})")
        except Exception as _e:
            _warn(f"[sparse-spec] Skipped: {_e}")

    if getattr(args, "sparse_verify", False):
        try:
            from squish.sparse_verify import SparseVerifyConfig
            _sparse_verify_config = SparseVerifyConfig()
            _sparse_verify_config._server_enabled = True
            _info("sparse-verify", f"sparse draft verification  (attn_sparsity={_sparse_verify_config.attn_sparsity})")
        except Exception as _e:
            _warn(f"[sparse-verify] Skipped: {_e}")

    if getattr(args, "long_spec", False):
        try:
            from squish.long_spec import LongSpecConfig
            _long_spec_config = LongSpecConfig()
            _long_spec_config._server_enabled = True
            _info("long-spec", f"extended speculative decoding  (gamma={_long_spec_config.gamma}  max_context={_long_spec_config.max_context_len})")
        except Exception as _e:
            _warn(f"[long-spec] Skipped: {_e}")

    if getattr(args, "fr_spec", False):
        try:
            from squish.fr_spec import FRSpecConfig
            _fr_spec_config = FRSpecConfig()
            _fr_spec_config._server_enabled = True
            _info("fr-spec", f"frequency-token speculative  (top_k_fraction={_fr_spec_config.top_k_fraction})")
        except Exception as _e:
            _warn(f"[fr-spec] Skipped: {_e}")

    if getattr(args, "diffusion_draft", ""):
        try:
            from squish.diffusion_draft import DiffusionDraftModel
            _diffusion_draft_model = DiffusionDraftModel(
                model_path=args.diffusion_draft,
            )
            _info("diffusion-draft", f"diffusion speculative model: {args.diffusion_draft}")
        except Exception as _e:
            _warn(f"[diffusion-draft] Skipped: {_e}")

    # ── Token-importance / adaptive-layer strategies ──────────────────────────
    if getattr(args, "trail", False):
        try:
            from squish.trail import TrailConfig
            _trail_config = TrailConfig()
            _trail_config._server_enabled = True
            _info("trail", f"token-importance layer skipping  (probe_layer={_trail_config.probe_layer})")
        except Exception as _e:
            _warn(f"[trail] Skipped: {_e}")

    if getattr(args, "specontext", False):
        try:
            from squish.specontext import SpeContextConfig
            _specontext_config = SpeContextConfig()
            _specontext_config._server_enabled = True
            _info("specontext", f"speculative context retrieval  (topk={_specontext_config.retrieval_topk})")
        except Exception as _e:
            _warn(f"[specontext] Skipped: {_e}")

    if getattr(args, "forelen", False):
        try:
            from squish.forelen import ForelenConfig
            _forelen_config = ForelenConfig()
            _forelen_config._server_enabled = True
            _info("forelen", f"forward length prediction  (buckets={_forelen_config.n_length_buckets})")
        except Exception as _e:
            _warn(f"[forelen] Skipped: {_e}")

    if getattr(args, "ipw", False):
        try:
            from squish.ipw import IPWConfig
            _ipw_config = IPWConfig()
            _ipw_config._server_enabled = True
            _info("ipw", f"importance-weighted prefill  (quality_weight={_ipw_config.quality_weight})")
        except Exception as _e:
            _warn(f"[ipw] Skipped: {_e}")

    if getattr(args, "layer_skip", False):
        try:
            from squish.layer_skip import EarlyExitConfig
            _layer_skip_config = EarlyExitConfig()
            _layer_skip_config._server_enabled = True
            _info("layer-skip", f"early-exit adaptive decoding  (exit_layer={_layer_skip_config.exit_layer}  threshold={_layer_skip_config.confidence_threshold})")
        except Exception as _e:
            _warn(f"[layer-skip] Skipped: {_e}")

    if getattr(args, "lora_adapter", ""):
        try:
            from squish.lora_manager import LoRAManager
            _lora_mgr = LoRAManager()
            _lora_mgr.load(args.lora_adapter)
            _info("lora-adapter", f"{args.lora_adapter}")
        except Exception as _e:
            _warn(f"[lora-adapter] Skipped: {_e}")

    # ── Wave 12: Reasoning-aware KV quantisation ─────────────────────────────
    if getattr(args, "pm_kvq", False):
        try:
            from squish.pm_kvq import PMKVQConfig, PMKVQScheduler
            _pm_cfg = PMKVQConfig(
                n_blocks=getattr(args, "pm_kvq_blocks", 32),
            )
            _pm_kvq_scheduler = PMKVQScheduler(_pm_cfg)
            _info("pm-kvq", f"progressive KV quant  bits={_pm_cfg.min_bits_sensitive}→{_pm_cfg.min_bits}  "
                  f"blocks={_pm_cfg.n_blocks}")
        except Exception as _e:
            _warn(f"[pm-kvq] Skipped: {_e}")

    if getattr(args, "mix_kvq", False):
        try:
            from squish.mix_kvq import MixKVQConfig, MixKVQQuantizer
            _mx_cfg = MixKVQConfig()
            _mix_kvq_quantizer = MixKVQQuantizer(_mx_cfg)
            _info("mix-kvq", f"query-aware mixed-precision KV  "
                  f"fp16_ratio={_mx_cfg.fp16_channel_ratio}")
        except Exception as _e:
            _warn(f"[mix-kvq] Skipped: {_e}")

    if getattr(args, "cocktail_kv", False):
        try:
            from squish.cocktail_kv import CocktailConfig, CocktailKVStore
            _ck_cfg = CocktailConfig()
            _cocktail_kv_store = CocktailKVStore(_ck_cfg)
            _info("cocktail-kv", f"chunk-similarity adaptive KV  "
                  f"chunk_size={_ck_cfg.chunk_size}  fp16_fraction={_ck_cfg.fp16_fraction}")
        except Exception as _e:
            _warn(f"[cocktail-kv] Skipped: {_e}")

    if getattr(args, "agile_io", False):
        try:
            from squish.agile_io import AgileIOConfig, AgileIOManager
            _aio_cfg = AgileIOConfig(
                n_worker_threads=getattr(args, "agile_io_threads", 4),
                cache_size_mb=getattr(args, "agile_io_cache_mb", 256),
            )
            _agile_io_manager = AgileIOManager(_aio_cfg)
            _info("agile-io", f"async NVMe prefetch  threads={_aio_cfg.n_worker_threads}  "
                  f"cache={_aio_cfg.cache_size_mb}MB")
        except Exception as _e:
            _warn(f"[agile-io] Skipped: {_e}")

    if getattr(args, "milo", False):
        try:
            from squish.milo_quant import MiLoConfig, MiLoQuantizer
            _ml_cfg = MiLoConfig(
                target_bits=getattr(args, "milo_bits", 3),
                max_rank=getattr(args, "milo_rank", 16),
            )
            _milo_quantizer = MiLoQuantizer(_ml_cfg)
            _info("milo", f"INT{_ml_cfg.target_bits}+low-rank compensator  "
                  f"max_rank={_ml_cfg.max_rank}  snr≥{_ml_cfg.snr_threshold_db}dB")
        except Exception as _e:
            _warn(f"[milo] Skipped: {_e}")

    block_expert_dir = getattr(args, "block_expert", "")
    if block_expert_dir:
        try:
            from squish.block_expert_archive import BlockExpertArchive, BlockExpertConfig
            _be_path = Path(block_expert_dir).expanduser()
            if _be_path.is_dir():
                _block_expert_archive = BlockExpertArchive.load(_be_path)
                _info("block-expert", f"archive loaded  "
                      f"blocks={_block_expert_archive.stats.n_blocks}  "
                      f"experts={_block_expert_archive.stats.n_experts_total}  "
                      f"snr={_block_expert_archive.stats.avg_delta_snr_db:.1f}dB")
            else:
                _be_cfg = BlockExpertConfig(
                    n_clusters=getattr(args, "block_expert_clusters", 4),
                )
                _block_expert_archive = BlockExpertArchive(_be_path, _be_cfg)
                _be_path.mkdir(parents=True, exist_ok=True)
                _block_expert_archive.save()
                _info("block-expert", f"new archive created at {_be_path}")
        except Exception as _e:
            _warn(f"[block-expert] Skipped: {_e}")

    # ── Wave 13b: Ultra-Long Context + Adaptive Speculative Decoding ─────────
    if getattr(args, "duo_attention", False):
        try:
            from squish.duo_attention import DuoAttentionConfig, DuoKVManager
            _da_cfg = DuoAttentionConfig(
                num_layers=getattr(args, "duo_attention_layers", 32),
                num_heads=getattr(args, "duo_attention_heads", 32),
                head_dim=getattr(args, "duo_attention_head_dim", 128),
                local_window=getattr(args, "duo_attention_window", 512),
            )
            _duo_attn_manager = DuoKVManager(_da_cfg)
            _info("duo-attention", f"retrieval+streaming head separation  "
                  f"layers={_da_cfg.num_layers}  heads={_da_cfg.num_heads}  "
                  f"window={_da_cfg.local_window}")
        except Exception as _e:
            _warn(f"[duo-attention] Skipped: {_e}")

    if getattr(args, "shadow_kv", False):
        try:
            from squish.shadow_kv import ShadowKVCache, ShadowKVConfig
            _skv_cfg = ShadowKVConfig(
                svd_rank=getattr(args, "shadow_kv_rank", 128),
                n_landmarks=getattr(args, "shadow_kv_landmarks", 64),
            )
            _shadow_kv_cache = ShadowKVCache(_skv_cfg)
            _info("shadow-kv", f"low-rank pre-RoPE key cache + CPU value shadow  "
                  f"svd_rank={_skv_cfg.svd_rank}  landmarks={_skv_cfg.n_landmarks}")
        except Exception as _e:
            _warn(f"[shadow-kv] Skipped: {_e}")

    if getattr(args, "pq_cache", False):
        try:
            from squish.pq_cache import PQCacheConfig, PQKeyIndex
            _pq_cfg = PQCacheConfig(
                n_subvectors=getattr(args, "pq_cache_subvectors", 8),
                n_codes=getattr(args, "pq_cache_codes", 256),
            )
            _pq_cache_index = PQKeyIndex(_pq_cfg)
            _info("pq-cache", f"product-quantized KV ANN retrieval  "
                  f"subvectors={_pq_cfg.n_subvectors}  codes={_pq_cfg.n_codes}")
        except Exception as _e:
            _warn(f"[pq-cache] Skipped: {_e}")

    if getattr(args, "spe_cache", False):
        try:
            from squish.spe_cache import SpeCacheConfig, SpeCachePrefetcher
            _sc_cfg = SpeCacheConfig(
                block_size=getattr(args, "spe_cache_block_size", 16),
                prefetch_budget=getattr(args, "spe_cache_budget", 8),
            )
            _spe_cache_prefetcher = SpeCachePrefetcher(_sc_cfg)
            _info("spe-cache", f"speculative KV-cache prefetch for multi-turn  "
                  f"block={_sc_cfg.block_size}  budget={_sc_cfg.prefetch_budget}")
        except Exception as _e:
            _warn(f"[spe-cache] Skipped: {_e}")

    if getattr(args, "duo_decoding", False):
        try:
            from squish.duo_decoding import DuoDecodingConfig, DuoDecodingDecoder
            _dd_cfg = DuoDecodingConfig(
                gamma=getattr(args, "duo_decoding_gamma", 4),
                k_max=getattr(args, "duo_decoding_kmax", 8),
            )
            _duo_decoding_decoder = DuoDecodingDecoder(_dd_cfg)
            _info("duo-decoding", f"hardware-aware dynamic multi-sequence spec-decode  "
                  f"gamma={_dd_cfg.gamma}  k_max={_dd_cfg.k_max}")
        except Exception as _e:
            _warn(f"[duo-decoding] Skipped: {_e}")

    if getattr(args, "knapspec", False):
        try:
            from squish.knapspec import KnapSpecConfig, KnapSpecSelector
            _ks_cfg = KnapSpecConfig(
                num_layers=getattr(args, "knapspec_layers", 32),
                budget_fraction=getattr(args, "knapspec_budget", 0.7),
            )
            _knapspec_selector = KnapSpecSelector(_ks_cfg)
            _info("knapspec", f"knapsack-optimal self-speculative layer selection  "
                  f"layers={_ks_cfg.num_layers}  budget={_ks_cfg.budget_fraction:.0%}")
        except Exception as _e:
            _warn(f"[knapspec] Skipped: {_e}")

    if getattr(args, "token_merging", False):
        try:
            from squish.token_merging import TokenMergingConfig
            _tm_cfg = TokenMergingConfig(
                r=getattr(args, "token_merging_r", 8),
                start_layer=getattr(args, "token_merging_start", 4),
                end_layer=getattr(args, "token_merging_end", -1),
            )
            _token_merging_cfg = _tm_cfg
            _info("token-merging", f"ToMe prefill token dedup  "
                  f"r={_tm_cfg.r}  layers={_tm_cfg.start_layer}→{_tm_cfg.end_layer}")
        except Exception as _e:
            _warn(f"[token-merging] Skipped: {_e}")

    if getattr(args, "token_swift", False):
        try:
            from squish.token_swift import TokenSwiftConfig, TokenSwiftDecoder
            _ts_cfg = TokenSwiftConfig(
                n_heads=getattr(args, "token_swift_heads", 4),
                window_size=getattr(args, "token_swift_window", 512),
                vocab_size=getattr(args, "token_swift_vocab", 151936),
            )
            _token_swift_decoder = TokenSwiftDecoder(_ts_cfg)
            _info("token-swift", f"multi-token draft heads + partial KV reuse  "
                  f"heads={_ts_cfg.n_heads}  window={_ts_cfg.window_size}")
        except Exception as _e:
            _warn(f"[token-swift] Skipped: {_e}")

    if getattr(args, "c2t", False):
        try:
            from squish.c2t import AdaptiveTreeBuilder, C2TConfig
            _c2t_cfg = C2TConfig(
                tree_depth=getattr(args, "c2t_depth", 4),
                wide_branches=getattr(args, "c2t_wide", 3),
                narrow_branches=getattr(args, "c2t_narrow", 1),
            )
            _c2t_tree_builder = AdaptiveTreeBuilder(_c2t_cfg)
            _info("c2t", f"classifier-based candidate tree  "
                  f"depth={_c2t_cfg.tree_depth}  wide={_c2t_cfg.wide_branches}  "
                  f"narrow={_c2t_cfg.narrow_branches}")
        except Exception as _e:
            _warn(f"[c2t] Skipped: {_e}")

    if getattr(args, "clasp", False):
        try:
            from squish.clasp import CLaSPConfig, CLaSPDecoder
            _cl_cfg = CLaSPConfig(
                num_layers=getattr(args, "clasp_layers", 32),
                max_skip_layers=getattr(args, "clasp_max_skip", 8),
                draft_gamma=getattr(args, "clasp_gamma", 4),
            )
            _clasp_decoder = CLaSPDecoder(_cl_cfg)
            _info("clasp", f"in-context layer-skip adaptive spec-decode  "
                  f"layers={_cl_cfg.num_layers}  max_skip={_cl_cfg.max_skip_layers}  "
                  f"gamma={_cl_cfg.draft_gamma}")
        except Exception as _e:
            _warn(f"[clasp] Skipped: {_e}")

    # ── Wave 14: Quantization + Vocabulary-Adaptive Spec-Decode + Expert Mixing ─

    if getattr(args, "soup_experts", False):
        try:
            from squish.soup_experts import SoupOfExperts
            _soup_experts_mixer = SoupOfExperts(
                tolerance=getattr(args, "soup_experts_tolerance", 0.01),
            )
            _info("soup-experts", f"sparse LoRA-expert blending  "
                  f"tolerance={_soup_experts_mixer.tolerance}")
        except Exception as _e:
            _warn(f"[soup-experts] Skipped: {_e}")

    if getattr(args, "vision_cache", False):
        try:
            from squish.vision_cache import VisionPrefixCache
            _vision_prefix_cache = VisionPrefixCache(
                max_entries=getattr(args, "vision_cache_max_entries", 64),
            )
            _info("vision-cache", f"SHA-256 vision prefix dedup  "
                  f"max_entries={_vision_prefix_cache.max_entries}")
        except Exception as _e:
            _warn(f"[vision-cache] Skipped: {_e}")

    if getattr(args, "vector_index", False):
        try:
            from squish.vector_index import MRLIndex
            _full_dim   = getattr(args, "vector_index_dim", 512)
            _coarse_dim = getattr(args, "vector_index_coarse_dim", 64)
            _vector_index = MRLIndex(full_dim=_full_dim, coarse_dim=_coarse_dim)
            _info("vector-index", f"Matryoshka repr learning + HNSW ANN  "
                  f"full_dim={_full_dim}  coarse_dim={_coarse_dim}")
        except Exception as _e:
            _warn(f"[vector-index] Skipped: {_e}")

    if getattr(args, "del_decoder", False):
        try:
            from squish.del_decoder import DELConfig, DELDecoder
            _del_cfg = DELConfig(
                num_layers=32,
                min_exit_layer=getattr(args, "del_decoder_min_exit", 4),
                max_exit_layer=getattr(args, "del_decoder_max_exit", 8),
                gamma=getattr(args, "del_decoder_gamma", 4),
            )
            _del_decoder_inst = DELDecoder(
                forward_fn=lambda toks, layer=None: __import__("numpy").zeros((len(toks), 1)),
                config=_del_cfg,
            )
            _info("del-decoder", f"dynamic early-layer exit spec-decode  "
                  f"exit=[{_del_cfg.min_exit_layer},{_del_cfg.max_exit_layer}]  "
                  f"gamma={_del_cfg.gamma}")
        except Exception as _e:
            _warn(f"[del-decoder] Skipped: {_e}")

    if getattr(args, "dfloat11", False):
        try:
            from squish.dfloat11 import DFloat11Config
            _dfloat11_cfg = DFloat11Config(
                block_size=getattr(args, "dfloat11_block_size", 256),
            )
            _info("dfloat11", f"DFloat11 block-float compression  "
                  f"block_size={_dfloat11_cfg.block_size}  "
                  f"use_rans={_dfloat11_cfg.use_rans}")
        except Exception as _e:
            _warn(f"[dfloat11] Skipped: {_e}")

    if getattr(args, "rans_codec", False):
        try:
            from squish.rans_codec import RANSCodec
            _rans_codec_inst = RANSCodec(
                freq={},
                m_bits=getattr(args, "rans_codec_mbits", 14),
            )
            _info("rans-codec", f"rANS entropy codec for KV/weight compression  "
                  f"m_bits={_rans_codec_inst.m_bits}")
        except Exception as _e:
            _warn(f"[rans-codec] Skipped: {_e}")

    if getattr(args, "qspec", False):
        try:
            from squish.qspec import QSpecConfig, QSpecDecoder
            _qs_cfg = QSpecConfig(
                gamma=getattr(args, "qspec_gamma", 4),
                draft_act_bits=getattr(args, "qspec_draft_bits", 4),
                verify_act_bits=getattr(args, "qspec_verify_bits", 8),
            )
            _qspec_decoder = QSpecDecoder(
                w4a8_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                w4a16_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                config=_qs_cfg,
            )
            _info("qspec", f"quantisation-aware spec-decode  "
                  f"gamma={_qs_cfg.gamma}  draft={_qs_cfg.draft_act_bits}b  "
                  f"verify={_qs_cfg.verify_act_bits}b")
        except Exception as _e:
            _warn(f"[qspec] Skipped: {_e}")

    if getattr(args, "quant_spec", False):
        try:
            from squish.quant_spec import QuantSpecConfig, QuantSpecDecoder
            _qts_cfg = QuantSpecConfig(
                gamma=getattr(args, "quant_spec_gamma", 4),
                draft_quant_bits=getattr(args, "quant_spec_bits", 4),
            )
            _quant_spec_decoder = QuantSpecDecoder(
                draft_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                config=_qts_cfg,
            )
            _info("quant-spec", f"draft-quantised spec-decode  "
                  f"gamma={_qts_cfg.gamma}  bits={_qts_cfg.draft_quant_bits}")
        except Exception as _e:
            _warn(f"[quant-spec] Skipped: {_e}")

    if getattr(args, "copy_spec", False):
        try:
            from squish.copy_spec import CopySpecConfig, CopySpecDrafter
            _cs_cfg = CopySpecDrafter(
                config=CopySpecConfig(
                    max_draft_len=getattr(args, "copy_spec_max_draft", 8),
                    max_history_len=getattr(args, "copy_spec_history_len", 2048),
                ),
            )
            _copy_spec_drafter = _cs_cfg
            _info("copy-spec", f"copy-based spec-decode from history  "
                  f"max_draft={_copy_spec_drafter.config.max_draft_len}  "
                  f"history={_copy_spec_drafter.config.max_history_len}")
        except Exception as _e:
            _warn(f"[copy-spec] Skipped: {_e}")

    if getattr(args, "sub_spec", False):
        try:
            from squish.sub_spec import SubSpecConfig, SubSpecDecoder
            _ss_cfg = SubSpecConfig(
                gamma=getattr(args, "sub_spec_gamma", 4),
                n_gpu_layers=getattr(args, "sub_spec_gpu_layers", 16),
            )
            _sub_spec_decoder = SubSpecDecoder(
                draft_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                target_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                config=_ss_cfg,
            )
            _info("sub-spec", f"quantized-substitute spec-decode  "
                  f"gamma={_ss_cfg.gamma}  gpu_layers={_ss_cfg.n_gpu_layers}")
        except Exception as _e:
            _warn(f"[sub-spec] Skipped: {_e}")

    if getattr(args, "squeeze_llm", False):
        try:
            from squish.squeeze_llm import SqueezeLLMConfig, SqueezeLLMQuantizer
            _sq_cfg = SqueezeLLMConfig(
                quant_bits=getattr(args, "squeeze_llm_bits", 4),
                sparsity_ratio=getattr(args, "squeeze_llm_sparsity", 0.45),
            )
            _squeeze_llm_quant = SqueezeLLMQuantizer(config=_sq_cfg)
            _info("squeeze-llm", f"sparse+dense mixed-precision quantisation  "
                  f"bits={_sq_cfg.quant_bits}  sparsity={_sq_cfg.sparsity_ratio}")
        except Exception as _e:
            _warn(f"[squeeze-llm] Skipped: {_e}")

    if getattr(args, "hetero_vocab_sd", False):
        try:
            from squish.hetero_vocab_sd import (
                HeteroVocabConfig,
                HeteroVocabDecoder,
                HeteroVocabDrafter,
            )
            _hv_cfg = HeteroVocabConfig(
                gamma=getattr(args, "hetero_vocab_gamma", 4),
                draft_vocab_size=getattr(args, "hetero_vocab_draft_size", 32000),
            )
            _hetero_vocab_decoder = HeteroVocabDecoder(
                drafter=HeteroVocabDrafter(config=_hv_cfg),
                target_fn=lambda toks: __import__("numpy").zeros((len(toks), 1)),
                config=_hv_cfg,
            )
            _info("hetero-vocab-sd", f"mismatched-vocab spec-decode  "
                  f"gamma={_hv_cfg.gamma}  draft_vocab={_hv_cfg.draft_vocab_size}")
        except Exception as _e:
            _warn(f"[hetero-vocab-sd] Skipped: {_e}")

    if getattr(args, "head_infer", False):
        try:
            from squish.head_infer import HeadAwareKVStore, HeadInferConfig
            _hi_cfg = HeadInferConfig(
                n_layers=getattr(args, "head_infer_layers", 32),
                n_heads=getattr(args, "head_infer_heads", 32),
            )
            _head_aware_kv_store = HeadAwareKVStore(config=_hi_cfg)
            _info("head-infer", f"head-level KV separation  "
                  f"layers={_hi_cfg.n_layers}  heads={_hi_cfg.n_heads}")
        except Exception as _e:
            _warn(f"[head-infer] Skipped: {_e}")

    if getattr(args, "nf4_quant", False):
        try:
            from squish.nf4_quant import NF4_LEVELS  # noqa: F401
            _info("nf4-quant", f"NF4 normal-float 4-bit quantisation  "
                  f"levels={len(NF4_LEVELS)}")
        except Exception as _e:
            _warn(f"[nf4-quant] Skipped: {_e}")

    if getattr(args, "spin_quant", False):
        try:
            from squish.spin_quant import run_rotation  # noqa: F401
            _info("spin-quant", "SpinQuant Hadamard rotation for quantisation-friendly layout")
        except Exception as _e:
            _warn(f"[spin-quant] Skipped: {_e}")

    if getattr(args, "life_model", False):
        try:
            from squish.life_model import predict  # noqa: F401
            _info("life-model", "model lifecycle predictor for cache eviction guidance")
        except Exception as _e:
            _warn(f"[life-model] Skipped: {_e}")

    print()
    _section("")
    print(f"  {_C.B}{_gradient('  Server ready!', _LOGO_GRAD)}{_C.R}")
    print()
    _info("API endpoint",  f"{_C.T}http://{args.host}:{args.port}/v1{_C.R}")
    _info("Web chat UI",   f"{_C.T}http://{args.host}:{args.port}/chat{_C.R}")
    _info("Ollama compat", f"{_C.T}http://{args.host}:{args.port}/api/chat{_C.R}")
    print()
    print(f"  {_C.DIM}Set in any OpenAI client:{_C.R}")
    print(f"    {_C.MG}OPENAI_BASE_URL{_C.R}=http://{args.host}:{args.port}/v1")
    print(f"    {_C.MG}OPENAI_API_KEY{_C.R}=squish")
    print()

    uvicorn.run(
        app,
        host      = args.host,
        port      = args.port,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
