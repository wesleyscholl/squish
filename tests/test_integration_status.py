"""Integration status test — verifies every optimization module is:
  1. Importable as a standalone module
  2. Exported in squish.__all__
  3. Referenced by squish/server.py (flag or import)

Run with:  python -m pytest tests/test_integration_status.py -v
"""

import importlib
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module catalogue: (module_name, key_class_or_fn, server_flag_or_import)
# ---------------------------------------------------------------------------
# Tier A  — actually wired into server.py (import + init block)
# Tier B  — exported in __all__, server flag registered (activation deferred)
# Tier C  — exported in __all__, no server flag (library-only for now)
# ---------------------------------------------------------------------------

CATALOGUE = [
    # (module,              key_export,               server_flag,                tier)
    # ── Tier A: actively wired + called at runtime ──────────────────────────
    ("paged_attention",     "PagedKVCache",            "--paged-attention",        "A"),
    ("radix_cache",         "RadixTree",               "--no-prefix-cache",        "A"),
    # ── Tier B: flag registered, lazy init in main() ────────────────────────
    ("prompt_lookup",       "PromptLookupDecoder",     "--prompt-lookup",          "B"),
    ("seq_packing",         "SequencePacker",          "--seq-packing",            "B"),
    ("ada_serve",           "AdaServeScheduler",       "--ada-serve",              "B"),
    ("conf_spec",           "ConfSpecVerifier",        "--conf-spec",              "B"),
    ("kvsharer",            "KVShareMap",              "--kv-share",               "B"),
    ("kv_slab",             "KVSlabAllocator",         "--kv-slab",                "B"),
    ("paris_kv",            "ParisKVCodebook",         "--paris-kv",               "B"),
    ("streaming_sink",      "SinkKVCache",             "--streaming-sink",         "B"),
    ("diffkv",              "DiffKVPolicyManager",     "--diff-kv",                "B"),
    ("smallkv",             "SmallKVCache",            "--small-kv",               "B"),
    ("sage_attention",      "SageAttentionKernel",     "--sage-attention",         "B"),
    ("sage_attention2",     "SageAttention2Kernel",    "--sage-attention2",        "B"),
    ("sparge_attn",         "SpargeAttnEngine",        "--sparge-attention",       "B"),
    ("squeeze_attention",   "SqueezeKVCache",          "--squeeze-attention",      "B"),
    ("yoco",                "YOCOConfig",              "--yoco-kv",                "B"),
    ("cla",                 "CLAConfig",               "--cla",                    "B"),
    ("kvtuner",             "KVTunerConfig",           "--kvtuner",                "B"),
    ("robust_scheduler",    "AMaxScheduler",           "--robust-scheduler",       "B"),
    ("gemfilter",           "GemFilterConfig",         "--gemfilter",              "B"),
    ("svdq",                "SVDqConfig",              "--svdq",                   "B"),
    ("sparse_spec",         "SparseSpecConfig",        "--sparse-spec",            "B"),
    ("sparse_verify",       "SparseVerifyConfig",      "--sparse-verify",          "B"),
    ("trail",               "TrailConfig",             "--trail",                  "B"),
    ("specontext",          "SpeContextConfig",        "--specontext",             "B"),
    ("forelen",             "ForelenConfig",           "--forelen",                "B"),
    ("ipw",                 "IPWConfig",               "--ipw",                    "B"),
    ("layer_skip",          "EarlyExitConfig",         "--layer-skip",             "B"),
    ("lookahead_reasoning", "LookaheadReasoningEngine","--lookahead",              "B"),
    ("spec_reason",         "SpecReasonOrchestrator",  "--spec-reason",            "B"),
    ("long_spec",           "LongSpecConfig",          "--long-spec",              "B"),
    ("fr_spec",             "FRSpecConfig",            "--fr-spec",                "B"),
    ("lora_manager",        "LoRAManager",             "--lora-adapter",           "B"),
    ("diffusion_draft",     "DiffusionDraftModel",     "--diffusion-draft",        "B"),
    # ── Tier C: exported, no dedicated server flag ───────────────────────────
    ("kvsharer",            "KVSharerConfig",          None,                       "C"),
    ("diffkv",              "DiffKVConfig",            None,                       "C"),
    ("smallkv",             "SmallKVConfig",           None,                       "C"),
]

# Deduplicate to unique modules
MODULES = sorted({row[0] for row in CATALOGUE})

SERVER_PY = Path(__file__).parent.parent / "squish" / "server.py"
_server_src = SERVER_PY.read_text()

# Pre-parse __all__
import squish as _squish_pkg

_ALL = set(_squish_pkg.__all__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _server_has_flag(flag: str) -> bool:
    return flag in _server_src


def _server_imports_module(mod: str) -> bool:
    pat = re.compile(
        r'from\s+squish\.' + re.escape(mod) + r'\s+import'
        r'|from\s+\.' + re.escape(mod) + r'\s+import'
    )
    return bool(pat.search(_server_src))


# ---------------------------------------------------------------------------
# Test class 1: every module is importable
# ---------------------------------------------------------------------------

class TestModuleImportable:
    @pytest.mark.parametrize("mod", MODULES)
    def test_module_importable(self, mod):
        m = importlib.import_module(f"squish.{mod}")
        assert m is not None, f"squish.{mod} could not be imported"


# ---------------------------------------------------------------------------
# Test class 2: key export exists after import
# ---------------------------------------------------------------------------

class TestKeyExportsExist:
    @pytest.mark.parametrize("mod,export,_flag,_tier", CATALOGUE)
    def test_export_exists(self, mod, export, _flag, _tier):
        m = importlib.import_module(f"squish.{mod}")
        assert hasattr(m, export), (
            f"squish.{mod} is missing attribute '{export}'"
        )


# ---------------------------------------------------------------------------
# Test class 3: key classes are in squish.__all__
# ---------------------------------------------------------------------------

class TestExportsInAll:
    # Only check the primary (first-listed) export per module in __all__
    _primary = {row[0]: row[1] for row in CATALOGUE}

    @pytest.mark.parametrize("mod", MODULES)
    def test_in_squish_all(self, mod):
        export = self._primary[mod]
        assert export in _ALL, (
            f"'{export}' from squish.{mod} is missing from squish.__all__"
        )


# ---------------------------------------------------------------------------
# Test class 4: server.py has CLI flag for Tier B modules
# ---------------------------------------------------------------------------

_TIER_B_FLAGS = [(row[0], row[2]) for row in CATALOGUE if row[3] == "B" and row[2]]
# deduplicate
_seen = set()
_TIER_B_FLAGS_UNIQUE = []
for mod, flag in _TIER_B_FLAGS:
    key = (mod, flag)
    if key not in _seen:
        _seen.add(key)
        _TIER_B_FLAGS_UNIQUE.append((mod, flag))


class TestServerFlags:
    @pytest.mark.parametrize("mod,flag", _TIER_B_FLAGS_UNIQUE)
    def test_server_has_flag(self, mod, flag):
        assert _server_has_flag(flag), (
            f"squish/server.py is missing argparse flag '{flag}' for module '{mod}'"
        )


# ---------------------------------------------------------------------------
# Test class 5: Tier A modules are actively imported in server.py
# ---------------------------------------------------------------------------

_TIER_A = [(row[0], row[2]) for row in CATALOGUE if row[3] == "A"]

class TestTierAWired:
    @pytest.mark.parametrize("mod,_flag", _TIER_A)
    def test_server_imports(self, mod, _flag):
        assert _server_imports_module(mod), (
            f"squish/server.py does not import squish.{mod} — expected Tier A wiring"
        )


# ---------------------------------------------------------------------------
# Test class 6: integration smoke test — instantiate key objects
# ---------------------------------------------------------------------------

class TestIntegrationSmoke:
    """Instantiate the most important cross-wave objects with default config."""

    def test_ada_serve_scheduler_smoke(self):
        import time

        from squish.ada_serve import (
            BUILT_IN_SLOS,
            AdaServeConfig,
            AdaServeRequest,
            AdaServeScheduler,
        )
        sched = AdaServeScheduler(AdaServeConfig())
        sched.register_slo("general", BUILT_IN_SLOS["general"])
        req = AdaServeRequest(request_id="req-1", slo=BUILT_IN_SLOS["general"], arrival_time_ms=time.time() * 1000)
        sched.enqueue(req)
        gamma = sched.get_gamma("req-1")
        assert 1 <= gamma <= 8

    def test_conf_spec_verifier_smoke(self):
        import numpy as np

        from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
        verifier = ConfSpecVerifier(ConfSpecConfig())
        logits = np.zeros(32000, dtype=np.float32)
        logits[100] = 10.0
        decision = verifier.verify_step("hello world", "context", logits)
        assert decision.routing is not None

    def test_prompt_lookup_decoder_smoke(self):
        import numpy as np

        from squish.prompt_lookup import PromptLookupConfig, PromptLookupDecoder
        prompt_ids = [1, 2, 3, 4, 5, 1, 2]
        def _fwd(ids):
            return np.zeros(len(ids), dtype=np.float32)
        cfg = PromptLookupConfig(ngram_min=2, ngram_max=3, max_speculative=3)
        dec = PromptLookupDecoder(full_forward=_fwd, config=cfg)
        result = dec.generate(prompt_ids, max_new_tokens=8)
        assert result is not None

    def test_seq_packer_smoke(self):
        from squish.seq_packing import PackingConfig, SequencePacker
        packer = SequencePacker(PackingConfig(max_packed_length=512))
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        batches = packer.pack(seqs)
        assert isinstance(batches, list)

    def test_smallkv_smoke(self):
        import numpy as np

        from squish.smallkv import SmallKVCache, SmallKVConfig
        cache = SmallKVCache(SmallKVConfig(n_layers=4))
        k = np.random.randn(4, 64).astype(np.float32)
        v = np.random.randn(4, 64).astype(np.float32)
        scores = np.array([0.9, 0.1, 0.5, 0.8], dtype=np.float32)
        cache.ingest(layer_idx=0, token_indices=np.array([0, 1, 2, 3]), keys=k, values=v,
                     importance_scores=scores)
        assert cache.stats.total_critical > 0

    def test_diffkv_policy_mgr_smoke(self):
        import numpy as np

        from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
        mgr = DiffKVPolicyManager(DiffKVConfig(n_layers=4, n_heads=4))
        w = np.abs(np.random.randn(4, 8)).astype(np.float32)
        w /= w.sum(axis=-1, keepdims=True)
        mgr.record_attention(layer_idx=0, head_idx=0, attn_weights=w)
        policy = mgr.get_policy(layer_idx=0, head_idx=0)
        assert policy is not None

    def test_streaming_sink_smoke(self):
        import numpy as np

        from squish.streaming_sink import SinkConfig, SinkKVCache
        cache = SinkKVCache(SinkConfig(num_sinks=4, window_size=32, head_dim=32))
        k = np.random.randn(32).astype(np.float32)
        v = np.random.randn(32).astype(np.float32)
        cache.append(key=k, value=v)
        assert cache.size == 1

    def test_lookahead_engine_smoke(self):
        from squish.lookahead_reasoning import (
            LookaheadConfig,
            LookaheadReasoningEngine,
            LookaheadStep,
        )
        def _draft(ctx):
            return LookaheadStep(text="step", source="draft", confidence=0.8, tokens_used=5)
        engine = LookaheadReasoningEngine(
            config=LookaheadConfig(lookahead_k=2),
            draft_fn=_draft,
        )
        batch = engine.run_cycle("the quick brown fox")
        assert batch.n_steps == 2

    def test_spec_reason_smoke(self):
        from squish.spec_reason import ReasoningStep, SpecReasonConfig, SpecReasonOrchestrator
        def _draft(ctx):
            return ReasoningStep(text="draft answer", source="draft",
                                 confidence=0.9, tokens_used=10, step_idx=0)
        def _target(ctx):
            return ReasoningStep(text="target answer", source="target",
                                 confidence=0.95, tokens_used=12, step_idx=0)
        orch = SpecReasonOrchestrator(SpecReasonConfig(), _draft, _target)
        step, verdict = orch.generate_step("given context")
        assert step is not None

    def test_sage_attention_smoke(self):
        import numpy as np

        from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
        cfg = SageAttentionConfig(head_dim=32, n_heads=2, block_size=8)
        kern = SageAttentionKernel(cfg)
        q = np.random.randn(2, 16, 32).astype(np.float32)
        k = np.random.randn(2, 16, 32).astype(np.float32)
        v = np.random.randn(2, 16, 32).astype(np.float32)
        out, stats = kern.forward(q, k, v)
        assert out.shape == (2, 16, 32)

    def test_sparge_attn_smoke(self):
        import numpy as np

        from squish.sparge_attn import SpargeAttnConfig, SpargeAttnEngine
        cfg = SpargeAttnConfig(head_dim=32, n_heads=2, block_size=8)
        engine = SpargeAttnEngine(cfg)
        q = np.random.randn(2, 16, 32).astype(np.float32)
        k = np.random.randn(2, 16, 32).astype(np.float32)
        v = np.random.randn(2, 16, 32).astype(np.float32)
        out, stats = engine.forward(q, k, v)
        assert out.shape == (2, 16, 32)


# ---------------------------------------------------------------------------
# Print report to stdout when run directly
# ---------------------------------------------------------------------------

def _report():
    import squish as sq
    all_exports = set(sq.__all__)

    tiers = {"A": [], "B": [], "C": []}
    seen = set()
    for mod, export, flag, tier in CATALOGUE:
        if mod in seen:
            continue
        seen.add(mod)
        importable = True
        try:
            importlib.import_module(f"squish.{mod}")
        except Exception:
            importable = False
        in_all = export in all_exports
        has_flag = flag and _server_has_flag(flag)
        wired = _server_imports_module(mod)
        tiers[tier].append((mod, export, flag, importable, in_all, has_flag, wired))

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  squish INTEGRATION STATUS REPORT — March 2026")
    lines.append("=" * 72)

    tier_labels = {
        "A": "TIER A — Actively wired (runtime import + init in server.py)",
        "B": "TIER B — CLI flag registered, lazy init on startup",
        "C": "TIER C — Exported in __all__, no dedicated server flag",
    }
    for t in ("A", "B", "C"):
        lines.append("")
        lines.append(f"  {tier_labels[t]}")
        lines.append(f"  {'Module':<22} {'Key class':<26} {'Flag':<24} {'Import'} {'__all__'} {'Flag ok'} {'Wired'}")
        lines.append(f"  {'-'*22} {'-'*26} {'-'*24} {'-'*6} {'-'*7} {'-'*7} {'-'*5}")
        for mod, export, flag, imp, ia, hf, wired in tiers[t]:
            flag_str = (flag or "—")[:24]
            lines.append(
                f"  {mod:<22} {export:<26} {flag_str:<24} "
                f"{'✅' if imp else '❌'}      "
                f"{'✅' if ia else '❌'}      "
                f"{'✅' if hf else '—'}      "
                f"{'✅' if wired else '—'}"
            )
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Total exported in __all__: {len(all_exports)}")
    lines.append(f"  Total optimization modules tracked: {len(seen)}")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_report())
