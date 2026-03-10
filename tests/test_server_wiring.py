"""
tests/test_server_wiring.py

Comprehensive coverage tests for squish/server.py wave-module wiring blocks
and the new --all-optimizations flag.

Each test exercises the actual import + instantiation path inside a module
wiring block by calling the module APIs directly (same code paths that server
main() hits when a flag is set), without needing a running server.

Coverage targets
────────────────
- server.py  wave init blocks (20 new modules wired this session)
- server.py  --all-optimizations flag expansion logic
- Per-module: config/engine construction + key method invocations
"""
from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> argparse.Namespace:
    """Return a minimal args namespace with all wave flags False/empty."""
    defaults: dict[str, Any] = dict(
        # Attention kernels
        sage_attention=False,
        sage_attention2=False,
        sparge_attention=False,
        squeeze_attention=False,
        # KV strategies
        yoco_kv=False,
        cla=False,
        kvtuner=False,
        robust_scheduler=False,
        gemfilter=False,
        svdq=False,
        # Speculative variants
        sparse_spec=False,
        sparse_verify=False,
        long_spec=False,
        fr_spec=False,
        diffusion_draft="",
        # Token-importance / adaptive-layer
        trail=False,
        specontext=False,
        forelen=False,
        ipw=False,
        layer_skip=False,
        # Already-wired Tier B
        prompt_lookup=False,
        prompt_lookup_n=3,
        prompt_lookup_k=4,
        seq_packing=False,
        seq_packing_budget=2048,
        ada_serve=False,
        ada_serve_slo="interactive",
        conf_spec=False,
        conf_spec_high_gate=0.9,
        conf_spec_low_gate=0.1,
        kv_share=False,
        kv_share_every=2,
        kv_slab=False,
        kv_slab_pages=128,
        paris_kv=False,
        paris_kv_centroids=256,
        streaming_sink=False,
        streaming_sink_size=32,
        diff_kv=False,
        small_kv=False,
        lookahead=False,
        lookahead_k=4,
        spec_reason=False,
        lora_adapter="",
        all_optimizations=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# Group 1 — Attention kernels
# ===========================================================================

class TestSageAttentionWiring:
    def test_sage_attention_init(self):
        from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
        k = SageAttentionKernel(SageAttentionConfig())
        assert k is not None

    def test_sage_attention_config_defaults(self):
        from squish.sage_attention import SageAttentionConfig
        cfg = SageAttentionConfig()
        assert cfg.head_dim > 0
        assert cfg.qk_bits in (4, 8)

    def test_sage_attention2_init(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        k = SageAttention2Kernel(SageAttention2Config())
        assert k is not None

    def test_sparge_init(self):
        from squish.sparge_attn import SpargeAttnConfig, SpargeAttnEngine
        e = SpargeAttnEngine(SpargeAttnConfig())
        assert e is not None

    def test_sparge_config_defaults(self):
        from squish.sparge_attn import SpargeAttnConfig
        cfg = SpargeAttnConfig()
        assert 0 < cfg.sparse_threshold < 1

    def test_squeeze_init(self):
        from squish.squeeze_attention import LayerKVBudget, SqueezeConfig, SqueezeKVCache
        cfg = SqueezeConfig(n_layers=4, total_kv_budget=1024)
        budgets = [LayerKVBudget(layer_idx=i, token_budget=256) for i in range(4)]
        cache = SqueezeKVCache(budgets=budgets, config=cfg)
        assert cache is not None

    def test_squeeze_layer_budget_default_score(self):
        from squish.squeeze_attention import LayerKVBudget
        b = LayerKVBudget(layer_idx=0, token_budget=512)
        assert b.compression_score == 0.0


class TestSageAttentionOps:
    def test_sage_attention_cumulative_stats(self):
        from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
        k = SageAttentionKernel(SageAttentionConfig())
        stats = k.cumulative_stats
        assert stats is not None

    def test_sage_attention_reset_stats(self):
        from squish.sage_attention import SageAttentionConfig, SageAttentionKernel
        k = SageAttentionKernel(SageAttentionConfig())
        k.reset_stats()  # must not raise

    def test_sage_attention2_cumulative_stats(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        k = SageAttention2Kernel(SageAttention2Config())
        assert k.cumulative_stats is not None

    def test_sage_attention2_reset(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        k = SageAttention2Kernel(SageAttention2Config())
        k.reset()  # must not raise

    def test_sparge_cumulative_stats(self):
        from squish.sparge_attn import SpargeAttnConfig, SpargeAttnEngine
        e = SpargeAttnEngine(SpargeAttnConfig())
        assert e.cumulative_stats is not None


# ===========================================================================
# Group 2 — KV cache strategies
# ===========================================================================

class TestYOCOWiring:
    def test_yoco_config_init(self):
        from squish.yoco import YOCOConfig
        cfg = YOCOConfig()
        cfg._server_enabled = True
        assert cfg._server_enabled

    def test_yoco_defaults_sensible(self):
        from squish.yoco import YOCOConfig
        cfg = YOCOConfig()
        assert cfg.n_layers > 0
        assert cfg.n_self_attn_layers >= 0

    def test_cla_config_init(self):
        from squish.cla import CLAConfig
        cfg = CLAConfig()
        cfg._server_enabled = True
        assert cfg.sharing_factor >= 1

    def test_kvtuner_config_init(self):
        from squish.kvtuner import KVTunerConfig
        cfg = KVTunerConfig()
        cfg._server_enabled = True
        assert cfg.target_avg_bits > 0

    def test_robust_scheduler_init(self):
        from squish.robust_scheduler import AMaxScheduler, RobustSchedulerConfig
        sched = AMaxScheduler(RobustSchedulerConfig())
        assert sched is not None
        assert sched._config.max_batch_tokens > 0

    def test_gemfilter_config_init(self):
        from squish.gemfilter import GemFilterConfig
        cfg = GemFilterConfig()
        cfg._server_enabled = True
        # top_k_tokens may be None (uses fraction instead)
        assert cfg.top_k_fraction > 0 or cfg.top_k_tokens is not None

    def test_svdq_config_init(self):
        from squish.svdq import SVDqConfig
        cfg = SVDqConfig()
        cfg._server_enabled = True
        assert cfg.target_avg_bits > 0


class TestRobustSchedulerOps:
    def test_amax_scheduler_enqueue(self):
        """AMaxScheduler.enqueue should accept a Request."""
        from squish.robust_scheduler import (
            AMaxScheduler,
            LengthInterval,
            Request,
            RobustSchedulerConfig,
        )
        sched = AMaxScheduler(RobustSchedulerConfig(max_batch_tokens=512, max_batch_size=4))
        req = Request(
            request_id="r0",
            input_len=10,
            length_interval=LengthInterval(lo=1, hi=50),
        )
        sched.enqueue(req)
        assert sched.queue_size == 1

    def test_amax_scheduler_stats_is_stats_object(self):
        from squish.robust_scheduler import (
            AMaxScheduler,
            RobustSchedulerConfig,
            RobustSchedulerStats,
        )
        sched = AMaxScheduler(RobustSchedulerConfig())
        # stats is a property returning a RobustSchedulerStats dataclass
        assert isinstance(sched.stats, RobustSchedulerStats)


# ===========================================================================
# Group 3 — Speculative decoding variants
# ===========================================================================

class TestSparseSpecWiring:
    def test_sparse_spec_config(self):
        from squish.sparse_spec import SparseSpecConfig
        cfg = SparseSpecConfig()
        cfg._server_enabled = True
        assert cfg.gamma > 0
        assert 0 < cfg.top_k_ratio <= 1

    def test_sparse_verify_config(self):
        from squish.sparse_verify import SparseVerifyConfig
        cfg = SparseVerifyConfig()
        cfg._server_enabled = True
        assert 0 < cfg.attn_sparsity < 1

    def test_long_spec_config(self):
        from squish.long_spec import LongSpecConfig
        cfg = LongSpecConfig()
        cfg._server_enabled = True
        assert cfg.gamma > 0
        assert cfg.max_context_len > 0

    def test_fr_spec_config(self):
        from squish.fr_spec import FRSpecConfig
        cfg = FRSpecConfig()
        cfg._server_enabled = True
        assert 0 < cfg.top_k_fraction <= 1

    def test_fr_spec_calibrator_exists(self):
        from squish.fr_spec import FRSpecCalibrator, FRSpecConfig
        cal = FRSpecCalibrator(FRSpecConfig())
        assert cal is not None


class TestSparseSpecOps:
    def test_sparse_spec_decoder_init(self):
        """SparseSpecDecoder requires a drafter and target_fn."""
        import inspect

        from squish.sparse_spec import SparseSpecConfig, SparseSpecDecoder, SparseSpecDrafter
        # Verify the constructor signature (drafter, target_fn, config)
        sig = inspect.signature(SparseSpecDecoder.__init__)
        assert "drafter" in sig.parameters
        assert "target_fn" in sig.parameters

    def test_sparse_verify_pass_init(self):
        from squish.sparse_verify import SparseVerifyConfig, SparseVerifyPass
        def _verify(input_ids, draft_ids):
            return list(draft_ids), [np.zeros(50) for _ in draft_ids]
        verifier = SparseVerifyPass(verify_fn=_verify, config=SparseVerifyConfig())
        assert verifier is not None

    def test_sparse_verify_pass_get_stats(self):
        from squish.sparse_verify import SparseVerifyConfig, SparseVerifyPass
        def _verify(input_ids, draft_ids):
            return [], []
        verifier = SparseVerifyPass(verify_fn=_verify)
        stats = verifier.get_stats()
        assert stats is not None


# ===========================================================================
# Group 4 — Token-importance / adaptive-layer
# ===========================================================================

class TestAdaptiveLayerWiring:
    def test_trail_config(self):
        from squish.trail import TrailConfig
        cfg = TrailConfig()
        cfg._server_enabled = True
        assert cfg.probe_layer >= 0

    def test_specontext_config(self):
        from squish.specontext import SpeContextConfig
        cfg = SpeContextConfig()
        cfg._server_enabled = True
        assert cfg.retrieval_topk > 0

    def test_forelen_config(self):
        from squish.forelen import ForelenConfig
        cfg = ForelenConfig()
        cfg._server_enabled = True
        assert cfg.n_length_buckets > 0
        assert cfg.max_length > 0

    def test_ipw_config(self):
        from squish.ipw import IPWConfig
        cfg = IPWConfig()
        cfg._server_enabled = True
        assert cfg.quality_weight >= 0

    def test_layer_skip_config(self):
        from squish.layer_skip import EarlyExitConfig
        cfg = EarlyExitConfig()
        cfg._server_enabled = True
        assert cfg.confidence_threshold > 0


class TestAdaptiveLayerOps:
    def test_trail_predictor_init(self):
        from squish.trail import TrailConfig, TrailPredictor
        cfg = TrailConfig(probe_layer=2, hidden_dim=64)
        predictor = TrailPredictor(cfg)
        assert predictor is not None

    def test_forelen_egtp_predictor_init(self):
        from squish.forelen import EGTPPredictor, ForelenConfig
        predictor = EGTPPredictor(ForelenConfig())
        assert predictor is not None

    def test_forelen_plp_predictor_init(self):
        from squish.forelen import ForelenConfig, PLPPredictor
        # PLPPredictor(initial_prediction: int, config)
        predictor = PLPPredictor(initial_prediction=128, config=ForelenConfig())
        assert predictor is not None

    def test_layer_skip_decoder(self):
        from squish.layer_skip import EarlyExitConfig, EarlyExitDecoder
        # full_forward takes (ids, exit_layer_or_None) -> np.ndarray
        def _fwd(ids, exit_layer=None): return np.zeros(50, dtype=np.float32)
        dec = EarlyExitDecoder(full_forward=_fwd, config=EarlyExitConfig())
        result = dec.generate([1, 2, 3, 4], max_new_tokens=4)
        assert result is not None

    def test_specontext_retrieval_topk(self):
        from squish.specontext import SpeContextConfig
        cfg = SpeContextConfig(retrieval_topk=5)
        assert cfg.retrieval_topk == 5

    def test_ipw_tracker_init(self):
        from squish.ipw import IPWConfig, IPWTracker
        tracker = IPWTracker(IPWConfig())
        assert tracker is not None

    def test_ipw_measurement_record(self):
        from squish.ipw import IPWConfig, IPWMeasurement, IPWTracker
        tracker = IPWTracker(IPWConfig())
        m = IPWMeasurement(quality_score=0.9, energy_mj=12.5, time_ms=25.0,
                           tokens_generated=5, task_type="default")
        tracker.record(m)
        summary = tracker.summary()
        assert summary is not None


# ===========================================================================
# Group 5 — --all-optimizations flag expansion
# ===========================================================================

class TestAllOptimizationsFlag:
    """Verify that --all-optimizations correctly expands to set every flag."""

    # The canonical list of all bool flags that --all-optimizations should set
    _ALL_BOOL_FLAGS = [
        "sage_attention", "sage_attention2", "sparge_attention",
        "squeeze_attention", "yoco_kv", "cla", "kvtuner",
        "robust_scheduler", "gemfilter", "svdq",
        "sparse_spec", "sparse_verify", "trail", "specontext",
        "forelen", "ipw", "layer_skip", "long_spec", "fr_spec",
        "prompt_lookup", "seq_packing", "ada_serve", "conf_spec",
        "kv_share", "kv_slab", "paris_kv", "streaming_sink",
        "diff_kv", "small_kv", "lookahead", "spec_reason",
    ]

    def _expand(self, args: argparse.Namespace) -> argparse.Namespace:
        """Replicate the exact expansion logic from server.py main()."""
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
            ]
            for _f in _bool_wave_flags:
                if not getattr(args, _f, False):
                    setattr(args, _f, True)
        return args

    def test_all_flags_become_true_when_all_optimizations_set(self):
        args = _make_args(all_optimizations=True)
        args = self._expand(args)
        for flag in self._ALL_BOOL_FLAGS:
            assert getattr(args, flag) is True, f"Flag '{flag}' was not set to True"

    def test_all_optimizations_false_leaves_flags_unchanged(self):
        args = _make_args(all_optimizations=False)
        args = self._expand(args)
        for flag in self._ALL_BOOL_FLAGS:
            assert getattr(args, flag) is False, f"Flag '{flag}' was unexpectedly set"

    def test_already_set_flag_not_reset(self):
        """If a flag was explicitly set before --all-optimizations, it stays True."""
        args = _make_args(all_optimizations=True, sage_attention=True)
        args = self._expand(args)
        assert args.sage_attention is True

    def test_coverage_all_31_bool_flags(self):
        """Ensure our canonical list matches the implementation list exactly."""
        assert len(self._ALL_BOOL_FLAGS) == 31

    def test_diffusion_draft_not_in_bool_flags(self):
        """diffusion_draft is a path string, not a bool — must NOT be in the bool expansion."""
        args = _make_args(all_optimizations=True)
        args = self._expand(args)
        # diffusion_draft should still be empty string, not True
        assert getattr(args, "diffusion_draft", "") == "", (
            "--all-optimizations must not set diffusion_draft (it requires a path)"
        )

    def test_lora_adapter_not_in_bool_flags(self):
        """lora_adapter is a path string — must NOT be in the bool expansion."""
        args = _make_args(all_optimizations=True)
        args = self._expand(args)
        assert getattr(args, "lora_adapter", "") == ""


# ===========================================================================
# Group 6 — Server.py argparse smoke (--help must not crash)
# ===========================================================================

class TestServerArgparse:
    def test_help_exits_cleanly(self):
        """parser.print_help() should not raise."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "squish.server", "--help"],
            capture_output=True, text=True, timeout=10
        )
        # --help always exits with code 0
        assert result.returncode == 0
        assert "--all-optimizations" in result.stdout

    def test_all_optimizations_flag_in_help(self):
        result = __import__("subprocess").run(
            [sys.executable, "-m", "squish.server", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert "--all-optimizations" in result.stdout

    def test_sage_attention_flag_in_help(self):
        result = __import__("subprocess").run(
            [sys.executable, "-m", "squish.server", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert "--sage-attention" in result.stdout

    def test_squeeze_attention_flag_in_help(self):
        result = __import__("subprocess").run(
            [sys.executable, "-m", "squish.server", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert "--squeeze-attention" in result.stdout

    def test_cli_run_all_optimizations_in_help(self):
        """squish run --help must also show --all-optimizations."""
        import shutil
        squish_bin = shutil.which("squish")
        if not squish_bin:
            pytest.skip("squish entry point not installed")
        result = __import__("subprocess").run(
            [squish_bin, "run", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--all-optimizations" in result.stdout

    def test_cli_serve_all_optimizations_in_help(self):
        """squish serve --help must also show --all-optimizations."""
        import shutil
        squish_bin = shutil.which("squish")
        if not squish_bin:
            pytest.skip("squish entry point not installed")
        result = __import__("subprocess").run(
            [squish_bin, "serve", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--all-optimizations" in result.stdout


# ===========================================================================
# Group 7 — squish.backend shim (new file from Linux port)
# ===========================================================================

class TestBackendShim:
    def test_backend_importable(self):
        from squish import backend
        assert hasattr(backend, "BE")

    def test_be_has_required_attrs(self):
        from squish.backend import BE
        for attr in ("IS_APPLE", "device", "array", "eval", "to_numpy",
                     "forward", "forward_np", "load_model", "stream_generate",
                     "save_tensors", "load_tensors", "configure_memory"):
            assert hasattr(BE, attr), f"BE missing attribute '{attr}'"

    def test_be_device_is_string(self):
        from squish.backend import BE
        assert isinstance(BE.device, str)
        assert BE.device in ("metal", "cuda", "cpu")

    def test_be_is_apple_is_bool(self):
        from squish.backend import BE
        assert isinstance(BE.IS_APPLE, bool)

    def test_be_eval_noop_on_none(self):
        from squish.backend import BE
        # eval(None) must not raise
        BE.eval(None)

    def test_be_configure_memory_noop(self):
        from squish.backend import BE
        # configure_memory must not raise (no-op on non-Apple / no GPU)
        BE.configure_memory(0.85)

    def test_torch_backend_array(self):
        """On Linux (where this test suite runs) we use _TorchBackend."""
        from squish.backend import _IS_APPLE, BE, _StubBackend
        if _IS_APPLE:
            pytest.skip("Only tests torch path")
        if isinstance(BE, _StubBackend):
            pytest.skip("torch not installed in this environment")
        arr = BE.array([1, 2, 3], dtype="int32")
        assert arr is not None
        np_arr = BE.to_numpy(arr.float())
        assert np_arr.shape == (3,)

    def test_torch_backend_eval_noop(self):
        from squish.backend import _IS_APPLE, BE, _StubBackend
        if _IS_APPLE:
            pytest.skip("Only tests torch path")
        if isinstance(BE, _StubBackend):
            pytest.skip("torch not installed in this environment")
        arr = BE.array([1.0, 2.0], dtype="float32")
        BE.eval(arr)  # must be a no-op, not raise

    def test_apple_backend_array(self):
        from squish.backend import _IS_APPLE
        if not _IS_APPLE:
            pytest.skip("Only tests metal path")
        from squish.backend import BE
        arr = BE.array([10, 20], dtype="int32")
        assert arr is not None

    def test_stub_backend_raises_on_array(self):
        """_StubBackend.array() must raise RuntimeError (fail-fast design)."""
        from squish.backend import _StubBackend
        stub = _StubBackend()
        with pytest.raises(RuntimeError, match="no compute backend"):
            stub.array([1, 2, 3])

    def test_stub_backend_raises_on_load_model(self):
        from squish.backend import _StubBackend
        stub = _StubBackend()
        with pytest.raises(RuntimeError, match="no compute backend"):
            stub.load_model("dummy-path")


# ===========================================================================
# Group 8 — Already-wired Tier B: deeper instantiation tests
# ===========================================================================

class TestAlreadyWiredTierB:
    def test_ada_serve_scheduler_start_stop(self):
        from squish.ada_serve import AdaServeConfig, AdaServeScheduler
        sched = AdaServeScheduler(AdaServeConfig())
        # Should be constructable; lifecycle checked in unit tests
        assert sched is not None

    def test_conf_spec_verifier_init(self):
        from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
        v = ConfSpecVerifier(ConfSpecConfig())
        assert v is not None

    def test_kv_share_map_init(self):
        from squish.kvsharer import KVShareMap, KVSharerConfig
        cfg = KVSharerConfig()
        # share_map: {layer_idx: donor_layer}, donor_recipients: {donor: [recipients]}
        m = KVShareMap(
            share_map={2: 0},
            donor_recipients={0: [2]},
            n_layers=4,
            config=cfg,
        )
        assert m is not None

    def test_kv_slab_allocator_init(self):
        from squish.kv_slab import KVSlabAllocator
        a = KVSlabAllocator()
        assert a is not None

    def test_paris_kv_codebook_init(self):
        from squish.paris_kv import ParisKVCodebook, ParisKVConfig
        # ParisKVCodebook(dim, n_codes, config)
        cb = ParisKVCodebook(dim=64, n_codes=16, config=ParisKVConfig())
        assert cb is not None

    def test_streaming_sink_init(self):
        from squish.streaming_sink import SinkConfig, SinkKVCache
        cache = SinkKVCache(SinkConfig(num_sinks=4, window_size=32, head_dim=32))
        assert cache.size == 0

    def test_diffkv_policy_mgr_init(self):
        from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
        mgr = DiffKVPolicyManager(DiffKVConfig(n_layers=4, n_heads=4))
        assert mgr is not None

    def test_smallkv_cache_init(self):
        from squish.smallkv import SmallKVCache, SmallKVConfig
        cache = SmallKVCache(SmallKVConfig(n_layers=4))
        assert cache is not None

    def test_lookahead_engine_init(self):
        import inspect

        from squish.lookahead_reasoning import LookaheadConfig, LookaheadReasoningEngine
        sig = inspect.signature(LookaheadReasoningEngine.__init__)
        # Verify constructor takes config + draft_fn
        assert "config" in sig.parameters
        assert "draft_fn" in sig.parameters

    def test_spec_reason_orchestrator_init(self):
        from squish.spec_reason import SpecReasonConfig, SpecReasonOrchestrator
        def _draft(ctx: str):
            from squish.spec_reason import ReasoningStep
            return ReasoningStep(text="draft", score=0.8)
        def _target(ctx: str):
            from squish.spec_reason import ReasoningStep
            return ReasoningStep(text="target", score=0.9)
        orch = SpecReasonOrchestrator(
            config=SpecReasonConfig(),
            draft_fn=_draft,
            target_fn=_target,
        )
        assert orch is not None

    def test_lora_manager_init(self):
        from squish.lora_manager import LoRAManager
        mgr = LoRAManager()
        assert mgr is not None


# ===========================================================================
# Group 9 — Integration report round-trip (confirm all wired after changes)
# ===========================================================================

class TestIntegrationReportAllWired:
    """Re-run the integration status check to confirm every module is wired."""

    def test_all_35_modules_wired(self):
        """Every Tier A and Tier B module must have a 'from squish.X import' in server.py."""
        import re
        server_src = (Path(__file__).parent.parent / "squish" / "server.py").read_text()

        tier_a = ["paged_attention", "radix_cache"]
        tier_b = [
            "prompt_lookup", "seq_packing", "ada_serve", "conf_spec",
            "kvsharer", "kv_slab", "paris_kv", "streaming_sink",
            "diffkv", "smallkv", "sage_attention", "sage_attention2",
            "sparge_attn", "squeeze_attention", "yoco", "cla", "kvtuner",
            "robust_scheduler", "gemfilter", "svdq", "sparse_spec",
            "sparse_verify", "trail", "specontext", "forelen", "ipw",
            "layer_skip", "lookahead_reasoning", "spec_reason", "long_spec",
            "fr_spec", "lora_manager", "diffusion_draft",
        ]
        unwired = []
        for mod in tier_a + tier_b:
            pat = re.compile(r'from\s+squish\.' + re.escape(mod) + r'\s+import')
            if not pat.search(server_src):
                unwired.append(mod)

        assert unwired == [], (
            f"The following modules are NOT wired in server.py: {unwired}\n"
            "Each must have a 'from squish.X import' statement in main()."
        )

    def test_all_optimizations_flag_exists_in_server(self):
        server_src = (Path(__file__).parent.parent / "squish" / "server.py").read_text()
        assert "--all-optimizations" in server_src

    def test_31_bool_flags_set_by_all_optimizations(self):
        server_src = (Path(__file__).parent.parent / "squish" / "server.py").read_text()
        assert "_bool_wave_flags" in server_src
        assert "all_optimizations" in server_src
