"""tests/test_fused_kernels_unit.py — 100% coverage for squish/fused_kernels.py"""
from __future__ import annotations

import importlib
import math
import sys

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not available (requires Apple Silicon)")
import squish.fused_kernels as fk
from squish.fused_kernels import FusedAttention, FusedFFNGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qkv(B=1, H=2, S_q=3, S_kv=5, d=4):
    k = mx.random.key(0)
    q = mx.random.normal((B, H, S_q, d), key=k)
    k_ = mx.random.normal((B, H, S_kv, d), key=mx.random.key(1))
    v = mx.random.normal((B, H, S_kv, d), key=mx.random.key(2))
    return q, k_, v


# ---------------------------------------------------------------------------
# Module-level: except ImportError block (lines 56-57)
# ---------------------------------------------------------------------------

class TestModuleLevelImportError:
    def test_mlx_unavailable_sets_flags_false(self):
        """Reload squish.fused_kernels with mlx mocked as None → covers lines 56-57."""
        saved_mlx_core = sys.modules.get("mlx.core")
        saved_mlx_nn   = sys.modules.get("mlx.nn")
        saved_fk       = sys.modules.pop("squish.fused_kernels")

        sys.modules["mlx.core"] = None
        sys.modules["mlx.nn"]   = None
        try:
            mod = importlib.import_module("squish.fused_kernels")
            assert mod._HAS_MLX is False
            assert mod._HAS_METAL_KERNEL is False
        finally:
            # Restore mlx modules
            if saved_mlx_core is None:
                sys.modules.pop("mlx.core", None)
            else:
                sys.modules["mlx.core"] = saved_mlx_core
            if saved_mlx_nn is None:
                sys.modules.pop("mlx.nn", None)
            else:
                sys.modules["mlx.nn"] = saved_mlx_nn
            # Restore the original squish.fused_kernels module
            sys.modules["squish.fused_kernels"] = saved_fk


# ---------------------------------------------------------------------------
# FusedAttention.__init__ and _get_scale
# ---------------------------------------------------------------------------

class TestFusedAttentionInit:
    def test_default_scale_none(self):
        fa = FusedAttention()
        assert fa._scale is None
        assert fa._kernel is None

    def test_explicit_scale_stored(self):
        fa = FusedAttention(scale=0.25)
        assert fa._scale == 0.25

    def test_use_mlx_fast_is_bool(self):
        fa = FusedAttention()
        assert isinstance(fa._use_mlx_fast, bool)


class TestFusedAttentionGetScale:
    def test_with_explicit_scale(self):
        fa = FusedAttention(scale=0.5)
        assert fa._get_scale(16) == 0.5

    def test_none_scale_derives_inv_sqrt(self):
        fa = FusedAttention()
        assert fa._get_scale(16) == pytest.approx(1.0 / math.sqrt(16))


# ---------------------------------------------------------------------------
# FusedAttention.__call__
# ---------------------------------------------------------------------------

class TestFusedAttentionCall:
    def test_fast_path_no_mask(self):
        """_use_mlx_fast=True, mask=None → mx.fast.scaled_dot_product_attention."""
        fa = FusedAttention()
        fa._use_mlx_fast = True
        q, k, v = _qkv()
        out = fa(q, k, v)
        assert out.shape == q.shape

    def test_fast_path_with_mask(self):
        """_use_mlx_fast=True, mask provided → kwargs includes mask."""
        fa = FusedAttention()
        fa._use_mlx_fast = True
        q, k, v = _qkv(S_q=3, S_kv=5)
        mask = mx.zeros((1, 2, 3, 5))
        out = fa(q, k, v, mask=mask)
        assert out.shape == q.shape

    def test_fallback_no_mask(self):
        """_use_mlx_fast=False → three-step path, mask=None."""
        fa = FusedAttention()
        fa._use_mlx_fast = False
        q, k, v = _qkv()
        out = fa(q, k, v)
        assert out.shape == q.shape

    def test_fallback_with_mask(self):
        """_use_mlx_fast=False → three-step path with mask applied."""
        fa = FusedAttention()
        fa._use_mlx_fast = False
        q, k, v = _qkv(S_q=3, S_kv=5)
        mask = mx.zeros((1, 2, 3, 5))
        out = fa(q, k, v, mask=mask)
        assert out.shape == q.shape

    def test_no_mlx_raises(self, monkeypatch):
        """_HAS_MLX=False → RuntimeError."""
        monkeypatch.setattr(fk, "_HAS_MLX", False)
        fa = FusedAttention()
        q, k, v = _qkv()
        with pytest.raises(RuntimeError, match="mlx not available"):
            fa(q, k, v)


# ---------------------------------------------------------------------------
# FusedFFNGate.__init__
# ---------------------------------------------------------------------------

class TestFusedFFNGateInit:
    def test_kernel_initially_none(self):
        g = FusedFFNGate()
        assert g._kernel is None


# ---------------------------------------------------------------------------
# FusedFFNGate._build_kernel
# ---------------------------------------------------------------------------

class TestFusedFFNGateBuildKernel:
    def test_no_metal_kernel_returns_early(self, monkeypatch):
        """_HAS_METAL_KERNEL=False → returns without touching self._kernel."""
        monkeypatch.setattr(fk, "_HAS_METAL_KERNEL", False)
        g = FusedFFNGate()
        g._build_kernel()
        assert g._kernel is None   # unchanged (early return)

    def test_compile_success_sets_kernel(self):
        """Default: _HAS_METAL_KERNEL=True → Metal kernel object stored."""
        g = FusedFFNGate()
        g._build_kernel()
        assert g._kernel is not None

    def test_compile_exception_sets_kernel_none(self, monkeypatch):
        """mx.fast.metal_kernel raises → except block → self._kernel stays None."""
        def bad_metal_kernel(**kwargs):
            raise RuntimeError("compile fail")

        monkeypatch.setattr(mx.fast, "metal_kernel", bad_metal_kernel)
        g = FusedFFNGate()
        g._build_kernel()
        assert g._kernel is None


# ---------------------------------------------------------------------------
# FusedFFNGate.__call__
# ---------------------------------------------------------------------------

class TestFusedFFNGateCall:
    def test_no_mlx_raises(self, monkeypatch):
        """_HAS_MLX=False → RuntimeError."""
        monkeypatch.setattr(fk, "_HAS_MLX", False)
        g = FusedFFNGate()
        with pytest.raises(RuntimeError, match="mlx not available"):
            g(mx.ones((4,)), mx.ones((4,)))

    def test_metal_kernel_success(self):
        """Pre-set a mock kernel that returns successfully → covers line 281
        (return out.reshape(shape).astype(gate.dtype))."""
        g = FusedFFNGate()
        n = 8
        result_buf = mx.zeros((n,), dtype=mx.float32)

        def mock_kernel_success(**kwargs):
            return (result_buf,)   # simulate successful kernel return

        g._kernel = mock_kernel_success
        gate = mx.ones((n,))
        up   = mx.ones((n,))
        out  = g(gate, up)
        assert out.shape == (n,)

    def test_metal_kernel_auto_build_first_call(self):
        """auto-build on first __call__: kernel is None before call → build runs."""
        g = FusedFFNGate()
        assert g._kernel is None
        gate = mx.ones((8,))
        up   = mx.ones((8,))
        out  = g(gate, up)   # triggers _build_kernel(); may succeed or fallback
        assert out.shape == (8,)

    def test_metal_kernel_call_exception_fallback(self):
        """Kernel call raises → except block (lines 282-283) → fallback silu*up."""
        g = FusedFFNGate()
        g._build_kernel()   # pre-build so _kernel is not None

        def bad_kernel(**kwargs):
            raise RuntimeError("kernel runtime fail")

        g._kernel = bad_kernel   # replace with bad one
        gate = mx.array([0.0, 1.0, -1.0, 0.5])
        up   = mx.array([1.0, 1.0,  1.0, 1.0])
        out  = g(gate, up)   # must not raise; falls back to nn.silu * up
        assert out.shape == (4,)

    def test_no_metal_kernel_fallback(self, monkeypatch):
        """_HAS_METAL_KERNEL=False → if block skipped → nn.silu(gate)*up (line 286)."""
        monkeypatch.setattr(fk, "_HAS_METAL_KERNEL", False)
        g = FusedFFNGate()
        gate = mx.array([0.0, 1.0, -1.0])
        up   = mx.array([1.0, 1.0,  1.0])
        out  = g(gate, up)
        assert out.shape == (3,)

    def test_kernel_build_fails_in_call_then_fallback(self, monkeypatch):
        """_HAS_METAL_KERNEL=True but compile fails during first __call__ →
        _kernel stays None → if self._kernel is not None branch False → fallback."""
        def bad_metal_kernel(**kwargs):
            raise RuntimeError("compile error in call")

        monkeypatch.setattr(mx.fast, "metal_kernel", bad_metal_kernel)
        g = FusedFFNGate()
        assert g._kernel is None   # not built yet
        gate = mx.ones((4,))
        up   = mx.ones((4,))
        # First call: _kernel is None → build → fails → still None → fallback
        out = g(gate, up)
        assert out.shape == (4,)
