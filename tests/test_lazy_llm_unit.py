"""
tests/test_lazy_llm_unit.py

Unit tests for squish/lazy_llm.py — LazyLLM dynamic token pruning.

Covers:
  • LazyLLMConfig construction and validation
  • _importance_scores: shape and dtype (mocked mx)
  • _build_keep_mask: keep_ratio, revive_window edge cases
  • _PruneState: initial state, mutation
  • _LazyLLMLayerWrapper: decode passthrough, prefill masking/update
  • patch_model_lazy_llm / unpatch_model_lazy_llm: layer replacement,
    layer passthrough for < start_layer, reversibility
  • _get_layers: both attribute paths
  • End-to-end: multi-layer forward with importance-based pruning
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

# ── import under test ─────────────────────────────────────────────────────────

from squish.lazy_llm import (
    LazyLLMConfig,
    _PruneState,
    _build_keep_mask,
    _get_layers,
    _LazyLLMLayerWrapper,
    patch_model_lazy_llm,
    unpatch_model_lazy_llm,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_hidden(T: int, D: int = 8, seed: int = 0) -> object:
    """
    Return a fake ``mx.array`` with shape (1, T, D).

    Uses a real mx.array when MLX is installed (required for importance-score
    path tests), otherwise falls back to a numpy-backed stub.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((1, T, D)).astype(np.float32)
    try:
        import mlx.core as mx
        return mx.array(data)
    except ImportError:
        return _NpArray3D(data)


class _NpArray3D:
    """Thin mx.array stub for shape (1, T, D)."""

    def __init__(self, data: np.ndarray):
        assert data.ndim == 3
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, idx):
        sliced = self._data[idx]
        return _NpArray2D(sliced)

    def __mul__(self, other):
        if isinstance(other, _NpArray3D):
            return _NpArray3D(self._data * other._data)
        return _NpArray3D(self._data * np.array(other._data))

    def astype(self, _dtype):
        return self


class _NpArray2D:
    """Thin mx.array stub for shape (T, D)."""

    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    def astype(self, _dtype):
        return self

    def __mul__(self, other):
        return _NpArray2D(self._data * other._data)


def _patch_mx_for_lazy_llm(monkeypatch):
    """
    Inject a minimal mlx.core stub so that _importance_scores can run
    without a real MLX installation.
    """
    import sys
    import squish.lazy_llm as mod

    # Only patch if mlx is not present
    try:
        import mlx.core  # noqa: F401 — real MLX present; no stub needed
        return
    except ImportError:
        pass

    mx_stub = types.SimpleNamespace(
        float32 = np.float32,
        sqrt    = lambda x: _NpArray1D(np.sqrt(np.array(x._data))),
        sum     = lambda x, axis: _NpArray1D(np.sum(x._data, axis=axis)),
        eval    = lambda *a: None,
        array   = lambda x, dtype=None: x,
    )
    mlx_mod    = types.ModuleType("mlx")
    mlx_core   = types.ModuleType("mlx.core")
    mlx_core.__dict__.update(mx_stub.__dict__)
    sys.modules["mlx"]      = mlx_mod
    sys.modules["mlx.core"] = mlx_core
    monkeypatch.setattr(mod, "mx", mlx_core, raising=False)


class _NpArray1D:
    def __init__(self, data):
        self._data = np.asarray(data)

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# LazyLLMConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestLazyLLMConfig:
    def test_defaults(self):
        cfg = LazyLLMConfig()
        assert cfg.keep_ratio == 0.70
        assert cfg.start_layer == 2
        assert cfg.revive_window == 4
        assert cfg.verbose is False

    def test_custom_values(self):
        cfg = LazyLLMConfig(keep_ratio=0.5, start_layer=4, revive_window=8, verbose=True)
        assert cfg.keep_ratio == 0.5
        assert cfg.start_layer == 4
        assert cfg.revive_window == 8
        assert cfg.verbose is True

    def test_invalid_keep_ratio_zero(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            LazyLLMConfig(keep_ratio=0.0)

    def test_invalid_keep_ratio_above_one(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            LazyLLMConfig(keep_ratio=1.1)

    def test_negative_start_layer_raises(self):
        with pytest.raises(ValueError, match="start_layer"):
            LazyLLMConfig(start_layer=-1)

    def test_negative_revive_window_raises(self):
        with pytest.raises(ValueError, match="revive_window"):
            LazyLLMConfig(revive_window=-1)

    def test_keep_ratio_boundary_one(self):
        cfg = LazyLLMConfig(keep_ratio=1.0)
        assert cfg.keep_ratio == 1.0

    def test_keep_ratio_just_above_zero(self):
        cfg = LazyLLMConfig(keep_ratio=0.01)
        assert cfg.keep_ratio == 0.01


# ─────────────────────────────────────────────────────────────────────────────
# _PruneState
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneState:
    def test_initial_mask_is_none(self):
        state = _PruneState()
        assert state.active_mask is None

    def test_mask_can_be_set(self):
        state = _PruneState()
        mask = np.array([True, False, True])
        state.active_mask = mask
        assert np.array_equal(state.active_mask, mask)

    def test_mask_can_be_reset_to_none(self):
        state = _PruneState()
        state.active_mask = np.ones(4, dtype=bool)
        state.active_mask = None
        assert state.active_mask is None


# ─────────────────────────────────────────────────────────────────────────────
# _build_keep_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildKeepMask:
    def _scores(self, T: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random(T).astype(np.float32)

    def test_output_shape(self):
        mask = _build_keep_mask(self._scores(10), keep_ratio=0.7, revive_window=2)
        assert mask.shape == (10,)
        assert mask.dtype == bool

    def test_keep_ratio_respected_approximately(self):
        T = 20
        scores = self._scores(T)
        mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=0)
        # Should keep at least ceil(0.5 * 20) = 10
        assert mask.sum() >= 10

    def test_revive_window_always_kept(self):
        T = 10
        # Give trailing tokens the worst scores
        scores = np.zeros(T, dtype=np.float32)
        scores[-4:] = -1.0   # lowest importance
        mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=4)
        # Last 4 tokens must always be kept
        assert all(mask[-4:])

    def test_zero_revive_window(self):
        T = 10
        scores = self._scores(T)
        mask = _build_keep_mask(scores, keep_ratio=0.6, revive_window=0)
        assert mask.shape == (10,)

    def test_revive_window_larger_than_T(self):
        T = 3
        scores = self._scores(T)
        mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=10)
        # All tokens kept (revive window covers everything)
        assert mask.sum() == T

    def test_keep_ratio_one_keeps_all(self):
        T = 8
        scores = self._scores(T)
        mask = _build_keep_mask(scores, keep_ratio=1.0, revive_window=0)
        assert mask.sum() == T

    def test_at_least_one_token_kept(self):
        T = 6
        scores = np.zeros(T, dtype=np.float32)
        mask = _build_keep_mask(scores, keep_ratio=0.001, revive_window=0)
        assert mask.sum() >= 1

    def test_single_token(self):
        scores = np.array([0.5], dtype=np.float32)
        mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=1)
        assert mask.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# _get_layers
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLayers:
    def test_model_model_layers_path(self):
        inner = types.SimpleNamespace(layers=[1, 2, 3])
        model = types.SimpleNamespace(model=inner)
        assert _get_layers(model) == [1, 2, 3]

    def test_direct_layers_path(self):
        model = types.SimpleNamespace(layers=[4, 5])
        # model.model not present → should fall to model.layers
        assert _get_layers(model) == [4, 5]

    def test_model_model_layers_takes_priority(self):
        """model.model.layers takes priority over model.layers."""
        inner = types.SimpleNamespace(layers=["inner"])
        model = types.SimpleNamespace(model=inner, layers=["outer"])
        assert _get_layers(model) == ["inner"]

    def test_no_layers_returns_none(self):
        model = types.SimpleNamespace()
        assert _get_layers(model) is None

    def test_empty_layers_list(self):
        model = types.SimpleNamespace(layers=[])
        assert _get_layers(model) == []


# ─────────────────────────────────────────────────────────────────────────────
# _LazyLLMLayerWrapper — passthrough / masking logic
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLayer:
    """Minimal transformer-layer stub that echoes its input."""

    def __init__(self):
        self.call_count = 0
        self.last_x = None
        self.attribute_A = "A"

    def __call__(self, x, *args, **kwargs):
        self.call_count += 1
        self.last_x = x
        return x   # echo → hidden == input, shape preserved


class TestLazyLLMLayerWrapper:
    def _make_wrapper(self, layer_idx=3, start_layer=2, keep_ratio=0.7, revive_window=2):
        cfg   = LazyLLMConfig(start_layer=start_layer, keep_ratio=keep_ratio,
                              revive_window=revive_window)
        state = _PruneState()
        layer = _FakeLayer()
        return _LazyLLMLayerWrapper(layer, layer_idx, cfg, state), layer, state, cfg

    # ── decode passthrough (T == 1) ──────────────────────────────────────────

    def test_decode_passthrough_no_mask_update(self):
        wrapper, orig, state, _ = self._make_wrapper()
        x = _fake_hidden(T=1)
        out = wrapper(x)
        assert out is x
        assert state.active_mask is None          # mask not updated for T==1

    def test_decode_passthrough_calls_orig(self):
        wrapper, orig, state, _ = self._make_wrapper()
        x = _fake_hidden(T=1)
        wrapper(x)
        assert orig.call_count == 1

    # ── layer below start_layer — no masking, but still updates mask ──────────

    def test_below_start_layer_no_mask_applied(self):
        """Layer index < start_layer: original receives the unmasked hidden."""
        wrapper, orig, state, _ = self._make_wrapper(layer_idx=0, start_layer=2)
        state.active_mask = np.array([True, False, True, True], dtype=bool)
        x_orig = _fake_hidden(T=4)
        wrapper(x_orig)
        # The original layer should have received x_orig (no gating)
        assert orig.last_x is x_orig

    # ── attribute delegation ──────────────────────────────────────────────────

    def test_attribute_delegation(self):
        wrapper, orig, _, _ = self._make_wrapper()
        assert wrapper.attribute_A == "A"

    # ── prefill path (T > 1) — mask updated ───────────────────────────────────

    def test_prefill_updates_state_mask(self):
        """After processing a T=8 sequence, active_mask must be set."""
        wrapper, orig, state, _ = self._make_wrapper(layer_idx=3, start_layer=2)
        x = _fake_hidden(T=8)
        try:
            import mlx.core  # noqa — real MLX: test fully exercised
            wrapper(x)
            assert state.active_mask is not None
            assert len(state.active_mask) == 8
        except ImportError:
            pytest.skip("MLX not available — skipping importance-score path")

    def test_prefill_with_existing_mask_applies_gate(self):
        """When state.active_mask is set, incoming x is gated before orig()."""
        wrapper, orig, state, _ = self._make_wrapper(layer_idx=3, start_layer=2)
        # Force a mask that zeros position 1
        T = 4
        state.active_mask = np.array([True, False, True, True], dtype=bool)
        x = _fake_hidden(T=T)
        try:
            import mlx.core  # noqa
            wrapper(x)
            assert orig.call_count == 1
        except ImportError:
            pytest.skip("MLX not available")


# ─────────────────────────────────────────────────────────────────────────────
# patch_model_lazy_llm / unpatch_model_lazy_llm
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchUnpatch:
    def _make_model(self, n_layers: int = 4):
        layers = [_FakeLayer() for _ in range(n_layers)]
        inner  = types.SimpleNamespace(layers=layers)
        model  = types.SimpleNamespace(model=inner)
        return model, layers

    def test_patch_returns_prune_state(self):
        model, _ = self._make_model()
        state = patch_model_lazy_llm(model, LazyLLMConfig(start_layer=1))
        assert isinstance(state, _PruneState)

    def test_patch_replaces_layers_from_start_layer(self):
        model, orig = self._make_model(4)
        cfg = LazyLLMConfig(start_layer=2)
        patch_model_lazy_llm(model, cfg)
        layers = model.model.layers
        # Layers 0, 1 unchanged; 2, 3 wrapped
        assert layers[0] is orig[0]
        assert layers[1] is orig[1]
        assert isinstance(layers[2], _LazyLLMLayerWrapper)
        assert isinstance(layers[3], _LazyLLMLayerWrapper)

    def test_patch_start_layer_zero_wraps_all(self):
        model, orig = self._make_model(3)
        patch_model_lazy_llm(model, LazyLLMConfig(start_layer=0))
        for lay in model.model.layers:
            assert isinstance(lay, _LazyLLMLayerWrapper)

    def test_patch_missing_layers_returns_none(self):
        model = types.SimpleNamespace()   # no .layers attribute at all
        result = patch_model_lazy_llm(model, LazyLLMConfig())
        assert result is None

    def test_patch_sets_model_attrs(self):
        model, _ = self._make_model()
        patch_model_lazy_llm(model)
        assert hasattr(model, "_lazy_llm_state")
        assert hasattr(model, "_lazy_llm_orig")

    def test_unpatch_restores_original_layers(self):
        model, orig = self._make_model(4)
        patch_model_lazy_llm(model, LazyLLMConfig(start_layer=1))
        unpatch_model_lazy_llm(model)
        assert model.model.layers is orig

    def test_unpatch_removes_model_attrs(self):
        model, _ = self._make_model(4)
        patch_model_lazy_llm(model, LazyLLMConfig())
        unpatch_model_lazy_llm(model)
        assert not hasattr(model, "_lazy_llm_state")
        assert not hasattr(model, "_lazy_llm_orig")

    def test_unpatch_safe_on_unpatched_model(self):
        model, _ = self._make_model()
        unpatch_model_lazy_llm(model)   # should not raise

    def test_default_config_used_when_none(self):
        model, _ = self._make_model(4)
        state = patch_model_lazy_llm(model, config=None)
        assert state is not None

    def test_state_active_mask_starts_none(self):
        model, _ = self._make_model(3)
        state = patch_model_lazy_llm(model)
        assert state.active_mask is None

    def test_state_reset_between_requests(self):
        model, _ = self._make_model(3)
        state = patch_model_lazy_llm(model)
        mask = np.ones(5, dtype=bool)
        state.active_mask = mask
        # Simulate per-request reset (server.py does this)
        state.active_mask = None
        assert state.active_mask is None

    def test_original_layers_stashed(self):
        model, orig = self._make_model(3)
        patch_model_lazy_llm(model)
        assert model._lazy_llm_orig is orig

    def test_direct_model_layers_path(self):
        """Patch works when model exposes .layers (not .model.layers)."""
        orig    = [_FakeLayer() for _ in range(3)]
        model   = types.SimpleNamespace(layers=list(orig))

        cfg = LazyLLMConfig(start_layer=1)
        state = patch_model_lazy_llm(model, cfg)
        assert state is not None
        assert isinstance(model.layers[1], _LazyLLMLayerWrapper)
        assert isinstance(model.layers[2], _LazyLLMLayerWrapper)
        assert model.layers[0] is orig[0]

    def test_roundtrip_patch_unpatch_patch(self):
        """Double-patch via unpatch→repatch works cleanly."""
        model, orig = self._make_model(3)
        patch_model_lazy_llm(model)
        unpatch_model_lazy_llm(model)
        state2 = patch_model_lazy_llm(model, LazyLLMConfig(start_layer=0))
        assert state2 is not None
        for lay in model.model.layers:
            assert isinstance(lay, _LazyLLMLayerWrapper)

    def test_unpatch_direct_layers_path(self):
        """unpatch_model_lazy_llm restores layers when model uses .layers (not .model.layers)."""
        orig  = [_FakeLayer() for _ in range(3)]
        # Use the same list object (not a copy) so identity check works
        model = types.SimpleNamespace(layers=orig)

        patch_model_lazy_llm(model, LazyLLMConfig(start_layer=0))
        assert isinstance(model.layers[0], _LazyLLMLayerWrapper)

        unpatch_model_lazy_llm(model)
        # After unpatch the original list is restored
        assert model.layers is orig

    def test_unpatch_neither_model_nor_layers_attr(self):
        """
        When a model has _lazy_llm_orig set but neither model.model.layers nor
        model.layers exists after patching, unpatch must still clean up safely.
        This exercises the elif-False → del branch (line 327->330).
        """
        model = types.SimpleNamespace()
        # Manually inject patch state to simulate an exotic model layout
        model._lazy_llm_orig  = []
        model._lazy_llm_state = _PruneState()
        unpatch_model_lazy_llm(model)
        assert not hasattr(model, "_lazy_llm_state")
        assert not hasattr(model, "_lazy_llm_orig")


# ─────────────────────────────────────────────────────────────────────────────
# Extra branch coverage: non-tuple return & verbose output
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerWrapperNonTupleAndVerbose:
    """Covers lines 218 (tuple return) and 224->240 (hidden.shape[1]<=1 branch)."""

    def _make_wrapper(self, layer_idx=3, start_layer=2, verbose=False):
        cfg   = LazyLLMConfig(start_layer=start_layer, keep_ratio=0.7,
                              revive_window=2, verbose=verbose)
        state = _PruneState()
        return cfg, state

    def test_tuple_return_covers_hidden_out0(self):
        """
        When the wrapped layer returns (hidden, kv) tuple, line 218 ``hidden = out[0]``
        must be taken.
        """
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        cfg, state = self._make_wrapper(verbose=False)

        class _TupleLayer:
            """Returns (hidden, dummy_kv) — a tuple like attention layers in mlx_lm."""
            def __call__(self, x, *a, **kw):
                dummy_kv = x[:, :1, :]   # arbitrary second element
                return (x, dummy_kv)

        wrapper = _LazyLLMLayerWrapper(_TupleLayer(), 3, cfg, state)
        x = _fake_hidden(T=8)
        out = wrapper(x)
        # out should be the original tuple returned by _TupleLayer
        assert isinstance(out, tuple)
        assert state.active_mask is not None     # mask must have been updated

    def test_hidden_shape1_le1_skips_importance_scores(self):
        """
        When the orig layer compresses the sequence to T=1 (e.g. a pooling layer),
        hidden.shape[1] <= 1 → the importance-scoring branch (224->240) is skipped.
        """
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        cfg, state = self._make_wrapper()

        class _PoolingLayer:
            """Reduce T>1 → T=1 by mean-pooling (simulates a layer that collapses seq)."""
            def __call__(self, x, *a, **kw):
                # x: (1, T, D) → (1, 1, D)
                import mlx.core as mx
                return mx.mean(x, axis=1, keepdims=True)

        wrapper = _LazyLLMLayerWrapper(_PoolingLayer(), 3, cfg, state)
        x = _fake_hidden(T=8)
        # Should not raise; active_mask NOT updated because hidden.shape[1]==1
        wrapper(x)
        assert state.active_mask is None   # no update when T collapses to 1

    def test_verbose_print_on_prefill(self):
        """
        When verbose=True and T>1, the importance-score block must print to stdout.
        Covers lines 232-234 (the verbose print path).
        """
        try:
            import mlx.core  # noqa — real MLX required for importance scores
        except ImportError:
            pytest.skip("MLX not available — skipping verbose path")

        cfg, state = self._make_wrapper(verbose=True)

        class _EchoTupleLayer:
            def __call__(self, x, *a, **kw):
                return (x, None)   # returns tuple so hidden = out[0] = x

        wrapper = _LazyLLMLayerWrapper(_EchoTupleLayer(), 3, cfg, state)
        x = _fake_hidden(T=8)
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            wrapper(x)
        finally:
            sys.stdout = old_stdout
        printed = buf.getvalue()
        assert "lazy_llm" in printed
        assert state.active_mask is not None
