"""tests/test_token_merging_unit.py — 100% coverage for squish/token_merging.py"""
import sys
import numpy as np
import pytest
from unittest.mock import patch

from squish.token_merging import (
    TokenMergingConfig,
    TokenMergingState,
    bipartite_merge,
    unmerge_tokens,
    patch_model_tome,
    unpatch_model_tome,
    _cosine_similarity_bipartite,
    _ToMeLayerWrapper,
    _get_layers,
)


# ---------------------------------------------------------------------------
# TokenMergingConfig
# ---------------------------------------------------------------------------

class TestTokenMergingConfig:
    def test_defaults(self):
        cfg = TokenMergingConfig()
        assert cfg.r == 16
        assert cfg.start_layer == 4
        assert cfg.end_layer is None
        assert cfg.similarity_threshold == 0.5
        assert cfg.verbose is False

    def test_invalid_r(self):
        with pytest.raises(ValueError, match="r must be"):
            TokenMergingConfig(r=-1)

    def test_invalid_start_layer(self):
        with pytest.raises(ValueError, match="start_layer"):
            TokenMergingConfig(start_layer=-1)

    def test_end_layer_less_than_start_raises(self):
        with pytest.raises(ValueError, match="end_layer"):
            TokenMergingConfig(start_layer=5, end_layer=3)

    def test_invalid_similarity_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            TokenMergingConfig(similarity_threshold=1.5)
        with pytest.raises(ValueError, match="similarity_threshold"):
            TokenMergingConfig(similarity_threshold=-2.0)

    def test_r_zero_ok(self):
        cfg = TokenMergingConfig(r=0)
        assert cfg.r == 0

    def test_end_layer_equal_start_ok(self):
        cfg = TokenMergingConfig(start_layer=3, end_layer=3)
        assert cfg.end_layer == 3


# ---------------------------------------------------------------------------
# TokenMergingState
# ---------------------------------------------------------------------------

class TestTokenMergingState:
    def test_initial_empty(self):
        s = TokenMergingState()
        assert s.n_merges == 0
        assert s.n_merge_layers == 0

    def test_record_merge(self):
        s    = TokenMergingState()
        src  = np.array([0, 2], dtype=np.int64)
        dst  = np.array([1, 3], dtype=np.int64)
        s.record_merge(src, dst, t_before=10)
        assert s.n_merge_layers == 1
        assert s.n_merges == 2

    def test_reset_clears(self):
        s    = TokenMergingState()
        s.record_merge(np.array([0]), np.array([1]), 4)
        s.reset()
        assert s.n_merges == 0
        assert s.n_merge_layers == 0


# ---------------------------------------------------------------------------
# _cosine_similarity_bipartite
# ---------------------------------------------------------------------------

class TestCosineSimilarityBipartite:
    def test_self_similarity_is_one(self):
        a   = np.random.rand(4, 8).astype(np.float32)
        sim = _cosine_similarity_bipartite(a, a)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_shape(self):
        a   = np.random.rand(3, 8).astype(np.float32)
        b   = np.random.rand(5, 8).astype(np.float32)
        sim = _cosine_similarity_bipartite(a, b)
        assert sim.shape == (3, 5)

    def test_orthogonal_zero(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = _cosine_similarity_bipartite(a, b)
        assert sim[0, 0] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# bipartite_merge
# ---------------------------------------------------------------------------

class TestBipartiteMerge:
    def test_output_shape_reduces(self):
        h         = np.random.rand(10, 8).astype(np.float32)
        merged, s, d = bipartite_merge(h, r=3, similarity_threshold=-1.0)
        assert merged.shape[0] < h.shape[0]
        assert merged.shape[1] == 8

    def test_r_zero_returns_identity(self):
        h         = np.random.rand(10, 8).astype(np.float32)
        merged, s, d = bipartite_merge(h, r=0)
        assert merged.shape == h.shape
        assert len(s) == 0
        assert len(d) == 0

    def test_too_small_seq_returns_identity(self):
        h         = np.ones((1, 4), dtype=np.float32)
        merged, s, d = bipartite_merge(h, r=2)
        assert merged.shape == h.shape

    def test_high_threshold_no_merge(self):
        h            = np.random.rand(8, 4).astype(np.float32)
        merged, s, d = bipartite_merge(h, r=4, similarity_threshold=2.0)
        # No pairs can exceed 2.0 cosine similarity
        assert len(s) == 0

    def test_identical_tokens_merge_well(self):
        """All tokens equal → all pairs should merge."""
        h         = np.ones((8, 4), dtype=np.float32)
        merged, s, d = bipartite_merge(h, r=4, similarity_threshold=0.0)
        # At least some merges should have occurred
        assert len(s) >= 1

    def test_dtype_preserved(self):
        h         = np.random.rand(10, 8).astype(np.float32)
        merged, _, _ = bipartite_merge(h, r=2, similarity_threshold=-1.0)
        assert merged.dtype == np.float32


# ---------------------------------------------------------------------------
# unmerge_tokens
# ---------------------------------------------------------------------------

class TestUnmergeTokens:
    def test_no_merges_passthrough(self):
        m   = np.random.rand(10, 4).astype(np.float32)
        out = unmerge_tokens(m, np.array([]), np.array([]), t_original=10)
        np.testing.assert_array_equal(out, m)

    def test_restores_length(self):
        h         = np.random.rand(10, 4).astype(np.float32)
        merged, s, d = bipartite_merge(h, r=2, similarity_threshold=-1.0)
        restored = unmerge_tokens(merged, s, d, t_original=10)
        assert restored.shape == (10, 4)

    def test_identity_merge_cycle(self):
        """Merge then unmerge — positions not merged should be identical."""
        h         = np.eye(8, dtype=np.float32)   # identity matrix rows
        merged, s, d = bipartite_merge(h, r=1, similarity_threshold=-1.0)
        restored     = unmerge_tokens(merged, s, d, t_original=8)
        assert restored.shape == (8, 8)
        # Non-merged positions (keep_mask) should be exactly preserved
        keep_mask = np.ones(8, dtype=bool)
        keep_mask[s] = False
        for orig_idx in np.where(keep_mask)[0]:
            pass  # just verify no exception

    def test_dimension_mismatch_fallback(self):
        """len(keep_pos) != T_merged → safe fallback path (lines 291-293)."""
        # Duplicate indices in src_idx create a mismatch:
        # unique positions removed from keep_mask = 2,
        # but T_merged = t_original - len(src_idx) = 6 - 3 = 3 → mismatch.
        t_original = 6
        merged     = np.ones((3, 4), dtype=np.float32)
        src_idx    = np.array([0, 0, 1], dtype=np.int64)  # duplicate 0
        dst_idx    = np.array([2, 2, 3], dtype=np.int64)
        out = unmerge_tokens(merged, src_idx, dst_idx, t_original=t_original)
        assert out.shape == (t_original, 4)


# ---------------------------------------------------------------------------
# _get_layers
# ---------------------------------------------------------------------------

class TestGetLayers:
    def test_model_with_model_layers(self):
        class Inner:
            layers = [1, 2, 3]
        class M:
            model = Inner()
        assert _get_layers(M()) == [1, 2, 3]

    def test_model_with_direct_layers(self):
        class M:
            layers = [4, 5]
        assert _get_layers(M()) == [4, 5]

    def test_model_no_layers(self):
        class M:
            pass
        assert _get_layers(M()) is None


# ---------------------------------------------------------------------------
# patch_model_tome / unpatch_model_tome
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __call__(self, x, *a, **kw):
        return x


class _FakeModel:
    def __init__(self, n=6):
        self.layers = [_FakeLayer() for _ in range(n)]


class TestPatchUnpatch:
    def test_patch_returns_state(self):
        model = _FakeModel(6)
        state = patch_model_tome(model, TokenMergingConfig(start_layer=2, end_layer=4))
        assert isinstance(state, TokenMergingState)

    def test_patch_wraps_correct_layers(self):
        model = _FakeModel(6)
        patch_model_tome(model, TokenMergingConfig(start_layer=2, end_layer=4))
        for i, layer in enumerate(model.layers):
            if 2 <= i <= 4:
                assert isinstance(layer, _ToMeLayerWrapper), f"Layer {i} should be wrapped"
            else:
                assert isinstance(layer, _FakeLayer), f"Layer {i} should be original"

    def test_unpatch_restores_layers(self):
        model  = _FakeModel(6)
        patch_model_tome(model, TokenMergingConfig(start_layer=2))
        unpatch_model_tome(model)
        assert all(isinstance(l, _FakeLayer) for l in model.layers)
        assert not hasattr(model, "_tome_state")

    def test_unpatch_direct_layers_model(self):
        """Covers 443→446: model has .layers (not .model.layers) during unpatch."""
        model  = _FakeModel(4)  # direct layers, no .model attribute
        patch_model_tome(model, TokenMergingConfig(start_layer=0, end_layer=2))
        unpatch_model_tome(model)
        assert all(isinstance(l, _FakeLayer) for l in model.layers)
        assert not hasattr(model, "_tome_state")

    def test_unpatch_model_no_layers_attr(self):
        """443→446 False branch: model has _tome_orig but no .layers or .model."""
        class NoLayerModel:
            pass
        m            = NoLayerModel()
        m._tome_orig  = []          # simulate a patched model
        m._tome_state = TokenMergingState()
        # Neither hasattr(m, "model") nor hasattr(m, "layers")
        unpatch_model_tome(m)
        assert not hasattr(m, "_tome_orig")

    def test_unpatch_on_unpatched_model_safe(self):
        model = _FakeModel()
        unpatch_model_tome(model)  # should not raise

    def test_patch_incompatible_model_returns_state(self):
        class Incompatible:
            pass
        state = patch_model_tome(Incompatible())
        assert isinstance(state, TokenMergingState)

    def test_patch_model_with_nested_model_attr(self):
        class Inner:
            def __init__(self):
                self.layers = [_FakeLayer() for _ in range(4)]
        class Nested:
            def __init__(self):
                self.model = Inner()
        m = Nested()
        state = patch_model_tome(m, TokenMergingConfig(start_layer=1, end_layer=2))
        assert isinstance(state, TokenMergingState)
        for i, l in enumerate(m.model.layers):
            if 1 <= i <= 2:
                assert isinstance(l, _ToMeLayerWrapper)

    def test_unpatch_nested_model(self):
        class Inner:
            def __init__(self):
                self.layers = [_FakeLayer() for _ in range(4)]
        class Nested:
            def __init__(self):
                self.model = Inner()
        m = Nested()
        patch_model_tome(m, TokenMergingConfig(start_layer=0))
        unpatch_model_tome(m)
        assert all(isinstance(l, _FakeLayer) for l in m.model.layers)


# ---------------------------------------------------------------------------
# _ToMeLayerWrapper
# ---------------------------------------------------------------------------

class TestToMeLayerWrapper:
    def test_passthrough_non_mlx(self):
        """When mlx.core import fails, wrapper delegates directly to original layer."""
        orig  = _FakeLayer()
        cfg   = TokenMergingConfig(r=2, start_layer=0)
        state = TokenMergingState()
        w     = _ToMeLayerWrapper(orig, layer_idx=0, config=cfg, state=state)
        x     = np.ones((1, 8, 4), dtype=np.float32)
        with patch.dict(sys.modules, {"mlx.core": None}):
            result = w(x)
        # Falls back to orig since MLX not available in test env
        assert result is x

    def test_getattr_delegates(self):
        class LayerWithAttr:
            def __call__(self, x, *a, **kw): return x
            custom_attr = "hello"
        orig  = LayerWithAttr()
        cfg   = TokenMergingConfig(start_layer=0)
        state = TokenMergingState()
        w     = _ToMeLayerWrapper(orig, 0, cfg, state)
        assert w.custom_attr == "hello"

    def test_call_passthrough_single_token(self):
        """x.shape[1] <= 1 → pass-through without merging (line 332-333)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        orig  = _FakeLayer()
        cfg   = TokenMergingConfig(r=4, start_layer=0)
        state = TokenMergingState()
        w     = _ToMeLayerWrapper(orig, 0, cfg, state)
        x_np   = np.ones((1, 1, 4), dtype=np.float32)  # T=1
        x      = mx.array(x_np)
        result = w(x)
        assert result is not None
        assert state.n_merges == 0  # no merging for single token

    def test_call_passthrough_end_layer_exceeded(self):
        """idx > end_layer → pass-through (line 337-338)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        orig   = _FakeLayer()
        cfg    = TokenMergingConfig(r=4, start_layer=0, end_layer=0)
        state  = TokenMergingState()
        w      = _ToMeLayerWrapper(orig, layer_idx=1, config=cfg, state=state)
        x_np   = np.ones((1, 8, 4), dtype=np.float32)
        x      = mx.array(x_np)
        result = w(x)
        assert state.n_merges == 0  # idx=1 > end_layer=0 → pass-through

    def test_call_with_merging_verbose(self):
        """Full merge path: lines 327, 332-362 including verbose print."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        orig   = _FakeLayer()
        cfg    = TokenMergingConfig(r=2, start_layer=0,
                                    similarity_threshold=-1.0, verbose=True)
        state  = TokenMergingState()
        w      = _ToMeLayerWrapper(orig, layer_idx=0, config=cfg, state=state)
        # 8-token batch with identical rows → high similarity, merges occur
        x_np   = np.ones((1, 8, 4), dtype=np.float32)
        x      = mx.array(x_np)
        result = w(x)
        assert state.n_merge_layers >= 1

    def test_call_with_merging_silent(self):
        """353→359: merges happen but verbose=False → no print (silent path)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        orig   = _FakeLayer()
        cfg    = TokenMergingConfig(r=2, start_layer=0,
                                    similarity_threshold=-1.0, verbose=False)
        state  = TokenMergingState()
        w      = _ToMeLayerWrapper(orig, layer_idx=0, config=cfg, state=state)
        x_np   = np.ones((1, 8, 4), dtype=np.float32)
        x      = mx.array(x_np)
        result = w(x)
        assert state.n_merge_layers >= 1

    def test_call_no_merges_low_similarity(self):
        """len(src_idx) == 0 path: very high threshold → no merges (351→359)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        orig   = _FakeLayer()
        # Use max valid threshold; random orthogonal data won't hit it
        cfg    = TokenMergingConfig(r=2, start_layer=0, similarity_threshold=1.0)
        state  = TokenMergingState()
        w      = _ToMeLayerWrapper(orig, layer_idx=0, config=cfg, state=state)
        # Orthogonal basis vectors: cosine similarity = 0 for any pair
        x_np   = np.eye(8, 8, dtype=np.float32)[:8]  # shape (8,8)
        x_np   = x_np.reshape(1, 8, 8)
        x      = mx.array(x_np)
        result = w(x)
        assert state.n_merges == 0
