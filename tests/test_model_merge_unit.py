"""
tests/test_model_merge_unit.py

Unit tests for squish/model_merge.py — 100% coverage.
"""

import math

import numpy as np
import pytest

from squish.model_merge import (
    MergeConfig,
    MergeStats,
    ModelMerger,
    dare_merge,
    slerp,
    ties_merge,
)


# ---------------------------------------------------------------------------
# MergeConfig
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_defaults(self):
        cfg = MergeConfig()
        assert cfg.method == "slerp"
        assert cfg.t == 0.5
        assert cfg.dare_density == 0.5
        assert cfg.ties_k == 0.2
        assert cfg.base_weights is None

    def test_custom_valid(self):
        cfg = MergeConfig(method="dare", t=0.8, dare_density=0.6, ties_k=0.5)
        assert cfg.method == "dare"
        assert cfg.t == 0.8

    @pytest.mark.parametrize("method", ["slerp", "dare", "ties"])
    def test_all_valid_methods(self, method):
        MergeConfig(method=method)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"method": "lerp"}, "method"),
            ({"t": -0.1}, "t"),
            ({"t": 1.1}, "t"),
            ({"dare_density": 0.0}, "dare_density"),
            ({"dare_density": 1.1}, "dare_density"),
            ({"ties_k": 0.0}, "ties_k"),
            ({"ties_k": 1.1}, "ties_k"),
        ],
    )
    def test_validation_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MergeConfig(**kwargs)

    def test_t_boundary_values(self):
        MergeConfig(t=0.0)
        MergeConfig(t=1.0)

    def test_dare_density_boundary(self):
        MergeConfig(dare_density=1.0)

    def test_frozen(self):
        cfg = MergeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.t = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MergeStats
# ---------------------------------------------------------------------------


class TestMergeStats:
    def test_avg_keys_per_merge_normal(self):
        ms = MergeStats(n_merges=3, total_keys=9, method="slerp")
        assert ms.avg_keys_per_merge == 3.0

    def test_avg_keys_per_merge_zero_merges(self):
        ms = MergeStats(n_merges=0, total_keys=0, method="slerp")
        assert ms.avg_keys_per_merge == 0.0


# ---------------------------------------------------------------------------
# slerp
# ---------------------------------------------------------------------------


class TestSlerp:
    def test_t_zero_returns_a(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        out = slerp(a, b, 0.0)
        assert np.allclose(out, a, atol=1e-5)

    def test_t_one_returns_b(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        out = slerp(a, b, 1.0)
        assert np.allclose(out, b, atol=1e-5)

    def test_preserves_norm_at_midpoint(self):
        a = np.array([2.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 2.0], dtype=np.float32)
        out = slerp(a, b, 0.5)
        assert abs(np.linalg.norm(out) - np.linalg.norm(a)) < 1e-4

    def test_parallel_fallback_to_lerp(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        out = slerp(a, b, 0.5)
        assert np.allclose(out, a, atol=1e-5)

    def test_zero_vector_fallback(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = slerp(a, b, 0.5)
        # Falls back to lerp: 0.5 * b
        assert np.allclose(out, 0.5 * b, atol=1e-5)

    def test_preserves_shape(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        out = slerp(a, b, 0.5)
        assert out.shape == (3, 4)

    def test_output_dtype_matches_input(self):
        a = np.random.randn(8).astype(np.float64)
        b = np.random.randn(8).astype(np.float64)
        out = slerp(a, b, 0.5)
        # implementation casts to float32 then back to input dtype
        assert out.dtype == a.dtype


# ---------------------------------------------------------------------------
# dare_merge
# ---------------------------------------------------------------------------


class TestDareMerge:
    def test_shape_preserved(self):
        base = np.zeros((4, 4), dtype=np.float32)
        da = np.ones((4, 4), dtype=np.float32)
        db = np.ones((4, 4), dtype=np.float32) * 2
        out = dare_merge(base, da, db, density=1.0, t=0.5)
        assert out.shape == (4, 4)

    def test_density_one_no_drop(self):
        base = np.zeros(10, dtype=np.float32)
        da = np.ones(10, dtype=np.float32)
        db = np.zeros(10, dtype=np.float32)
        out = dare_merge(base, da, db, density=1.0, t=0.0, seed=0)
        # t=0 → only da contributes; density=1 → no drop
        assert np.allclose(out, da, atol=1e-5)

    def test_density_reduces_nonzero(self):
        rng = np.random.default_rng(42)
        base = np.zeros(1000, dtype=np.float32)
        da = np.ones(1000, dtype=np.float32)
        db = np.ones(1000, dtype=np.float32)
        out = dare_merge(base, da, db, density=0.5, t=0.5, seed=42)
        # With rescaling, mean should be close to 1.0
        assert 0.5 < float(np.mean(out)) < 1.5

    def test_seed_reproducibility(self):
        base = np.zeros(100, dtype=np.float32)
        da = np.ones(100, dtype=np.float32)
        db = np.ones(100, dtype=np.float32)
        out1 = dare_merge(base, da, db, 0.5, 0.5, seed=7)
        out2 = dare_merge(base, da, db, 0.5, 0.5, seed=7)
        assert np.allclose(out1, out2)

    def test_dtype_preserved(self):
        base = np.zeros(10, dtype=np.float16)
        da = np.ones(10, dtype=np.float16)
        db = np.ones(10, dtype=np.float16)
        out = dare_merge(base, da, db, 1.0, 0.5)
        assert out.dtype == np.float16


# ---------------------------------------------------------------------------
# ties_merge
# ---------------------------------------------------------------------------


class TestTiesMerge:
    def test_empty_deltas_returns_base(self):
        base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = ties_merge(base, [], k=0.5)
        assert np.array_equal(out, base)

    def test_single_delta_with_positive_sign(self):
        base = np.zeros(5, dtype=np.float32)
        finetuned = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        out = ties_merge(base, [(base, finetuned)], k=1.0)
        # All elements positive → all kept → base + delta = finetuned
        assert np.allclose(out, finetuned, atol=1e-5)

    def test_sign_election_agrees(self):
        base = np.zeros(4, dtype=np.float32)
        ft1 = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        ft2 = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        ft3 = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        # Position 0: +,+,+ → positive; position 1: +,-,+ → positive; etc.
        out = ties_merge(base, [(base, ft1), (base, ft2), (base, ft3)], k=1.0)
        assert out.shape == (4,)

    def test_top_k_trims_small_magnitudes(self):
        base = np.zeros(10, dtype=np.float32)
        finetuned = np.array([100.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        out = ties_merge(base, [(base, finetuned)], k=0.1)
        # Only the top-10% (1 element = index 0) survives
        assert abs(out[0]) > 0.0

    def test_shape_and_dtype_preserved(self):
        base = np.zeros((3, 3), dtype=np.float16)
        ft = np.ones((3, 3), dtype=np.float16)
        out = ties_merge(base, [(base, ft)], k=0.5)
        assert out.shape == (3, 3)
        assert out.dtype == np.float16


# ---------------------------------------------------------------------------
# ModelMerger
# ---------------------------------------------------------------------------


class TestModelMerger:
    def _make_weights(self, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "layer1": rng.standard_normal((4, 4)).astype(np.float32),
            "layer2": rng.standard_normal((2, 2)).astype(np.float32),
        }

    def test_slerp_merge_returns_all_keys(self):
        merger = ModelMerger(MergeConfig(method="slerp"))
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        merged = merger.merge(wa, wb)
        assert set(merged.keys()) == set(wa.keys())

    def test_dare_merge(self):
        merger = ModelMerger(MergeConfig(method="dare", t=0.5))
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        merged = merger.merge(wa, wb)
        assert "layer1" in merged

    def test_ties_merge(self):
        merger = ModelMerger(MergeConfig(method="ties", t=0.5))
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        merged = merger.merge(wa, wb)
        assert "layer1" in merged

    def test_passthrough_extra_keys(self):
        merger = ModelMerger(MergeConfig())
        wa = {"shared": np.ones(4, dtype=np.float32), "only_a": np.zeros(2, dtype=np.float32)}
        wb = {"shared": np.ones(4, dtype=np.float32), "only_b": np.zeros(3, dtype=np.float32)}
        merged = merger.merge(wa, wb)
        assert "shared" in merged
        assert "only_a" in merged
        assert "only_b" in merged

    def test_n_merged_increments(self):
        merger = ModelMerger(MergeConfig())
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        assert merger.n_merged == 0
        merger.merge(wa, wb)
        assert merger.n_merged == 1
        merger.merge(wa, wb)
        assert merger.n_merged == 2

    def test_last_merge_stats_none_before_merge(self):
        merger = ModelMerger(MergeConfig())
        assert merger.last_merge_stats is None

    def test_last_merge_stats_content(self):
        merger = ModelMerger(MergeConfig(method="slerp", t=0.3))
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        merger.merge(wa, wb)
        s = merger.last_merge_stats
        assert s is not None
        assert s["keys_merged"] == 2
        assert s["method"] == "slerp"
        assert s["t"] == 0.3

    def test_stats_cumulative(self):
        merger = ModelMerger(MergeConfig())
        wa = self._make_weights(0)
        wb = self._make_weights(1)
        merger.merge(wa, wb)
        merger.merge(wa, wb)
        st = merger.stats()
        assert st.n_merges == 2
        assert st.total_keys == 4
        assert st.avg_keys_per_merge == 2.0
        assert st.method == "slerp"

    def test_dare_uses_base_weights(self):
        base = {"w": np.zeros(4, dtype=np.float32)}
        wa = {"w": np.ones(4, dtype=np.float32)}
        wb = {"w": np.ones(4, dtype=np.float32) * 2}
        merger = ModelMerger(MergeConfig(method="dare", dare_density=1.0, t=0.0, base_weights=base))
        merged = merger.merge(wa, wb)
        # t=0, density=1 → delta_a / density → base + delta_a = wa
        assert np.allclose(merged["w"], wa["w"], atol=1e-4)

    def test_ties_uses_base_weights_override(self):
        base = {"w": np.zeros(4, dtype=np.float32)}
        wa = {"w": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}
        wb = {"w": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)}
        merger = ModelMerger(MergeConfig(method="ties", ties_k=1.0))
        merged = merger.merge(wa, wb, base_weights=base)
        assert merged["w"].shape == (4,)
