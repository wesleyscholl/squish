"""tests/test_paris_kv_unit.py — 100% coverage for squish/paris_kv.py"""
import numpy as np
import pytest

from squish.paris_kv import (
    ParisKVConfig,
    ParisKVCodebook,
    ema_update_centroids,
    _kmeans_plus_plus,
    _pairwise_sq_dist,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# ParisKVConfig
# ---------------------------------------------------------------------------

class TestParisKVConfig:
    def test_defaults(self):
        cfg = ParisKVConfig()
        assert cfg.learning_rate == pytest.approx(0.05)
        assert cfg.min_count     == 2
        assert cfg.drift_window  == 50
        assert cfg.refine_iters  == 20

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ParisKVConfig(learning_rate=-0.1)
        with pytest.raises(ValueError, match="learning_rate"):
            ParisKVConfig(learning_rate=1.1)

    def test_invalid_min_count(self):
        with pytest.raises(ValueError, match="min_count"):
            ParisKVConfig(min_count=0)

    def test_invalid_drift_window(self):
        with pytest.raises(ValueError, match="drift_window"):
            ParisKVConfig(drift_window=0)

    def test_invalid_refine_iters(self):
        with pytest.raises(ValueError, match="refine_iters"):
            ParisKVConfig(refine_iters=0)


# ---------------------------------------------------------------------------
# _pairwise_sq_dist
# ---------------------------------------------------------------------------

class TestPairwiseSqDist:
    def test_zero_dist_self(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        d = _pairwise_sq_dist(a, a)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-5)

    def test_known_distance(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        d = _pairwise_sq_dist(a, b)
        assert d[0, 0] == pytest.approx(25.0, abs=1e-4)

    def test_shape(self):
        a = np.random.rand(5, 8).astype(np.float32)
        b = np.random.rand(3, 8).astype(np.float32)
        d = _pairwise_sq_dist(a, b)
        assert d.shape == (5, 3)


# ---------------------------------------------------------------------------
# _kmeans_plus_plus
# ---------------------------------------------------------------------------

class TestKMeansPlusPlus:
    def test_returns_k_centers(self):
        vecs = np.random.rand(50, 4).astype(np.float32)
        rng  = np.random.default_rng(1)
        ctrs = _kmeans_plus_plus(vecs, 8, rng)
        assert ctrs.shape == (8, 4)

    def test_k_equals_n(self):
        vecs = np.random.rand(4, 4).astype(np.float32)
        rng  = np.random.default_rng(2)
        ctrs = _kmeans_plus_plus(vecs, 4, rng)
        assert ctrs.shape == (4, 4)

    def test_single_center(self):
        vecs = np.random.rand(10, 4).astype(np.float32)
        rng  = np.random.default_rng(3)
        ctrs = _kmeans_plus_plus(vecs, 1, rng)
        assert ctrs.shape == (1, 4)


# ---------------------------------------------------------------------------
# ema_update_centroids
# ---------------------------------------------------------------------------

class TestEmaUpdateCentroids:
    def _simple_inputs(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        vectors   = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
        assigns   = np.array([0, 1])
        return centroids, vectors, assigns

    def test_output_shape(self):
        c, v, a = self._simple_inputs()
        out = ema_update_centroids(c, v, a, learning_rate=0.1, min_count=1)
        assert out.shape == c.shape

    def test_centroids_move_toward_vectors(self):
        c, v, a = self._simple_inputs()
        out = ema_update_centroids(c, v, a, learning_rate=0.5, min_count=1)
        # Centroid 0 should move toward [0.1, 0.1]
        assert out[0, 0] > 0.0

    def test_low_count_centroid_not_updated(self):
        """Centroids with count < min_count should be skipped."""
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        vectors   = np.array([[5.0, 5.0]], dtype=np.float32)
        assigns   = np.array([0])   # centroid 1 gets no vectors → count=0
        out = ema_update_centroids(centroids, vectors, assigns, learning_rate=0.5, min_count=2)
        # Centroid 1 should remain unchanged
        np.testing.assert_allclose(out[1], [10.0, 10.0], atol=1e-5)

    def test_all_centroids_updated_with_min_count_1(self):
        centroids = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        vectors   = np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float32)
        assigns   = np.array([0, 1])
        out = ema_update_centroids(centroids, vectors, assigns, learning_rate=1.0, min_count=1)
        np.testing.assert_allclose(out[0], [1.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(out[1], [4.0, 4.0], atol=1e-4)


# ---------------------------------------------------------------------------
# ParisKVCodebook
# ---------------------------------------------------------------------------

class TestParisKVCodebook:
    def _make(self, n_codes=8, dim=4, **cfg_kw):
        cfg = ParisKVConfig(**cfg_kw) if cfg_kw else ParisKVConfig()
        return ParisKVCodebook(dim=dim, n_codes=n_codes, config=cfg)

    def test_not_fitted_initially(self):
        cb = self._make()
        assert not cb.is_fitted

    def test_properties_available_unfitted(self):
        cb = self._make(dim=4, n_codes=8)
        assert cb.n_codes == 8
        assert cb.dim == 4

    def test_encode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError):
            cb.encode(np.zeros((1, 4), dtype=np.float32))

    def test_decode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError):
            cb.decode(np.array([0], dtype=np.uint16))

    def test_fit_marks_fitted(self):
        cb    = self._make()
        vecs  = np.random.rand(50, 4).astype(np.float32)
        cb.fit(vecs, seed=0)
        assert cb.is_fitted
        assert cb.centroids is not None
        assert cb.centroids.shape == (8, 4)

    def test_fit_returns_self(self):
        cb   = self._make()
        vecs = np.random.rand(20, 4).astype(np.float32)
        ret  = cb.fit(vecs, seed=1)
        assert ret is cb

    def test_encode_decode_roundtrip(self):
        cb   = self._make(n_codes=16)
        vecs = np.random.rand(50, 4).astype(np.float32)
        cb.fit(vecs, seed=2)
        indices = cb.encode(vecs)
        assert indices.dtype == np.uint16
        assert indices.shape == (50,)
        decoded = cb.decode(indices)
        assert decoded.shape == (50, 4)
        err = np.mean((decoded - vecs) ** 2)
        # Reconstruction error should be smaller than a naive baseline
        baseline = np.mean((vecs - 0.5) ** 2)
        assert err < baseline

    def test_quantization_error_updated(self):
        cb   = self._make()
        vecs = np.random.rand(30, 4).astype(np.float32)
        cb.fit(vecs, seed=3)
        assert cb.quantization_error >= 0.0

    def test_online_update_changes_centroids(self):
        cb   = self._make(learning_rate=0.9)
        vecs = np.random.rand(30, 4).astype(np.float32)
        cb.fit(vecs, seed=4)
        old_ctrs = cb.centroids.copy()
        new_vecs = np.ones((20, 4), dtype=np.float32) * 99.0
        cb.online_update(new_vecs)
        assert not np.allclose(cb.centroids, old_ctrs)

    def test_drift_score_non_negative(self):
        cb   = self._make()
        vecs = np.random.rand(30, 4).astype(np.float32)
        cb.fit(vecs, seed=5)
        cb.online_update(vecs)
        assert cb.drift_score >= 0.0

    def test_online_update_before_fit_raises(self):
        cb = self._make()
        with pytest.raises(RuntimeError):
            cb.online_update(np.random.rand(5, 4).astype(np.float32))


class TestParisKVValidationAndEdgeCases:
    def test_dim_lt_1_raises(self):
        """Line 167: dim < 1 → ValueError."""
        with pytest.raises(ValueError, match="dim"):
            ParisKVCodebook(dim=0)

    def test_n_codes_lt_1_raises(self):
        """Line 169: n_codes < 1 → ValueError."""
        with pytest.raises(ValueError, match="n_codes"):
            ParisKVCodebook(dim=4, n_codes=0)

    def test_fit_wrong_shape_raises(self):
        """Line 200: vectors with wrong dim → ValueError."""
        cb   = ParisKVCodebook(dim=4)
        bad  = np.ones((10, 9), dtype=np.float32)   # second dim = 9 ≠ 4
        with pytest.raises(ValueError, match="Expected"):
            cb.fit(bad)

    def test_fit_loop_exhausts_without_convergence(self):
        """Branch 209→223: refine_iters=1 → loop runs once without triggering
        the early break (data doesn't converge fully in 1 step)."""
        cfg  = ParisKVConfig(refine_iters=1)
        cb   = ParisKVCodebook(dim=4, n_codes=4, config=cfg)
        rng  = np.random.default_rng(0)
        vecs = rng.random((20, 4)).astype(np.float32)
        cb.fit(vecs)   # 1 iter → loop body runs once then exits naturally
        assert cb.is_fitted

    def test_fit_empty_cluster_handled(self, monkeypatch):
        """Line 218: empty cluster during fit refinement → centroid unchanged."""
        import squish.paris_kv as _pk

        call_count = [0]
        real_dist  = _pk._pairwise_sq_dist

        def biased_dist(a, b):
            call_count[0] += 1
            d = real_dist(a, b)
            if call_count[0] == 1:
                # Force all points to assign to centroid 0 by making its
                # distances tiny; all other columns become huge.
                d[:, 1:] = 1e9
            return d

        monkeypatch.setattr(_pk, "_pairwise_sq_dist", biased_dist)
        cb   = ParisKVCodebook(dim=4, n_codes=4)
        vecs = np.random.default_rng(7).random((20, 4)).astype(np.float32)
        cb.fit(vecs, seed=7)   # centroids 1,2,3 are empty → line 218 executed
        assert cb.is_fitted

    def test_drift_score_empty_history(self):
        """Line 312: drift_score when _drift_history is empty → 0.0."""
        cb   = ParisKVCodebook(dim=4)
        vecs = np.random.rand(20, 4).astype(np.float32)
        cb.fit(vecs)   # fit does NOT populate drift_history
        assert cb.drift_score == 0.0

    def test_quantization_error_after_online_update(self):
        """Line 320: quantization_error after online_update → non-zero float."""
        cb   = ParisKVCodebook(dim=4)
        vecs = np.random.rand(20, 4).astype(np.float32)
        cb.fit(vecs)
        cb.online_update(vecs)
        assert cb.quantization_error >= 0.0

    def test_drift_history_pop_when_window_exceeded(self):
        """Line 299: _drift_history.pop(0) when len > drift_window."""
        cfg  = ParisKVConfig(drift_window=2)         # tiny window
        cb   = ParisKVCodebook(dim=4, config=cfg)
        vecs = np.random.rand(20, 4).astype(np.float32)
        cb.fit(vecs)
        # Call online_update 3 times → history will exceed drift_window=2
        for _ in range(3):
            cb.online_update(vecs)
        assert len(cb._drift_history) <= 2
