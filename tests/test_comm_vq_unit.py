"""tests/test_comm_vq_unit.py — 100% coverage for squish/comm_vq.py"""
import numpy as np
import pytest

from squish.comm_vq import CommVQCodebook, MultiCodebookVQ, fit_kv_codebooks


RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# CommVQCodebook — construction
# ---------------------------------------------------------------------------

class TestCommVQCodebookInit:
    def test_defaults(self):
        cb = CommVQCodebook(dim=64, n_codes=16)
        assert cb.dim     == 64
        assert cb.n_codes == 16
        assert cb.centroids is None
        assert cb.bits    == 4   # log2(16)

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim"):
            CommVQCodebook(dim=0)

    def test_invalid_n_codes_not_power_of_2(self):
        with pytest.raises(ValueError, match="n_codes"):
            CommVQCodebook(dim=8, n_codes=3)
        with pytest.raises(ValueError, match="n_codes"):
            CommVQCodebook(dim=8, n_codes=1)

    def test_bits_2bit(self):
        cb = CommVQCodebook(dim=8, n_codes=4)
        assert cb.bits == 2

    def test_bits_8bit(self):
        cb = CommVQCodebook(dim=8, n_codes=256)
        assert cb.bits == 8

    def test_repr(self):
        cb = CommVQCodebook(dim=4, n_codes=4)
        r  = repr(cb)
        assert "unfitted" in r
        assert "dim=4" in r


# ---------------------------------------------------------------------------
# CommVQCodebook — encode/decode before fit raises
# ---------------------------------------------------------------------------

class TestCommVQCodebookUnfitted:
    def test_encode_raises(self):
        cb = CommVQCodebook(dim=8, n_codes=4)
        with pytest.raises(RuntimeError, match="fitted"):
            cb.encode(np.zeros((3, 8), dtype=np.float32))

    def test_decode_raises(self):
        cb = CommVQCodebook(dim=8, n_codes=4)
        with pytest.raises(RuntimeError, match="fitted"):
            cb.decode(np.array([0, 1], dtype=np.uint16))

    def test_quantization_error_raises(self):
        cb = CommVQCodebook(dim=8, n_codes=4)
        with pytest.raises(RuntimeError, match="fitted"):
            cb.quantization_error(np.zeros((3, 8), dtype=np.float32))


# ---------------------------------------------------------------------------
# CommVQCodebook — fit
# ---------------------------------------------------------------------------

class TestCommVQCodebookFit:
    def test_fit_sets_centroids(self):
        cb   = CommVQCodebook(dim=8, n_codes=4)
        vecs = np.random.rand(20, 8).astype(np.float32)
        cb.fit(vecs, n_iters=10, seed=0)
        assert cb.centroids is not None
        assert cb.centroids.shape == (4, 8)

    def test_fit_returns_self(self):
        cb  = CommVQCodebook(dim=8, n_codes=4)
        ret = cb.fit(np.random.rand(20, 8).astype(np.float32))
        assert ret is cb

    def test_fit_wrong_dim_raises(self):
        cb = CommVQCodebook(dim=8, n_codes=4)
        with pytest.raises(ValueError, match="dim"):
            cb.fit(np.random.rand(10, 16).astype(np.float32))

    def test_fit_too_few_vectors_raises(self):
        cb = CommVQCodebook(dim=8, n_codes=16)
        with pytest.raises(ValueError, match="calibration vectors"):
            cb.fit(np.random.rand(4, 8).astype(np.float32))

    def test_repr_fitted(self):
        cb   = CommVQCodebook(dim=4, n_codes=4)
        cb.fit(np.random.rand(10, 4).astype(np.float32))
        assert "fitted" in repr(cb)


# ---------------------------------------------------------------------------
# CommVQCodebook — encode / decode round-trip
# ---------------------------------------------------------------------------

class TestCommVQCodebookEncodeDecode:
    def _make_fitted(self, dim=8, n_codes=8, n=50):
        cb   = CommVQCodebook(dim=dim, n_codes=n_codes)
        vecs = np.random.rand(n, dim).astype(np.float32)
        cb.fit(vecs, seed=0)
        return cb, vecs

    def test_encode_shape_and_dtype(self):
        cb, vecs = self._make_fitted()
        idx      = cb.encode(vecs)
        assert idx.shape == (len(vecs),)
        assert idx.dtype == np.uint16

    def test_decode_shape(self):
        cb, vecs = self._make_fitted()
        idx      = cb.encode(vecs)
        dec      = cb.decode(idx)
        assert dec.shape == (len(vecs), cb.dim)

    def test_encode_1d_vector(self):
        cb, vecs = self._make_fitted()
        idx      = cb.encode(vecs[0])   # 1-D input
        assert idx.shape == (1,)

    def test_reconstruction_better_than_random(self):
        cb, vecs = self._make_fitted(dim=16, n_codes=16, n=100)
        err      = cb.quantization_error(vecs)
        baseline = float(np.mean((vecs - 0.5) ** 2))
        assert err < baseline

    def test_decode_clips_out_of_range_index(self):
        cb, vecs = self._make_fitted()
        # Index 999 is out of range → should be clipped, not raise
        idx = np.array([999], dtype=np.uint16)
        out = cb.decode(idx)
        assert out.shape == (1, cb.dim)


# ---------------------------------------------------------------------------
# CommVQCodebook — helpers
# ---------------------------------------------------------------------------

class TestCommVQHelpers:
    def test_pairwise_sq_dist_self(self):
        vecs = np.eye(4, dtype=np.float32)
        d    = CommVQCodebook._pairwise_sq_dist(vecs, vecs)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-5)

    def test_pairwise_sq_dist_known(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        d = CommVQCodebook._pairwise_sq_dist(a, b)
        assert d[0, 0] == pytest.approx(25.0, abs=1e-4)

    def test_kmeans_plus_plus_k_centers(self):
        vecs = np.random.rand(50, 4).astype(np.float32)
        rng  = np.random.default_rng(1)
        ctrs = CommVQCodebook._kmeans_plus_plus_init(vecs, 8, rng)
        assert ctrs.shape == (8, 4)

    def test_kmeans_plus_plus_degenerate(self):
        """All-same vectors → all-zero squared distances → random fallback."""
        vecs = np.ones((10, 4), dtype=np.float32)
        rng  = np.random.default_rng(0)
        ctrs = CommVQCodebook._kmeans_plus_plus_init(vecs, 3, rng)
        assert ctrs.shape == (3, 4)

    def test_compression_ratio(self):
        cb   = CommVQCodebook(dim=128, n_codes=4)
        # fp16: 128*2 = 256 bytes; vq: 2 bytes (uint16)
        assert cb.compression_ratio == pytest.approx(128.0, rel=0.01)


# ---------------------------------------------------------------------------
# MultiCodebookVQ
# ---------------------------------------------------------------------------

class TestMultiCodebookVQ:
    def _make(self, dim=16, n_sub=4, n_codes=4, n=40):
        mcvq = MultiCodebookVQ(dim=dim, n_subvectors=n_sub, n_codes=n_codes)
        vecs = np.random.rand(n, dim).astype(np.float32)
        return mcvq, vecs

    def test_invalid_dim_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiCodebookVQ(dim=10, n_subvectors=3)

    def test_is_fitted_false_before_fit(self):
        mcvq, _ = self._make()
        assert not mcvq.is_fitted

    def test_fit_returns_self(self):
        mcvq, vecs = self._make()
        ret        = mcvq.fit(vecs)
        assert ret is mcvq

    def test_is_fitted_true_after_fit(self):
        mcvq, vecs = self._make()
        mcvq.fit(vecs)
        assert mcvq.is_fitted

    def test_encode_raises_before_fit(self):
        mcvq, vecs = self._make()
        with pytest.raises(RuntimeError):
            mcvq.encode(vecs)

    def test_encode_shape(self):
        mcvq, vecs = self._make()
        mcvq.fit(vecs)
        idx = mcvq.encode(vecs)
        assert idx.shape == (len(vecs), mcvq.n_subvectors)
        assert idx.dtype == np.uint16

    def test_decode_shape(self):
        mcvq, vecs = self._make()
        mcvq.fit(vecs)
        idx = mcvq.encode(vecs)
        dec = mcvq.decode(idx)
        assert dec.shape == (len(vecs), mcvq.dim)

    def test_roundtrip_lower_error_than_baseline(self):
        mcvq, vecs = self._make(dim=32, n_sub=4, n_codes=16, n=120)
        mcvq.fit(vecs)
        err      = mcvq.quantization_error(vecs)
        baseline = float(np.mean((vecs - 0.5) ** 2))
        assert err < baseline


# ---------------------------------------------------------------------------
# fit_kv_codebooks
# ---------------------------------------------------------------------------

class TestFitKVCodebooks:
    def test_returns_two_codebooks(self):
        keys = np.random.rand(30, 16).astype(np.float32)
        vals = np.random.rand(30, 16).astype(np.float32)
        k_cb, v_cb = fit_kv_codebooks(keys, vals, n_codes=4, n_iters=5)
        assert isinstance(k_cb, CommVQCodebook)
        assert isinstance(v_cb, CommVQCodebook)

    def test_codebooks_are_fitted(self):
        keys = np.random.rand(30, 8).astype(np.float32)
        vals = np.random.rand(30, 8).astype(np.float32)
        k_cb, v_cb = fit_kv_codebooks(keys, vals, n_codes=4)
        assert k_cb.centroids is not None
        assert v_cb.centroids is not None

    def test_codebook_dim_matches(self):
        dim  = 32
        keys = np.random.rand(50, dim).astype(np.float32)
        vals = np.random.rand(50, dim).astype(np.float32)
        k_cb, v_cb = fit_kv_codebooks(keys, vals, n_codes=8)
        assert k_cb.dim == dim
        assert v_cb.dim == dim

    def test_key_and_value_codebooks_differ(self):
        """Different data → different centroids."""
        keys = np.zeros((30, 8), dtype=np.float32)
        vals = np.ones((30, 8), dtype=np.float32)
        k_cb, v_cb = fit_kv_codebooks(keys, vals, n_codes=4)
        assert not np.allclose(k_cb.centroids, v_cb.centroids)


class TestFitLoopBranches:
    def test_fit_loop_exhausts_without_convergence(self):
        """Branch 128→150: loop runs all n_iters without early break."""
        # Use random data that is unlikely to converge in exactly 1 step.
        rng  = np.random.default_rng(42)
        vecs = rng.random((20, 4)).astype(np.float32)
        cb   = CommVQCodebook(dim=4, n_codes=4)
        cb.fit(vecs, n_iters=1, seed=42)   # single iter → shift check never met
        assert cb.centroids.shape == (4, 4)
