"""
tests/test_matryoshka_emb_unit.py

Unit tests for squish/matryoshka_emb.py — 100% coverage.
"""

import math

import numpy as np
import pytest

from squish.matryoshka_emb import MRLConfig, MRLStats, MatryoshkaEmbedding


# ---------------------------------------------------------------------------
# MRLConfig
# ---------------------------------------------------------------------------


class TestMRLConfig:
    def test_defaults(self):
        cfg = MRLConfig()
        assert cfg.full_dim == 1536
        assert isinstance(cfg.nested_dims, list)
        assert cfg.full_dim in cfg.nested_dims
        assert cfg.normalize is True

    def test_nested_dims_sorted_ascending(self):
        cfg = MRLConfig(full_dim=512)
        assert cfg.nested_dims == sorted(cfg.nested_dims)

    def test_custom_nested_dims(self):
        cfg = MRLConfig(full_dim=256, nested_dims=[64, 128, 256])
        assert cfg.nested_dims == [64, 128, 256]

    def test_full_dim_added_to_nested_dims(self):
        cfg = MRLConfig(full_dim=128, nested_dims=[32, 64])
        assert 128 in cfg.nested_dims

    def test_no_duplicate_full_dim_in_nested(self):
        cfg = MRLConfig(full_dim=256, nested_dims=[64, 128, 256])
        assert cfg.nested_dims.count(256) == 1

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"full_dim": 0}, "full_dim"),
            ({"full_dim": -1}, "full_dim"),
            ({"full_dim": 100, "nested_dims": [200]}, "full_dim"),
            ({"full_dim": 100, "nested_dims": [0, 50]}, "0"),
        ],
    )
    def test_validation_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MRLConfig(**kwargs)

    def test_frozen(self):
        cfg = MRLConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.full_dim = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MRLStats
# ---------------------------------------------------------------------------


class TestMRLStats:
    def test_most_used_dim_empty(self):
        s = MRLStats(n_embeds=0, dims_used={})
        assert s.most_used_dim is None

    def test_most_used_dim(self):
        s = MRLStats(n_embeds=5, dims_used={64: 1, 128: 3, 256: 1})
        assert s.most_used_dim == 128


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding.embed
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbeddingEmbed:
    def _emb(self, full_dim=256, nested_dims=None):
        cfg = MRLConfig(full_dim=full_dim, nested_dims=nested_dims)
        return MatryoshkaEmbedding(cfg)

    def test_embed_1d_shape(self):
        emb = self._emb(256, [64, 128, 256])
        x = np.random.randn(256).astype(np.float32)
        out = emb.embed(x, target_dim=64)
        assert out.shape == (64,)

    def test_embed_1d_normalized(self):
        emb = self._emb(256, [64, 128, 256])
        x = np.random.randn(256).astype(np.float32) * 100
        out = emb.embed(x, target_dim=64)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_embed_no_normalize(self):
        cfg = MRLConfig(full_dim=256, nested_dims=[64, 256], normalize=False)
        emb = MatryoshkaEmbedding(cfg)
        x = np.ones(256, dtype=np.float32) * 2.0
        out = emb.embed(x, target_dim=64)
        assert abs(np.linalg.norm(out) - 2.0 * math.sqrt(64)) < 1e-3

    def test_embed_default_dim_is_full_dim(self):
        emb = self._emb(256, [64, 256])
        x = np.random.randn(256).astype(np.float32)
        out = emb.embed(x)
        assert out.shape == (256,)

    def test_embed_2d_batch(self):
        emb = self._emb(128, [32, 64, 128])
        X = np.random.randn(5, 128).astype(np.float32)
        out = emb.embed(X, target_dim=32)
        assert out.shape == (5, 32)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embed_invalid_target_dim_raises(self):
        emb = self._emb(256, [64, 128, 256])
        x = np.random.randn(256).astype(np.float32)
        with pytest.raises(ValueError, match="target_dim"):
            emb.embed(x, target_dim=100)

    def test_embed_wrong_1d_size_raises(self):
        emb = self._emb(256, [64, 256])
        x = np.random.randn(128).astype(np.float32)
        with pytest.raises(ValueError):
            emb.embed(x, target_dim=64)

    def test_embed_wrong_2d_width_raises(self):
        emb = self._emb(256, [64, 256])
        X = np.random.randn(3, 128).astype(np.float32)
        with pytest.raises(ValueError):
            emb.embed(X, target_dim=64)

    def test_embed_3d_raises(self):
        emb = self._emb(256, [64, 256])
        X = np.random.randn(2, 3, 256).astype(np.float32)
        with pytest.raises(ValueError):
            emb.embed(X, target_dim=64)


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding.batch_embed
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbeddingBatchEmbed:
    def test_batch_embed_shape(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[16, 32, 64])
        emb = MatryoshkaEmbedding(cfg)
        X = np.random.randn(4, 64).astype(np.float32)
        out = emb.batch_embed(X, target_dim=16)
        assert out.shape == (4, 16)

    def test_batch_embed_1d_raises(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[32, 64])
        emb = MatryoshkaEmbedding(cfg)
        with pytest.raises(ValueError, match="2-D"):
            emb.batch_embed(np.random.randn(64).astype(np.float32))


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding.similarity
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbeddingSimilarity:
    def _emb(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[16, 32, 64])
        return MatryoshkaEmbedding(cfg)

    def test_self_similarity_is_one(self):
        emb = self._emb()
        x = np.random.randn(64).astype(np.float32)
        s = emb.similarity(x, x, dim=32)
        assert abs(s - 1.0) < 1e-4

    def test_anti_similarity_is_neg_one(self):
        emb = self._emb()
        x = np.random.randn(64).astype(np.float32)
        s = emb.similarity(x, -x, dim=32)
        assert abs(s + 1.0) < 1e-4

    def test_similarity_default_dim(self):
        emb = self._emb()
        x = np.random.randn(64).astype(np.float32)
        s = emb.similarity(x, x)
        assert abs(s - 1.0) < 1e-4

    def test_similarity_no_normalize(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[16, 64], normalize=False)
        emb = MatryoshkaEmbedding(cfg)
        x = np.ones(64, dtype=np.float32)
        s = emb.similarity(x, x, dim=16)
        assert abs(s - 1.0) < 1e-4

    def test_zero_vector_similarity(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[16, 64], normalize=False)
        emb = MatryoshkaEmbedding(cfg)
        x = np.zeros(64, dtype=np.float32)
        s = emb.similarity(x, x, dim=16)
        assert s == 0.0


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding.nearest_dim
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbeddingNearestDim:
    def _emb(self):
        cfg = MRLConfig(full_dim=512, nested_dims=[64, 128, 256, 512])
        return MatryoshkaEmbedding(cfg)

    def test_exact_match(self):
        assert self._emb().nearest_dim(128) == 128

    def test_rounds_up(self):
        assert self._emb().nearest_dim(100) == 128

    def test_smaller_than_all_returns_smallest(self):
        assert self._emb().nearest_dim(1) == 64

    def test_larger_than_all_returns_largest(self):
        assert self._emb().nearest_dim(1000) == 512


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding.stats tracking
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbeddingStats:
    def test_stats_tracked(self):
        cfg = MRLConfig(full_dim=64, nested_dims=[16, 32, 64])
        emb = MatryoshkaEmbedding(cfg)
        x = np.random.randn(64).astype(np.float32)
        emb.embed(x, target_dim=16)
        emb.embed(x, target_dim=16)
        emb.embed(x, target_dim=32)
        s = emb.stats()
        assert s.n_embeds == 3
        assert s.dims_used.get(16) == 2
        assert s.dims_used.get(32) == 1
        assert s.most_used_dim == 16
