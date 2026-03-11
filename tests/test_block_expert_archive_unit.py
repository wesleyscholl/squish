"""
tests/test_block_expert_archive_unit.py

Unit tests for squish.block_expert_archive — Block-Expert Archive + K-means.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.block_expert_archive import (
    ArchiveStats,
    BlockExpertArchive,
    BlockExpertConfig,
    ExpertRouter,
    ExpertRoutingStats,
    _delta_snr_db,
    archive_manifest_hash,
    cluster_block_weights,
    pack_expert_delta,
    unpack_expert_delta,
)

# ─────────────────────────────────────────────────────────────────────────────
# BlockExpertConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockExpertConfig:
    def test_defaults(self):
        cfg = BlockExpertConfig()
        assert cfg.n_clusters == 4
        assert cfg.n_iter == 20
        assert cfg.similarity_metric == "cosine"
        assert cfg.delta_bits == 8
        assert cfg.min_delta_snr_db == 30.0
        assert cfg.router_temperature == 0.5

    def test_custom_values(self):
        cfg = BlockExpertConfig(n_clusters=8, delta_bits=4, similarity_metric="l2")
        assert cfg.n_clusters == 8
        assert cfg.delta_bits == 4
        assert cfg.similarity_metric == "l2"

    def test_invalid_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters"):
            BlockExpertConfig(n_clusters=0)

    def test_invalid_n_iter(self):
        with pytest.raises(ValueError, match="n_iter"):
            BlockExpertConfig(n_iter=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="similarity_metric"):
            BlockExpertConfig(similarity_metric="dot")

    def test_invalid_delta_bits(self):
        with pytest.raises(ValueError, match="delta_bits"):
            BlockExpertConfig(delta_bits=16)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="router_temperature"):
            BlockExpertConfig(router_temperature=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# pack_expert_delta / unpack_expert_delta
# ─────────────────────────────────────────────────────────────────────────────


class TestPackUnpack:
    def _make_weights(self, rows=8, cols=16, seed=0):
        rng = np.random.default_rng(seed)
        base    = rng.standard_normal((rows, cols)).astype(np.float32)
        expert  = base + rng.standard_normal((rows, cols)).astype(np.float32) * 0.1
        return base, expert

    def test_pack_unpack_8bit_roundtrip(self):
        base, expert = self._make_weights()
        delta_q, scales, zeros = pack_expert_delta(expert, base, bits=8)
        recon = unpack_expert_delta(delta_q, scales, zeros, base, bits=8)
        assert recon.shape == expert.shape
        snr = _delta_snr_db(expert, recon)
        assert snr > 20.0, f"SNR too low: {snr:.1f} dB"

    def test_pack_unpack_4bit_roundtrip(self):
        base, expert = self._make_weights(rows=4, cols=8)
        orig_ncols = expert.shape[1]
        delta_q, scales, zeros = pack_expert_delta(expert, base, bits=4)
        recon = unpack_expert_delta(delta_q, scales, zeros, base, bits=4,
                                    original_ncols=orig_ncols)
        assert recon.shape == expert.shape
        snr = _delta_snr_db(expert, recon)
        assert snr > 10.0, f"SNR too low for 4-bit: {snr:.1f} dB"

    def test_pack_identity_delta(self):
        """When expert == base, delta should be zero → SNR = inf."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal((4, 8)).astype(np.float32)
        delta_q, scales, zeros = pack_expert_delta(base, base, bits=8)
        recon = unpack_expert_delta(delta_q, scales, zeros, base, bits=8)
        snr = _delta_snr_db(base, recon)
        assert snr > 30.0

    def test_pack_shape_mismatch_raises(self):
        rng = np.random.default_rng(1)
        base   = rng.standard_normal((4, 8)).astype(np.float32)
        expert = rng.standard_normal((4, 9)).astype(np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            pack_expert_delta(expert, base, bits=8)

    def test_scales_positive(self):
        base, expert = self._make_weights()
        _, scales, _ = pack_expert_delta(expert, base, bits=8)
        assert np.all(scales >= 0)

    def test_8bit_delta_dtype(self):
        base, expert = self._make_weights()
        delta_q, _, _ = pack_expert_delta(expert, base, bits=8)
        assert delta_q.dtype == np.int8


# ─────────────────────────────────────────────────────────────────────────────
# cluster_block_weights
# ─────────────────────────────────────────────────────────────────────────────


class TestClusterBlockWeights:
    def _random_snapshots(self, n=12, rows=8, cols=16, seed=42):
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((rows, cols)).astype(np.float32) for _ in range(n)]

    def test_returns_correct_k(self):
        snaps = self._random_snapshots(n=10)
        labels, centroids = cluster_block_weights(snaps, n_clusters=4)
        assert len(centroids) == 4
        assert len(labels) == 10

    def test_labels_in_range(self):
        snaps = self._random_snapshots(n=8)
        labels, _ = cluster_block_weights(snaps, n_clusters=3)
        assert all(0 <= l < 3 for l in labels)

    def test_centroids_shape(self):
        snaps = self._random_snapshots(n=6, rows=4, cols=8)
        _, centroids = cluster_block_weights(snaps, n_clusters=2)
        for c in centroids:
            assert c.shape == (4, 8)

    def test_single_snapshot(self):
        snap = np.ones((4, 8), dtype=np.float32)
        labels, centroids = cluster_block_weights([snap], n_clusters=3)
        assert len(labels) == 1
        assert len(centroids) == 3  # padded to k

    def test_l2_metric(self):
        snaps = self._random_snapshots(n=8)
        labels, centroids = cluster_block_weights(snaps, n_clusters=3, metric="l2")
        assert len(centroids) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="weight_snapshots must not be empty"):
            cluster_block_weights([], n_clusters=2)

    def test_reproducible_with_seed(self):
        snaps = self._random_snapshots(n=12)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        labels1, _ = cluster_block_weights(snaps, n_clusters=3, rng=rng1)
        labels2, _ = cluster_block_weights(snaps, n_clusters=3, rng=rng2)
        assert labels1 == labels2


# ─────────────────────────────────────────────────────────────────────────────
# ExpertRouter
# ─────────────────────────────────────────────────────────────────────────────


class TestExpertRouter:
    def _make_router(self, n_blocks=3, n_clusters=4, dim=16):
        rng = np.random.default_rng(0)
        table = {}
        for bi in range(n_blocks):
            centroids = []
            for _ in range(n_clusters):
                v = rng.standard_normal(dim).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                centroids.append(v)
            table[bi] = centroids
        cfg = BlockExpertConfig(n_clusters=n_clusters)
        return ExpertRouter(table, cfg)

    def test_route_returns_valid_cluster(self):
        router = self._make_router(n_clusters=4, dim=16)
        rng = np.random.default_rng(1)
        weight = rng.standard_normal((8, 16)).astype(np.float32)
        cluster, stats = router.route(0, weight)
        assert 0 <= cluster < 4

    def test_route_stats_type(self):
        router = self._make_router()
        weight = np.ones((8, 16), dtype=np.float32)
        _, stats = router.route(1, weight)
        assert isinstance(stats, ExpertRoutingStats)
        assert stats.routing_time_us >= 0

    def test_route_invalid_block_raises(self):
        router = self._make_router(n_blocks=2)
        weight = np.ones((4, 8), dtype=np.float32)
        with pytest.raises(KeyError):
            router.route(99, weight)

    def test_n_blocks(self):
        router = self._make_router(n_blocks=5)
        assert router.n_blocks == 5

    def test_call_count_increments(self):
        router = self._make_router()
        w = np.ones((4, 16), dtype=np.float32)
        router.route(0, w)
        router.route(0, w)
        assert router.call_count == 2

    def test_reset_stats(self):
        router = self._make_router()
        router.route(0, np.ones((4, 16), dtype=np.float32))
        router.reset_stats()
        assert router.call_count == 0

    def test_logit_scores_sum_to_one(self):
        router = self._make_router()
        _, stats = router.route(0, np.ones((4, 16), dtype=np.float32))
        total = sum(stats.logit_scores)
        assert abs(total - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# ArchiveStats
# ─────────────────────────────────────────────────────────────────────────────


class TestArchiveStats:
    def test_defaults(self):
        s = ArchiveStats()
        assert s.n_blocks == 0
        assert s.n_experts_total == 0
        assert s.avg_experts_per_block == 0.0

    def test_avg_experts_per_block(self):
        s = ArchiveStats(n_blocks=4, n_experts_total=16)
        assert s.avg_experts_per_block == 4.0

    def test_avg_experts_with_zero_blocks(self):
        s = ArchiveStats(n_blocks=0, n_experts_total=0)
        assert s.avg_experts_per_block == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BlockExpertArchive — create / save / load / round-trip
# ─────────────────────────────────────────────────────────────────────────────


def _make_block_weights(n_blocks=3, n_snaps=6, rows=8, cols=16, seed=0):
    rng = np.random.default_rng(seed)
    block_weights = {}
    base_weights  = {}
    for bi in range(n_blocks):
        base = rng.standard_normal((rows, cols)).astype(np.float32)
        snaps = [base + rng.standard_normal((rows, cols)).astype(np.float32) * 0.05
                 for _ in range(n_snaps)]
        block_weights[bi] = snaps
        base_weights[bi]  = base
    return block_weights, base_weights


class TestBlockExpertArchiveCreate:
    def test_create_populates_stats(self):
        bw, base = _make_block_weights(n_blocks=2, n_snaps=5)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            assert arc.stats.n_blocks == 2
            assert arc.stats.n_experts_total > 0

    def test_create_num_blocks(self):
        bw, base = _make_block_weights(n_blocks=4)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            assert arc.num_blocks() == 4

    def test_create_num_experts_per_block(self):
        cfg = BlockExpertConfig(n_clusters=3)
        bw, base = _make_block_weights(n_blocks=2, n_snaps=6)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base, config=cfg)
            for bi in range(2):
                assert arc.num_experts(bi) == 3

    def test_get_expert_weight_valid(self):
        bw, base = _make_block_weights(n_blocks=2, n_snaps=5)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            w = arc.get_expert_weight(0, 0, base[0])
            assert w.shape == base[0].shape

    def test_get_expert_weight_invalid_block_raises(self):
        bw, base = _make_block_weights(n_blocks=1)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            with pytest.raises(KeyError):
                arc.get_expert_weight(99, 0, base[0])

    def test_summary_contains_keys(self):
        bw, base = _make_block_weights(n_blocks=2)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            s = arc.summary()
            for key in ("n_blocks", "n_experts_total", "avg_delta_snr_db", "config"):
                assert key in s


class TestBlockExpertArchivePersistence:
    def test_save_creates_manifest(self):
        bw, base = _make_block_weights(n_blocks=2)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            arc.save()
            assert (Path(td) / "manifest.json").is_file()

    def test_save_creates_experts_dir(self):
        bw, base = _make_block_weights(n_blocks=2)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            arc.save()
            assert (Path(td) / "experts").is_dir()

    def test_save_creates_router_json(self):
        bw, base = _make_block_weights(n_blocks=2)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            arc.save()
            assert (Path(td) / "router.json").is_file()

    def test_load_roundtrip(self):
        bw, base = _make_block_weights(n_blocks=3, n_snaps=4)
        with tempfile.TemporaryDirectory() as td:
            arc1 = BlockExpertArchive.create(td, bw, base)
            arc1.save()
            arc2 = BlockExpertArchive.load(td)
            assert arc2.stats.n_blocks == arc1.stats.n_blocks
            assert arc2.stats.n_experts_total == arc1.stats.n_experts_total

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            BlockExpertArchive.load("/tmp/does_not_exist_squish_test")

    def test_manifest_hash(self):
        bw, base = _make_block_weights(n_blocks=2)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            arc.save()
            h = archive_manifest_hash(td)
            assert len(h) == 16

    def test_4bit_save_load_roundtrip(self):
        cfg = BlockExpertConfig(delta_bits=4, n_clusters=2)
        bw, base = _make_block_weights(n_blocks=2, n_snaps=4, rows=4, cols=8)
        with tempfile.TemporaryDirectory() as td:
            arc1 = BlockExpertArchive.create(td, bw, base, config=cfg)
            arc1.save()
            arc2 = BlockExpertArchive.load(td)
            assert arc2.config.delta_bits == 4


class TestBlockExpertArchiveAbsorb:
    def test_absorb_snapshot_new_block(self):
        bw, base = _make_block_weights(n_blocks=1, n_snaps=4)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            rng = np.random.default_rng(99)
            new_w = rng.standard_normal((8, 16)).astype(np.float32)
            # Block 99 not in archive — should create new entry
            k = arc.absorb_snapshot(99, new_w, np.zeros_like(new_w))
            assert k == 0

    def test_absorb_snapshot_existing_block(self):
        bw, base = _make_block_weights(n_blocks=2, n_snaps=5)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            rng = np.random.default_rng(42)
            new_w = rng.standard_normal((8, 16)).astype(np.float32)
            k = arc.absorb_snapshot(0, new_w, base[0])
            assert 0 <= k < arc.config.n_clusters

    def test_absorb_increments_learn_epochs(self):
        bw, base = _make_block_weights(n_blocks=1, n_snaps=4)
        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, bw, base)
            initial_epochs = arc.stats.learn_epochs
            rng = np.random.default_rng(5)
            arc.absorb_snapshot(0, rng.standard_normal((8, 16)).astype(np.float32), base[0])
            assert arc.stats.learn_epochs > initial_epochs
