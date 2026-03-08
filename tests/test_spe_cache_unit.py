"""tests/test_spe_cache_unit.py — 100% coverage for squish/spe_cache.py"""
import threading
import numpy as np
import pytest

from squish.spe_cache import (
    SpeCacheConfig,
    BlockScoreTracker,
    SpeCachePrefetcher,
    InMemoryBlockStore,
)


# ---------------------------------------------------------------------------
# SpeCacheConfig
# ---------------------------------------------------------------------------

class TestSpeCacheConfig:
    def test_defaults(self):
        cfg = SpeCacheConfig()
        assert cfg.block_size == 64
        assert cfg.prefetch_budget == 8
        assert cfg.sink_blocks == 1
        assert cfg.alpha_recency == 0.4
        assert cfg.beta_attention == 0.6
        assert cfg.top_k_attention == 32

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            SpeCacheConfig(block_size=0)

    def test_invalid_prefetch_budget(self):
        with pytest.raises(ValueError, match="prefetch_budget"):
            SpeCacheConfig(prefetch_budget=-1)

    def test_invalid_sink_blocks(self):
        with pytest.raises(ValueError, match="sink_blocks"):
            SpeCacheConfig(sink_blocks=-1)

    def test_invalid_alpha_recency(self):
        with pytest.raises(ValueError, match="alpha_recency"):
            SpeCacheConfig(alpha_recency=-0.1)
        with pytest.raises(ValueError, match="alpha_recency"):
            SpeCacheConfig(alpha_recency=1.1)

    def test_invalid_beta_attention(self):
        with pytest.raises(ValueError, match="beta_attention"):
            SpeCacheConfig(beta_attention=-0.1)

    def test_invalid_top_k_attention(self):
        with pytest.raises(ValueError, match="top_k_attention"):
            SpeCacheConfig(top_k_attention=0)


# ---------------------------------------------------------------------------
# BlockScoreTracker
# ---------------------------------------------------------------------------

class TestBlockScoreTracker:
    def test_record_accumulates(self):
        tracker = BlockScoreTracker(block_size=4, top_k=4)
        scores = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        tracker.record(scores)
        result = tracker.get_scores()
        assert len(result) > 0
        assert tracker.step_count == 1

    def test_reset_clears(self):
        tracker = BlockScoreTracker(block_size=4, top_k=4)
        tracker.record(np.ones(8, dtype=np.float32))
        tracker.reset()
        assert len(tracker.get_scores()) == 0
        assert tracker.step_count == 0

    def test_empty_input_no_error(self):
        tracker = BlockScoreTracker(block_size=4, top_k=4)
        tracker.record(np.array([], dtype=np.float32))
        assert tracker.step_count == 0

    def test_normalisation_uniform(self):
        tracker = BlockScoreTracker(block_size=1, top_k=10)
        scores = np.ones(10, dtype=np.float32)
        tracker.record(scores)
        result = tracker.get_scores()
        total = sum(result.values())
        # Should be normalised
        assert total <= 1.01

    def test_all_zero_scores_handled(self):
        tracker = BlockScoreTracker(block_size=4, top_k=4)
        scores = np.zeros(8, dtype=np.float32)
        tracker.record(scores)  # should not raise

    def test_top_k_limits_blocks(self):
        tracker = BlockScoreTracker(block_size=1, top_k=2)
        # 8 blocks, but only top-2 should be recorded per step
        scores = np.array([0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01],
                          dtype=np.float32)
        tracker.record(scores)
        result = tracker.get_scores()
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# InMemoryBlockStore
# ---------------------------------------------------------------------------

class TestInMemoryBlockStore:
    def test_store_and_load(self):
        store = InMemoryBlockStore(block_size=64)
        store.store(0, b"data")
        assert store.load(0) == b"data"

    def test_load_missing_returns_none(self):
        store = InMemoryBlockStore()
        assert store.load(99) is None

    def test_prefetch_marks_block(self):
        store = InMemoryBlockStore()
        store.prefetch(5)
        assert store.is_prefetched(5)
        assert not store.is_prefetched(6)

    def test_len(self):
        store = InMemoryBlockStore()
        store.store(0, "a")
        store.store(1, "b")
        assert len(store) == 2


# ---------------------------------------------------------------------------
# SpeCachePrefetcher
# ---------------------------------------------------------------------------

class TestSpeCachePrefetcher:
    def _make(self, **kw):
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=4, sink_blocks=1, **kw)
        store = InMemoryBlockStore()
        return SpeCachePrefetcher(cfg, store), store

    def test_predict_returns_list(self):
        pref, _ = self._make()
        blocks = pref.predict_next_turn_blocks(total_blocks=8)
        assert isinstance(blocks, list)

    def test_predict_empty_when_no_blocks(self):
        pref, _ = self._make()
        blocks = pref.predict_next_turn_blocks(total_blocks=0)
        assert blocks == []

    def test_prefetch_called(self):
        pref, store = self._make()
        pref.prefetch(3)
        assert store.is_prefetched(3)

    def test_prefetch_batch(self):
        pref, store = self._make()
        pref.prefetch_batch([0, 1, 2])
        for i in range(3):
            assert store.is_prefetched(i)

    def test_end_of_turn_resets_tracker(self):
        pref, _ = self._make()
        pref.record_attention(np.ones(16, dtype=np.float32))
        pref.end_of_turn(total_blocks=4)
        assert pref.step_count == 0   # reset

    def test_step_count_increments(self):
        pref, _ = self._make()
        pref.record_attention(np.ones(8, dtype=np.float32))
        assert pref.step_count == 1

    def test_sink_blocks_always_included(self):
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=4, sink_blocks=2,
                               alpha_recency=0.0, beta_attention=0.0)
        store = InMemoryBlockStore()
        pref  = SpeCachePrefetcher(cfg, store)
        blocks = pref.predict_next_turn_blocks(total_blocks=10)
        # Sink blocks 0 and 1 must be present
        assert 0 in blocks
        assert 1 in blocks

    def test_attention_based_scoring(self):
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=3, sink_blocks=0,
                               alpha_recency=0.0, beta_attention=1.0)
        store = InMemoryBlockStore()
        pref  = SpeCachePrefetcher(cfg, store)
        # Heavy attention on block 2 (tokens 8-11)
        attn = np.zeros(20, dtype=np.float32)
        attn[8:12] = 1.0
        pref.record_attention(attn)
        blocks = pref.predict_next_turn_blocks(total_blocks=5)
        assert 2 in blocks

    def test_thread_safety(self):
        pref, _ = self._make()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    pref.record_attention(np.random.rand(16).astype(np.float32))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_candidates_loop_exhausts_naturally(self):
        """Branch 298→305: for loop exits without break (budget > len(candidates))."""
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=100, sink_blocks=0,
                               alpha_recency=1.0, beta_attention=0.0)
        store = InMemoryBlockStore()
        pref  = SpeCachePrefetcher(cfg, store)
        # Only 3 blocks → 3 candidates, budget 100 → loop never breaks
        blocks = pref.predict_next_turn_blocks(total_blocks=3)
        assert sorted(blocks) == [0, 1, 2]

    def test_sink_block_skipped_in_candidates_loop(self):
        """Branch 301→298: block_id already in seen (sink) → skipped in second loop."""
        # Sink blocks 0 and 1 appear later in sorted candidates → False branch hit.
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=10, sink_blocks=2,
                               alpha_recency=1.0, beta_attention=0.0)
        store = InMemoryBlockStore()
        pref  = SpeCachePrefetcher(cfg, store)
        # 3 blocks total; candidates sorted descending by recency: [2, 1, 0]
        # After sinks (0, 1) are added to seen, iterating: block 2 added, then
        # block 1 → already in seen (False branch), block 0 → already in seen.
        blocks = pref.predict_next_turn_blocks(total_blocks=3)
        assert 0 in blocks
        assert 1 in blocks
        assert 2 in blocks
