"""tests/test_head_infer_unit.py — 100% coverage for squish/head_infer.py"""
import numpy as np
import pytest

from squish.head_infer import (
    HeadType,
    HeadInferConfig,
    HeadClassifier,
    HeadAwareKVStore,
    _HeadBuffer,
)


# ---------------------------------------------------------------------------
# HeadInferConfig
# ---------------------------------------------------------------------------

class TestHeadInferConfig:
    def test_defaults(self):
        cfg = HeadInferConfig()
        assert cfg.n_layers == 32
        assert cfg.n_heads  == 32
        assert cfg.window_size == 512
        assert cfg.sink_tokens == 4

    def test_custom(self):
        cfg = HeadInferConfig(n_layers=4, n_heads=4, window_size=16,
                              sink_tokens=2, retrieval_threshold=0.2,
                              top_k_retrieval=8)
        assert cfg.n_layers == 4

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            HeadInferConfig(n_layers=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            HeadInferConfig(n_heads=0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            HeadInferConfig(window_size=0)

    def test_invalid_sink_tokens(self):
        with pytest.raises(ValueError, match="sink_tokens"):
            HeadInferConfig(sink_tokens=-1)

    def test_invalid_retrieval_threshold_zero(self):
        with pytest.raises(ValueError, match="retrieval_threshold"):
            HeadInferConfig(retrieval_threshold=0.0)

    def test_invalid_retrieval_threshold_one(self):
        with pytest.raises(ValueError, match="retrieval_threshold"):
            HeadInferConfig(retrieval_threshold=1.0)

    def test_invalid_top_k_retrieval(self):
        with pytest.raises(ValueError, match="top_k_retrieval"):
            HeadInferConfig(top_k_retrieval=-1)


# ---------------------------------------------------------------------------
# HeadClassifier
# ---------------------------------------------------------------------------

class TestHeadClassifier:
    def _make(self, n_layers=2, n_heads=4):
        cfg = HeadInferConfig(n_layers=n_layers, n_heads=n_heads,
                              window_size=4, sink_tokens=1,
                              retrieval_threshold=0.1)
        return HeadClassifier(cfg)

    def test_defaults_to_unknown(self):
        hc = self._make()
        assert hc.head_types[0][0] == HeadType.UNKNOWN

    def test_label_unknown_returns_retrieval(self):
        hc = self._make()
        # Unknown defaults to RETRIEVAL
        assert hc.label(0, 0) == HeadType.RETRIEVAL

    def test_out_of_range_label(self):
        hc = self._make()
        assert hc.label(100, 0) == HeadType.RETRIEVAL
        assert hc.label(0, 100) == HeadType.RETRIEVAL

    def test_calibrate_streaming(self):
        # Attention concentrated at the end → streaming
        hc = self._make(n_layers=1, n_heads=2)
        seq = 20
        # Head 0: all mass on last 4 tokens (within window) → streaming
        attn = np.zeros((2, seq, seq), dtype=np.float32)
        attn[0, :, -4:] = 1.0   # recent mass
        attn[1, :, :4]  = 1.0   # remote mass → retrieval
        hc.calibrate([attn])
        assert hc.head_types[0][0] == HeadType.STREAMING
        assert hc.head_types[0][1] == HeadType.RETRIEVAL

    def test_calibrate_verbose(self, capsys):
        hc = self._make(n_layers=1, n_heads=2)
        seq = 10
        attn = np.zeros((2, seq, seq), dtype=np.float32)
        attn[:, :, :] = 1.0
        hc.calibrate([attn], verbose=True)
        out = capsys.readouterr().out
        assert "Layer 0" in out

    def test_calibrate_2d_input(self):
        hc = self._make(n_layers=1, n_heads=2)
        attn = np.zeros((2, 10), dtype=np.float32)
        attn[:, :2] = 1.0   # remote mass
        hc.calibrate([attn])
        # Both heads should be classified (not UNKNOWN)
        assert hc.head_types[0][0] != HeadType.UNKNOWN

    def test_calibrate_skips_extra_layers(self):
        hc = self._make(n_layers=1, n_heads=2)
        attn = np.zeros((2, 10, 10), dtype=np.float32)
        # Pass 3 layers, but index only has 1
        hc.calibrate([attn, attn, attn])
        # Should not raise

    def test_calibrate_wrong_ndim_skipped(self):
        hc = self._make(n_layers=1, n_heads=2)
        bad = np.zeros((2, 10, 10, 10), dtype=np.float32)
        hc.calibrate([bad])   # should not raise

    def test_to_labels_array(self):
        hc = self._make(n_layers=1, n_heads=2)
        hc.head_types[0][0] = HeadType.STREAMING
        hc.head_types[0][1] = HeadType.RETRIEVAL
        arr = hc.to_labels_array()
        assert arr[0, 0] == 0   # STREAMING
        assert arr[0, 1] == 1   # RETRIEVAL

    def test_from_labels_array(self):
        hc = self._make(n_layers=1, n_heads=2)
        arr = np.array([[0, 1]], dtype=np.int8)
        hc.from_labels_array(arr)
        assert hc.head_types[0][0] == HeadType.STREAMING
        assert hc.head_types[0][1] == HeadType.RETRIEVAL

    def test_label_returns_classified_type_directly(self):
        """Line 208: label() returns ht when ht != UNKNOWN (no early-return)."""
        hc = self._make(n_layers=1, n_heads=1)
        hc.head_types[0][0] = HeadType.STREAMING
        assert hc.label(0, 0) == HeadType.STREAMING

    def test_retrieval_score_zero_seq_len(self):
        """Line 219: _retrieval_score with seq_len=0 → 0.0 score → STREAMING."""
        hc    = self._make(n_layers=1, n_heads=1)
        # Pass attention with shape (1, 0): 1 head, 0 sequence positions
        empty = np.zeros((1, 0), dtype=np.float32)
        hc.calibrate([empty])
        # score = 0.0 < retrieval_threshold → classified as STREAMING
        assert hc.head_types[0][0] == HeadType.STREAMING


# ---------------------------------------------------------------------------
# HeadAwareKVStore branch: non-UNKNOWN head type provided
# ---------------------------------------------------------------------------

class TestHeadAwareKVStoreNonUnknownType:
    def test_non_unknown_head_type_used_directly(self):
        """Branch 367→371: head_types provided with non-UNKNOWN → False branch
        of 'if ht == HeadType.UNKNOWN' → ht used as-is."""
        cfg = HeadInferConfig(n_layers=1, n_heads=1, window_size=4, sink_tokens=1)
        store = HeadAwareKVStore(cfg, head_types=[[HeadType.RETRIEVAL]])
        # No error; buffer created with RETRIEVAL type
        store.put(0, 0, np.ones(4), np.ones(4))
        k, _ = store.get(0, 0)
        assert k.shape[0] == 1


# ---------------------------------------------------------------------------
# _HeadBuffer
# ---------------------------------------------------------------------------

class TestHeadBuffer:
    def test_streaming_eviction(self):
        buf = _HeadBuffer(HeadType.STREAMING, window=2, sinks=1, max_entries=1000)
        for i in range(10):
            buf.put(np.array([float(i)]), np.array([float(i)]))
        # Should keep sink (1) + window (2) = 3 entries
        assert len(buf) == 3

    def test_retrieval_eviction(self):
        buf = _HeadBuffer(HeadType.RETRIEVAL, window=2, sinks=1, max_entries=5)
        for i in range(10):
            buf.put(np.array([float(i)]), np.array([float(i)]))
        assert len(buf) == 5

    def test_get_empty_returns_empty(self):
        buf = _HeadBuffer(HeadType.STREAMING, window=4, sinks=1, max_entries=100)
        k, v = buf.get()
        assert k.shape[0] == 0

    def test_get_streaming_no_query(self):
        buf = _HeadBuffer(HeadType.STREAMING, window=4, sinks=1, max_entries=100)
        buf.put(np.ones(4), np.ones(4))
        k, v = buf.get()
        assert k.shape[0] == 1

    def test_get_retrieval_with_query(self):
        buf = _HeadBuffer(HeadType.RETRIEVAL, window=4, sinks=1, max_entries=100)
        for i in range(5):
            buf.put(np.random.rand(4).astype(np.float32),
                    np.random.rand(4).astype(np.float32))
        q = np.ones(4, dtype=np.float32)
        k, v = buf.get(query=q, top_k=3)
        assert k.shape[0] == 3

    def test_get_retrieval_no_query_returns_all(self):
        buf = _HeadBuffer(HeadType.RETRIEVAL, window=4, sinks=1, max_entries=100)
        buf.put(np.ones(2), np.ones(2))
        k, v = buf.get(query=None, top_k=0)
        assert k.shape[0] == 1


# ---------------------------------------------------------------------------
# HeadAwareKVStore
# ---------------------------------------------------------------------------

class TestHeadAwareKVStore:
    def _make(self, n_layers=2, n_heads=2, window=4):
        cfg = HeadInferConfig(n_layers=n_layers, n_heads=n_heads,
                              window_size=window, sink_tokens=1,
                              top_k_retrieval=5)
        return HeadAwareKVStore(cfg)

    def test_put_and_get_round_trip(self):
        store = self._make()
        k = np.array([1.0, 2.0], dtype=np.float32)
        v = np.array([3.0, 4.0], dtype=np.float32)
        store.put(0, 0, k, v)
        keys, vals = store.get(0, 0)
        assert keys.shape[0] == 1

    def test_get_out_of_range(self):
        store = self._make()
        keys, vals = store.get(99, 99)
        # Returns empty
        assert keys.shape[0] == 0

    def test_put_out_of_range_no_error(self):
        store = self._make()
        store.put(99, 99, np.ones(2), np.ones(2))  # should not raise

    def test_head_size(self):
        store = self._make()
        store.put(0, 0, np.ones(2), np.ones(2))
        assert store.head_size(0, 0) == 1
        assert store.head_size(99, 99) == 0

    def test_total_entries(self):
        store = self._make()
        store.put(0, 0, np.ones(2), np.ones(2))
        store.put(0, 1, np.ones(2), np.ones(2))
        assert store.total_entries() >= 2

    def test_reset_clears_all(self):
        store = self._make()
        store.put(0, 0, np.ones(2), np.ones(2))
        store.reset()
        assert store.total_entries() == 0

    def test_default_head_types_all_retrieval(self):
        cfg   = HeadInferConfig(n_layers=1, n_heads=2, window_size=4, sink_tokens=1)
        store = HeadAwareKVStore(cfg, head_types=None)
        # All should be RETRIEVAL by default (no error on put/get)
        store.put(0, 0, np.ones(4), np.ones(4))
        k, v = store.get(0, 0)
        assert k.shape[0] == 1

    def test_unknown_head_types_mapped_to_retrieval(self):
        from squish.head_infer import HeadType
        n_layers, n_heads = 1, 1
        head_types = [[HeadType.UNKNOWN]]
        cfg = HeadInferConfig(n_layers=n_layers, n_heads=n_heads,
                              window_size=4, sink_tokens=1)
        store = HeadAwareKVStore(cfg, head_types=head_types)
        store.put(0, 0, np.ones(4), np.ones(4))
        k, _ = store.get(0, 0)
        assert k.shape[0] == 1
