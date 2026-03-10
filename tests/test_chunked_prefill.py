"""
tests/test_chunked_prefill.py

Coverage for squish/chunked_prefill.py:
  - ChunkedPrefillConfig dataclass (default values)
  - chunk_prefill generator: single chunk, multi-chunk, explicit config
  - is_final_chunk True/False path
  - config=None triggers default construction
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not available (requires Apple Silicon)")

from squish.chunked_prefill import ChunkedPrefillConfig, chunk_prefill

# ── helpers ───────────────────────────────────────────────────────────────────

VOCAB = 32


def _fake_model(x, cache=None):
    """Return a zero-logit tensor of shape (1, seq_len, VOCAB)."""
    b, seq = x.shape
    return mx.zeros((b, seq, VOCAB))


# ── ChunkedPrefillConfig ──────────────────────────────────────────────────────

class TestChunkedPrefillConfig:

    def test_default_chunk_size(self):
        cfg = ChunkedPrefillConfig()
        assert cfg.chunk_size == 512

    def test_default_interleave_decode(self):
        cfg = ChunkedPrefillConfig()
        assert cfg.interleave_decode is True

    def test_custom_values(self):
        cfg = ChunkedPrefillConfig(chunk_size=128, interleave_decode=False)
        assert cfg.chunk_size == 128
        assert cfg.interleave_decode is False


# ── chunk_prefill ─────────────────────────────────────────────────────────────

class TestChunkPrefill:

    def test_single_chunk_is_final(self):
        """len(input_ids) <= chunk_size → one yield with is_final=True."""
        results = list(chunk_prefill(_fake_model, [1, 2, 3], None,
                                     config=ChunkedPrefillConfig(chunk_size=512)))
        assert len(results) == 1
        logit, is_final = results[0]
        assert is_final is True

    def test_uses_default_config_when_none(self):
        """config=None should default to ChunkedPrefillConfig()."""
        results = list(chunk_prefill(_fake_model, [10, 20], None, config=None))
        assert len(results) == 1
        _, is_final = results[0]
        assert is_final is True

    def test_multi_chunk_yields_correct_count(self):
        """10 tokens with chunk_size=4 → ceil(10/4)=3 yields."""
        ids = list(range(10))
        cfg = ChunkedPrefillConfig(chunk_size=4)
        results = list(chunk_prefill(_fake_model, ids, None, config=cfg))
        assert len(results) == 3

    def test_is_final_only_on_last_chunk(self):
        """All intermediate chunks have is_final=False; last has is_final=True."""
        ids = list(range(9))
        cfg = ChunkedPrefillConfig(chunk_size=4)
        results = list(chunk_prefill(_fake_model, ids, None, config=cfg))
        # 3 chunks: [0..3], [4..7], [8]
        assert results[0][1] is False
        assert results[1][1] is False
        assert results[2][1] is True

    def test_chunk_size_exactly_divides_sequence(self):
        """8 tokens with chunk_size=4 → exactly 2 yields."""
        ids = list(range(8))
        cfg = ChunkedPrefillConfig(chunk_size=4)
        results = list(chunk_prefill(_fake_model, ids, None, config=cfg))
        assert len(results) == 2
        assert results[-1][1] is True

    def test_logit_shape(self):
        """Each yielded logit has shape (VOCAB,)."""
        results = list(chunk_prefill(_fake_model, [1, 2, 3, 4, 5], None,
                                     config=ChunkedPrefillConfig(chunk_size=2)))
        for logit, _ in results:
            assert logit.shape == (VOCAB,)

    def test_chunk_size_one_token(self):
        """chunk_size=1 yields one result per token."""
        ids = [1, 2, 3]
        cfg = ChunkedPrefillConfig(chunk_size=1)
        results = list(chunk_prefill(_fake_model, ids, None, config=cfg))
        assert len(results) == 3
        assert results[-1][1] is True

    def test_cache_arg_passed_to_model(self):
        """The layer_caches argument is forwarded to the model callable."""
        received = {}

        def tracking_model(x, cache=None):
            received["cache"] = cache
            b, seq = x.shape
            return mx.zeros((b, seq, VOCAB))

        sentinel = object()
        list(chunk_prefill(tracking_model, [1, 2], sentinel,
                          config=ChunkedPrefillConfig(chunk_size=512)))
        assert received.get("cache") is sentinel
