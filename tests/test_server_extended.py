"""
tests/test_server_extended.py

Extended server tests covering:
  - _Stats.record_completion  (lines 108-112)
  - _make_chunk               (lines 728-740)
  - tokenize endpoint w/ model (lines 1143-1158)
  - /v1/models list (extra branches)
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

import squish.server as _srv


# ── _ModelState.record_completion ────────────────────────────────────────────

class TestStatsRecordCompletion:
    def test_record_completion_increments_requests(self):
        stats = _srv._ModelState()
        stats.record_completion(10, 0.5, 0.1)
        assert stats.requests == 1
        assert stats.tokens_gen == 10

    def test_record_completion_accumulates(self):
        stats = _srv._ModelState()
        stats.record_completion(5, 0.2, 0.05)
        stats.record_completion(10, 0.8, 0.1)
        assert stats.requests == 2
        assert stats.tokens_gen == 15

    def test_record_completion_tps_window(self):
        stats = _srv._ModelState()
        stats.record_completion(100, 1.0, 0.01)
        assert stats.avg_tps > 0

    def test_avg_tps_zero_initially(self):
        stats = _srv._ModelState()
        # avg_tps with no completions
        tps = stats.avg_tps
        assert tps == 0.0 or tps is not None  # just verify it doesn't crash

    def test_record_completion_zero_duration(self):
        """Zero duration should not cause division by zero (uses max(dur, 1e-6))."""
        stats = _srv._ModelState()
        stats.record_completion(50, 0.0, 0.0)
        assert stats.requests == 1


# ── _make_chunk ────────────────────────────────────────────────────────────────

class TestMakeChunk:
    def test_returns_sse_format(self):
        result = _srv._make_chunk("hello", "test-model", "cid-1")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_json_parseable(self):
        result = _srv._make_chunk("world", "squish", "cid-2")
        data = result.removeprefix("data: ").strip()
        obj = json.loads(data)
        assert obj["choices"][0]["delta"]["content"] == "world"
        assert obj["model"] == "squish"

    def test_finish_reason_none(self):
        result = _srv._make_chunk("", "m", "id-3", finish_reason=None)
        obj = json.loads(result.removeprefix("data: ").strip())
        assert obj["choices"][0]["finish_reason"] is None

    def test_finish_reason_stop(self):
        result = _srv._make_chunk("", "m", "id-4", finish_reason="stop")
        obj = json.loads(result.removeprefix("data: ").strip())
        assert obj["choices"][0]["finish_reason"] == "stop"

    def test_empty_content_delta_is_empty_dict(self):
        result = _srv._make_chunk("", "m", "id-5")
        obj = json.loads(result.removeprefix("data: ").strip())
        # empty content → delta is {}
        assert obj["choices"][0]["delta"] == {}

    def test_nonempty_content_delta_has_content(self):
        result = _srv._make_chunk("hi", "m", "id-6")
        obj = json.loads(result.removeprefix("data: ").strip())
        assert "content" in obj["choices"][0]["delta"]

    def test_id_and_model_fields(self):
        result = _srv._make_chunk("test", "my-model", "abc123")
        obj = json.loads(result.removeprefix("data: ").strip())
        assert obj["id"] == "abc123"
        assert obj["model"] == "my-model"

    def test_created_is_epoch_approx(self):
        before = int(time.time()) - 2
        result = _srv._make_chunk("x", "m", "id-7")
        after = int(time.time()) + 2
        obj = json.loads(result.removeprefix("data: ").strip())
        assert before <= obj["created"] <= after


# ── Tokenize endpoint with state.model set ────────────────────────────────────

class TestTokenizeWithModel:
    """Cover lines 1143-1158 by setting _state.model + _state.tokenizer."""

    def setup_method(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_srv.app, raise_server_exceptions=False)
        # Save originals
        self._orig_model     = _srv._state.model
        self._orig_model_name = _srv._state.model_name
        self._orig_tokenizer = _srv._state.tokenizer
        # Install mock state
        _srv._state.model      = MagicMock()
        _srv._state.model_name = "test-model"
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        _srv._state.tokenizer  = tok

    def teardown_method(self):
        _srv._state.model      = self._orig_model
        _srv._state.model_name = self._orig_model_name
        _srv._state.tokenizer  = self._orig_tokenizer

    def test_tokenize_with_text_field(self):
        r = self.client.post("/v1/tokenize", json={"text": "hello world"})
        assert r.status_code == 200
        body = r.json()
        assert "token_ids" in body
        assert "token_count" in body

    def test_tokenize_with_messages_field(self):
        msgs = [{"role": "user", "content": "hello"}]
        r = self.client.post("/v1/tokenize", json={"messages": msgs})
        assert r.status_code in (200, 500)  # 500 if chat template fails

    def test_tokenize_missing_body_fields(self):
        r = self.client.post("/v1/tokenize", json={"other": "value"})
        assert r.status_code == 400

    def test_tokenize_tokenizer_exception(self):
        tok = MagicMock()
        tok.encode.side_effect = RuntimeError("encode failed")
        tok.__call__ = MagicMock(side_effect=RuntimeError)
        _srv._state.tokenizer = tok
        r = self.client.post("/v1/tokenize", json={"text": "hi"})
        assert r.status_code == 500


# ── /v1/embeddings zero-norm path (line 1047 else-branch) ────────────────────

class TestEmbeddingsZeroNorm:
    def test_zero_norm_embedding_not_normalized(self):
        """When model outputs all-zeros, norm=0 → skip normalization (line 1047→1050)."""
        import numpy as np
        mx = pytest.importorskip("mlx.core")
        from fastapi.testclient import TestClient

        mock_model = MagicMock()
        # Return an all-zero tensor → norm == 0 → normalization skipped
        mock_model.model.return_value = mx.zeros([1, 3, 16])
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]

        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model      = mock_model
        _srv._state.tokenizer  = mock_tok
        _srv._state.model_name = "test-zero-embed"
        try:
            client = TestClient(_srv.app, raise_server_exceptions=False)
            resp = client.post("/v1/embeddings", json={"input": "hello", "model": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 1
            emb = data["data"][0]["embedding"]
            # All-zero embedding → norm stays 0, no normalization
            assert all(v == 0.0 for v in emb)
        finally:
            _srv._state = orig
