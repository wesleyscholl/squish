"""
tests/test_server_unit.py

Unit tests for squish/server.py.

Strategy: tests that do NOT require a real MLX model use the TestClient directly
with _state.model = None (the default). Tests for pure helper functions
are called directly. Tests for auth use _API_KEY patching.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

import squish.server as _srv


# ── TestClient fixture (no model) ────────────────────────────────────────────

@pytest.fixture()
def client():
    """Fresh TestClient with _API_KEY=None and model=None."""
    orig_state  = _srv._state
    orig_apikey = _srv._API_KEY

    _srv._state   = _srv._ModelState()
    _srv._API_KEY = None

    c = TestClient(_srv.app, raise_server_exceptions=False)
    yield c

    _srv._state   = orig_state
    _srv._API_KEY = orig_apikey


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_always_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_no_model_status(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "no_model"
        assert data["loaded"] is False

    def test_has_required_fields(self, client):
        r = client.get("/health")
        data = r.json()
        for key in ("status", "model", "loaded", "requests", "tokens_gen", "inflight", "uptime_s"):
            assert key in data, f"Missing field: {key}"

    def test_uptime_zero_when_no_model(self, client):
        r = client.get("/health")
        assert r.json()["uptime_s"] == 0


# ── /v1/metrics ───────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/v1/metrics")
        assert r.status_code == 200

    def test_content_is_text_plain(self, client):
        r = client.get("/v1/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_contains_prometheus_metrics(self, client):
        r = client.get("/v1/metrics")
        body = r.text
        assert "squish_requests_total" in body
        assert "squish_tokens_generated_total" in body
        assert "squish_inflight_requests" in body
        assert "squish_uptime_seconds" in body

    def test_contains_help_and_type_comments(self, client):
        r = client.get("/v1/metrics")
        body = r.text
        assert "# HELP" in body
        assert "# TYPE" in body


# ── /v1/models ────────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_empty_list_without_model(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_model_id_404_without_model(self, client):
        r = client.get("/v1/models/some-model")
        assert r.status_code == 404

    def test_model_list_with_mock_model(self):
        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model      = MagicMock()
        _srv._state.model_name = "test-model"
        _srv._state.loaded_at  = 0.0
        try:
            c = TestClient(_srv.app, raise_server_exceptions=False)
            r = c.get("/v1/models")
            assert r.status_code == 200
            data = r.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "test-model"
        finally:
            _srv._state = orig


# ── /v1/chat/completions ─────────────────────────────────────────────────────

class TestChatCompletionsEndpoint:
    def test_503_without_model(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 503

    def test_400_empty_messages_without_model(self, client):
        # 503 not loaded takes precedence over 400 validation in this implementation
        r = client.post("/v1/chat/completions", json={"messages": []})
        # Either 503 or 400 is acceptable
        assert r.status_code in (400, 503)


# ── /v1/completions ───────────────────────────────────────────────────────────

class TestCompletionsEndpoint:
    def test_503_without_model(self, client):
        r = client.post("/v1/completions", json={"prompt": "hello"})
        assert r.status_code == 503


# ── /v1/tokenize ──────────────────────────────────────────────────────────────

class TestTokenizeEndpoint:
    def test_503_without_model(self, client):
        r = client.post("/v1/tokenize", json={"text": "hello world"})
        assert r.status_code == 503


# ── /v1/embeddings ────────────────────────────────────────────────────────────

class TestEmbeddingsEndpointNoModel:
    def test_503_without_model(self, client):
        r = client.post("/v1/embeddings", json={"input": "hello"})
        assert r.status_code == 503


# ── /chat ─────────────────────────────────────────────────────────────────────

class TestWebChatUI:
    def test_returns_200_or_404(self, client):
        r = client.get("/chat")
        assert r.status_code in (200, 404)


# ── _apply_chat_template ──────────────────────────────────────────────────────

class TestApplyChatTemplate:
    def test_fallback_manual_format(self):
        """Without apply_chat_template on tokenizer, uses ChatML fallback."""
        messages = [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        tok = object()  # no apply_chat_template attribute
        result = _srv._apply_chat_template(messages, tok)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result

    def test_uses_tokenizer_apply_chat_template(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<prompt>hi</prompt>"
        messages = [{"role": "user", "content": "hi"}]
        result = _srv._apply_chat_template(messages, tok)
        assert result == "<prompt>hi</prompt>"

    def test_fallback_when_apply_raises(self):
        """If apply_chat_template raises, fall back gracefully."""
        tok = MagicMock()
        tok.apply_chat_template.side_effect = ValueError("unsupported")
        messages = [{"role": "user", "content": "test"}]
        result = _srv._apply_chat_template(messages, tok)
        # Fallback should still produce something
        assert "test" in result

    def test_empty_messages(self):
        tok = object()
        result = _srv._apply_chat_template([], tok)
        assert isinstance(result, str)

    def test_assistant_suffix_added_in_fallback(self):
        """Fallback appends the assistant start tag."""
        messages = [{"role": "user", "content": "Q"}]
        result = _srv._apply_chat_template(messages, object())
        assert "<|im_start|>assistant" in result


# ── _count_tokens ─────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_none_tokenizer_word_split(self):
        orig = _srv._state.tokenizer
        _srv._state.tokenizer = None
        try:
            n = _srv._count_tokens("hello world foo bar")
            assert n == 4
        finally:
            _srv._state.tokenizer = orig

    def test_uses_tokenizer_encode(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        _srv._state.tokenizer = tok
        try:
            n = _srv._count_tokens("test text")
            assert n == 5
        finally:
            _srv._state.tokenizer = orig

    def test_fallback_on_encode_exception(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.side_effect = RuntimeError("cannot encode")
        _srv._state.tokenizer = tok
        try:
            n = _srv._count_tokens("hello world")
            assert n == 2  # word-split fallback
        finally:
            _srv._state.tokenizer = orig


# ── _check_auth ───────────────────────────────────────────────────────────────

class TestCheckAuth:
    def _wrap(self):
        """Get a creds stub."""
        creds = MagicMock(spec=HTTPAuthorizationCredentials)
        creds.credentials = "secret"
        return creds

    def test_no_api_key_always_passes(self):
        orig = _srv._API_KEY
        _srv._API_KEY = None
        try:
            _srv._check_auth(None)  # should not raise
            _srv._check_auth(self._wrap())  # should not raise
        finally:
            _srv._API_KEY = orig

    def test_correct_key_passes(self):
        orig = _srv._API_KEY
        _srv._API_KEY = "secret"
        try:
            _srv._check_auth(self._wrap())  # should not raise
        finally:
            _srv._API_KEY = orig

    def test_wrong_key_raises_401(self):
        orig = _srv._API_KEY
        _srv._API_KEY = "secret"
        try:
            creds = MagicMock(spec=HTTPAuthorizationCredentials)
            creds.credentials = "wrong"
            with pytest.raises(HTTPException) as exc:
                _srv._check_auth(creds)
            assert exc.value.status_code == 401
        finally:
            _srv._API_KEY = orig

    def test_missing_creds_raises_401_when_key_set(self):
        orig = _srv._API_KEY
        _srv._API_KEY = "secret"
        try:
            with pytest.raises(HTTPException) as exc:
                _srv._check_auth(None)
            assert exc.value.status_code == 401
        finally:
            _srv._API_KEY = orig


# ── _get_stop_ids ─────────────────────────────────────────────────────────────

class TestGetStopIds:
    def test_none_returns_empty(self):
        orig = _srv._state.tokenizer
        _srv._state.tokenizer = None
        try:
            result = _srv._get_stop_ids(None)
            assert result == []
        finally:
            _srv._state.tokenizer = orig

    def test_string_input(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.return_value = [10, 20]
        _srv._state.tokenizer = tok
        try:
            result = _srv._get_stop_ids("</s>")
            assert result == [[10, 20]]
        finally:
            _srv._state.tokenizer = orig

    def test_list_input(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.side_effect = lambda s, **kw: [ord(c) for c in s]
        _srv._state.tokenizer = tok
        try:
            result = _srv._get_stop_ids(["</s>", "<end>"])
            assert len(result) == 2
        finally:
            _srv._state.tokenizer = orig

    def test_skips_empty_token_ids(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.return_value = []  # empty ids
        _srv._state.tokenizer = tok
        try:
            result = _srv._get_stop_ids("empty")
            assert result == []
        finally:
            _srv._state.tokenizer = orig

    def test_exception_in_encode_skips(self):
        orig = _srv._state.tokenizer
        tok = MagicMock()
        tok.encode.side_effect = RuntimeError("boom")
        _srv._state.tokenizer = tok
        try:
            result = _srv._get_stop_ids("bad")
            assert result == []
        finally:
            _srv._state.tokenizer = orig


# ── _system_fingerprint ────────────────────────────────────────────────────────

class TestSystemFingerprint:
    def test_returns_string_starting_sq(self):
        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model_name = "mymodel"
        _srv._state.loaded_at  = 12345.0
        try:
            fp = _srv._system_fingerprint()
            assert isinstance(fp, str)
            assert fp.startswith("sq-")
        finally:
            _srv._state = orig

    def test_fingerprint_is_deterministic(self):
        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model_name = "mymodel"
        _srv._state.loaded_at  = 99999.0
        try:
            fp1 = _srv._system_fingerprint()
            fp2 = _srv._system_fingerprint()
            assert fp1 == fp2
        finally:
            _srv._state = orig

    def test_different_models_different_fingerprints(self):
        orig = _srv._state

        _srv._state = _srv._ModelState()
        _srv._state.model_name = "modelA"
        _srv._state.loaded_at  = 1.0
        fp1 = _srv._system_fingerprint()

        _srv._state = _srv._ModelState()
        _srv._state.model_name = "modelB"
        _srv._state.loaded_at  = 1.0
        fp2 = _srv._system_fingerprint()

        _srv._state = orig
        assert fp1 != fp2


# ── _PrefixCache ───────────────────────────────────────────────────────────────

class TestPrefixCache:
    def _cache(self, maxsize=4):
        return _srv._PrefixCache(maxsize=maxsize)

    def test_get_miss_returns_none(self):
        c = self._cache()
        assert c.get("hello world") is None

    def test_put_then_get(self):
        c = self._cache()
        c.put("hello", "Hi there!", "stop")
        result = c.get("hello")
        assert result == ("Hi there!", "stop")

    def test_size_zero_initially(self):
        c = self._cache()
        assert c.size == 0

    def test_size_increments_on_put(self):
        c = self._cache()
        c.put("prompt1", "response1", "stop")
        assert c.size == 1
        c.put("prompt2", "response2", "length")
        assert c.size == 2

    def test_evicts_lru_when_full(self):
        c = self._cache(maxsize=2)
        c.put("a", "resp_a", "stop")
        c.put("b", "resp_b", "stop")
        # Access 'a' to make it recently-used
        c.get("a")
        # Insert 'c' → 'b' should be evicted (LRU)
        c.put("c", "resp_c", "stop")
        assert c.get("b") is None
        assert c.get("a") is not None
        assert c.get("c") is not None

    def test_overwrite_existing_key(self):
        c = self._cache()
        c.put("prompt", "first", "stop")
        c.put("prompt", "second", "length")
        result = c.get("prompt")
        assert result == ("second", "length")

    def test_clear_empties_cache(self):
        c = self._cache()
        c.put("a", "r", "stop")
        c.put("b", "r", "stop")
        c.clear()
        assert c.size == 0
        assert c.get("a") is None

    def test_hits_tracked(self):
        c = self._cache()
        c.get("missing")      # miss
        c.put("x", "y", "s")
        c.get("x")            # hit
        c.get("x")            # hit
        assert c.hits == 2

    def test_misses_tracked(self):
        c = self._cache()
        c.get("a")  # miss
        c.get("b")  # miss
        assert c.misses == 2

    def test_same_prompt_same_key(self):
        c = self._cache()
        c.put("test prompt", "response", "stop")
        assert c.get("test prompt") is not None
        assert c.get("test prompt  ") is None  # Different → miss

    def test_different_prompts_different_entries(self):
        c = self._cache()
        c.put("prompt A", "resp A", "stop")
        c.put("prompt B", "resp B", "stop")
        assert c.get("prompt A") == ("resp A", "stop")
        assert c.get("prompt B") == ("resp B", "stop")


# ── _ModelState dataclass defaults ───────────────────────────────────────────

class TestModelState:
    def test_defaults(self):
        s = _srv._ModelState()
        assert s.model is None
        assert s.tokenizer is None
        assert s.requests == 0
        assert s.tokens_gen == 0
        assert s.inflight == 0

    def test_increment_requests(self):
        s = _srv._ModelState()
        s.requests += 1
        s.requests += 1
        assert s.requests == 2


# ── auth middleware on endpoints ─────────────────────────────────────────────

class TestAuthOnEndpoints:
    def test_list_models_no_auth_needed_by_default(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200  # no API key set → always passes

    def test_health_no_auth_required(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_chat_with_wrong_bearer_returns_401(self):
        orig_key   = _srv._API_KEY
        orig_state = _srv._state
        _srv._API_KEY = "secret-key"
        _srv._state   = _srv._ModelState()
        try:
            c = TestClient(_srv.app, raise_server_exceptions=False)
            r = c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert r.status_code == 401
        finally:
            _srv._API_KEY = orig_key
            _srv._state   = orig_state

    def test_models_with_wrong_bearer_returns_401(self):
        orig_key   = _srv._API_KEY
        orig_state = _srv._state
        _srv._API_KEY = "secret-key"
        _srv._state   = _srv._ModelState()
        try:
            c = TestClient(_srv.app, raise_server_exceptions=False)
            r = c.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert r.status_code == 401
        finally:
            _srv._API_KEY = orig_key
            _srv._state   = orig_state

    def test_chat_with_correct_bearer_passes(self):
        orig_key   = _srv._API_KEY
        orig_state = _srv._state
        _srv._API_KEY = "secret-key"
        _srv._state   = _srv._ModelState()
        try:
            c = TestClient(_srv.app, raise_server_exceptions=False)
            r = c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer secret-key"},
            )
            # 503 expected since no model, but auth should pass
            assert r.status_code in (200, 503)
        finally:
            _srv._API_KEY = orig_key
            _srv._state   = orig_state

    def test_models_with_correct_bearer_passes(self):
        orig_key   = _srv._API_KEY
        orig_state = _srv._state
        _srv._API_KEY = "secret-key"
        _srv._state   = _srv._ModelState()
        try:
            c = TestClient(_srv.app, raise_server_exceptions=False)
            r = c.get(
                "/v1/models",
                headers={"Authorization": "Bearer secret-key"},
            )
            assert r.status_code == 200
        finally:
            _srv._API_KEY = orig_key
            _srv._state   = orig_state