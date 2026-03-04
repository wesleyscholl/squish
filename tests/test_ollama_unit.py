"""
Unit tests for squish/ollama_compat.py — Ollama-compatible API layer.

Tests mount the routes on a fresh FastAPI app with mocked model state and
call them through the HTTPX test client.  All network/MLX paths are mocked.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from squish.ollama_compat import mount_ollama


# ── helpers ────────────────────────────────────────────────────────────────

def _make_client(
    model_loaded: bool = True,
    model_name: str = "test-model",
    mock_generate=None,
    mock_tokenizer=None,
    models_dir=None,
) -> tuple[TestClient, MagicMock]:
    """Build a TestClient with ollama routes mounted."""
    app = FastAPI()
    state = MagicMock()
    state.model = MagicMock() if model_loaded else None
    state.model_name = model_name if model_loaded else None

    if mock_generate is None:
        mock_generate = lambda *a, **kw: iter([])  # noqa: E731

    mount_ollama(
        app,
        get_state=lambda: state,
        get_generate=lambda: mock_generate,
        get_tokenizer=lambda: mock_tokenizer,
        models_dir=models_dir,
    )
    client = TestClient(app, raise_server_exceptions=False)
    return client, state


def _simple_generate(tokens):
    """Return a generator that yields (token, finish_reason) pairs."""
    def _gen(prompt, max_tokens, temp, top_p, stop, seed):
        for i, tok in enumerate(tokens):
            finish = "stop" if i == len(tokens) - 1 else None
            yield tok, finish
    return _gen


# ── /api/version ───────────────────────────────────────────────────────────

class TestOllamaVersion:
    def test_version_returns_200(self):
        client, _ = _make_client()
        resp = client.get("/api/version")
        assert resp.status_code == 200

    def test_version_has_version_field(self):
        client, _ = _make_client()
        data = client.get("/api/version").json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ── /api/tags ──────────────────────────────────────────────────────────────

class TestOllamaTags:
    def test_tags_returns_200(self):
        client, _ = _make_client()
        resp = client.get("/api/tags")
        assert resp.status_code == 200

    def test_tags_has_models_list(self):
        client, _ = _make_client()
        data = client.get("/api/tags").json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1

    def test_tags_model_card_format(self):
        """Single-model card should have required Ollama fields."""
        client, _ = _make_client()
        data = client.get("/api/tags").json()
        card = data["models"][0]
        assert "name" in card
        assert "model" in card
        assert "size" in card
        assert "details" in card

    def test_tags_with_models_dir_empty(self):
        """Empty models_dir yields single card fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_path = Path(tmpdir) / "models"
            models_path.mkdir()
            client, _ = _make_client(models_dir=models_path)
            data = client.get("/api/tags").json()
            assert len(data["models"]) >= 1

    def test_tags_with_models_dir_containing_dirs(self):
        """models_dir with subdirs → lists them in tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_path = Path(tmpdir)
            (models_path / "llama3-8b").mkdir()
            (models_path / "qwen2-7b").mkdir()
            # Add a file inside one to test _model_size_bytes
            (models_path / "llama3-8b" / "config.json").write_text('{}')
            client, _ = _make_client(models_dir=models_path)
            data = client.get("/api/tags").json()
            names = [m["name"] for m in data["models"]]
            assert any("llama" in n for n in names)

    def test_tags_skips_dot_dirs(self):
        """Hidden directories are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_path = Path(tmpdir)
            (models_path / ".hidden").mkdir()
            (models_path / "visible-model").mkdir()
            client, _ = _make_client(models_dir=models_path)
            data = client.get("/api/tags").json()
            names = [m["name"] for m in data["models"]]
            assert not any(".hidden" in n for n in names)


# ── /api/show ──────────────────────────────────────────────────────────────

class TestOllamaShow:
    def test_show_returns_200(self):
        client, _ = _make_client()
        resp = client.post("/api/show", json={"name": "test-model"})
        assert resp.status_code == 200

    def test_show_details_present(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "qwen2-7b"}).json()
        assert "details" in data
        assert "family" in data["details"]

    def test_show_guess_family_qwen(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "Qwen2.5-14B"}).json()
        assert data["details"]["family"] == "qwen2"

    def test_show_guess_family_llama(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "llama3.2-3B-Instruct"}).json()
        assert data["details"]["family"] == "llama"

    def test_show_guess_family_mistral(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "Mistral-7B"}).json()
        assert data["details"]["family"] == "mistral"

    def test_show_guess_family_gemma(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "gemma-2b"}).json()
        assert data["details"]["family"] == "gemma"

    def test_show_guess_family_unknown(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "some-random-model"}).json()
        assert data["details"]["family"] == "unknown"

    def test_show_guess_params(self):
        client, _ = _make_client()
        data = client.post("/api/show", json={"name": "llama3-70B"}).json()
        # parameter_size should contain the number (70B)
        assert "7" in data["details"]["parameter_size"]

    def test_show_no_name_uses_loaded(self):
        """When no name in body, uses the loaded model name."""
        client, _ = _make_client(model_name="my-model")
        data = client.post("/api/show", json={}).json()
        assert "details" in data


# ── /api/pull ──────────────────────────────────────────────────────────────

class TestOllamaPull:
    def test_pull_returns_200(self):
        client, _ = _make_client()
        resp = client.post("/api/pull", json={"name": "some-model"})
        assert resp.status_code == 200

    def test_pull_returns_ndjson(self):
        client, _ = _make_client()
        resp = client.post("/api/pull", json={"name": "llama3"})
        assert resp.status_code == 200
        # Parse ndjson lines
        lines = [ln for ln in resp.text.splitlines() if ln.strip()]
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)
            assert "status" in obj

    def test_pull_has_done_line(self):
        client, _ = _make_client()
        resp = client.post("/api/pull", json={"name": "any"})
        lines = [json.loads(ln) for ln in resp.text.splitlines() if ln.strip()]
        last = lines[-1]
        assert last.get("status", "").lower() == "done" or last.get("completed") is not None


# ── /api/delete ────────────────────────────────────────────────────────────

class TestOllamaDelete:
    def test_delete_returns_400(self):
        client, _ = _make_client()
        resp = client.request("DELETE", "/api/delete", json={"name": "some-model"})
        assert resp.status_code == 400


# ── /api/generate ──────────────────────────────────────────────────────────

class TestOllamaGenerate:
    def test_generate_no_model_503(self):
        client, _ = _make_client(model_loaded=False)
        resp = client.post("/api/generate", json={"prompt": "hello"})
        assert resp.status_code == 503

    def test_generate_empty_prompt_400(self):
        client, _ = _make_client()
        resp = client.post("/api/generate", json={"prompt": ""})
        assert resp.status_code == 400

    def test_generate_non_streaming(self):
        gen = _simple_generate(["Hello", " there"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/generate", json={
            "prompt": "What is 2+2?",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "Hello there"
        assert data["done"] is True

    def test_generate_non_streaming_has_timing(self):
        gen = _simple_generate(["hi"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/generate", json={"prompt": "hi", "stream": False})
        data = resp.json()
        assert "total_duration" in data
        assert "eval_count" in data

    def test_generate_streaming(self):
        gen = _simple_generate(["tok1", "tok2"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/generate", json={
            "prompt": "count",
            "stream": True,
        })
        assert resp.status_code == 200
        lines = [json.loads(ln) for ln in resp.text.splitlines() if ln.strip()]
        # At least some lines with response content
        assert any(ln.get("response") for ln in lines)
        # Last line should be done
        assert lines[-1]["done"] is True

    def test_generate_with_options(self):
        gen = _simple_generate(["ok"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/generate", json={
            "prompt": "test",
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 100, "top_p": 0.95},
        })
        assert resp.status_code == 200


# ── /api/chat ──────────────────────────────────────────────────────────────

class TestOllamaChat:
    def test_chat_no_model_503(self):
        client, _ = _make_client(model_loaded=False)
        resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 503

    def test_chat_empty_messages_400(self):
        client, _ = _make_client()
        resp = client.post("/api/chat", json={"messages": []})
        assert resp.status_code == 400

    def test_chat_non_streaming(self):
        gen = _simple_generate(["Hello", " friend"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello friend"
        assert data["done"] is True

    def test_chat_non_streaming_has_timing(self):
        gen = _simple_generate(["hi"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        data = resp.json()
        assert "total_duration" in data

    def test_chat_streaming(self):
        gen = _simple_generate(["tok1", "tok2"])
        client, _ = _make_client(mock_generate=gen)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "tell me something"}],
            "stream": True,
        })
        assert resp.status_code == 200
        lines = [json.loads(ln) for ln in resp.text.splitlines() if ln.strip()]
        assert any(ln.get("message", {}).get("content") for ln in lines)
        assert lines[-1]["done"] is True

    def test_chat_with_chat_template(self):
        """Uses tokenizer.apply_chat_template when available."""
        gen = _simple_generate(["response"])
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<prompt>"
        client, _ = _make_client(mock_generate=gen, mock_tokenizer=mock_tok)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        mock_tok.apply_chat_template.assert_called_once()

    def test_chat_falls_back_to_concat_when_no_template(self):
        """Falls back to simple concat when tokenizer has no apply_chat_template."""
        gen = _simple_generate(["answer"])
        mock_tok = MagicMock(spec=[])  # no apply_chat_template
        client, _ = _make_client(mock_generate=gen, mock_tokenizer=mock_tok)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "yo"}],
            "stream": False,
        })
        assert resp.status_code == 200

    def test_chat_fallback_when_template_raises(self):
        """Falls back to concat when apply_chat_template raises."""
        gen = _simple_generate(["ok"])
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = RuntimeError("fail")
        client, _ = _make_client(mock_generate=gen, mock_tokenizer=mock_tok)
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
