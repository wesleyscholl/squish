#!/usr/bin/env python3
"""
squish/ollama_compat.py

Ollama-compatible HTTP API layer for Squish.

Mounts onto the existing FastAPI app and exposes the subset of Ollama's API
used by common tooling (Open WebUI, Continue.dev, aider, fabric, etc.).

Endpoints
─────────
  GET  /api/tags            → list local models (like `ollama list`)
  GET  /api/version         → {"version": "0.1.25"}
  POST /api/show            → model card / details
  POST /api/generate        → text completion (ndjson streaming)
  POST /api/chat            → chat completion (ndjson streaming)
  POST /api/pull            → politely redirect (models are managed locally)
  POST /api/embeddings      → embeddings (delegates to /v1/embeddings)
  DELETE /api/delete        → stub (local models aren't deleted via API)

Wire this into server.py at startup:
    from squish.ollama_compat import mount_ollama
    mount_ollama(app, state_getter, tokenizer_getter)

Ollama ndjson stream format:
    Each line is a JSON object ending with a newline.
    The final line has "done": true.

References:
    https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


def mount_ollama(
    app: FastAPI,
    get_state,           # callable → _ModelState
    get_generate,        # callable → _generate_tokens function
    get_tokenizer,       # callable → tokenizer
    models_dir: Path | None = None,
):
    """
    Register all Ollama-compatible routes on `app`.

    Parameters
    ----------
    app          : the FastAPI application from server.py
    get_state    : zero-arg callable returning the _ModelState global
    get_generate : zero-arg callable returning the _generate_tokens function
    get_tokenizer: zero-arg callable returning the loaded tokenizer
    models_dir   : path to search for local models (default: ~/models)
    """
    if models_dir is None:
        models_dir = Path.home() / "models"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _model_name() -> str:
        state = get_state()
        return state.model_name or "squish"

    def _model_size_bytes(model_name: str) -> int:
        """Rough size in bytes from disk — used in /api/tags listing."""
        candidate = models_dir / model_name
        if not candidate.exists():  # pragma: no cover
            return 0
        try:
            return sum(f.stat().st_size for f in candidate.rglob("*") if f.is_file())
        except Exception:  # pragma: no cover
            return 0

    def _local_models() -> list[dict]:
        """Scan ~/models/ and return Ollama-style model cards."""
        if not models_dir.exists():
            return [_single_model_card()]
        rows = []
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            size = _model_size_bytes(d.name)
            rows.append({
                "name":         d.name + ":latest",
                "model":        d.name + ":latest",
                "modified_at":  "2026-01-01T00:00:00Z",
                "size":         size,
                "digest":       "squish-local",
                "details": {
                    "parent_model":     "",
                    "format":           "squish",
                    "family":           _guess_family(d.name),
                    "families":         [_guess_family(d.name)],
                    "parameter_size":   _guess_params(d.name),
                    "quantization_level": "4-bit",
                },
            })
        if not rows:
            rows.append(_single_model_card())
        return rows

    def _single_model_card() -> dict:
        state = get_state()
        return {
            "name":        (state.model_name or "squish") + ":latest",
            "model":       (state.model_name or "squish") + ":latest",
            "modified_at": "2026-01-01T00:00:00Z",
            "size":        4_000_000_000,
            "digest":      "squish-local",
            "details": {
                "format":              "squish",
                "family":              "qwen2",
                "parameter_size":      "7B",
                "quantization_level":  "4-bit",
            },
        }

    def _guess_family(name: str) -> str:
        name_lower = name.lower()
        if "qwen" in name_lower:
            return "qwen2"
        if "llama" in name_lower:
            return "llama"
        if "mistral" in name_lower:
            return "mistral"
        if "gemma" in name_lower:
            return "gemma"
        return "unknown"

    def _guess_params(name: str) -> str:
        import re
        m = re.search(r'(\d+\.?\d*)[Bb]', name)
        return m.group(0).upper() if m else "?"

    def _messages_to_prompt(messages: list[dict]) -> str:
        """Convert Ollama messages to a flat prompt string using chat template."""
        try:
            tok = get_tokenizer()
            if hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        # Fallback: simple concatenation
        parts = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _ndjson(obj: dict) -> str:
        return json.dumps(obj) + "\n"

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/api/version")
    async def ollama_version():
        """Ollama version endpoint — tools check this to detect an Ollama server."""
        return JSONResponse({"version": "0.3.0"})

    @app.get("/api/tags")
    async def ollama_tags():
        """List all local models (equivalent to `ollama list`)."""
        return JSONResponse({"models": _local_models()})

    @app.post("/api/show")
    async def ollama_show(request: Request):
        """Return model details card."""
        body = await request.json()
        name = body.get("name", _model_name())
        return JSONResponse({
            "modelfile":  f"# Squish compressed model\nFROM {name}",
            "parameters": "",
            "template":   "{{ .Prompt }}",
            "details": {
                "parent_model":       "",
                "format":             "squish",
                "family":             _guess_family(name),
                "parameter_size":     _guess_params(name),
                "quantization_level": "4-bit",
            },
            "model_info": {
                "general.architecture":      _guess_family(name),
                "general.file_type":         7,
                "general.parameter_count":   7_000_000_000,
            },
        })

    @app.post("/api/pull")
    async def ollama_pull(request: Request):
        """
        Ollama pull stub.  Squish manages models locally via mlx_lm.convert —
        remote pulling is not supported.  Return helpful message.
        """
        body = await request.json()
        name = body.get("name", "")

        async def _stream():
            yield _ndjson({"status": "Squish manages models locally."})
            yield _ndjson({"status": f"To add '{name}', use mlx_lm to download + convert:"})
            yield _ndjson({"status": "  python3 -m mlx_lm.convert --hf-path <HF_MODEL_ID> -q --q-bits 4"})
            yield _ndjson({"status": "done", "completed": 0, "total": 0})

        return StreamingResponse(_stream(), media_type="application/x-ndjson")

    @app.delete("/api/delete")
    async def ollama_delete(request: Request):
        raise HTTPException(400, "Model deletion not supported via API. Remove the directory from ~/models/ manually.")

    @app.post("/api/generate")
    async def ollama_generate(request: Request):
        """
        POST /api/generate

        Ollama text-completion endpoint.  Supports streaming and non-streaming.
        Request:  {"model": str, "prompt": str, "stream": bool, "options": {...}}
        Response: ndjson stream (or single JSON if stream=false)
        """
        state = get_state()
        if state.model is None:
            raise HTTPException(503, "Model not loaded")

        body: dict[str, Any] = await request.json()
        prompt      = body.get("prompt", "")
        options     = body.get("options", {})
        stream      = body.get("stream", True)
        max_tokens  = int(options.get("num_predict", body.get("max_tokens", 512)))
        temperature = float(options.get("temperature", 0.7))
        top_p       = float(options.get("top_p", 0.9))
        stop        = options.get("stop", body.get("stop", None))
        seed        = options.get("seed", body.get("seed", None))
        model_name  = body.get("model", _model_name())

        if not prompt:
            raise HTTPException(400, "'prompt' must be non-empty")

        _generate = get_generate()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if stream:
            async def _stream_gen() -> AsyncIterator[str]:
                t0 = time.perf_counter()
                n_eval = 0
                try:
                    for tok_text, finish in _generate(prompt, max_tokens, temperature,
                                                      top_p, stop, seed):
                        if tok_text:
                            n_eval += 1
                            yield _ndjson({
                                "model":      model_name,
                                "created_at": created_at,
                                "response":   tok_text,
                                "done":       False,
                            })
                        if finish is not None:
                            break
                except Exception as exc:  # pragma: no cover
                    yield _ndjson({"error": str(exc)})
                    return

                elapsed_ns = int((time.perf_counter() - t0) * 1e9)
                yield _ndjson({
                    "model":                model_name,
                    "created_at":           created_at,
                    "response":             "",
                    "done":                 True,
                    "done_reason":          "stop",
                    "total_duration":       elapsed_ns,
                    "eval_count":           n_eval,
                    "eval_duration":        elapsed_ns,
                })

            return StreamingResponse(_stream_gen(),
                                     media_type="application/x-ndjson",
                                     headers={"X-Content-Type-Options": "nosniff"})
        else:
            full_text = ""
            t0 = time.perf_counter()
            n_eval = 0
            for tok_text, finish in _generate(prompt, max_tokens, temperature,
                                              top_p, stop, seed):
                if tok_text:
                    n_eval += 1
                    full_text += tok_text
                if finish is not None:
                    break
            elapsed_ns = int((time.perf_counter() - t0) * 1e9)
            return JSONResponse({
                "model":          model_name,
                "created_at":     created_at,
                "response":       full_text,
                "done":           True,
                "done_reason":    "stop",
                "total_duration": elapsed_ns,
                "eval_count":     n_eval,
            })

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        """
        POST /api/chat

        Ollama chat endpoint.  Translates to /v1/chat/completions internally.
        Request:  {"model": str, "messages": [...], "stream": bool, "options": {...}}
        Response: ndjson stream
        """
        state = get_state()
        if state.model is None:
            raise HTTPException(503, "Model not loaded")

        body: dict[str, Any] = await request.json()
        messages    = body.get("messages", [])
        options     = body.get("options", {})
        stream      = body.get("stream", True)
        max_tokens  = int(options.get("num_predict", body.get("max_tokens", 512)))
        temperature = float(options.get("temperature", 0.7))
        top_p       = float(options.get("top_p", 0.9))
        stop        = options.get("stop", body.get("stop", None))
        seed        = options.get("seed", body.get("seed", None))
        model_name  = body.get("model", _model_name())

        if not messages:
            raise HTTPException(400, "'messages' must be non-empty")

        prompt     = _messages_to_prompt(messages)
        _generate  = get_generate()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if stream:
            async def _chat_stream() -> AsyncIterator[str]:
                t0 = time.perf_counter()
                n_eval = 0
                full   = ""
                try:
                    for tok_text, finish in _generate(prompt, max_tokens, temperature,
                                                      top_p, stop, seed):
                        if tok_text:
                            n_eval += 1
                            full   += tok_text
                            yield _ndjson({
                                "model":      model_name,
                                "created_at": created_at,
                                "message":    {"role": "assistant", "content": tok_text},
                                "done":       False,
                            })
                        if finish is not None:
                            break
                except Exception as exc:  # pragma: no cover
                    yield _ndjson({"error": str(exc)})
                    return
                elapsed_ns = int((time.perf_counter() - t0) * 1e9)
                yield _ndjson({
                    "model":                model_name,
                    "created_at":           created_at,
                    "message":              {"role": "assistant", "content": ""},
                    "done":                 True,
                    "done_reason":          "stop",
                    "total_duration":       elapsed_ns,
                    "eval_count":           n_eval,
                })

            return StreamingResponse(_chat_stream(),
                                     media_type="application/x-ndjson",
                                     headers={"X-Content-Type-Options": "nosniff"})
        else:
            full_text = ""
            t0 = time.perf_counter()
            n_eval = 0
            for tok_text, finish in _generate(prompt, max_tokens, temperature,
                                              top_p, stop, seed):
                if tok_text:
                    n_eval += 1
                    full_text += tok_text
                if finish is not None:
                    break
            elapsed_ns = int((time.perf_counter() - t0) * 1e9)
            return JSONResponse({
                "model":          model_name,
                "created_at":     created_at,
                "message":        {"role": "assistant", "content": full_text},
                "done":           True,
                "done_reason":    "stop",
                "total_duration": elapsed_ns,
                "eval_count":     n_eval,
            })

    @app.post("/api/embeddings")
    async def ollama_embeddings(request: Request):  # pragma: no cover
        """
        POST /api/embeddings  (Ollama format)
        Computes mean-pooled, L2-normalised embeddings from the loaded model.
        """
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(400, "'prompt' must be non-empty")
        state = get_state()
        if state.model is None:
            raise HTTPException(503, "Model not loaded")
        import mlx.core as mx
        import numpy as np
        tok = get_tokenizer()
        ids = tok.encode(prompt, add_special_tokens=True)
        if len(ids) > 4096:
            ids = ids[:4096]
        id_arr = mx.array(ids, dtype=mx.int32)[None]
        logits = state.model(id_arr)
        # Mean-pool last hidden states (use logits as proxy — model.model() not always accessible)
        pooled = mx.mean(logits[0].astype(mx.float32), axis=0)
        # L2 normalise
        norm   = mx.sqrt(mx.sum(pooled * pooled) + 1e-8)
        pooled = pooled / norm
        mx.eval(pooled)
        emb = np.array(pooled).tolist()
        return JSONResponse({
            "embedding": emb,
            "model":     body.get("model", state.model_name),
        })
