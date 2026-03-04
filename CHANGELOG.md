# Changelog

All notable changes to Squish are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- **`eval_output/eval_report.md`** — Replaced physically impossible benchmark numbers
  (+14.1% ARC, +15.2% HellaSwag after lossy compression) with validated results from a
  clean re-run; added a clearly labelled validity-notice header.
- **`KVLayerCache.update_and_fetch` / `.offset`** — Added the `update_and_fetch(keys, values)`
  method and read-only `offset` property required by the mlx_lm per-layer cache protocol.
  Without these, `--kv-cache-mode int8/snap` silently had no effect on generation.
- **`QuantizedKVCache.__getitem__`** — Now returns `self._layers[idx]` (a `KVLayerCache`
  with `update_and_fetch`) instead of a `_LayerCacheView` wrapper that lacked the protocol
  method.
- **`server.py` `_sample_mx()`** — Added module-level temperature + nucleus-sampling helper
  used by the quantized KV cache generation path.
- **`server.py` KV cache generation path** — Wired the quantized cache into `_stream_tokens`;
  `--kv-cache-mode int8/snap` now routes through `model(x, cache=layer_caches)` per decode
  step with graceful fallback to `mlx_lm.stream_generate` on error.
- **`server.py` `/v1/embeddings`** — Semantic embeddings now use `model.model(x)` (last
  hidden state) as the preferred path, falling back to `embed_tokens` then logits mean-pool.
  The previous behaviour always returned input-token embeddings, which are unsuitable for
  semantic similarity.
- **`server.py` `--log-level`** — Added argument to control uvicorn log verbosity
  (choices: `critical` / `error` / `warning` / `info` / `debug` / `trace`; default:
  `warning`).  Previously hardcoded.
- **`cli.py compress --awq / --awq-samples`** — AWQ activation-calibration pass now exposed
  on the `squish compress` subcommand.  Loads the full model, collects activation scales,
  and passes `--awq-scales` to the conversion subprocess automatically.
- **`cli.py run/serve --log-level`** — Log-level argument forwarded from `squish run` /
  `squish serve` to the server process.

---

## [1.0.0] — 2026-03-03

**Initial public release**, accompanying the research paper.

### Added

- **Three-tier compressed weight loader** — INT8 Vectro → float16 npy → bf16 MLX safetensors
- **OpenAI-compatible API server** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`)
- **Ollama drop-in compatibility** (`/api/generate`, `/api/chat`, `/api/tags`, `/api/embeddings`)
- **Web chat UI** at `/chat` — dark-themed, streaming, multi-session history, offline
- **CLI** — `squish run` / `squish serve`, `squish chat`, `squish models`, `squish bench`, `squish info`, `squish rm`, `squish search`, `squish pull`, `squish --version`
- **Speculative decoding** — target + draft model acceleration
- **Batch scheduler** — dynamic batching with priority queues
- **KV cache quantisation** — KIVI INT8 + SnapKV compression
- **Prefix cache** — prompt prefix reuse across requests
- **Tool / function calling** — OpenAI-format `tools` → `tool_calls` round-trip
- **Rust/PyO3 INT8 quantiser** (`squish_quant_rs`) — ARM NEON SIMD vectorised
- **AWQ calibration** pass for activation-guided mixed-precision
- Integrations: Continue.dev, aider, LiteLLM (config templates in `configs/`)
- Evaluation harness wrapper (`squish[eval]`) — lm-evaluation-harness compatible

### Benchmark (Qwen2.5-1.5B-Instruct, Apple Silicon M-series)

| Metric | mlx_lm (cold) | Squish (cached) | Improvement |
|---|---:|---:|---:|
| Load time | 28.81 s | 0.53 s | **54×** |
| Peak load RAM | ~2600 MB | 402 MB | **6×** |
| Accuracy delta | — | ≤1.5% on all tasks | ✅ |
