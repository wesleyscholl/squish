# Changelog

All notable changes to Squish are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2025-03-02

**Initial public release**, accompanying the research paper.

### Added

- **Three-tier compressed weight loader** — INT8 Vectro → float16 npy → bf16 MLX safetensors
- **OpenAI-compatible API server** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`)
- **Ollama drop-in compatibility** (`/api/generate`, `/api/chat`, `/api/tags`, `/api/embeddings`)
- **Web chat UI** at `/chat` — dark-themed, streaming, multi-session history, offline
- **CLI** — `squish run`, `squish chat`, `squish models`, `squish bench`, `squish info`
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
