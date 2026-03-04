# Squish

**Run 70B language models on a MacBook.**

Squish compresses model weights into memory-mapped INT8 tensors that load in **milliseconds**, then serves them through a fully OpenAI-compatible REST API — all on Apple Silicon, no GPU required.

---

## Why Squish?

| | Ollama | LM Studio | **Squish** |
|---|---|---|---|
| Cold-start time | ~30 s | ~20 s | **< 2 s** |
| RAM for 70B | ~40 GB | ~40 GB | **~18 GB** |
| OpenAI API | ✅ | ✅ | ✅ |
| Batch requests | ❌ | ❌ | **✅** |
| Pre-compressed weights | ❌ | ❌ | **✅ HuggingFace** |
| Quantisation format | GGUF | GGUF | **INT8 mmap** |
| Platform | macOS/Linux | macOS/Windows | macOS (M1–M5) |

---

## Key Features

- **Instant load** — memory-mapped weights skip all decoding overhead
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, `/v1/completions`
- **Batch inference** — parallel requests in a single call
- **INT8 + INT4** — two quantisation tiers for accuracy vs. size trade-offs
- **Zero-copy mmap** — model data never fully loaded into RAM
- **CLI first** — `squish pull`, `squish run`, `squish serve`, `squish rm`, `squish search`

---

## Quick Demo

```bash
# Install
brew install wesleyscholl/squish/squish

# Pull a compressed model from the community hub
squish pull llama3.1:8b

# Chat interactively
squish run llama3.1:8b

# Or start the API server and query it like OpenAI
squish serve &
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Platform

!!! warning "macOS + Apple Silicon only"
    Squish uses [Apple MLX](https://github.com/ml-explore/mlx) for inference and requires an M1–M5 chip.
    Linux/CUDA support is on the roadmap — [watch the repo](https://github.com/wesleyscholl/squish) for updates.

---

## Community

- [Discord](https://discord.gg/squish) — get help, share benchmarks, discuss models  
- [GitHub Discussions](https://github.com/wesleyscholl/squish/discussions) — Q&A, ideas, show & tell  
- [HuggingFace](https://huggingface.co/squish-community) — pre-squished model weights  
- [Contributing](contributing.md) — good first issues, dev setup, PR guidelines  
