# API Reference

Squish exposes an **OpenAI-compatible REST API** on `http://localhost:11435` by default.

---

## Authentication

By default the server accepts requests without an API key.

To require a key, set the environment variable before starting the server:

```bash
export SQUISH_API_KEY=my-secret-key
squish serve
```

Pass the key using the standard `Authorization: Bearer <key>` header.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SQUISH_API_KEY` | _(none)_ | When set, all API requests must supply `Authorization: Bearer <key>` |
| `HF_TOKEN` | _(none)_ | HuggingFace access token — required for gated models |
| `SQUISH_OFFLINE` | `0` | Set to `1` to disable all network access (model must already be cached) |
| `SQUISH_CACHE_DIR` | `~/.squish/models` | Override the default model cache directory |

---

## Endpoints

### `GET /v1/models`

Lists all locally available models.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.1:8b",
      "object": "model",
      "created": 1720000000,
      "owned_by": "squish"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

OpenAI-compatible chat completion.

**Request body**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | ✅ | — | Model ID (e.g. `llama3.1:8b`) |
| `messages` | array | ✅ | — | Array of `{"role": "...", "content": "..."}` |
| `max_tokens` | integer | | 512 | Maximum tokens to generate |
| `temperature` | float | | 0.7 | Sampling temperature |
| `top_p` | float | | 0.9 | Top-p nucleus sampling |
| `stream` | boolean | | false | Stream tokens via SSE |
| `stop` | string/array | | null | Stop sequence(s) |

**Example**

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is MLX?"}
    ],
    "max_tokens": 128,
    "temperature": 0.5
  }'
```

**Response**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1720000001,
  "model": "llama3.1:8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "MLX is Apple's machine-learning framework optimised for Apple Silicon..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 42,
    "total_tokens": 66
  }
}
```

---

### `POST /v1/completions`

Text completion (non-chat). Supports single prompts and **batched** requests.

**Request body**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | ✅ | — | Model ID |
| `prompt` | string | ✅* | — | Single prompt text |
| `batch` | array | ✅* | — | Array of prompt strings (mutually exclusive with `prompt`) |
| `max_tokens` | integer | | 256 | Maximum tokens per completion |
| `temperature` | float | | 0.7 | Sampling temperature |
| `top_p` | float | | 0.9 | Top-p nucleus sampling |

*Either `prompt` or `batch` is required.

**Single prompt example**

```bash
curl http://localhost:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Once upon a time"}'
```

**Batch example**

```bash
curl http://localhost:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "batch": ["The sky is", "The ocean is", "The forest is"],
    "max_tokens": 32
  }'
```

---

### `GET /health`

Liveness probe.

```bash
curl http://localhost:11435/health
# {"status": "ok"}
```

---

## Streaming

Set `"stream": true` in a `/v1/chat/completions` request to receive tokens via **Server-Sent Events** (SSE), exactly like the OpenAI API:

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

Each SSE event is a JSON delta. The stream ends with `data: [DONE]`.

---

## Error responses

| HTTP status | Meaning |
|---|---|
| 400 | Bad request — missing or invalid fields |
| 401 | Unauthorized — invalid or missing API key |
| 404 | Model not found — run `squish pull <model>` first |
| 429 | Too many requests — queue is full, back off and retry |
| 500 | Internal server error — check server logs |

---

## CLI reference

| Command | Description |
|---|---|
| `squish pull <model>` | Download + compress a model |
| `squish run <model>` | Interactive chat REPL |
| `squish run <model> --prompt "..."` | Single-turn inference |
| `squish serve` | Start the API server |
| `squish serve --port N` | Custom port |
| `squish list` | List downloaded models |
| `squish rm <model>` | Delete a model |
| `squish search [query]` | Search the community hub |
