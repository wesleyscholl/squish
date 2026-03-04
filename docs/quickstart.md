# Quickstart

Get from zero to a running 8B chat model in under two minutes.

---

## 1. Pull a model

Squish downloads pre-compressed INT8 weights from the [squish-community](https://huggingface.co/squish-community) HuggingFace org:

```bash
squish pull llama3.1:8b
```

Progress is shown as weights are streamed. Models are cached in `~/.squish/models/`.

To see all available models:

```bash
squish search
# or
squish search llama
```

---

## 2. Chat interactively

```bash
squish run llama3.1:8b
```

Opens a REPL-style chat loop. Type your message and press Enter. Use `Ctrl+D` or `/exit` to quit.

---

## 3. Single-turn prompt

```bash
squish run llama3.1:8b --prompt "Explain gradient descent in one sentence."
```

---

## 4. Start the API server

```bash
squish serve
# or specify port / host
squish serve --port 11435 --host 0.0.0.0
```

The server binds to `http://localhost:11435` by default and is **OpenAI-compatible**.

---

## 5. Call the API

=== "curl"
    ```bash
    curl http://localhost:11435/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3.1:8b",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ]
      }'
    ```

=== "Python (openai SDK)"
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:11435/v1",
        api_key="not-needed",  # squish ignores the key by default
    )

    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(response.choices[0].message.content)
    ```

=== "Python (requests)"
    ```python
    import requests

    resp = requests.post(
        "http://localhost:11435/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    print(resp.json()["choices"][0]["message"]["content"])
    ```

---

## 6. Batch inference

Send multiple prompts in a single request with the `batch` field:

```bash
curl http://localhost:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "batch": [
      "The capital of France is",
      "The largest planet is",
      "Water boils at"
    ]
  }'
```

---

## 7. Manage local models

```bash
squish list          # show downloaded models
squish rm llama3.1:8b   # delete a model
```

---

## Next steps

- [API Reference](api.md) — full endpoint documentation  
- [Architecture](architecture.md) — how INT8 mmap compression works  
- [Contributing](contributing.md) — add a model, fix a bug, write a test  
