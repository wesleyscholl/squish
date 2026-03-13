# Squish Benchmark Guide

**Phase 11 — 5-Track Cross-Engine Comparison Suite**

The `squish bench --track <name>` command runs structured benchmarks against any
OpenAI-compatible inference engine and saves results to `eval_output/` as JSON.

---

## Quick Start

```bash
# Start a local model server first
squish run qwen3:8b

# Run a single track
squish bench --track perf --model qwen3:8b

# Run with multiple engines and compare
squish bench --track quality --engines squish,ollama --model qwen3:8b --compare

# Generate a full report after running all tracks
squish bench --track perf --model qwen3:8b --report
```

---

## Tracks

### Track A — Quality (`--track quality`)

Evaluates factual accuracy and reasoning using standard NLP benchmarks via
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

| Dataset | Few-shot | Metric |
|---------|----------|--------|
| MMLU | 5-shot | acc |
| ARC Challenge | 25-shot | acc_norm |
| HellaSwag | 10-shot | acc_norm |
| WinoGrande | 5-shot | acc |
| TruthfulQA MC1 | 0-shot | acc |
| GSM8K | 8-shot | exact_match |

**Requirements:** `pip install lm-eval`

```bash
squish bench --track quality --model qwen3:8b --limit 100
```

### Track B — Code (`--track code`)

Evaluates code generation quality using HumanEval and MBPP benchmarks.

> **Note:** Running code execution requires `--sandbox`. Without it, predictions
> are collected as JSON but not executed (safe default).

**Requirements:** `pip install lm-eval`

```bash
# Safe (no code execution)
squish bench --track code --model qwen3:8b

# With execution sandbox
squish bench --track code --model qwen3:8b --sandbox
```

### Track C — Tool Use (`--track tool`)

Tests function-calling accuracy against 20 canonical tool schemas (BFCL-style).

Metrics:
- `schema_compliance_pct` — response conforms to tool-call JSON schema
- `function_name_match_pct` — correct function name selected
- `exact_match_pct` — exact name + arguments match
- `arg_match_pct` — fraction of expected arguments present

No extra dependencies required.

```bash
squish bench --track tool --model qwen3:8b
```

### Track D — Agent (`--track agent`)

Runs 20 multi-turn agentic scenarios across four categories: file operations,
data lookup, code tasks, and multi-step workflows.

Metrics:
- `completion_rate` — fraction of scenarios where the final answer was reached
- `sequence_accuracy` — fraction of tool calls matching expected sequence
- `mean_step_efficiency` — optimal steps / actual steps (1.0 = optimal)
- `total_tokens_consumed` — sum of tokens across all turns

No extra dependencies required.

```bash
squish bench --track agent --model qwen3:8b
```

### Track E — Performance (`--track perf`)

Measures latency, throughput, and efficiency of a running inference server.

| Metric | Description |
|--------|-------------|
| `warm_ttft_ms` | Warm time-to-first-token (ms, median of 3 runs) |
| `tps` | Output tokens per second (median of 3 runs) |
| `ram_delta_mb` | RSS memory increase during benchmark (MB) |
| `long_ctx_tps` | TPS at 8× base prompt length |
| `batch_p50_ms` | P50 end-to-end latency with 8 concurrent requests |
| `batch_p99_ms` | P99 end-to-end latency with 8 concurrent requests |
| `batch_throughput_tps` | Total TPS across all concurrent requests |
| `tokens_per_watt` | Tokens/watt via `powermetrics` (macOS only; 0.0 elsewhere) |

No extra dependencies required.

```bash
squish bench --track perf --model qwen3:8b
```

---

## Supported Engines

| Name | Default URL | Notes |
|------|-------------|-------|
| `squish` | `http://localhost:11434` | Default squish server |
| `ollama` | `http://localhost:11434` | Ollama local server |
| `lmstudio` | `http://localhost:1234` | LM Studio local server |
| `mlxlm` | `http://localhost:8080` | mlx-lm server |
| `llamacpp` | `http://localhost:8080` | llama.cpp server |

Custom engines can be specified with `name=url` syntax:

```bash
squish bench --track perf --engines "custom=http://myhost:9000" --model qwen3:8b
```

---

## Output Format

Each run produces a JSON file in `eval_output/`:

```
eval_output/
  perf_qwen3_8b_squish_20260313_120000.json
  quality_qwen3_8b_squish_20260313_120100.json
```

File structure:
```json
{
  "track": "perf",
  "engine": "squish",
  "model": "qwen3:8b",
  "timestamp": "2026-03-13T12:00:00Z",
  "metrics": {
    "warm_ttft_ms": 42.5,
    "tps": 85.3,
    ...
  },
  "metadata": {
    "platform": "darwin",
    "max_tokens": 128,
    ...
  }
}
```

---

## Comparison and Reports

After collecting results from multiple engines, generate a comparison table:

```bash
squish bench --track perf --engines squish,ollama --model qwen3:8b --compare
```

Generate a full markdown report:

```bash
squish bench --track perf --model qwen3:8b --report
```

Reports are saved to `docs/benchmark_<date>.md`.

---

## Running All Tracks

```bash
MODEL=qwen3:8b
for track in quality code tool agent perf; do
    squish bench --track $track --model $MODEL --limit 50
done
squish bench --track perf --model $MODEL --compare --report
```

---

## Programmatic API

```python
from squish.benchmarks.base import SQUISH_ENGINE
from squish.benchmarks.perf_bench import PerfBenchRunner, PerfBenchConfig

runner = PerfBenchRunner(PerfBenchConfig(warm_reps=3, batch_concurrency=4))
record = runner.run(SQUISH_ENGINE, "qwen3:8b")
print(record.metrics)
record.save("eval_output/my_perf_result.json")
```

---

## Phase Gate Checklist (Phase 11)

- [x] `squish/benchmarks/base.py` — EngineConfig, EngineClient, ResultRecord, BenchmarkRunner
- [x] `squish/benchmarks/quality_bench.py` — Track A (MMLU/ARC/HellaSwag/WinoGrande/TruthfulQA/GSM8K)
- [x] `squish/benchmarks/code_bench.py` — Track B (HumanEval/MBPP)
- [x] `squish/benchmarks/tool_bench.py` — Track C (20 canonical tool schemas)
- [x] `squish/benchmarks/agent_bench.py` — Track D (20 agentic scenarios)
- [x] `squish/benchmarks/perf_bench.py` — Track E (TTFT/TPS/RAM/tokens-per-watt/batch)
- [x] `squish/benchmarks/compare.py` — Cross-engine comparison table generator
- [x] `squish/benchmarks/report.py` — Full benchmark report generator
- [x] `squish/cli.py` — `squish bench --track` / `--engines` / `--model` / `--compare` / `--report` / `--limit`
- [x] `eval_output/eval_meta.json` — Template metadata for benchmark runs
- [x] All Phase 11 unit tests pass
