# Squish — Multi-Model Benchmark Results

## Load Time & Throughput

| Model | Size (GB) | Compressed (GB) | Ref Load | Squish Load | Speedup | Tok/s Δ |
|-------|-----------|-----------------|----------|-------------|---------|---------|
| Qwen2.5-1.5B | 3.1 | 11.4 | 1.45s | 0.43s | **3.4×** | -21.9% |
| Qwen2.5-7B | 15.2 | 13.6 | — | 2.01s | — | — |

## Notes

- **Squish load** uses Tier 2 cache (`squish_weights.safetensors`) — sub-second on warm runs
- **Tok/s Δ** = throughput change vs reference (Vectro INT8 has near-zero quality impact)
- Benchmarks run on Apple M3 16GB unified memory
- lm-eval harness: EleutherAI lm-evaluation-harness v0.4.x

