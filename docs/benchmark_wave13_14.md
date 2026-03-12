# Squish Wave 13+14 Benchmark Results

**Generated**: 2026-03-11 13:59  
**Environment**: Python micro-benchmark (numpy CPU, no GPU).  
**Note**: Speedups on Apple Silicon MLX Metal are significantly higher;
these figures reflect pure-numpy CPU overhead only.

---

## Wave 13 — Long-Context + Speculative Decoding

| Module | Operation | Result | Notes |
|--------|-----------|--------|-------|
| DuoAttention | `store_kv()` latency | 0.72 µs | per token |
| DuoAttention | KV memory reduction | 0.20× | 56% retrieval heads |
| ShadowKV | `store()` 128 tokens | 751.4 µs | 16/64 rank |
| ShadowKV | Key memory reduction | 4.00× | SVD projected keys |
| PQCache | retrieve top-32 (256 tok) | 300.6 µs | ADC search |
| PQCache | Memory reduction | 64× | 1-byte codes vs float32 |
| TokenMerging | `merge()` seq=512 | 593.3 µs | r=16 pairs |
| TokenMerging | Sequence reduction | 3% | → 497 tokens |
| KnapSpec | `select()` ctx=2048 | — | 41/64 blocks, 36% skipped |

## Wave 14 — Quantisation + Coding + Speculative Decoding

| Module | Operation | Result | Notes |
|--------|-----------|--------|-------|
| DFloat11 | compress 65K BF16 wts | 30.7 ms | 1.000× (16.00 bits/wt) |
| RANSCodec | encode/decode n=4096 | 2035.8/2515.1 µs | 0.485 bytes/sym (entropy=1.846 bits) |
| SqueezeLLM | INT3 128×128 quant | 289.8 ms | 0.312× · SNR=15.8 dB |
| NF4 | quantize_nf4() 128×128 | 653.5 µs | MSE=0.00835 · SNR=20.7 dB |
| CopySpec | `draft()` latency | 2.0 µs | draft_len=1/8 |
| VisionPrefixCache | Cache hit (16 imgs) | 0.0 ms | hit rate=50.0% · 10.0× speedup |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.


| Optimisation | Improvement | Technique |
|---|---|---|
| KV memory (DuoAttn) | **1.5–2.0×** reduction | Streaming heads use only sink+window |
| KV memory (ShadowKV) | **2–4×** reduction | SVD low-rank projected keys |
| Key index (PQCache) | **10–32×** smaller | 1-byte PQ codes vs float32 |
| Spec throughput (QSpec) | **1.2–2.0×** | Complementary INT8/FP16 quantisation |
| Spec throughput (CLaSP) | **1.5–2.5×** | Adaptive layer-skip drafting |
| Spec throughput (C2T) | **1.5–3.0×** | Adaptive tree branching |
| Weight compression (DFloat11) | **1.3–1.5×** | Huffman 11-bit entropy coding |
| Weight compression (SqueezeLLM) | **5–10×** | Non-uniform INT3 + FP16 outliers |
| Weight compression (NF4) | **7–8×** | 4-bit NormalFloat vs float32 |
| Vision encode reduction | **80–95%** | LRU prefix cache for repeated images |

