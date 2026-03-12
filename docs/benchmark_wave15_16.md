# Squish v4 — Wave 15+16 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 15 — Serving Intelligence + KV Architecture Evolution

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| AdaServe | `get_gamma()` tight SLO | 0.91 | SLO-customized gamma selection |
| AdaServe | `get_gamma()` relaxed SLO | 0.83 | |
| ConfSpec | `verify_step()` flat logits | 131.89 | Full verification path |
| ConfSpec | `verify_step()` peaked logits | 127.56 | Auto-accept path (high confidence) |
| SeqPacking | `pack()` 32 short seqs | 2754.3 | 8–64 token sequences |
| SeqPacking | `pack()` 8 long seqs | 43442.3 | 128–512 token sequences |
| MetaReasoner | `compute_entropy()` 32k | 448.64 | Static method |
| MetaReasoner | `step()` 32k vocab | 0.12 | Per-token thinking budget decision |
| YOCO | `append()` seq=64 dim=128 | 0.59 | KV append to shared store |
| YOCO | `get_shared_kv()` | 2230.51 | Retrieve cached KV for cross-decoder layers |
| DiffKV | `get_policy()` | 1.37 | Per-head precision policy lookup |
| DiffKV | `record_attention()` 4×4 | 5.66 | Attention pattern accumulation |
| ParisKV | `encode()` batch=32 dim=128 | 29.4 | Online codebook assignment |
| ParisKV | `decode()` batch=32 | 3.9 | Codebook reconstruction |
| ParisKV | `online_update()` batch=8 | 107.2 | Drift-corrected centroid update |
| KVTuner | `search()` 32 layers | 4212.2 | Sensitivity-aware bit assignment |
| CLA | `CLASchedule.from_config()` | 19.49 | Cross-layer attention schedule gen |

---

## Wave 16 — Heterogeneous Compute + Advanced Spec-Decode

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| Dovetail | `verify_one()` vocab=32k | 602.4 | CPU target verification |
| PIPO | `run_layer()` in=out=4096 | 1376.2 | INT4 dequant + matmul w/ prefetch |
| MobileMoE | `route()` single 128 experts | 14.88 | Expert selection |
| MobileMoE | `route_batch()` 32 tokens | 486.7 | |
| OnlineSD | `record()` hidden=4096 | 1.40 | Trace buffer append |
| LookaheadReasoning | `run_cycle()` k=4 | 13.7 | Parallel step verification cycle |
| SparseSpec | `PillarAttnCache.update()` cap=4096 | 1.20 | Attention pillar accumulation |
| SparseSpec | `top_k_indices()` k=205 | 24.4 | Sparse position selection |
| FRSpec | `head.forward()` top-25% vocab | 4095.0 | Compressed draft logits |
| FRSpec | `compress_logits()` 32k→subset | 12.6 | Vocab projection |
| FRSpec | `expand_logits()` subset→32k | 21.8 | Full-vocab restore |
| LongSpec | `LongSpecHead.forward()` h=4096 | 12434.7 | Shared-KV draft head |
| ForeLen | `EGTPPredictor.predict()` | 99.12 | Entropy histogram → length |
| ForeLen | `PLPPredictor.update()` | 1.42 | Exponential decay estimate |
| RASD | `CorpusIndex.search()` 1k seqs | 0.72 | Prefix-tree lookup |
| RASD | `build_retrieval_tree()` | 1.83 | Draft tree construction |

---

## Reference: Paper-Reported Technique Improvements
> **Note:** These are technique-level estimates derived from published papers.
> End-to-end validation on Squish with a loaded model on Apple Silicon
> has not yet been run for this wave.
> See `dev/benchmarks/bench_eoe.py` for the real-hardware benchmark harness.


| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| KV memory (YOCO) | **50%** reduction | YOCO — only cross-decoder layers use KV |
| KV memory (DiffKV) | **2.7–5.7×** compression | DiffKV asymmetric K/V precision |
| KV memory (KVTuner) | **2×** vs naive quant | KVTuner mixed-precision calibration |
| CoT decode energy | **44–89%** saving | MetaReasoner dynamic thinking budget |
| Batch throughput | **1.8×** effective | SeqPacking barrel-effect elimination |
| Spec decode throughput | **2.13×** | SparseSpec dynamic sparse self-speculation |
| Reasoning throughput | **2.1×** | LookaheadReasoning parallel step verification |
| Offloaded model throughput | **1.7×** | PIPO pipelined prefetch offloading |
| Heterogeneous throughput | **2×** | Dovetail CPU+GPU spec decode |
| Draft acceptance | **+5–8 pp** | OnlineSD continuous adaptation |
| Length prediction (MAE) | **29% ↓** vs TRAIL | ForeLen entropy-guided prediction |
| Corpus hit rate | **40–60%** | RASD retrieval-augmented spec decode |

---

## Accuracy Baseline (unchanged — v4 operates on KV / serving paths)

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
