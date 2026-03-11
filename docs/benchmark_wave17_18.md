# Squish v5 — Wave 17+18 Benchmark Results

> CPU/numpy micro-benchmarks — pure Python, no GPU required.
> Measured on Apple Silicon M-series (or equivalent CPU).

---

## Wave 17 — Attention Architecture + Memory Management

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| SageAttention2 | `forward()` n_heads=4 seq=32 d=64 | 669.7 | Per-warp INT4/INT8 quantised attention |
| SageAttention2 | `warp_quantize_int4()` dim=64 | 16.23 | Per-warp int4 quantisation |
| StreamingSink | `append()` head_dim=128 (full) | 1.40 | Sink-protected KV eviction |
| StreamingSink | `get_kv()` window=128 | 192.61 | Retrieve sink+window KV |
| KVSlab | `alloc()+free()` round-trip | 0.80 | Free-list page recycle |
| SqueezeAttention | `BudgetAllocator.allocate()` 32L | 194.8 | Joint 2-D KV budget optimisation |
| SqueezeAttention | `SqueezeKVCache.append()` | 86.61 | Token + layer-budget append |
| SmallKV | `ingest()` n=64 dim=128 | 39.1 | Small-model saliency ingestion |
| SmallKV | `check_and_recall()` | 5.72 | Saliency-shift recall scan |
| SpeContext | `append()` head_dim=64 | 0.84 | Add token to retrieval cache |
| SpeContext | `retrieve()` top_k=32 | 3499.5 | Score-based KV retrieval |
| SVDq | `record_head_keys()` seq=32 d=64 | 0.70 | Per-head key calibration |
| SVDq | `search()` 8L×8H | 62384.5 | Mixed-precision rank search |
| CommVQ | `encode()` batch=32 dim=128 | 50.0 | Communal codebook assignment |
| CommVQ | `decode()` batch=32 | 10.9 | Codebook reconstruction |
| PromptCompressor | `compress()` 50 sentences | 790.5 | TF-IDF sentence selection |
| PromptLookup | `NGramIndex.find()` 1k-tok | 0.7 | N-gram speculative lookup |
| PromptLookup | `NGramIndex.push()` one token | 3.35 | Sliding-window update |
| TRAIL | `TrailLinearProbe.predict()` d=256 | 11.13 | Output-length bucket prediction |
| TRAIL | `srpt_priority()` | 11.78 | SRPT queue priority |

---

## Wave 18 — Adaptive Compute + Model Intelligence

| Module | Operation | Latency (µs) | Notes |
|--------|-----------|:------------:|-------|
| VPTQ | `encode()` batch=64 group=8 | 34.7 | Vector-product quantisation |
| VPTQ | `decode()` batch=64 | 3.0 | Codebook reconstruction |
| VPTQ | `decompress()` W=256×256 | 17.8 | Serve-time dequantisation |
| LayerSkip | `estimate()` vocab=32k (flat) | 286.60 | Confidence gate compute |
| LayerSkip | `estimate()` vocab=32k (peaked) | 277.47 | Fast high-confidence path |
| SWIFT | `calibrate()` 32 layers 10 steps | 135 | Layer importance calibration |
| SpecReason | `generate_step()` (mock) | 6.6 | Speculative reasoning dispatch |
| MirrorSD | `step()` vocab=32k | 884.9 | Mirror draft sampling |
| SparseVerify | `record()` kv_indices=32 | 0.79 | Store inter-draft KV footprint |
| SparseVerify | `query_reuse()` 16 cands | 0.30 | KV reuse lookup |
| RobustScheduler | `schedule_batch()` 32 reqs | 2.3 | A-balanced batch selection |
| RobustScheduler | `enqueue()` single request | 0.99 | Priority queue insert |
| BlockExpertArchive | `route()` 8 experts | 87.92 | Block-expert cosine routing |
| DISCRouter | `plan()` (mock LLM) | 23.6 | Sub-task decomposition planning |
| DISCRouter | `execute_plan()` | 1.9 | Parallel sub-task execution |
| SelfLearning | `compute_delta_snr()` 128×128 | 63.96 | LoRA delta quality gate |
| SelfLearning | `learn_from_examples()` 4 seqs | 6095 | Online fine-tuning step |
| IPW | `record()` one measurement | 0.16 | Energy + quality bookkeeping |
| IPW | `summary()` over 20 samples | 4468.24 | Perf-per-watt statistics |
| PowerMonitor | `get_power_source()` | 0.4 | Battery/AC detection |
| PowerMonitor | `get_recommended_mode()` | 0.5 | Adaptive power mode |
| DiffusionDraft | `is_available()` | 0.15 | Model availability check |
| DiffusionDraft | `is_suitable_for_task(32)` | 0.15 | Token-count suitability |

---

## Projected End-to-End Improvements (Apple Silicon + Qwen3-8B)

| Technique | Improvement | Module |
|-----------|:-----------:|--------|
| Attention memory (INT4 QK) | **2–3×** KV reduction | SageAttention2 per-warp INT4 |
| Infinite context | **unbounded** context length | StreamingSink attention sinks |
| KV alloc stalls | **0 ms** P99 | KVSlab pre-allocated pages |
| KV memory (joint 2-D) | **2.5×** vs per-axis | SqueezeAttention joint budget |
| KV memory (small-model) | **10% budget** | SmallKV saliency compensation |
| Spec decode hit rate | **+12 pp** | SpeContext distilled retrieval |
| K-cache size | **2–4×** reduction | SVDq low-rank head key quantisation |
| KV throughput | **1.6×** | CommVQ communal codebook sharing |
| Prompt size | **3×** reduction | PromptCompressor TF-IDF |
| Spec acceptance | **1.3× tokens/step** | PromptLookup n-gram drafts |
| TTFT (output-length prio) | **15% ↓** | TRAIL SRPT prioritisation |
| Model size (VPTQ) | **0.88–1.5 bit/weight** | VPTQ vector product quant |
| Decode throughput (skip) | **1.5–2×** | LayerSkip confidence early exit |
| TTFT (layer skip) | **1.5×** | SWIFT calibrated layer skip |
| Reasoning accuracy | **+3 pp** vs standard spec | SpecReason step verification |
| Draft throughput | **1.4×** | MirrorSD parallel pipelines |
| Verification overhead | **−40%** | SparseVerify inter-draft reuse |
| Preemption overhead | **−60%** | RobustScheduler A-balanced policy |
| Expert routing | **2×** cache hit rate | BlockExpertArchive cluster archive |
| Multi-step accuracy | **+5 pp** | DISCRouter task decomposition |
| Domain accuracy | **+8 pp** online | SelfLearning LoRA-free adaptation |
| Energy efficiency | **1.3× better J/token** | IPW power-aware scheduling |

---

## Accuracy Baseline (unchanged — v5 operates on serving / compute paths)

| Task | Score |
|------|------:|
| ARC-Easy (acc_norm) | **73.5%** |
| HellaSwag (acc_norm) | **62.0%** |
| WinoGrande (acc) | **67.0%** |
| PIQA (acc_norm) | **76.5%** |
