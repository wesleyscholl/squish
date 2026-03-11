# Squish — Development Plan

> Last updated: 2026-03-11 (v5 plan added)

This document tracks completed waves, the current release, and the next phase.

---

## Versioning Convention

| Version | Waves | Theme |
|---------|-------|-------|
| **v1** | 1–11 | Core baseline — loader, quantizer, server, API, CLI, speculative decode |
| **v2** | 12 | Reasoning-Aware KV · INT3 · Async I/O |
| **v3** | 13–14 | Ultra-Long Context · Adaptive Spec-Decode · Quantisation |
| **v4** | 15–16 | Serving Intelligence · KV Architecture Evolution · Heterogeneous Compute |
| **v5** | 17–18 | Attention Architecture · Memory Management · Adaptive Compute · Model Intelligence |

---

## ✅ v1 — Core Baseline (Released 2026-03-03)

- Three-tier compressed weight loader (INT8 → f16 → bf16 MLX safetensors)
- OpenAI-compatible API server (`/v1/*`) + Ollama drop-in (`/api/*`)
- Web chat UI at `/chat`
- CLI — `squish run/serve/chat/pull/models/info/bench/catalog/compress`
- Speculative decoding, batch scheduler, KV cache quantisation, prefix cache
- Tool / function calling, Rust/PyO3 INT8 quantiser

---

## ✅ v2 — Wave 12 (Released 2026-03-04)

Modules: PM-KVQ, MixKVQ, CocktailKV, MiLo INT3, AgileIO, SageAttn, SpargeAttn

Key results: 4.2× KV memory · 5.3× weight compression · 40–60% I/O latency reduction

---

## ✅ v3 — Waves 13+14 (Released 2026-03-11)

Wave 13 (10 modules): DuoAttention, ShadowKV, PQCache, SpeCache, DuoDecoding,
KnapSpec, TokenMerging, TokenSwift, C2T, CLaSP

Wave 14 (16 modules): DFloat11, SqueezeLLM, NF4, rANS, QSpec, QuantSpec,
CopySpec, SpinQuant, VisionPrefixCache, MRLIndex, SubSpec, DELDecoder,
HeteroVocab, HeadInfer, LifeModel, SoupOfExperts

Key results: 10–30× KV memory · 55% draft acceptance · 5–10× weight compression

---

## ✅ v4 — Waves 15+16 (Released 2026-03-12)

Theme: **Serving Intelligence · KV Architecture Evolution · Heterogeneous Compute**

### Wave 15 — Serving Intelligence + KV Architecture Evolution (10 modules)

| Module | Flag | Key Result |
|--------|------|-----------|
| AdaServe | `--ada-serve` | SLO-customized spec decode trees → 30% latency ↓ for tight SLOs |
| ConfSpec | `--conf-spec` | Confidence-gated verification → 54% verification cost ↓ |
| SeqPacking | `--seq-packing` | Barrel effect elimination → 1.8× effective throughput |
| MetaReasoner | `--meta-reasoner` | Dynamic thinking budget → 44–89% energy saved on CoT |
| YOCO | `--yoco-kv` | You Only Cache Once → 50% KV memory reduction |
| DiffKV | `--diff-kv` | Asymmetric K/V precision → 2.7–5.7× KV memory, 1.9–5.4× throughput |
| KVTuner | `--kvtuner` | Sensitivity-aware mixed-precision KV → 2× compression vs naive |
| KVSharer | `--kv-share` | Cross-layer KV sharing → 30% KV memory reduction |
| ParisKV | `--paris-kv` | Drift-robust online KV quantisation → 4× KV compression |
| CLA | `--cla` | Cross-Layer Attention sharing → 10–30% KV memory reduction |

### Wave 16 — Heterogeneous Compute + Advanced Spec-Decode (11 modules)

| Module | Flag | Key Result |
|--------|------|-----------|
| Dovetail | `--dovetail` | CPU+GPU heterogeneous spec decode → 2× throughput |
| SwiftSpec | `--swift-spec` | Async disaggregated decode → minimal overlap overhead |
| PIPO | `--pipo` | Pipelined prefetch offloading → 1.7× throughput >VRAM models |
| MobileMoE | `--mobile-moe` | MoE balanced layer skip → 1.4× throughput on MoE models |
| OnlineSD | `--online-sd` | Continuous draft adaptation → +5–8 pp acceptance rate |
| LookaheadReasoning | `--lookahead` | Parallel step verification → 2.1× throughput on reasoning |
| SparseSpec | `--sparse-spec` | Dynamic sparse self-speculation → 2.13× throughput |
| FRSpec | `--fr-spec` | Frequency-ranked vocab compression → 13% draft latency ↓ |
| LongSpec | `--long-spec` | Shared-KV draft head → zero draft KV overhead at any context |
| ForeLen | `--forelen` | Entropy-guided length prediction → 29% MAE ↓ vs TRAIL |
| RASD | `--rasd` | Retrieval-augmented spec decode → 40–60% corpus hit rate |

### Deliverables checklist

- [x] All 21 modules implemented and wired in `server.py`
- [x] `tests/test_wave15_server_wiring.py` — 44 tests, 44 passing
- [x] `tests/test_wave16_server_wiring.py` — 45 tests, 45 passing
- [x] `dev/benchmarks/bench_wave15_16.py` — micro-benchmark suite
- [x] `dev/results/wave15_16_bench.json` — benchmark results
- [x] `docs/benchmark_wave15_16.md` — human-readable results table
- [x] `dev/demos/record_v4_demo.py` — v4 demo GIF generator
- [x] `dev/demos/squish-v4-demo.gif` — demo GIF rendered
- [x] README.md — v4 module sections, Wave 15+16 tables, CLI examples
- [x] CHANGELOG.md — `[2.0.0]` entry

---

## 🚧 v5 — Waves 17+18 (Next Phase)

Theme: **Attention Architecture · Memory Management · Adaptive Compute · Model Intelligence · Evaluation**

28 fully-implemented modules across two waves — all available in `squish/` but not yet
documented, benchmarked, or wired into the CLI.

---

### Wave 17 — Attention Architecture + Memory Management (14 modules)

Focus: Next-generation attention kernels, zero-allocation KV memory, prompt and
token compression, and speculative context retrieval.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **SageAttn2** | `sage_attention2.py` | `SageAttention2Kernel`, `SageAttention2Config` | `--sage-attn2` | INT4 warp QK + FP8 PxV → **~3.1× vs FlashAttention2** |
| **StreamingSink** | `streaming_sink.py` | `SinkKVCache`, `SinkConfig` | `--streaming-sink` | Attention sink eviction → **infinite context** at fixed KV budget |
| **KVSlab** | `kv_slab.py` | `KVSlabAllocator`, `KVPage` | `--kv-slab` | Pre-allocated slab → **eliminates >10 ms** per-request heap stalls |
| **SqueezeAttn** | `squeeze_attention.py` | `SqueezeKVCache`, `BudgetAllocator` | `--squeeze-attn` | Dynamic per-layer KV budget → **configurable KV footprint** |
| **SmallKV** | `smallkv.py` | `SmallKVCache`, `SaliencyTracker` | `--small-kv` | Saliency-compensated 10% KV budget → **1.75–2.56× throughput** |
| **SpeContext** | `specontext.py` | `SpeContextCache`, `DistilledRetrievalHead` | `--spe-context` | Distilled retrieval head → **>90% param reduction**, 90% transfer ↓ |
| **SVDq** | `svdq.py` | `SVDqCalibrator`, `SVDqPrecisionMap` | `--svdq` | Per-head SVD key mixed precision → **calibrated rank-aware quantisation** |
| **CommVQ** | `comm_vq.py` | `CommVQCodebook`, `MultiCodebookVQ` | `--comm-vq` | Commutative VQ KV → **8× (2-bit) / 4× (4-bit) memory, near-lossless** |
| **ChunkedPrefill** | `chunked_prefill.py` | `ChunkedPrefillConfig` | `--chunked-prefill` | Interleaved chunk+decode → **O(chunk_size) prefill latency** |
| **GemFilter** | `gemfilter.py` | `GemSelector`, `AttentionScoreBuffer` | `--gemfilter` | Early-layer token compression → **2.4× speedup, 1000× @ 108K tokens** |
| **MInference** | `minference_patch.py` | *(monkey-patch)* | `--minference` | Dynamic sparse attention → **10× prefill speedup @ 1M context** |
| **PromptCompressor** | `prompt_compressor.py` | *(functional API)* | `--prompt-compress` | Token-budget long-context trimming → **~1 ms per 1K-word prompt** |
| **PromptLookup** | `prompt_lookup.py` | `PromptLookupDecoder`, `NGramIndex` | `--prompt-lookup` | N-gram spec decode from prompt → **zero draft model required** |
| **TRAIL** | `trail.py` | `TrailPredictor`, `TrailLinearProbe` | `--trail` | Probe-layer length predictor → **2.66× lower MAE** vs BERT, **1.66–2.01× lower latency** |

### Wave 18 — Adaptive Compute + Model Intelligence + Evaluation (14 modules)

Focus: Task-adaptive layer skipping, next-generation speculative decoding,
continuous self-improvement, serving intelligence, and battery-aware evaluation.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **VPTQ** | `vptq.py` | `VPTQQuantizer`, `VPTQCodebook` | `--vptq` | Vector post-training quant (NeurIPS 2025) → **sub-2-bit weights** near fp16 quality |
| **LayerSkip** | `layer_skip.py` | `EarlyExitDecoder`, `ConfidenceEstimator` | `--layer-skip` | Early exit self-spec decode → **(total−exit)/total compute saved** per easy token |
| **SWIFT** | `swift.py` | `SWIFTDecoder`, `SWIFTCalibrator` | `--swift` | Task-adaptive layer skip with calibration → **per-task skip schedules** |
| **SpecReason** | `spec_reason.py` | `SpecReasonOrchestrator`, `ReasoningStep` | `--spec-reason` | Step-level reasoning speculation → **1.4–3.0× speedup**, **8.8–58% token reduction** |
| **MirrorSD** | `mirror_sd.py` | `MirrorSDDecoder`, `MirrorDraftPipeline` | `--mirror-sd` | Overlapped dual-pipeline draft → **2.8–5.8× vs EAGLE-3** on SpecBench |
| **SparseVerify** | `sparse_verify.py` | `SparseVerifyPass`, `InterDraftReuseCache` | `--sparse-verify` | Sparse verification + inter-draft token reuse → **verification FLOPs ↓** |
| **RobustScheduler** | `robust_scheduler.py` | `ABalancedScheduler`, `AMaxScheduler` | `--robust-sched` | Interval-prediction adaptive batching → **balanced or max-throughput policy** |
| **BlockExpertArchive** | `block_expert_archive.py` | `BlockExpertArchive`, `ExpertRouter` | `--block-archive` | K-means cluster-delta expert compression → **MoE weight deduplication** |
| **DISCRouter** | `disc_router.py` | `DISCRouter`, `DISCPlan` | `--disc-router` | Task decomposition + parallel LLM routing → **multi-step agent acceleration** |
| **SelfLearning** | `self_learning.py` | *(LearnRequest API)* | `--self-learn` | Online LoRA-delta adaptation from feedback → **continuous quality improvement** |
| **SemanticCache** | `semantic_cache.py` | `SquishSemanticCache` | `--semantic-cache` | N-gram semantic prompt dedup → **zero-model cache hits** |
| **IPW** | `ipw.py` | `IPWTracker`, `IPWMeasurement` | `--ipw` | Intelligence-per-watt tracking → **quality ÷ energy metric for M-series** |
| **PowerMonitor** | `power_monitor.py` | `PowerMonitor`, `PowerModeConfig` | `--power-monitor` | pmset-based battery-adaptive mode selection → **auto power-aware scheduling** |
| **DiffusionDraft** | `diffusion_draft.py` | `DiffusionDraftModel` | `--diffusion-draft` | Non-autoregressive diffusion LLM drafting → **short-text parallel decode** |

### v5 Deliverables checklist

- [ ] `tests/test_wave17_server_wiring.py` — import + instantiation tests for 14 modules
- [ ] `tests/test_wave18_server_wiring.py` — import + instantiation tests for 14 modules
- [ ] `dev/benchmarks/bench_wave17_18.py` — micro-benchmark suite
- [ ] `dev/results/wave17_18_bench.json` — benchmark results
- [ ] `docs/benchmark_wave17_18.md` — human-readable results table
- [ ] `dev/demos/record_v5_demo.py` — v5 demo GIF generator
- [ ] `dev/demos/squish-v5-demo.gif` — demo GIF rendered
- [ ] README.md — v5 module sections, Wave 17+18 tables, CLI examples
- [ ] CHANGELOG.md — `[3.0.0]` entry
- [ ] PLAN.md updated to mark v5 complete

### v5 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 17 (Attention + Memory) | 14 |
| Wave 18 (Adaptive Compute + Intelligence) | 14 |
| Total new v5 modules | **28** |
| Total modules after v5 | **110** |
| Expected new tests | **~112** (4 per module) |
| Total tests after v5 | **~4 049** |

---

## 🔮 v6 — Waves 19+ (Future)

Candidate themes (not yet scoped):
- **Multi-GPU / tensor parallelism** — pipeline and tensor parallel inference
- **Custom Metal/WGPU kernels** — native kernels for M-series ANE/GPU
- **FP8 weight quantisation** — next-gen 8-bit float formats
- **Advanced distillation** — iteration-level knowledge transfer
- **Edge deployment** — compiled GGUF/MLIR export, CoreML packaging
- **Evaluation harness** — extended lm-eval tasks, PPL tracking, SpecBench CI
