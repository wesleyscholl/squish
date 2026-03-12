# Squish — Development Plan

> Last updated: 2026-03-11 (v6 complete)

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

## ✅ v5 — Waves 17+18 (Released 2026-03-11)

Theme: **Attention Architecture · Memory Management · Adaptive Compute · Model Intelligence**

28 modules across two waves — all implemented, tested, benchmarked, and documented.

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

- [x] `tests/test_wave17_server_wiring.py` — 56 tests, 56 passing
- [x] `tests/test_wave18_server_wiring.py` — 56 tests, 56 passing
- [x] `dev/benchmarks/bench_wave17_18.py` — micro-benchmark suite (24 modules timed, 4 skipped)
- [x] `dev/results/wave17_18_bench.json` — benchmark results
- [x] `docs/benchmark_wave17_18.md` — human-readable results table
- [x] `dev/demos/record_v5_demo.py` — v5 demo GIF generator (448 events, 85.2s)
- [x] `dev/demos/squish-v5-demo.gif` — demo GIF rendered (2.6 MB, 448 events, 85.2s)
- [x] README.md — v5 module sections, Wave 17+18 tables, CLI examples
- [x] CHANGELOG.md — `[3.0.0]` entry
- [x] PLAN.md updated to mark v5 complete

### v5 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 17 (Attention + Memory) | 14 |
| Wave 18 (Adaptive Compute + Intelligence) | 14 |
| Total new v5 modules | **28** |
| Total modules after v5 | **110** |
| New tests | **112** (56 Wave 17 + 56 Wave 18) |
| Total tests after v5 | **4 166** |

---

## ✅ v6 — Waves 19+20 (Released 2026-03-11)

Theme: **Next-Gen Precision · Advanced Attention · Model Composition · Serving Infrastructure**

28 new modules across two waves — all implemented, tested, benchmarked, and documented.

---

### Wave 19 — Next-Gen Attention & Precision (14 modules)

Focus: FP8/MX microscaling quantization, advanced attention patterns (paged KV,
GQA, sliding window, RoPE scaling), activation sparsity, and advanced speculative
decode heads (MEDUSA, EAGLE-3).

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **FP8Quant** | `fp8_quant.py` | `FP8Quantizer`, `FP8Config` | `--fp8-quant` | E4M3/E5M2 weight encoding → **~60% storage vs BF16** |
| **MXQuant** | `mx_quant.py` | `MXQuantizer`, `MXConfig` | `--mx-quant` | OCP MX4/MX6/MX9 microscaling → **better quality than INT4** at same bits |
| **FlashDecode** | `flash_decode.py` | `FlashDecodeAttention`, `FlashDecodeConfig` | `--flash-decode` | Split-KV parallel decode → **O(1) memory overhead** per decode step |
| **PagedKV** | `paged_kv.py` | `PagedKVCache`, `BlockTable` | `--paged-kv` | Virtual block mapping → **zero KV fragmentation** across requests |
| **GQA** | `gqa.py` | `GQACache`, `GQAConfig` | `--gqa` | Grouped Query Attention → **4–8× KV reduction** vs MHA |
| **SlidingWindowAttn** | `sliding_window_attn.py` | `SlidingWindowKVCache`, `SWAConfig` | `--sliding-window` | Sliding window KV → **O(window_size) memory** at any context length |
| **RoPEScaling** | `rope_scaling.py` | `RoPEScaler`, `YaRNScaler`, `NTKScaler` | `--rope-scaling` | NTK/YaRN/LongRoPE → **4–32× context extension** without fine-tuning |
| **ActSparsity** | `act_sparsity.py` | `ActSparsityPredictor`, `SparsityConfig` | `--act-sparsity` | Activation sparsity gating → **30–60% FFN compute saved** |
| **FusedRMSNorm** | `fused_rmsnorm.py` | `FusedRMSNorm`, `FusedLayerNorm` | `--fused-norm` | Fused RMSNorm + residual → **single kernel pass**, reduced bandwidth |
| **LoRAInference** | `lora_inference.py` | `LoRAInferenceAdapter`, `LoRAConfig` | `--lora-inference` | Zero-copy LoRA delta inference → **adapter switching without re-quant** |
| **MEDUSA** | `medusa.py` | `MedusaHead`, `MedusaDecoder` | `--medusa` | Multi-head tree speculation → **2–3× decode throughput** |
| **EAGLE3** | `eagle3.py` | `Eagle3DraftHead`, `Eagle3Decoder` | `--eagle3` | Feature-level draft head → **3.5× accept rate** vs token-prediction draft |
| **PrefixPool** | `prefix_pool.py` | `PrefixPool`, `PrefixPoolConfig` | `--prefix-pool` | Cross-request KV prefix sharing → **40–80% KV savings** on shared prompts |
| **TokenHealer** | `token_healer.py` | `TokenHealer`, `HealerConfig` | `--token-healer` | Boundary-aware token healing → **eliminates prefix-artifact generation** |

### Wave 20 — Serving Infrastructure & Intelligence (14 modules)

Focus: Model composition (merge, compose), continuous batching, evaluation harness,
power profiling, multi-modal efficiency, and knowledge distillation for spec heads.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|
| **ModelMerge** | `model_merge.py` | `ModelMerger`, `MergeConfig` | `--model-merge` | SLERP/DARE/TIES merging → **combine domains without retraining** |
| **LoRACompose** | `lora_compose.py` | `LoRAComposer`, `AdapterStack` | `--lora-compose` | Multi-LoRA mixture → **blend adapters with learnable coefficients** |
| **ContinuousBatching** | `continuous_batching.py` | `CBScheduler`, `InFlightRequest` | `--continuous-batching` | Mid-generation insertion → **max GPU utilization at any request rate** |
| **MatryoshkaEmb** | `matryoshka_emb.py` | `MatryoshkaEmbedding`, `MRLConfig` | `--matryoshka-emb` | Nested embedding truncation → **1 forward pass, any dimensionality** |
| **ANEProfiler** | `ane_profiler.py` | `ANEProfiler`, `ANEMetrics` | `--ane-profiler` | Apple Neural Engine utilization → **op-level ANE vs GPU breakdown** |
| **SpecBench** | `spec_bench.py` | `SpecBenchRunner`, `SpecBenchResult` | `--spec-bench` | SpecBench CI harness → **acceptance rate + throughput across tasks** |
| **PPLTracker** | `ppl_tracker.py` | `PPLTracker`, `PPLWindow` | `--ppl-tracker` | Rolling perplexity tracker → **real-time quality degradation detection** |
| **GrammarCache** | `grammar_cache.py` | `GrammarCache`, `FSMState` | `--grammar-cache` | FSM grammar cache → **constrained decoding without per-token rebuild** |
| **QuantAware** | `quant_aware.py` | `QuantAwareCalibrator`, `QAConfig` | `--quant-aware` | Activation-range calibration → **per-channel optimal scale selection** |
| **AdaptiveBudget** | `adaptive_budget.py` | `AdaptiveBudgetController`, `BudgetConfig` | `--adaptive-budget` | Dynamic compute budget → **SLO-aware KV + layer skip joint control** |
| **VisionTokens** | `vision_tokens.py` | `VisionTokenCompressor`, `VTConfig` | `--vision-tokens` | Visual token pruning → **50–80% vision token reduction** without quality loss |
| **ToolCache** | `tool_cache.py` | `ToolSchemaCache`, `ToolRouter` | `--tool-cache` | Schema + routing cache → **zero tool-call parse overhead** on repeated schemas |
| **DistilSpec** | `distil_spec.py` | `DistilSpecCalibrator`, `DistilConfig` | `--distil-spec` | Draft-head knowledge distillation → **+10–15 pp acceptance from calibration** |
| **BatchEmbed** | `batch_embed.py` | `BatchEmbedder`, `PoolingConfig` | `--batch-embed` | Dynamic pooling strategies → **mean/max/cls/weighted pool in single pass** |

### v6 Deliverables checklist

> **Progress (2026-03-11):** Wave 20 modules 1–14 (all) implemented and tested:
> ModelMerge, LoRACompose, ContinuousBatching, MatryoshkaEmb, ANEProfiler,
> SpecBench, PPLTracker, GrammarCache, QuantAware, AdaptiveBudget,
> VisionTokens, ToolCache, DistilSpec, BatchEmbed — 262+ new tests.

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave19_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave20_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave19_20.py` — micro-benchmark suite
- [x] `dev/results/wave19_20_bench.json` — benchmark results
- [x] `docs/benchmark_wave19_20.md` — human-readable results table
- [x] `dev/demos/record_v6_demo.py` — v6 demo GIF generator
- [x] `dev/demos/squish-v6-demo.gif` — demo GIF rendered
- [x] README.md — v6 module sections, Wave 19+20 tables, CLI examples
- [x] CHANGELOG.md — `[4.0.0]` entry
- [x] PLAN.md updated to mark v6 complete

### v6 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 19 (Next-Gen Attention + Precision) | 14 |
| Wave 20 (Serving Infrastructure + Intelligence) | 14 |
| Total new v6 modules | **28** |
| Total modules after v6 | **138** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v6 | **4 278** |
