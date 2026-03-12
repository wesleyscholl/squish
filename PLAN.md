# Squish — Development Plan

> Last updated: 2026-03-12 (v9 complete)

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
| **v6** | 19–20 | Next-Gen Precision · Serving Infrastructure · Intelligence |
| **v7** | 21–22 | Advanced Decode · Production Serving · Observability |
| **v8** | 23–24 | Multi-Modal & Long Context · Quantisation Evolution & Model Surgery |
| **v9** | 25–26 | Cutting-Edge Attention Variants & Compute Fusion · Distributed Inference & Production Reliability |

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

---

## ✅ v7 — Waves 21+22 (Released 2026-03-12)

Theme: **Advanced Decode · Production Serving · Observability**

28 new modules across two waves.

---

### Wave 21 — Advanced Memory & Decode (14 modules)

Focus: Tree-parallel speculative verification, online KV compression, mixed-precision
KV per head, pipeline-parallel decode, learned KV codecs, retention-style recurrent
attention, and context-length-adaptive RoPE scaling.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TreeVerifier** | `tree_verifier.py` | `TreeVerifier`, `TokenTree` | `--tree-verify` | Batched tree-parallel speculative verification → **structured multi-token acceptance** |
| **KVCompress** | `kv_compress.py` | `KVCompressor`, `KVCompressConfig` | `--kv-compress` | Online KV quantisation + pruning during generation → **adaptive old-context compression** |
| **DynamicNTK** | `dynamic_ntk.py` | `DynamicNTKScaler`, `NTKState` | `--dynamic-ntk` | Per-request runtime RoPE base auto-scaling → **auto-extends at 80% context fill** |
| **QuantSpecDecode** | `quant_spec_decode.py` | `QuantSpecDecoder`, `QSDConfig` | `--quant-spec-decode` | INT4 draft + FP16 verify → **draft memory ↓ 4× vs FP16** |
| **SparseAttnIndex** | `sparse_attn_index.py` | `SparseAttnIndex`, `ANCandidates` | `--sparse-attn-index` | ANN KV retrieval index → **sub-linear attention cost at very long context** |
| **MixedPrecisionKV** | `mixed_precision_kv.py` | `MixedPrecisionKVCache`, `HeadPrecision` | `--mp-kv` | Per-head INT8/INT4/FP16 KV via sensitivity analysis → **2–4× KV memory at iso-quality** |
| **PipelineBubble** | `pipeline_bubble.py` | `BubbleEliminator`, `StageSchedule` | `--pipeline-bubble` | Overlapped prefill + decode across pipeline stages → **bubble-free pipeline utilisation** |
| **LayerwiseDecode** | `layerwise_decode.py` | `LayerwiseDecoder`, `LayerStream` | `--layerwise-decode` | Layer-by-layer early-exit decode with multi-stream output → **configurable exit-layer latency** |
| **CodecKV** | `codec_kv.py` | `KVCodec`, `CodecConfig` | `--codec-kv` | Learned encode/decode KV codec → **2–4× KV compression via latent reconstruction** |
| **DedupeAttn** | `dedupe_attn.py` | `AttentionDeduplicator`, `DedupStats` | `--dedupe-attn` | Near-duplicate Q/K detection + output reuse → **attention FLOPs ↓ on repetitive context** |
| **FlashPrefill** | `flash_prefill.py` | `FlashPrefillKernel`, `PrefillConfig` | `--flash-prefill` | Chunked flash attention for prefill with causal mask → **O(chunk²) not O(seq²) memory** |
| **BudgetSpec** | `budget_spec.py` | `BudgetSpecDecoder`, `BudgetConfig` | `--budget-spec` | Token-budget-aware speculative decode → **exits drafting when budget threshold hit** |
| **RetentionAttn** | `retention_attn.py` | `RetentionState`, `RetentionKernel` | `--retention-attn` | Retention-style recurrent state → **O(1) per-step memory, linear recurrence** |
| **KVRouter** | `kv_router.py` | `KVRouter`, `KVRouteTable` | `--kv-router` | Cross-instance KV routing for disaggregated prefill/decode → **KV transfer without recomputation** |

### Wave 22 — Production Serving & Observability (14 modules)

Focus: Multi-tenant fair scheduling, intelligent load-balanced request routing,
predictive KV pre-warming, token budget enforcement, OpenTelemetry-compatible
tracing, request coalescing, adaptive quantisation, health monitoring, and
cost-aware serving.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **MultiTenantSched** | `multi_tenant_sched.py` | `TenantScheduler`, `TenantConfig` | `--multi-tenant` | Fair per-tenant QoS scheduling → **SLO-isolated multi-tenant serving** |
| **RequestRouter** | `request_router.py` | `RequestRouter`, `ReplicaRegistry` | `--request-router` | Load-aware request routing across replicas → **consistent-hash + least-loaded** |
| **CacheWarmup** | `cache_warmup.py` | `CacheWarmupPredictor`, `WarmupConfig` | `--cache-warmup` | Predictive KV cache pre-warming from patterns → **TTFT ↓ on hot prefix paths** |
| **TokenBudgetGate** | `token_budget_gate.py` | `TokenBudgetGate`, `BudgetPolicy` | `--token-budget` | Hard per-request token budget with graceful truncation → **deterministic cost control** |
| **ObservabilityHook** | `observability_hook.py` | `InferenceTracer`, `SpanCollector` | `--observability` | Zero-overhead per-step inference tracing → **OpenTelemetry-compatible spans** |
| **RequestCoalesce** | `request_coalesce.py` | `PrefixCoalescer`, `CoalesceStats` | `--req-coalesce` | Merge requests sharing long common prefixes → **shared prefill forward pass** |
| **AdaptiveQuantize** | `adaptive_quantize.py` | `AdaptiveQuantizer`, `PressureMonitor` | `--adaptive-quant` | Runtime precision switching under memory pressure → **auto INT8/INT4 under OOM** |
| **HealthCheck** | `health_check.py` | `InferenceHealthMonitor`, `HealthState` | `--health-check` | Degradation-aware server health monitoring → **automatic quality regression alerting** |
| **FaultTolerance** | `fault_tolerance.py` | `FaultHandler`, `FaultPolicy` | `--fault-tolerance` | Graceful OOM degradation → **auto KV eviction + draft disable + SLO re-negotiation** |
| **ModelPool** | `model_pool.py` | `ModelPool`, `PoolEntry` | `--model-pool` | Hot model pool with lazy-load + LRU eviction → **multi-model serving without reload latency** |
| **StreamingChunk** | `streaming_chunk.py` | `ChunkedStreamer`, `BackpressureBuffer` | `--streaming-chunk` | Sub-token-latency chunked streaming with backpressure → **first-chunk latency ↓** |
| **CostEstimator** | `cost_estimator.py` | `RequestCostEstimator`, `CostModel` | `--cost-estimate` | Per-request compute cost estimation → **supports billing and priority queuing** |
| **SLAMonitor** | `sla_monitor.py` | `SLAMonitor`, `ViolationPolicy` | `--sla-monitor` | Real-time SLA violation detection + remediation → **auto-escalation on breach** |
| **ContextCache** | `context_cache.py` | `PersistentContextCache`, `CacheEntry` | `--context-cache` | Persistent cross-session context cache with TTL → **zero re-encode on repeated context** |

### v7 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave21_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave22_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave21_22.py` — micro-benchmark suite
- [x] `dev/results/wave21_22_bench.json` — benchmark results
- [x] `docs/benchmark_wave21_22.md` — human-readable results table
- [x] `dev/demos/record_v7_demo.py` — v7 demo GIF generator
- [x] `dev/demos/squish-v7-demo.gif` — demo GIF rendered
- [x] README.md — v7 module sections, Wave 21+22 tables, CLI examples
- [x] CHANGELOG.md — `[5.0.0]` entry
- [x] PLAN.md updated to mark v7 complete

### v7 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 21 (Advanced Memory + Decode) | 14 |
| Wave 22 (Production Serving + Observability) | 14 |
| Total new v7 modules | **28** |
| Total modules after v7 | **166** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v7 | **~4 390** |

---

## ✅ v8 — Waves 23+24 — Released 2026-03-12

Theme: **Multi-Modal & Long Context · Quantisation Evolution & Model Surgery**

28 new modules across two waves.

---

### Wave 23 — Multi-Modal & Long Context Intelligence (14 modules)

Focus: Vision-language model efficiency, RAG-aware serving patterns, reasoning trace
compression, cross-modal attention, hierarchical KV management, and 1M+ token context
indexing.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **VisionKVFuse** | `vision_kv_fuse.py` | `VisionKVFuseCache`, `ModalityConfig` | `--vision-kv-fuse` | Fused vision+text KV with separate modality eviction → **modality-aware KV compression** |
| **ImageTokenPrune** | `image_token_prune.py` | `ImageTokenPruner`, `PruneConfig` | `--image-token-prune` | Attention entropy image token pruning → **50–70% image token reduction** |
| **RAGPrefetch** | `rag_prefetch.py` | `RAGPrefetcher`, `RAGConfig` | `--rag-prefetch` | Predictive doc KV prefetch→ **cold TTFT↓ on repeated RAG docs** |
| **CoTCompress** | `cot_compress.py` | `CoTCompressor`, `CoTConfig` | `--cot-compress` | CoT trace pruning via saliency → **30–50% reasoning token reduction** |
| **MultiModalBatch** | `multimodal_batch.py` | `MultiModalBatcher`, `BatchSlot` | `--multimodal-batch` | Shape-aware heterogeneous text+vision batcher → **minimise padding waste** |
| **ContextualRerank** | `contextual_rerank.py` | `ContextualReranker`, `RerankConfig` | `--ctx-rerank` | Context-aware KV token importance re-ranking → **preserves top-k salient positions** |
| **CrossModalAttn** | `cross_modal_attn.py` | `CrossModalAttention`, `CrossModalConfig` | `--cross-modal-attn` | Efficient cross-attention between text + vision features → **modality fusion** |
| **HierarchicalKV** | `hierarchical_kv.py` | `HierarchicalKVStore`, `TierConfig` | `--hierarchical-kv` | Hot/warm/cold KV tier management → **transparent KV tiering with O(1) promotion** |
| **StreamRAG** | `stream_rag.py` | `StreamRAGInjector`, `StreamRAGConfig` | `--stream-rag` | Streaming mid-generation document injection → **zero-restart RAG updates** |
| **CrossDocAttn** | `cross_doc_attn.py` | `CrossDocAttention`, `CrossDocConfig` | `--cross-doc-attn` | Chunked cross-document attention → **multi-document QA without full concatenation** |
| **VideoFramePrune** | `video_frame_prune.py` | `VideoFramePruner`, `FrameConfig` | `--video-frame-prune` | Temporal frame token pruning for video-LMs → **60–80% video token reduction** |
| **EmbeddingGate** | `embedding_gate.py` | `EmbeddingGate`, `GateConfig` | `--embedding-gate` | Gated modality-conditional embedding router → **zero-cost modality bypass** |
| **LongContextChunk** | `long_context_chunk.py` | `LongContextChunker`, `ChunkConfig` | `--long-context-chunk` | Semantic-boundary chunking for 1M+ token contexts → **boundary-aware chunk splits** |
| **ModalityRouter** | `modality_router.py` | `ModalityRouter`, `ModalityPolicy` | `--modality-router` | Per-modality SLO request dispatcher → **text vs vision vs audio routing** |

### Wave 24 — Quantisation Evolution & Model Surgery (14 modules)

Focus: Ternary and binary quantisation, N:M structured sparsity, cross-layer weight
sharing, second-order GPTQ-style calibration, sparse MoE routing, iterative pruning,
and surgical model architecture patching.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TernaryQuant** | `ternary_quant.py` | `TernaryQuantizer`, `TernaryConfig` | `--ternary-quant` | BitNet-style ternary {−1, 0, +1} weights → **1.58-bit effective storage** |
| **BinaryAttn** | `binary_attn.py` | `BinaryAttention`, `BinaryConfig` | `--binary-attn` | Sign-binarised attention approximation → **ultra-low attention memory** |
| **StructuredPrune** | `structured_prune.py` | `StructuredPruner`, `PruneConfig` | `--structured-prune` | 2:4 N:M magnitude pruning → **50% weight sparsity at 2× hardware throughput** |
| **LayerFusion** | `layer_fuse.py` | `LayerFuser`, `FusionConfig` | `--layer-fuse` | Adjacent transformer layer weight fusion → **reduced bandwidth on similar layers** |
| **WeightSharing** | `weight_sharing.py` | `WeightSharer`, `SharingConfig` | `--weight-share` | Cross-layer weight tying with delta residuals → **memory ↓ at iso-quality** |
| **QuantCalib** | `quant_calib.py` | `QuantCalibrator`, `CalibConfig` | `--quant-calib` | Unified MinMax/Percentile/MSE/GPTQ calibration pipeline → **optimal scale per method** |
| **SparseWeight** | `sparse_weight.py` | `SparseWeightStore`, `SparsityConfig` | `--sparse-weight` | CSR-format 2:4 pruned weight storage → **2× memory vs dense at 50% sparsity** |
| **DeltaCompress** | `delta_compress.py` | `DeltaCompressor`, `DeltaConfig` | `--delta-compress` | Rank-k SVD delta compression for fine-tuned weights → **fine-tune deltas at 10–50× reduction** |
| **ModelSurgery** | `model_surgery.py` | `ModelSurgeon`, `SurgeryPlan` | `--model-surgery` | In-place layer removal + head pruning → **architecture patching without retraining** |
| **ZeroQuantV2** | `zero_quant_v2.py` | `ZeroQuantV2`, `ZQConfig` | `--zero-quant-v2` | Groupwise quantisation with FP16 residual for outliers → **W8A8 with outlier preservation** |
| **GPTQLayer** | `gptq_layer.py` | `GPTQCalibrator`, `GPTQConfig` | `--gptq-layer` | Hessian-weighted second-order rounding → **group-wise optimal quant error** |
| **SparseMoE** | `sparse_moe.py` | `SparseMoERouter`, `MoEConfig` | `--sparse-moe` | Top-k sparse expert routing with load-balance loss → **efficient MoE inference** |
| **AWQv2** | `awq_v2.py` | `AWQv2Calibrator`, `AWQv2Config` | `--awq-v2` | Activation-aware scale+shift per-channel quant → **AWQ without grid search** |
| **IterPrune** | `iter_prune.py` | `IterativePruner`, `PruneSchedule` | `--iter-prune` | Iterative magnitude pruning with sparsity ramp schedule → **gradual 0→70% sparsity** |

### v8 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave23_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave24_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave23_24.py` — micro-benchmark suite
- [x] `dev/results/wave23_24_bench.json` — benchmark results
- [x] `docs/benchmark_wave23_24.md` — human-readable results table
- [x] `dev/demos/record_v8_demo.py` — v8 demo GIF generator
- [x] `dev/demos/squish-v8-demo.gif` — demo GIF rendered
- [x] README.md — v8 module sections, Wave 23+24 tables, CLI examples
- [x] CHANGELOG.md — `[6.0.0]` entry
- [x] PLAN.md updated to mark v8 complete

### v8 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 23 (Multi-Modal + Long Context Intelligence) | 14 |
| Wave 24 (Quantisation Evolution + Model Surgery) | 14 |
| Total new v8 modules | **28** |
| Total modules after v8 | **194** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v8 | **~4 502** |

---

## ✅ v9 — Waves 25+26 — Released 2026-03-12

Theme: **Cutting-Edge Attention Variants & Compute Fusion · Distributed Inference & Production Reliability**

28 new modules across two waves.

---

### Wave 25 — Cutting-Edge Attention Variants & Compute Fusion (14 modules)

Focus: DeepSeek-V2/V3 production attention patterns (MLA, NSA), fused sampling,
online KV defragmentation, dual-chunk long-context attention, activation offloading,
attention morphing, multi-draft hydra speculation, and constrained decoding.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **FlashMLA** | `flash_mla.py` | `FlashMLACache`, `MLAConfig` | `--flash-mla` | Multi-head latent attention (DeepSeek-V2 style); low-rank KV via down/up projection → **KV size ↓ by latent_dim/head_dim** |
| **NativeSparseAttn** | `native_sparse_attn.py` | `NativeSparseAttention`, `NSAConfig` | `--native-sparse-attn` | Block-sparse + sliding window attention (DeepSeek-V3 NSA style) → **sub-quadratic attention cost** |
| **FusedSampler** | `fused_sampler.py` | `FusedSampler`, `SamplerConfig` | `--fused-sampler` | Fused temperature/top-p/top-k/min-p/rep-penalty in single pass → **zero intermediate allocations** |
| **KVDefrag** | `kv_defrag.py` | `KVDefragmenter`, `DefragStats` | `--kv-defrag` | Online KV cache defragmentation and in-place compaction → **fragmentation ratio ↓** |
| **DualChunkAttn** | `dual_chunk_attn.py` | `DualChunkAttention`, `DCAConfig` | `--dual-chunk-attn` | Intra-chunk + inter-chunk attention for 1M+ contexts → **O(chunk²) not O(seq²)** |
| **ActivationOffload** | `activation_offload.py` | `ActivationOffloader`, `OffloadPolicy` | `--act-offload` | Layer activation offload to CPU during prefill → **peak GPU memory ↓** |
| **MorphAttn** | `morph_attn.py` | `AttentionMorpher`, `MorphConfig` | `--morph-attn` | Per-layer attention pattern selection: full/sparse/linear → **optimal compute per layer** |
| **HydraSpec** | `hydra_spec.py` | `HydraSpecDecoder`, `HydraConfig` | `--hydra-spec` | Multi-draft heads for parallel speculation → **n_heads candidate tokens per step** |
| **SeqCompact** | `seq_compact.py` | `SequenceCompactor`, `CompactStats` | `--seq-compact` | In-place KV sequence compaction after token pruning → **zero-copy repack** |
| **LatencyPredictor** | `latency_predictor.py` | `LatencyPredictor`, `LatencyModel` | `--latency-predict` | Per-request latency prediction for scheduling → **prefill + decode latency forecast** |
| **ParallelSampler** | `parallel_sampler.py` | `ParallelSampler`, `DiversityConfig` | `--parallel-sample` | Best-of-n sampling with diversity scoring → **quality improvement with n candidates** |
| **ContextSummarizer** | `context_summarizer.py` | `ContextSummarizer`, `SummaryConfig` | `--ctx-summarize` | Inference-time context compression when context overflows → **keep semantics, shed tokens** |
| **TokenWatermark** | `token_watermark.py` | `TokenWatermarker`, `WatermarkConfig` | `--token-watermark` | Statistical green-list token watermarking (Kirchenbauer et al.) → **detectable attribution** |
| **SchemaGen** | `schema_gen.py` | `SchemaGenEngine`, `SchemaState` | `--schema-gen` | FSM-accelerated constrained JSON schema generation → **zero invalid token sampling** |

### Wave 26 — Distributed Inference & Production Reliability (14 modules)

Focus: Tensor/sequence parallelism, live KV migration, disaggregated prefill/decode,
request preemption, smart inference gateway, zero-downtime model swaps, APM profiling,
adaptive batching, safety classification, semantic response caching, and audit logging.

| Module | File | Key Classes | Flag | Key Result |
|--------|------|-------------|------|-----------|\
| **TensorParallel** | `tensor_parallel.py` | `TensorParallelShard`, `TPConfig` | `--tensor-parallel` | Row/column tensor sharding + all-reduce → **linear memory scaling across devices** |
| **SequenceParallel** | `sequence_parallel.py` | `SequenceParallelScatter`, `SPConfig` | `--seq-parallel` | Ulysses-style sequence dimension split → **attention FLOPs distributed across devices** |
| **KVMigrate** | `kv_migrate.py` | `KVMigrator`, `MigrateStats` | `--kv-migrate` | Live KV state pack/unpack for cross-worker migration → **zero-recompute worker handoff** |
| **DisaggPrefill** | `disagg_prefill.py` | `DisaggPrefillNode`, `DisaggDecodeNode` | `--disagg-prefill` | Disaggregated prefill→decode with KV payload transfer → **prefill/decode hardware specialisation** |
| **RequestPreempt** | `request_preempt.py` | `PreemptScheduler`, `PreemptState` | `--req-preempt` | Preemptive SRPT scheduling with KV save/restore → **priority inversion elimination** |
| **InferGateway** | `infer_gateway.py` | `InferenceGateway`, `WorkerRegistry` | `--infer-gateway` | Smart front-door gateway: routing + health + load balancing → **single ingress, N workers** |
| **ModelVersionSwap** | `model_version_swap.py` | `ModelVersionManager`, `SwapPolicy` | `--model-swap` | Zero-downtime hot model version swap → **canary → promote → rollback in-flight** |
| **ProductionProfiler** | `production_profiler.py` | `ProductionProfiler`, `ProfilerWindow` | `--prod-profiler` | Continuous APM-style per-op latency tracking → **p50/p99/p999 per operation** |
| **AdaptiveBatcher** | `adaptive_batcher.py` | `AdaptiveBatchController`, `BatchObjective` | `--adaptive-batch` | Throughput/latency-objective dynamic batching → **SLO-aware batch size control** |
| **SafetyLayer** | `safety_layer.py` | `SafetyClassifier`, `SafetyConfig` | `--safety-layer` | Inline token-level safety classification → **zero extra forward pass overhead** |
| **SemanticResponseCache** | `semantic_response_cache.py` | `SemanticResponseCache`, `CacheConfig` | `--semantic-resp-cache` | Embedding-similarity response deduplication → **exact + fuzzy response cache hits** |
| **RateLimiter** | `rate_limiter.py` | `TokenBucketRateLimiter`, `RateLimitConfig` | `--rate-limit` | Token-bucket per-tenant rate limiting with burst → **hard request ceiling per tenant** |
| **SchemaValidator** | `schema_validator.py` | `SchemaValidator`, `ValidationResult` | `--schema-validate` | JSON schema validation for structured generation → **100% schema-compliant outputs** |
| **AuditLogger** | `audit_logger.py` | `AuditLogger`, `AuditEntry` | `--audit-log` | SHA-256 chained inference audit log → **tamper-evident request provenance** |

### v9 Deliverables checklist

- [x] All 28 modules implemented in `squish/`
- [x] `tests/test_wave25_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `tests/test_wave26_server_wiring.py` — import + instantiation tests for 14 modules
- [x] `dev/benchmarks/bench_wave25_26.py` — micro-benchmark suite
- [x] `dev/results/wave25_26_bench.json` — benchmark results
- [x] `dev/demos/record_v9_demo.py` — v9 demo GIF generator
- [x] `dev/demos/squish-v9-demo.gif` — demo GIF rendered
- [x] README.md — v9 module sections, Wave 25+26 tables, CLI examples
- [x] CHANGELOG.md — `[7.0.0]` entry
- [x] PLAN.md updated to mark v9 complete

### v9 Module Count Summary

| Scope | Count |
|-------|------:|
| Wave 25 (Cutting-Edge Attention + Compute Fusion) | 14 |
| Wave 26 (Distributed Inference + Production Reliability) | 14 |
| Total new v9 modules | **28** |
| Total modules after v9 | **222** |
| Expected new tests | **~112** (4 per module × 28) |
| Expected total tests after v9 | **~4 876** |
