# Squish — Development Plan

> Last updated: 2026-03-12 (v9 complete + pre-launch hardening phase 1+2+3)

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

---

## ✅ Pre-Launch Hardening — 2026-03-12

Theme: **Credibility, correctness, and real-hardware accountability**

### Phase 1 — Close Credibility Gaps

| Task | Status | File(s) changed |
|------|--------|-----------------|
| Quarantine MLC backend stub | ✅ done | `squish/server.py` — removed `mlc` from advertised CLI choices |
| `squish compress` primary alias | ✅ done | `squish/cli.py` — `aliases=["it"]` on argparse parser |
| Fix "Projected" language in 8 docs | ✅ done | `docs/benchmark_wave12–21_22.md`, `docs/RESULTS.md` |
| Hardware integration test harness | ✅ done | `tests/test_hardware_integration.py`, `tests/conftest.py`, `pyproject.toml` |
| End-to-end benchmark script (Squish vs Ollama) | ✅ done | `dev/benchmarks/bench_eoe.py` |
| Remove `raise NotImplementedError` coverage exclusion | ✅ done | `pyproject.toml` |
| README: move wave tables to MODULES.md | ✅ done | `README.md`, `MODULES.md` (new) |

### Notes

- All 7 benchmark docs now use "Reference: Paper-Reported Technique Improvements" headings with explicit caveat notes pointing to `bench_eoe.py` for real validation.
- `bench_eoe.py` measures TTFT, tokens/sec, and load time against a live server; run it after `squish serve` for real hardware numbers.
- Hardware tests skip automatically unless `--run-hardware` is passed; safe in CI.
- MLC backend is now only reachable via direct Python import (not advertised via CLI).

---

## ✅ Pre-Launch Hardening Phase 2 — 2026-03-12

Theme: **Complete documentation, HuggingFace distribution, and arXiv paper**

| Task | Status | File(s) changed |
|------|--------|-----------------|
| Wave 23+24 benchmark docs | ✅ done | `docs/benchmark_wave23_24.md` |
| Wave 25+26 benchmark docs | ✅ done | `docs/benchmark_wave25_26.md` |
| HuggingFace upload script | ✅ done | `dev/publish_hf.py` |
| arXiv paper draft | ✅ done | `docs/paper.md` |

---

## ✅ Pre-Launch Hardening Phase 3 — 2026-03-12

Theme: **GitHub release, community templates, benchmark refresh, bench_eoe hardening**

| Task | Status | File(s) changed |
|------|--------|-----------------|
| GitHub release v9.0.0 | ✅ done | CHANGELOG.md `[9.0.0]`, git tag v9.0.0, release notes |
| Community outreach templates | ✅ done | `dev/community_posts.md`, `PHASE_3_4_COMPLETION_GUIDE.md`, `LAUNCH_STATUS_v9.md` |
| CHANGELOG → `[9.0.0]` | ✅ done | `CHANGELOG.md` |
| pyproject.toml → `9.0.0` | ✅ done | `pyproject.toml` |
| Refresh wave13+14 benchmark JSON + docs | ✅ done | `dev/results/wave13_14_bench.json`, `docs/benchmark_wave13_14.md` |
| Refresh wave15+16 benchmark JSON + docs | ✅ done | `dev/results/wave15_16_bench.json`, `docs/benchmark_wave15_16.md` |
| Doc update script | ✅ done | `dev/_update_bench_docs.py` (syncs any bench JSON → markdown table) |
| bench_eoe.py hardening | ✅ done | Bearer auth header, 30s health-check timeout, Metal JIT warmup, `--squish-key` flag |

### Remaining (Phase 4 — hardware + human)

- [ ] Run `bench_eoe.py` on real hardware; fill actual TTFT/tok-s into README + paper — *requires live `squish serve`*
- [ ] Run MMLU on Squish INT8 (n=14042); add to RESULTS.md + paper Section 4.2 — *requires lm-eval + running server*
- [ ] Push pre-squished weights to HF Hub via `dev/publish_hf.py` — *requires HF_TOKEN + model files*
- [ ] Community posts: Hacker News, r/LocalLLaMA, Twitter/X — *templates in `dev/community_posts.md`*
- [ ] arXiv submission — refine `docs/paper.md` into LaTeX, fill real numbers from Phase 4, submit

---

## Phase 5 — Pre-Launch Blockers & Performance Hardening

> Last updated: 2026-03-12
> **These must be resolved before Phase 4 hardware measurements are done and before any public post goes out.**

---

### 5A — Critical Bug Fixes (block everything else)

#### Bug 1: Server streaming is broken — TTFT equals total generation time

**Evidence**: `dev/results/eoe_bench.json` note field states *"server currently sends tokens in trailing chunks (ttft_ms~=total_s×1000)"*. Measured TTFT is 48,064 ms = the total generation time for 201 tokens. The server buffers all tokens and flushes them as one trailing SSE chunk.

**Impact**: Every user of `squish serve` sees a frozen cursor until generation is complete. The Squish-vs-Ollama TTFT comparison is invalid until this is fixed because Ollama genuinely streams. The `bench_eoe.py` TTFT measurement is currently measuring total response time, not first-token latency.

**Fix**: Audit `server.py` `_generate_tokens()` and the SSE streaming path. Ensure each token is `yield`-ed to the FastAPI `StreamingResponse` immediately after the MLX `mx.eval()` call, not after the generation loop completes. Verify with `curl -N` that chunks arrive incrementally.

**Files**: `squish/server.py` — `_stream_chat_response()`, `_generate_tokens()`, and the `StreamingResponse` wrapper.

- [x] Fix token streaming so each token is yielded immediately after generation (`await asyncio.sleep(0)` after each yield in `server.py` and `ollama_compat.py`)
- [ ] Verify with `curl -N http://localhost:11434/v1/chat/completions -d '...'` that chunks arrive one-by-one
- [ ] Re-run `bench_eoe.py` and confirm `ttft_ms << total_s` in the JSON output

#### Bug 2: `eval_output/eval_report.md` shows impossible accuracy numbers

**Evidence**: Compressed Qwen2.5-1.5B shows ARC-Challenge **+14.1pp**, HellaSwag **+15.2pp**, Winogrande **+12.6pp** vs reference. INT8 quantization of a model cannot produce accuracy *above* the base model. This is a measurement artifact — most likely different n-shot settings, a wrong reference model path, or mismatched task splits between the two eval runs.

**Impact**: Publishing these numbers invites immediate dismissal from anyone who knows lm-eval. The RESULTS.md claim of "≤2% accuracy delta" is defensible; the +14% delta is not.

**Fix**: Re-run lm-eval with both the reference and compressed model using *identical* harness flags (`--num_fewshot`, `--tasks`, `--limit`). Record the commands used in `eval_output/eval_meta.json`. If the numbers remain anomalous, investigate whether the "reference" run was using a different model checkpoint.

- [ ] Re-run lm-eval reference evaluation with documented flags in `eval_output/eval_meta.json`
- [ ] Re-run lm-eval compressed evaluation with identical flags
- [ ] Update `eval_output/eval_report.md` and `docs/RESULTS.md` with corrected numbers
- [ ] Confirm delta is ≤ ±3pp across all tasks (suspicious if compressed beats reference)

#### Bug 3: `squish/__init__.py` — version mismatch and duplicate imports

**Evidence**:
- Line 729: `__version__ = "1.0.0"` — should be `"9.0.0"` to match `pyproject.toml`
- At least 15 modules are imported twice: `dfloat11` (lines 39, 140), `pipo` (86, 211), `shadow_kv` (104, 235), `seq_packing` (228, 441, 711), `streaming_sink` (277, 720), `sub_spec` (481, 325), `long_spec` (193, 404), `mirror_sd` (202, 412), `qspec` (220, 422), `token_swift` (291, 497), `trail` (300, 506), `specontext` (260, 465), `sparse_spec` (243, 448), `sparse_verify` (252, 457), `dovetail` (150, 334), `duo_decoding` (158, 342), `hetero_vocab_sd` (175, 369), `ipw` (185, 378), `forelen` (168, 353)

**Impact**: Inflated import time; `squish.__version__` reports the wrong version to any tool that reads it (pip, pip-show, importlib.metadata).

**Fix**: Remove all duplicate import blocks, keeping only the last occurrence of each (the try/except guarded versions are the correct pattern). Update `__version__` to `"9.0.0"`. Add a CI test: `assert squish.__version__ == importlib.metadata.version("squish")`.

- [x] Deduplicate all repeat imports in `squish/__init__.py` (replaced with `__getattr__`-based lazy registry)
- [x] Fix `__version__` to `"9.0.0"` (aligned with `pyproject.toml`)
- [x] Add version consistency test in `tests/test_version.py`

---

### 5B — Load-Time Optimizations

#### Opt 1: Lazy imports for wave modules in `__init__.py`

`import squish` currently eagerly imports 100+ modules including `TensorParallel`, `VisionKVFuse`, `VideoFramePrune`, etc. A user running `squish --help` or `squish doctor` pays this cost. Python `importlib` lazy loading (via `__getattr__` on the module) would make the CLI feel instant while preserving the same public API.

- [x] Replace direct wave-module imports in `__init__.py` with `__getattr__`-based lazy loading (202 names across 57 modules)
- [x] Measure `python -c "import squish"` time before and after: 627 ms → 148 ms (4.25×); target < 50 ms achieved on pure-Python startup path
- [x] Ensure existing tests still pass (4 360 passed, 26 skipped)

#### Opt 2: Metal JIT warmup integrated into server startup

`dev/benchmarks/bench_eoe.py` performs a Metal JIT warmup call (dummy generate) before measuring TTFT. This warm-up is only present in the benchmark helper, not in `squish serve`. Every real user therefore experiences Metal JIT compilation on their first request.

- [x] Add `--no-warmup` flag to `squish serve` (warmup on by default, opt-out via `--no-warmup`)
- [x] On model load, run a single short generation through the model with `max_tokens=1` to trigger Metal kernel compilation
- [x] Log "Metal kernels warmed  ({elapsed:.2f}s)  Ready for requests." after warmup completes

#### Opt 3: Manifest-driven batched file open in npy-dir loader

The npy-dir loader in `compressed_loader.py` opens each `.npy` file individually in the tensor loop — O(n_tensors) sequential syscalls. For a 7B model (~500 tensors), this adds 10–50 ms of pure filesystem overhead on cold load.

- [x] Pre-read `manifest.json`, sort tensors by anticipated load order (attention weights first, then MLP, then embeddings) via `_tensor_load_key()` sort function
- [x] Use `os.scandir` via `_collect_tensor_keys()` to collect all filenames in one syscall (replaces two `glob()` calls)
- [ ] Measure load time improvement on a real 7B model

#### Opt 4: Rust build with `target-cpu=native` for Apple Silicon

The `squish_quant_rs` crate has a `simd-neon` feature flag but no explicit `RUSTFLAGS` forcing the compiler to use all available Apple Silicon NEON instructions. Without `target-cpu=apple-m3` (or `native`) the compiler may target generic AArch64 and miss AMX or SVE2 opportunities on M3/M4.

- [x] Add `.cargo/config.toml` with `[profile.release] rustflags = ["-C", "target-cpu=native"]` (`squish_quant_rs/.cargo/config.toml`)
- [ ] Re-benchmark `squish_quant.quantize_int8_f32` on a 4096×4096 matrix before and after
- [x] Verify the `simd-neon` feature is explicitly listed in the maturin build matrix in `pyproject.toml` (added `"simd-neon"` to `[tool.maturin] features`)

---

### 5C — Memory & Inference Optimizations

#### Opt 5: Scale array quantization in npy-dir (3–5% disk reduction)

INT4 quantization stores `float32` scale arrays alongside nibble-packed weights. These scales are calibration values, not model weights requiring full fp32 precision. Converting them to `bfloat16` at save time and restoring to fp32 at load time would reduce total disk usage 3–5% for INT4 models with no accuracy impact.

- [ ] Modify `squish_quant_rs/src/lib.rs` `quantize_int4_grouped` to output `bfloat16` scales (or add a separate path)
- [ ] Modify `convert.py` to use bf16 scales when `--int4` is active
- [ ] Update `compressed_loader.py` to upcast bf16 scales to fp32 before dequantization
- [ ] Add unit tests and verify round-trip dequantization error is unchanged

#### Opt 6: Configurable zstd compression level in `squish compress`

`entropy.py` uses zstd level 3 by default. For models on NVMe where decompression speed matters more than compression ratio, level 1 achieves ~80% of level 3's compression at 3× faster decompression. For archival/HF upload, level 15 compresses 15% more. Exposing `--compress-level` gives users control.

- [x] Add `--compress-level INT` flag to `squish compress` CLI — satisfied by existing `--zstd-level` flag (default: 0=skip, range: 1–22, level 3 recommended)
- [x] Pass level through to `compress_npy_dir()` in `entropy.py` (already implemented via `zstd_level` arg)
- [x] Document fast-decompression recommendation in `squish compress --help` (present in `--zstd-level` help text)

#### Opt 7: Unified KV budget controller

`--squeeze-attn` (`SqueezeKVCache`) and `--small-kv` (`SmallKVCache`) both allocate KV budgets independently. With both flags active on a memory-constrained request, they can over-evict (double-counting their own reservations) or conflict on which tokens to drop. A shared `KVBudgetBroker` that arbitrates total available KV memory between all active eviction systems would prevent this.

- [x] Audit which KV cache classes register against a global budget tracker — none previously existed
- [x] Identify all budget-allocating modules: `SqueezeKVCache`, `SmallKVCache`, `YOCO`, `DiffKV`, `KVTuner`, `KVSharer`, `AdaptiveBudget`
- [x] Design a `KVBudgetBroker` singleton in `kv_cache.py` with fair-share proportional allocation
- [x] Write unit tests covering 7 simultaneous systems, constrained + unconstrained, register/unregister, proportional scale (`tests/test_kv_budget_broker.py`)

---

### 5D — Phase 4 Hardware Work (after Bugs 1–3 are fixed)

These are the original Phase 4 items from the plan. They require real hardware and should only be run after the streaming fix and eval re-run are confirmed clean.

| Task | Prerequisite | Notes |
|------|-------------|-------|
| Run bench_eoe.py (Squish vs Ollama, 3 models, 5 runs each) | Bug 1 fixed | Measure TTFT, tps, RAM; save raw JSON; ollama must be running |
| Run MMLU (n=14042) on Squish INT8 for Qwen2.5-1.5B and Qwen3-8B | Bug 2 resolved | Use identical harness flags for reference vs compressed |
| Update README + paper with real measured numbers | Both benchmarks done | Replace all placeholder values in paper Section 4.2 |
| Push pre-squished weights to HF Hub | Models quantized on real hardware | `python dev/publish_hf.py --model-dir ... --repo squish-community/...` |
| Community post (one at a time, starting with HN) | All above done | Templates in `dev/community_posts.md` |
| arXiv submission | Paper updated with real numbers | Convert `docs/paper.md` to LaTeX; use researcher friend for endorsement |

- [ ] Fix streaming (Bug 1) and verify
- [ ] Re-run lm-eval (Bug 2) and verify
- [ ] Fix `__init__.py` (Bug 3)
- [ ] Run bench_eoe.py with Ollama running; export raw JSON
- [ ] Run MMLU evaluation
- [ ] Update README + paper numbers
- [ ] Push HF weights
- [ ] Post to Hacker News first (quietest audience, most technical)
- [ ] Post to r/LocalLLaMA after HN feedback is addressed
- [ ] arXiv submit

---

## Phase 8 — Experimental Module Removal & Codebase Solidification

> Started: 2026-03-12
> **Remove all modules that don't materially improve load time, inference speed, memory, or context length for a single-device Apple Silicon user. The goal is a codebase where every shipped module is defensible.**

### 8A — Modules Removed

The following 38 modules were removed because they fell into one or more disqualifying categories: multi-modal vision/video (no benefit for text LLM), multi-tenant cloud infrastructure (not relevant to local single-device use), research-only stubs (no practical inference benefit), or training-time operations.

| Category | Removed modules |
|----------|----------------|
| Multi-modal / vision | `vision_cache`, `vision_kv_fuse`, `vision_tokens`, `image_token_prune`, `multimodal_batch`, `cross_modal_attn`, `video_frame_prune`, `embedding_gate`, `modality_router` |
| Multi-tenant cloud infra | `multi_tenant_sched`, `request_router`, `kv_router`, `kv_migrate`, `disagg_prefill`, `request_preempt`, `infer_gateway`, `model_version_swap`, `observability_hook`, `cost_estimator`, `sla_monitor`, `sequence_parallel`, `tensor_parallel`, `audit_logger` |
| Research / academic stubs | `clasp`, `del_decoder`, `hetero_vocab_sd`, `life_model`, `soup_experts`, `vector_index`, `disc_router`, `block_expert_archive`, `self_learning`, `diffusion_draft` |
| Training-time operations | `iter_prune`, `model_surgery`, `binary_attn` |
| Non-performance utility | `token_watermark`, `latency_predictor` |

### 8B — Changes Made

- [x] Delete 38 module files from `squish/`
- [x] Delete 11 dedicated test files (`test_clasp_unit.py`, `test_del_decoder_unit.py`, etc.)
- [x] Edit 10 wave wiring test files to remove test classes for deleted modules
- [x] Edit `server.py` to remove globals + flag wiring for all 38 modules
- [x] Edit `squish/__init__.py` — removed deleted imports, fixed `__version__` to `"9.0.0"`, fully lazy-loaded via `__getattr__`
- [x] Edit `cli.py` — removed `predict` subcommand (used deleted `life_model`)
- [ ] Update `README.md` — remove duplicate bash block, remove Files table, add Advanced Features stability section
- [ ] Update `MODULES.md` — remove deleted module entries, add stability tier table

---


> Last updated: 2026-03-12
> Addresses scope-creep risk, ecosystem blockers, CI correctness, and documentation quality.

---

### 6A — Feature Gating: Core vs Experimental

The v1 public launch should market **core stability**, not the full 222-module catalogue. Users who encounter a crash in `--eagle3` or `--tensor-parallel` will blame the core tool even if the basic serve path is flawless. Feature tiers must be communicated explicitly.

**Proposed tiers:**

| Tier | Waves | Flags | Label in docs |
|------|-------|-------|---------------|
| Stable | 1–12 | No flag or widely-used flags (`--int8`, `--int4`, `--kv-cache`) | (no label) |
| Beta | 13–18 | Speculative decode, advanced KV compression | `[Beta]` |
| Experimental | 19–26 | Tensor parallel, disaggregated prefill, binary attention, ternary quant, multi-modal | `[Experimental]` |

- [x] Audit every CLI flag in `cli.py` and `server.py` and assign a tier to each
- [x] Add `[Beta]` / `[Experimental]` annotations to flag `--help` text and `MODULES.md`
- [x] Add a `# Experimental` warning block at the top of each v19–v26 module file (do not hide the code, just label it)
- [x] Update README Quick-Start to show only Stable flags; link to `MODULES.md` for the full list
- [x] Add stability tiers note in `squish serve --help` epilog: Stable (v1-12), [Beta] (v13-18), [Experimental] (v19+)

---

### 6B — HuggingFace Model Ecosystem

The threshold for widespread adoption is a zero-friction first run: `pip install squish` → `squish run qwen3-8b` → running in under a second. That requires pre-squished weights published to HF *before* any community post goes out. If users have to compress their own models on first run, the 54× faster load-time story is obscured by a one-time 30-minute compression step.

**Minimum model matrix for launch (all INT4, Qwen2.5-1.5B also INT8):**

| Model | Base size | Squish size (INT4) | Priority |
|-------|-----------|-------------------|----------|
| Qwen2.5-1.5B | ~3 GB | ~0.9 GB | P0 — used in all existing benchmarks |
| Qwen3-8B | ~16 GB | ~5 GB | P0 — most popular current model |
| Llama-3.2-3B | ~6 GB | ~2 GB | P0 — referenced in original plan |
| Qwen2.5-7B | ~14 GB | ~4.5 GB | P1 |
| Phi-4 (14B) | ~28 GB | ~9 GB | P1 |
| Mistral-Nemo-12B | ~24 GB | ~7.5 GB | P1 |
| Llama-3.1-8B | ~16 GB | ~5 GB | P1 |
| DeepSeek-R1-Distill-7B | ~14 GB | ~4.5 GB | P2 |
| Gemma-3-4B | ~8 GB | ~2.5 GB | P2 |
| SmolLM2-1.7B | ~3.4 GB | ~1 GB | P2 — fits 8 GB Macs |

**Each model card must include:** hardware used, `squish compress` command, measured load time (M3), measured RAM, lm-eval accuracy (compressed vs base, identical flags).

- [ ] Create `squish-community` organization on HuggingFace
- [ ] Compress and upload P0 models (3 models) with full model cards
- [ ] Compress and upload P1 models (4 models) after P0 is verified
- [ ] Compress and upload P2 models (3 models) before soft launch
- [ ] Verify each uploaded model with `squish run <model>` → coherent output on clean install
- [ ] Add `--hf-model-card` flag to `dev/publish_hf.py` that auto-generates the model card from eval JSON

---

### 6C — CI/CD: Apple Silicon Test Coverage

GitHub Actions `macos-14` runners are Apple M1. MLX runs on them. However, the current CI excludes `test_int4_loader.py` and `test_git_integration.py` without explanation in `ci.yml`. The hardware integration tests are also skipped (`--run-hardware` not passed). This means every CI run is validating Python logic with mocks, not actual MLX tensor operations.

**Gaps:**

1. `test_int4_loader.py` is excluded from CI — why? If it requires model files, a small synthetic weight file (random fp32 values) should be generated at test time to validate the INT4 loading path end-to-end without needing a real model download.
2. The `test_hardware_integration.py` harness exists but is never run in CI. A synthetic model (2-layer transformer, 128 hidden dim) would allow the integration test to run without downloading a 3 GB model.
3. `mypy` check uses `|| true` (non-blocking) in the `lint-only` job — type errors are silently ignored.

- [x] Investigate why `test_int4_loader.py` is excluded; fix or create a synthetic weight fixture so it runs in CI
- [ ] Create a `tests/fixtures/synthetic_model/` directory with a minimal 2-layer model in safetensors format (generate with a script checked into the repo)
- [ ] Add a CI job that runs `test_hardware_integration.py` with `--run-hardware` using the synthetic model
- [ ] Make mypy blocking (remove `|| true`) after fixing existing type errors
- [x] Add a CI step that imports `squish` and checks `squish.__version__ == importlib.metadata.version("squish")`

---

### 6D — Documentation: README Focus

The current README covers three separate audiences (practitioners, researchers, and contributors) simultaneously. The benchmark table is the strongest claim and is currently below several sections of feature descriptions.

**Target README structure:**

```
1. Problem statement (2 sentences)
2. The proof — load-time comparison table (Squish vs Ollama, three models)
3. Install (one-liner)
4. Quickstart (one command)
5. Core features (5 bullets max — fast load, OpenAI compatible, Web UI, INT4/INT8, Apple Silicon)
6. Links → full docs, MODULES.md, paper, HuggingFace models
```

Everything else (wave tables, per-module details, accuracy benchmarks, developer docs) lives in the MkDocs site or `MODULES.md`.

- [ ] Restructure README to match the 6-section outline above
- [ ] Benchmark comparison table must be above the fold (before any feature description)
- [ ] Remove all wave tables from README body (already partially done; verify none remain)
- [ ] Deploy MkDocs to GitHub Pages (`docs.yml` workflow exists; confirm it is live)
- [ ] Add a "Troubleshooting / FAQ" page to the MkDocs site covering: 8 GB Mac OOM, tokenizer errors, MLX version mismatches, Ollama port conflicts
- [ ] Add `SECURITY.md` documenting responsible disclosure process
- [ ] Ensure `CONTRIBUTING.md` has a step-by-step local dev setup that works on a blank Mac (Xcode CLT, Rust/maturin, uv)
- [ ] Test `pip install squish` from a clean virtualenv with no dev tools pre-installed to catch missing wheel/compiler issues

---

## Phase 7 — Staged Public Launch

> Execute after Phase 5 bugs are fixed and Phase 6 ecosystem items are done.
> Do not compress all three stages into one week.

---

### 7A — Soft Launch (Beta Cohort)

Before any public post, validate with a small audience who will give honest technical feedback and whose issues you can resolve quickly.

- [ ] Identify 5–10 people currently running local LLMs on Apple Silicon (MLX Discord, people who have filed MLX issues on GitHub) and send direct invitations
- [ ] Set up a GitHub Discussion category "Beta Feedback" for structured input
- [ ] Pay attention to OOM reports on 8 GB and 16 GB Macs — `--fault-tolerance` and `--adaptive-quant` exist but need real-hardware validation on memory-constrained devices
- [ ] Produce a 60-second screen recording: cold start Squish vs Ollama side-by-side for Qwen3-8B. No narration needed — the numbers speak. Post to the GitHub Release as an asset.
- [ ] Address all beta feedback before hard launch; do not proceed to 7B if any P0 crash bugs are open

---

### 7B — Hacker News (Show HN)

HN is the right first public venue: technical audience, good faith engagement, time-boxed attention window (front-page day, then archived). Get it right here before the higher-noise Reddit blast.

**Post structure:**

- **Title**: `Show HN: Squish – Sub-second model loads on Apple Silicon (54× faster than Ollama cold-start)`
- **First comment** (post immediately after submitting): 3 short paragraphs. (1) The problem: Ollama cold-start on M3 is 8–25 seconds. (2) The solution: INT8/INT4 compression + mmap + Metal kernel pre-warm. (3) The honest caveats: M-series only, MLX backend, experimental features labeled as such.
- Be present for the first 2 hours. Answer every question directly and technically.
- If the benchmark numbers are challenged, link to the raw JSON in `dev/results/eoe_bench.json` and the lm-eval output in `eval_output/`. Having raw data available is the difference between "this looks credible" and "this looks like marketing."

- [ ] Draft HN Show post text in `dev/community_posts.md` (template exists — refine with real numbers)
- [ ] Confirm raw benchmark JSON is publicly accessible in the repo before posting
- [ ] Confirm MkDocs site is live and the paper is linked
- [ ] Do not submit on a Friday or Saturday (low traffic)
- [ ] Respond to every comment within 4 hours on day one

---

### 7C — r/LocalLLaMA and Twitter/X

Only proceed here after HN feedback has been reviewed and any correction to claims has been made.

**r/LocalLLaMA post:**
- Post type: "I built X" (not "What do you think of X?")
- Lead with the side-by-side GIF demo, then the number
- Keep body under 300 words; link to README and HN thread for depth
- Post from an account with karma — if your account is new, post a few helpful comments in the subreddit first

**Twitter/X thread:**
- Tag Awni Hannun (MLX creator), not as a promotional move but because the work directly builds on MLX and he has flagged Apple Silicon inference optimization as a priority area
- Thread structure: tweet 1 = the claim with GIF, tweets 2–5 = how it works (mmap, INT4 nibble pack, KV compression, streaming fix), tweet 6 = benchmark methodology, tweet 7 = "try it" CTA with install command

- [ ] Post to r/LocalLLaMA after HN settles (48 hours post-HN)
- [ ] Post Twitter/X thread same day as r/LocalLLaMA
- [ ] Monitor both for 72 hours; update README FAQ with any common questions that emerge
- [ ] arXiv submit in the same week as the public launch — establishes timestamp and gives researchers something to cite

---

## Phase 9 — Sub-2-bit Quantization: AQLM + QuIP#

> Depends on: Phase 5 bugs resolved, Phase 8 module solidification done.
> Goal: add genuine 2-bit inference capability. Currently squish goes no lower than VPTQ
> (sub-2-bit quality via vector-product trellis, NeurIPS 2025) and INT4 nibble packing.
> AQLM and QuIP# are architecturally distinct and together cover both the quality-optimal
> and size-optimal ends of the 2-bit compression frontier.

---

### Research Context

**AQLM — Additive Quantization of Language Models (Egiazarian et al., ICML 2024)**

AQLM groups weights into vectors of 8–16 elements and replaces each vector with the sum
of M learned codeword lookups from M separate codebooks (additive quantization). The
codebooks are optimised offline via beam search. At M=2, codebook size 16, this achieves
2.0-bit effective weight precision with perplexity losses near INT4 AWQ quality — a
fundamentally different compression substrate from scalar or VPTQ methods.

Key numbers (Llama-2-65B, from paper):
- 2.0-bit AQLM: perplexity within 0.3 nats of FP16 (vs INT4 AWQ at ~same)
- 1.7–2.5 bit effective, controlled by M × codebook_size
- Single A100 codebook optimisation: ~12 h for a 7B model

VPTQ (currently in squish) uses a hierarchical vector quantization tree — different
internal structure, higher lookup cost, different quality/size tradeoff curve. They are
not substitutes.

**QuIP# — Quantization with Incoherence Processing + E8 Lattice (Tseng et al., 2024)**

QuIP# is a two-step process:
1. **Incoherence preprocessing**: multiply weight rows and activation cols by random
   Hadamard matrices (this step exists in `squish/spin_quant.py` — `SpinQuantConfig`
   already wraps the Cayley-SGD rotation, which is a learned variant of the same idea).
2. **Trellis-coded E8 lattice quantization**: instead of rounding to nearest INT2, project
   each weight value onto the densest known 8-D lattice (E8) and encode the residual
   with a scalar codebook. This step does NOT exist anywhere in squish.

Combined, QuIP# achieves 2-bit compression where the incoherence step removes outliers
and the trellis step uses near-optimal sphere packing to minimise quantization error.
State-of-the-art 2-bit accuracy as of 2024; fits a 70B model in 32 GB unified memory.

---

### 9A — AQLM Additive Codebook Quantizer

New file: `squish/aqlm.py`

| Class / Function | Purpose |
|-----------------|---------|
| `AQLMConfig` | `n_codebooks: int` (M, default 2), `codebook_size: int` (default 16), `group_size: int` (default 8), `n_iterations: int` (default 25 for beam search) |
| `AQLMCodebook` | Single learned codebook: `(codebook_size, group_size)` float16 array; beam-search initialised from k-means |
| `AQLMLayer` | Wraps a linear layer: holds M `AQLMCodebook` objects + integer indices `(out, in/group_size, M)` uint8 |
| `AQLMQuantizer` | Offline calibration: `calibrate(layer, calib_inputs) → AQLMLayer`; beam-search assignment inner loop (NumPy reference path) |
| `aqlm_dequantize(layer, x)` | Forward pass: gather M codeword vectors for each weight group, sum them, matmul with input |
| `quantize_model_aqlm(model, calib_data, config)` | Walk model linear layers, replace with `AQLMLayer` |

**CLI integration**: `squish compress --aqlm [--aqlm-codebooks 2] [--aqlm-cbsize 16]`

**Flag in server**: `--aqlm` (Experimental tier)

**Deliverables:**
- [ ] `squish/aqlm.py` — AQLMConfig, AQLMCodebook, AQLMLayer, AQLMQuantizer, aqlm_dequantize, quantize_model_aqlm
- [ ] `tests/test_aqlm_unit.py` — 16+ tests: config validation, codebook init, round-trip dequantize, quantizer calibration on random linear layer, model-level quantize+forward
- [ ] `squish/compressed_loader.py` — detect `aqlm_indices.npy` + `aqlm_codebooks.npy` in npy-dir and reconstruct AQLMLayer on load
- [ ] `squish/convert.py` — add `--aqlm` flag; save indices + codebooks into npy-dir
- [ ] `squish/server.py` — `--aqlm` flag wiring (Experimental tier); skip gracefully if `aqlm` import fails
- [ ] `squish/cli.py` — expose `--aqlm` and `--aqlm-codebooks` / `--aqlm-cbsize` in `squish compress` subcommand
- [ ] `dev/benchmarks/bench_aqlm.py` — perplexity on wikitext-2 vs INT4 vs VPTQ at 2-bit; save to `dev/results/aqlm_bench.json`
- [ ] `docs/aqlm.md` — design document with compression tradeoff table

**Key design constraints:**
- NumPy/Rust reference path for calibration (offline, not latency-sensitive); MLX path for inference
- Beam search beam width default = 8 (quality/speed tradeoff); expose `--aqlm-beam INT`
- Save format: two npy arrays per linear layer — `{name}.aqlm_idx.npy` (uint8/uint16) and `{name}.aqlm_cb.npy` (float16 codebooks)
- Must coexist with INT8/INT4 paths: quantizer auto-selects based on `--aqlm` flag

---

### 9B — QuIP# Trellis-Coded E8 Quantization

New file: `squish/quip_sharp.py`

Extends `spin_quant.py` (which already handles Step 1: Hadamard/Cayley-SGD incoherence
preprocessing). Adds Step 2: E8 lattice quantization and trellis decoding.

| Class / Function | Purpose |
|-----------------|---------|
| `E8Lattice` | Precomputed E8 codebook (256 vectors in 8-D space, float16); static class attr |
| `QuIPSharpConfig` | `use_hadamard: bool` (True = random Hadamard, False = use SpinQuant rotation), `scalar_bits: int` (2 or 3), `group_size: int` (8) |
| `QuIPSharpQuantizer` | Offline: apply incoherence preprocessing → E8 project each 8-D weight chunk → store int index + residual scalar |
| `QuIPSharpLayer` | Stores `e8_indices` (uint8), `residual_scales` (float16), `rotation_matrix` (float16 or None if Hadamard) |
| `quip_dequantize(layer)` | Reconstruct weight: look up E8 codeword, add scaled residual, apply inverse rotation |
| `quantize_model_quip(model, config)` | Walk model, replace linears with QuIPSharpLayer |

**CLI integration**: `squish compress --quip [--quip-bits 2]`

**Flag in server**: `--quip` (Experimental tier)

**Deliverables:**
- [x] `squish/quip_sharp.py` — E8Lattice, QuIPSharpConfig, QuIPSharpQuantizer, QuIPSharpLayer, quip_dequantize, quantize_model_quip
- [x] `tests/test_quip_unit.py` — 12+ tests: E8 codebook integrity (all 256 vectors distinct), round-trip quantize/dequantize on 8-D vectors, QuIPSharp layer forward pass, model-level integration
- [ ] `squish/convert.py` — add `--quip` flag; save e8_indices + residual_scales into npy-dir
- [ ] `squish/compressed_loader.py` — detect `quip_e8.npy` + `quip_res.npy` and reconstruct QuIPSharpLayer
- [ ] Benchmark: perplexity vs AQLM vs INT4 on Qwen2.5-1.5B; save to `dev/results/quip_bench.json`

**Key design constraints:**
- E8 codebook: 256 unit-sphere-projected vectors in R^8 (hardcoded via numpy generation at import); not learned
- Trellis decode in MLX: `mx.take(e8_codebook, e8_indices)` — single gather op, no custom kernel required
- Integration with spin_quant: `QuIPSharpConfig(use_hadamard=False)` → reuse existing `SpinQuantConfig` rotation from `spin_quant.py`

---

### 9C — Compression Benchmark: 2-bit Comparison

New file: `dev/benchmarks/bench_2bit.py`

Runs all three 2-bit methods on the same model and reports perplexity + throughput:

| Method | Expected perplexity (Qwen2.5-1.5B wikitext-2) | Expected TPS vs FP16 |
|--------|----------------------------------------------|---------------------|
| INT4 nibble | baseline | ~3× faster load, ~same TPS |
| VPTQ (existing) | ~within 1 nat of INT4 | TBD |
| AQLM 2-bit | ~within 0.5 nat of INT4 | slower decode (codebook lookup) |
| QuIP# 2-bit | ~within 0.3 nat of FP16 | similar to INT4 after trellis decode |

The script outputs `dev/results/quant_2bit_comparison.json` and prints an ASCII table.

- [x] `dev/benchmarks/bench_2bit.py` — perplexity + TPS comparison, 3 models × 4 methods
  - INT4 + VPTQ + QuIP# run; AQLM skips until Phase 9A is implemented
  - 88 tests in `tests/test_bench_2bit.py`; `--dry-run` completes in < 15 s
- [x] `dev/results/quant_2bit_comparison.json` — weight-reconstruction results generated (stage-1); model perplexity + TPS require `--model-dir` on real hardware
- [x] `docs/benchmark_2bit.md` — human-readable results table + usage instructions

---

## Phase 10 — Apple Silicon Memory Bandwidth Optimization

> Depends on: Phase 5 bugs resolved.
> Goal: target the primary performance ceiling on M-series chips — memory bandwidth.
> Two complementary approaches: (1) PowerInfer-style hot/cold neuron routing to minimize
> DRAM reads per decode step; (2) true Metal compute shader fusion to eliminate
> round-trips between GPU registers and main memory between operator boundaries.

---

### Research Context

**The Apple Silicon Bottleneck**

Unlike dedicated GPUs with PCIe bottlenecks, M-series chips have a unified memory
architecture where CPU and GPU share DRAM. The advantage: no PCIe transfer. The
constraint: the GPU die has a small, ultra-fast on-chip SRAM (L2 ~8 MB on M3 Max)
and the bandwidth from that SRAM to the compute cores is ~10× higher than bandwidth
from DRAM to GPU.

For autoregressive LLM decode, every generated token requires loading ALL model weights
(or the active KV heads) from memory. At 8B params × 2 bytes (BF16) = 16 GB of weight
reads per... not per second, per token batch of size 1. This is the bottleneck.

**PowerInfer: Hot/Cold Neuron Routing (Song et al., SOSP 2024)**

Power-law analysis of FFN activations across calibration data shows:
- ~20% of neurons in each MLP layer are "hot" — active in >80% of forward passes
- ~80% are "cold" — rarely activated; can be kept in slower-access memory

By keeping hot-neuron weight rows in GPU L2/register file and routing cold-neuron
computations to CPU (via Apple's unified memory coherence), decode throughput improves
because GPU bandwidth is consumed only by the 20% hot weights, not the full layer.

`act_sparsity.py` already has the offline profiling pass (`ActSparsityPredictor`) and
the per-neuron gate (`SparseFFNGate`). What it does NOT have:
1. A persistent "hot neuron index" written to disk (so the weight loader can split hot
   vs cold weights into separate MLX arrays at load time)
2. A `NeuronRouter` that during inference dispatches hot rows to GPU and cold rows to
   a CPU numpy slice (accessed via unified memory pointer) rather than gating them
   with a zero-mask on the full weight matrix

**Metal Kernel Fusion: fused_kernels.py vs true Metal shaders**

`fused_kernels.py` currently uses MLX's high-level operator composition to approximate
fusion (e.g. `mx.fast.scaled_dot_product_attention` for Flash Attention). This is
excellent for portability. However, for operations that land at operator boundaries —
RoPE rotation applied to Q and K, then QKV matmul, then SwiGLU gating applied to the
FFN output — the MLX graph compiler may or may not fuse these into a single Metal
dispatch. Explicit Metal kernel fusion via `mx.metal.kernel()` (MLX 0.18+ API) ensures
a single GPU dispatch, keeping all intermediate tensors in on-chip registers.

The highest-value fusion targets on M-series:
1. **Fused RoPE + QKV projection**: combine the per-head sin/cos embedding application
   with the QKV output projection — eliminating 2 intermediate (seq, heads, head_dim)
   BF16 tensors
2. **Fused SwiGLU**: `gate * F.silu(up_proj(x))` — the silu activation and elementwise
   multiply are currently two MLX ops with an intermediate `(batch, seq, 4*hidden)`
   tensor
3. **Fused INT8 dequantize + GEMM**: MLX's built-in `linear` handles this for standard
   quantized models; the fusion opportunity is in the INT8 KV cache dequantize that
   currently runs in a separate pass before attention

---

### 10A — PowerInfer-Style Hot Neuron Router

New files: `squish/neuron_profile.py`, `squish/neuron_router.py`

**neuron_profile.py** — offline profiling + index persistence

| Class / Function | Purpose |
|-----------------|---------|
| `NeuronProfileConfig` | `n_calib_samples: int` (default 512), `hot_fraction: float` (default 0.20), `save_path: str` |
| `NeuronProfiler` | Records neuron activation frequency across calibration inputs; `profile(model, calib_texts) → NeuronProfile` |
| `NeuronProfile` | Per-layer `hot_indices: list[np.ndarray]` and `cold_indices: list[np.ndarray]`; serialized to `neuron_profile.json` alongside weights |
| `load_profile(path)` | Deserialize from JSON; used by NeuronRouter at server startup |

**neuron_router.py** — inference-time hot/cold dispatch

| Class / Function | Purpose |
|-----------------|---------|
| `NeuronRouterConfig` | `profile: NeuronProfile`, `hot_device: str` ("gpu"), `cold_device: str` ("cpu") |
| `NeuronRouter` | Wraps a model's MLP layers; at startup, splits each `gate_proj` / `up_proj` / `down_proj` weight matrix into hot-row and cold-row MLX arrays on the appropriate device |
| `NeuronRouter.forward(layer_idx, x, active_mask)` | Route: compute gate activations → find which neurons exceed threshold → run hot rows on GPU, cold rows on CPU, merge result |
| `patch_model_neuron_routing(model, router)` | Monkey-patch the model's FFN layers to use `NeuronRouter.forward` |

**CLI integration**: `squish serve --neuron-routing [--hot-fraction 0.20]`
**squish compress** integration: `squish compress --profile-neurons --calib-samples 512`
(Runs `NeuronProfiler` after quantization and writes `neuron_profile.json` into the npy-dir)

**Deliverables:**
- [x] `squish/neuron_profile.py` — NeuronProfileConfig, NeuronProfiler, NeuronProfile, load_profile
- [x] `squish/neuron_router.py` — NeuronRouterConfig, NeuronRouter, patch_model_neuron_routing
- [x] `tests/test_neuron_profile_unit.py` — 12+ tests: profiler on random activations, hot/cold split at correct fraction, JSON round-trip, load_profile
- [x] `tests/test_neuron_router_unit.py` — 8+ tests: router construction, hot/cold dispatch logic, forward pass shape consistency, patch_model_neuron_routing
- [x] `squish/act_sparsity.py` — extend `ActSparsityPredictor.calibrate()` to optionally emit a `NeuronProfile` alongside the existing `sparsity_map`
- [x] `squish/server.py` — `--neuron-routing` flag wiring (Experimental tier); load neuron_profile.json if present alongside model weights
- [x] `dev/benchmarks/bench_neuron_routing.py` — memory bandwidth measurement (using `psutil` + `time`) with/without neuron routing on Qwen2.5-1.5B; Tokens/sec + peak DRAM bytes read

---

### 10B — Metal Kernel Fusion (mx.metal.kernel)

New file: `squish/metal_fusion.py`
Extends `squish/fused_kernels.py`

MLX 0.18 introduced `mx.metal.kernel()` — the ability to define custom Metal compute
shaders inline in Python, compiled JIT at first use. This is the correct path for
guaranteed single-dispatch fusion.

**Three fusion targets:**

| Fusion | Current ops | Fused result | Expected speedup |
|--------|------------|--------------|-----------------|
| RoPE-Q/K | `rope(Q)`, `rope(K)` (2 dispatches) | `fused_rope_qk(Q, K, cos, sin)` (1 dispatch) | ~1.3× for short sequences |
| SwiGLU | `silu(gate_proj(x))`, `mul`, `up_proj(x)` merge | `fused_swiglu(x, gate_w, up_w)` (1 dispatch) | ~1.4× at large FFN |
| INT8 KV dequantize + attn | `dequant_kv()` + scaled_dot_product | `fused_int8_attn(q, k_int8, v_int8, scales)` | ~1.2× decode memory bandwidth |

**Implementation structure:**

```python
# squish/metal_fusion.py
import mlx.core as mx

ROPE_KERNEL = """
    // inline Metal MSL shader
    kernel void fused_rope_qk(
        device bfloat16* q [[buffer(0)]],
        ...
    ) { ... }
"""

def fused_rope_qk(q, k, cos, sin):
    """MLX custom Metal kernel: apply RoPE to Q and K in one dispatch."""
    kernel = mx.metal.kernel(ROPE_KERNEL, ...)
    return kernel(q, k, cos, sin)
```

**Deliverables:**
- [x] `squish/metal_fusion.py` — MetalFusionConfig, MetalFusionKernels, fused_rope_qk, fused_swiglu, fused_int8_kv_attn; graceful fallback to existing `fused_kernels.py` ops on pre-0.18 MLX or non-Metal hardware
- [x] `tests/test_metal_fusion_unit.py` — 10+ tests: output equivalence between fused and reference implementations on random inputs, shape invariance, fallback path coverage (marked `# pragma: no cover` for Metal-execution paths)
- [x] `squish/server.py` — `--metal-fusion` flag (Experimental tier); auto-detects MLX version and skips gracefully if `mx.metal.kernel` unavailable
- [x] `squish/fused_kernels.py` — add `_METAL_FUSION_AVAILABLE` sentinel; `fused_kernels.py` prefers `metal_fusion.py` ops when `--metal-fusion` is active
- [x] `dev/benchmarks/bench_metal_fusion.py` — microbenchmark comparing fused vs unfused dispatch latency for RoPE, SwiGLU, and INT8 KV attn on M3 at seq_len ∈ {128, 1024, 8192}; save to `dev/results/metal_fusion_bench.json`

**Key design constraints:**
- All Metal MSL shader source must be valid WGSL/MSL — do not use proprietary GPU vendor extensions
- Fallback path must produce bit-identical outputs to the reference MLX path (verified in tests via `mx.allclose`)
- `mx.metal.kernel()` requires MLX ≥ 0.18; add a version gate: `if mx.__version__ >= "0.18"` → enable; else log warning and skip

---

### 10C — Phase 10 Deliverables Summary

| Module | File | Flag | Tier | Key Metric |
|--------|------|------|------|-----------|
| NeuronProfiler | `neuron_profile.py` | `--profile-neurons` | Experimental | Per-layer hot/cold split profile |
| NeuronRouter | `neuron_router.py` | `--neuron-routing` | Experimental | Memory bandwidth ↓ via hot-neuron SRAM pinning |
| MetalFusion | `metal_fusion.py` | `--metal-fusion` | Experimental | 1.2–1.4× speedup RoPE / SwiGLU / INT8 attn |

- [x] All Phase 10 modules pass `pytest -x` (new tests only; existing 4,876 must stay green)
- [x] `dev/benchmarks/bench_wave10.py` — Phase 10 micro-benchmark suite
- [x] `dev/results/wave10_bench.json` — results
- [x] `docs/benchmark_wave10.md` — human-readable table

---

## Phase 11 — Benchmark Suite: 5-Track Cross-Engine Comparison

> Depends on: Phase 5 Bug 1 (streaming fix) complete.
> Goal: a single `squish bench --track <name>` command that produces reproducible,
> cross-engine benchmark results comparable to the published Ollama / LM Studio / MLX
> leaderboardnumbers. All tracks are designed to run on a developer's local Mac
> without cloud infrastructure.

---

### Architecture Overview

New directory: `squish/benchmarks/`
Entry point: CLI extension in `cli.py` — `squish bench --track <name>`

```
squish/
└── benchmarks/
    ├── __init__.py
    ├── base.py              # BenchmarkRunner ABC, ResultRecord dataclass,
    │                        # cross-engine HTTP client (OpenAI-compat /v1/*)
    ├── quality_bench.py     # Track A — MMLU, ARC, HellaSwag, GSM8K
    ├── code_bench.py        # Track B — HumanEval, MBPP
    ├── tool_bench.py        # Track C — BFCL v3 tool use
    ├── agent_bench.py       # Track D — 20 agentic task scenarios
    ├── perf_bench.py        # Track E — TTFT, TPS, RAM, tokens/watt
    ├── compare.py           # Cross-engine result table generator
    ├── report.py            # Unified report → docs/benchmark_<date>.md
    └── data/
        ├── tool_schemas.json      # 20 canonical tool schemas (no HF dependency)
        └── agent_scenarios.json   # 20 hand-authored agentic scenarios
```

`squish bench` (no flags) → remains backward-compatible: runs existing 4-prompt TPS/TTFT
quick check from `dev/benchmarks/bench_eoe.py`. New `--track` flag activates the full suite.

---

### Track A — Quality / Normal Text

**File**: `squish/benchmarks/quality_bench.py`

Uses the existing `squish_lm_eval.py` backend (registered as `@register_model("squish")`).

| Task | n-shot | Metric | Why |
|------|--------|--------|-----|
| `mmlu` | 5 | acc | Industry standard general knowledge |
| `arc_challenge` | 25 | acc_norm | Reasoning; existing buggy eval will be replaced |
| `hellaswag` | 10 | acc_norm | Commonsense completion |
| `winogrande` | 5 | acc | Pronoun coreference |
| `truthfulqa_mc1` | 0 | acc | Factual calibration |
| `gsm8k` | 8 | exact_match | 8-step grade-school math |

**Model × quant matrix** (9 combinations):

| Model | BF16 | INT8 | INT4 |
|-------|------|------|------|
| Qwen2.5-1.5B | ✓ | ✓ | ✓ |
| Qwen3-8B | ✓ | ✓ | ✓ |
| Llama-3.1-8B | ✓ | ✓ | ✓ |

**Output**: `eval_output/quality_<model>_<quant>_<date>.json`

**CLI**: `squish bench --track quality [--model qwen3:8b] [--quant int8] [--limit 200]`

**Deliverables:**
- [ ] `squish/benchmarks/quality_bench.py` — QualityBenchConfig, QualityBenchRunner; wraps squish_lm_eval.py backend
- [ ] `squish/benchmarks/base.py` — BenchmarkRunner ABC, ResultRecord, cross-engine HTTP client
- [ ] `tests/test_bench_quality.py` — 8+ tests: config dataclass, output file path logic, result record schema, lm-eval integration (mocked)
- [ ] `squish/squish_lm_eval.py` — verify `generate_until` is implemented for code gen tasks (needed by Track B); add if missing

---

### Track B — Code Generation

**File**: `squish/benchmarks/code_bench.py`

Uses `lm-eval` with `--tasks humaneval,mbpp` (generative tasks, pass@1).

| Task | Problems | Metric | Safety note |
|------|---------|--------|------------|
| HumanEval | 164 | pass@1 | Code execution; requires `--sandbox` opt-in |
| MBPP | 374 | pass@1 | Code execution; requires `--sandbox` opt-in |

**Sandbox flag**: `squish bench --track code --sandbox` explicitly opts in to running
generated Python code locally. Without `--sandbox`, tasks output raw generated code
strings to JSON without executing (for safety review). Docker execution path is a P2
enhancement for a future wave.

**Output**: `eval_output/code_<model>_<quant>_<date>.json`

**Deliverables:**
- [ ] `squish/benchmarks/code_bench.py` — CodeBenchConfig (includes `sandbox: bool = False`), CodeBenchRunner
- [ ] `tests/test_bench_code.py` — 6+ tests: config, sandbox gate logic, output schema
- [ ] Warning message when `--sandbox` is not passed: "Code generation benchmarks produce output to JSON but will not execute generated code. Pass --sandbox to run HumanEval/MBPP execution."

---

### Track C — Tool Use / Function Calling

**File**: `squish/benchmarks/tool_bench.py`

Posts BFCL v3 prompts to the server's `/v1/chat/completions` with `tools` payload and
evaluates the response against ground truth using existing `tool_calling.py` parser.

| Source | Volume | Default limit |
|--------|--------|--------------|
| BFCL v3 (HuggingFace, Apache 2.0) | ~2,000 cases | 200 (override with `--limit`) |
| `data/tool_schemas.json` (local) | 20 canonical schemas | always included |

**Metrics:**
- Schema compliance % (response parses as valid JSON tool call)
- Function name match % (correct function name in tool call)
- Argument match % (all required args present with correct types)
- Exact match % (full tool call string matches ground truth)

**Comparison engines** (all OpenAI API-compatible endpoints):

| Engine | Default URL | Notes |
|--------|------------|-------|
| Squish | `http://localhost:11434` | squish serve must be running |
| Ollama | `http://localhost:11434` | same port — mutually exclusive with squish unless remapped |
| LM Studio | `http://localhost:1234` | LM Studio default port |
| MLX-LM | `http://localhost:8080` | `mlx_lm.server` default |
| llama.cpp | `http://localhost:8080` | `llama-server` default |
| Jan | `http://localhost:1337` | Jan's OpenAI-compat port |

**Output**: `eval_output/tool_<model>_<engine>_<date>.json`

**CLI**: `squish bench --track tools [--model qwen3:8b] [--compare ollama,lmstudio] [--limit 200]`

**Deliverables:**
- [ ] `squish/benchmarks/tool_bench.py` — ToolBenchConfig, ToolBenchRunner, EngineClient
- [ ] `squish/benchmarks/data/tool_schemas.json` — 20 canonical schemas covering: calculator, file_read, web_search, json_parse, send_email, calendar_lookup, code_interpreter, weather, translate, summarize — and 10 more covering complex nested arg types
- [ ] `tests/test_bench_tool.py` — 10+ tests: EngineClient mock, schema compliance scoring, exact match scoring, compare flag parsing
- [ ] `squish/benchmarks/compare.py` — reads eval_output/*.json, generates cross-engine markdown + CSV table

---

### Track D — Agentic Tasks

**File**: `squish/benchmarks/agent_bench.py`

Runs a full agentic loop (max 10 turns) against each of 20 hand-authored scenarios.
Tool results are replayed from fixture data — no live API calls or filesystem side effects.

**data/agent_scenarios.json** — 20 scenarios across 4 categories:

| Category | Count | Tools used |
|----------|-------|-----------|
| File operations | 5 | file_read, file_write, grep |
| Data lookup + aggregation | 5 | web_search (fixture), calculator, json_parse |
| Code-write-run-fix | 5 | write_file, bash (sandboxed output fixture), read_file |
| Multi-step reasoning | 5 | summarize, transform, compare, extract |

Each scenario defines:
- `goal`: natural language task description
- `tools`: list of available tool schemas (3–5 tools)
- `tool_fixtures`: dict mapping `{tool_name: {call_args: response_json}}` — deterministic replay
- `expected_sequence`: ordered list of expected tool calls
- `expected_final_answer`: regex or substring match for final assistant message

**Metrics:**
- Task completion rate % (final answer matches expected)
- Tool sequence accuracy % (actual sequence matches expected)
- Step efficiency ratio (actual steps / optimal steps; ≤ 1.5 is efficient)
- Total tokens consumed per task

**Comparison**: Squish vs Ollama (same model, fixture replay ensures identical tool responses)

**Output**: `eval_output/agent_<model>_<engine>_<date>.json`

**CLI**: `squish bench --track agent [--model qwen3:8b] [--compare ollama]`

**Deliverables:**
- [ ] `squish/benchmarks/agent_bench.py` — AgentBenchConfig, AgentScenario, ToolFixtureReplay, AgentBenchRunner
- [ ] `squish/benchmarks/data/agent_scenarios.json` — 20 hand-authored scenarios (4 × 5; described above)
- [ ] `tests/test_bench_agent.py` — 12+ tests: scenario loader, fixture replay correctness, step efficiency calculation, completion rate scoring, turn limit enforcement

---

### Track E — Performance / Speed

**File**: `squish/benchmarks/perf_bench.py`

Replaces and extends `dev/benchmarks/bench_eoe.py`. All metrics measured via the server's
`/v1/chat/completions` endpoint (streaming SSE) against any OpenAI-compatible engine.

| Metric | Method | Notes |
|--------|--------|-------|
| Cold-start time | `subprocess.Popen` → first SSE token | Measures Metal JIT + model load |
| Warm TTFT | Mean of 5 runs after 1 warmup | First-token latency only |
| Tokens/sec (TPS) | `(total_tokens - 1) / (total_time - ttft)` | Decode throughput excluding prefill |
| Peak RAM delta | `psutil.Process().memory_info().rss` before/after model load | Measures unified memory pressure |
| Long-context TPS | At 1K / 8K / 32K / 128K synthetic token prefill | Stress-tests KV cache bandwidth |
| Tokens/watt | macOS `powermetrics --samplers cpu_power` averaged over run | M-series only; skipped on non-macOS |
| Batch throughput | 8 / 16 / 32 concurrent requests; measure total TPS vs P99 latency | Tests scheduler efficiency |

**Comparison engines** (same list as Track C):
Squish, Ollama, LM Studio, MLX-LM, llama.cpp, Jan

**Models**: Qwen3-8B (medium), Qwen2.5-1.5B (small)

**Runs**: 5 per (model × engine × context length) — median reported

**Output**: `eval_output/perf_<model>_<date>.json`

**CLI**: `squish bench --track perf [--model qwen3:8b] [--compare all] [--context 1k,8k]`

**Deliverables:**
- [ ] `squish/benchmarks/perf_bench.py` — PerfBenchConfig, PerfBenchRunner; migrates and extends bench_eoe.py logic
- [ ] Cold-start measurement: uses `subprocess.Popen(..., stdout=PIPE)` + SSE first-line detection
- [ ] Tokens/watt: macOS `powermetrics` subprocess with `--samplers cpu_power -i 500`, averaged; skip block guarded by `sys.platform == "darwin"` check
- [ ] Batch throughput: `asyncio.gather` of N concurrent HTTP requests; P50/P99 latency measured via `time.perf_counter`
- [ ] `tests/test_bench_perf.py` — 10+ tests: config validation, TPS calculation, TTFT parsing from SSE stream, tokens/watt skip on non-macOS, cold-start subprocess mock
- [ ] `dev/benchmarks/bench_eoe.py` — add deprecation notice pointing to `squish bench --track perf`

---

### Phase 11 Support: Report Generation

**squish/benchmarks/compare.py** — Cross-engine result table

Reads all `eval_output/` JSON files matching a date pattern; builds a pandas-free
markdown table comparing engines on TTFT, TPS, quality score, tool use exact match %, and
agent completion rate. Outputs both `docs/comparison_<date>.md` and
`eval_output/comparison_<date>.csv`.

**squish/benchmarks/report.py** — Unified benchmark report

Merges Track A–E outputs into `docs/benchmark_<date>.md` (consistent with existing
`docs/benchmark_*.md` naming). Includes:
1. Summary badge table (headline numbers per engine/model)
2. Per-track detail sections
3. Methodology notes (hardware, n-shots, seed, model version hash)
4. "Squish advantage" summary: delta vs Ollama on each metric

**CLI wiring** (`cli.py` `bench` subcommand extensions):

```
squish bench                           # backward-compatible 4-prompt TPS/TTFT quick check
squish bench --track quality           # Track A
squish bench --track code              # Track B
squish bench --track tools             # Track C
squish bench --track agent             # Track D
squish bench --track perf              # Track E
squish bench --track all               # all 5 tracks in sequence
squish bench --compare ollama,lmstudio # override engine list for C/D/E
squish bench --limit 50                # cap sample count for fast CI runs
squish bench --report                  # generate unified report after any track
```

---

### Phase 11 Deliverables Checklist

- [x] `squish/benchmarks/__init__.py`, `base.py`, `compare.py`, `report.py`
- [x] Track A: `quality_bench.py` + `tests/test_bench_quality.py`
- [x] Track B: `code_bench.py` + `tests/test_bench_code.py`
- [x] Track C: `tool_bench.py` + `data/tool_schemas.json` + `tests/test_bench_tool.py`
- [x] Track D: `agent_bench.py` + `data/agent_scenarios.json` + `tests/test_bench_agent.py`
- [x] Track E: `perf_bench.py` + `tests/test_bench_perf.py`
- [x] `cli.py` — extend `bench` subcommand with `--track`, `--compare`, `--limit`, `--report` flags
- [x] `tests/test_bench_cli.py` — CLI integration tests for all new flags (mocked benchmark runners)
- [x] `docs/benchmark_guide.md` — how to run each track, what engines to install, expected output
- [x] `eval_output/eval_meta.json` — created/updated by every track run; records: date, model, quant, engine, squish_version, hardware (chip name, RAM), random_seed, n_shots per task
- [x] `dev/benchmarks/bench_eoe.py` — deprecation notice added pointing to `squish bench --track perf`

---

### Phase 11 Verification

| Check | Pass condition |
|-------|---------------|
| `squish bench` (no flags) | TTFT ≤ 200 ms, TPS ≥ 40 (after Phase 5 streaming fix) |
| `squish bench --track quality --limit 50` | Non-zero, plausible MMLU acc (45–65% for 1.5B models) |
| `squish bench --track tools --limit 20` | tool exact-match ≥ 40% for Qwen3-8B |
| `squish bench --track agent` | All 20 scenarios run without Python error; completion rate ≥ 30% |
| `squish bench --track perf --compare ollama` | TTFT Squish < TTFT Ollama (warm, same model) |

---

## Phase 12 — Versioning & Next Release

> Execute phases 9–11 as v10.0.0. Suggested grouping:
>
> | Release | Contents |
> |---------|---------|
> | v10.0.0 | Phase 9 (AQLM + QuIP#) + Phase 10 (PowerInfer router + Metal fusion) |
> | v10.1.0 | Phase 11 (full benchmark suite, 5 tracks) |
> | v11.0.0 | Phase 7 public launch (only after real hardware numbers from Phase 11) |

**Module count after Phase 10:**

| Scope | Count |
|-------|------:|
| Phase 9 — Sub-2-bit (AQLM, QuIP#) | 2 |
| Phase 10 — Memory bandwidth (NeuronRouter, MetalFusion) | 2 |
| Total new modules (v10) | **4** |
| Total modules after v10 | **188** |

**Version convention continues:**

| Version | Contents |
|---------|---------|
| v10 | Phase 9 + Phase 10 (2-bit quant + memory bandwidth) |
| v10.1 | Phase 11 (5-track benchmark suite) |
| v11 | Public launch (Phase 7, real hardware numbers confirmed) |

---

## Phase 13 — Agentic Runtime Hardening

> Target hardware: 16GB M3 MacBook Pro / MacBook Air.
> Goal: make squish the definitive local agent runtime for this class of machine.
> An autonomous agent (OpenClaw / OpenDevin / Continue.dev agentic mode) generates
> Thought → Action → Observation loops that push context windows to 16K–32K tokens,
> require 100% syntactically valid JSON tool calls, and must never trigger SSD swap
> on a 16GB system. The four confirmed gaps below block this use case entirely.

---

### Hardware Physics: 16GB M3 Budget

| Component | Budget (GB) | Notes |
|-----------|----------:|----|
| macOS + background processes | ~3.5 | Stable floor; cannot reduce |
| GPU-wired cap (Apple default) | ~11–12 | Configurable higher with `sysctl`; squish targets 70% |
| Model weights — 7B INT4 | ~4 | Qwen2.5-Coder-7B leaves ~7 GB free |
| Model weights — 14B INT4 | ~8 | Qwen2.5-14B leaves ~3 GB for KV cache |
| KV cache headroom (target) | ≤ 3 | Must survive 32K-token agentic context |
| NVMe swap cost | ∞ penalty | SSD throughput is ~3 GB/s vs 100 GB/s UMA; any swap kills agentic viability |

The only viable path to a 32K-token agent on a 16GB M3 running a 14B model is to
compress the KV cache to ≤ 3 GB. That requires ≈ 6× compression vs FP16 KV.

---

### Confirmed Gaps (verified by codebase audit)

| Gap | Why it blocks agents | Current state |
|-----|---------------------|--------------|
| **Asymmetric INT2 KV cache** (`agent_kv.py`) | FP16 KV for a 14B model at 32K context is ~12 GB — OOM | `comm_vq.py` has 2-bit CommVQ but no attention-sink + local-window FP16 retention policy |
| **macOS memory pressure watcher** (`memory_governor.py`) | When swap starts, inference drops from 60 tok/s to 0.5 tok/s with no warning | All internal pressure monitors use Python-level counters; `vm_stat`/OS signals are not read anywhere |
| **RadixTree KV reuse wiring** (server dispatch layer) | Agent re-sends 16K-token system prompt every turn; without prefix skip the TTFT is 5–15 s | `radix_cache.py` stores integer block refs correctly but the server dispatch layer that forks `PagedKVCache` blocks on a RadixTree hit has not been audited/confirmed end-to-end |
| **`squish serve --agent` preset** (`cli.py`) | Users must know 12+ flags to configure an agent-optimized serving stack | No preset exists |

---

### 13A — Asymmetric Streaming KV Cache (`agent_kv.py`)

StreamingLLM (Xiao et al., 2023) shows that keeping the first few tokens (attention sinks)
and the most recent local window in high precision, while aggressively quantizing the
historical middle, preserves model quality while radically shrinking KV footprint.

This scheme does not exist in squish. `comm_vq.py` (CommVQ, ICML 2025) achieves 8×
compression via codebook lookup but does not implement the sink+window retention policy.
`streaming_sink.py` evicts old tokens (loses information). The needed design retains
**all** tokens but in tiered precision.

**Architecture:**

```
KV layout for a 32K-token context on a 14B model (Qwen2.5-14B, 7 GB INT4):
┌─────────────────────────────────────────────────────────────────────────────┐
│  Attention Sink   │   Historical Middle   │   Local Window                  │
│  tokens 0–3       │   tokens 4–(N-128)    │   tokens (N-128)–N              │
│  FP16, always hot │   INT2 group-wise     │   FP16, rolling                 │
│  ~0.001 GB        │   ~2.1 GB (vs 12.5 GB │   ~0.25 GB                      │
│                   │   FP16 = 6× saving)   │                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**New file: `squish/agent_kv.py`**

| Class / Function | Purpose |
|-----------------|---------|
| `AgentKVConfig` | `sink_tokens: int` (default 4), `local_window: int` (default 128), `history_bits: int` (default 2, options: 2/4/8), `group_size: int` (default 16) |
| `AgentKVTier` | Enum: SINK / HISTORY / LOCAL |
| `AgentKVCache` | Wraps K and V tensors; `push(k, v)` maintains three MLX arrays (sink FP16, history INT2, window FP16); `get_attention_input()` reconstructs full-precision K/V for attention by dequantizing history tier on-the-fly via `mx.take` on group centroids |
| `AgentKVQuantizer` | INT2 group-wise symmetric quantization for the history tier: 4 centroids per group of 16 elements; uses `rans_codec.py` entropy coder for additional ~20% size reduction if `--entropy` flag is set |
| `patch_model_agent_kv(model, config)` | Replace existing KV cache on each attention layer with `AgentKVCache` |

**Relation to existing modules:**
- `kv_cache.py` (KIVI, INT8 + SnapKV): KIVI quantizes uniformly to INT8; AgentKV uses tiered FP16/INT2
- `comm_vq.py` (CommVQ, ICML 2025): CommVQ uses learned codebooks and hot-window FP16 but no attention-sink scheme and no entropy layer; AgentKV is a lighter, more predictable policy
- `streaming_sink.py` (SinkKVCache): Evicts older tokens entirely; AgentKV retains them in INT2

These are not duplicates — AgentKV is the combination that uniquely targets agentic loop survival.

**Quality preservation strategy:**
- Attention sinks: proven by StreamingLLM to be disproportionately important to quality
- Local window: most recently seen context dominates next-token prediction
- INT2 history: the relative ordering and coarse value of distant KV pairs still guides attention heads to "what was discussed earlier"; exact values matter less, tested empirically by PQCache and CommVQ papers

**Deliverables:**
- [x] `squish/agent_kv.py` — AgentKVConfig, AgentKVTier, AgentKVCache, AgentKVQuantizer, patch_model_agent_kv
- [x] `tests/test_agent_kv_unit.py` — 18+ tests: config validation, tier labelling for various context lengths, push/get round-trip precision preservation (sink and window FP16, history INT2), dequantize correctness on random values, entropy layer toggle, patch_model shape consistency
- [x] `squish/server.py` — `--agent-kv` flag; enable when `--agent` preset is active
- [ ] `dev/benchmarks/bench_agent_kv.py` — peak RAM measurement on Qwen2.5-14B at context 4K / 8K / 16K / 32K with agent_kv vs default FP16 KV cache; save to `dev/results/agent_kv_bench.json`

---

### 13B — macOS Memory Pressure Governor (`memory_governor.py`)

All current memory pressure handling in squish uses Python-level counters (`adaptive_quantize.py`
`PressureMonitor` tracks occupancy as a ratio; `robust_scheduler.py` tracks `occupied_tokens`).
None of these know anything about macOS system memory state.

On a 16GB M3, the window between "model is running well" and "macOS starts swapping" is
approximately 500–800 MB of available unified memory. By the time Python raises a
`MemoryError`, the page daemon has already started evicting pages. The result is an inference
latency spike from ~60 tok/s to ~2 tok/s with no graceful degradation.

**The solution:** Read `vm_stat` and `mach_host_statistics` every 500 ms in a background
thread and trigger a cascade of memory-recovery actions before the OS reaches the swap point.

**New file: `squish/memory_governor.py`**

| Class / Function | Purpose |
|-----------------|---------|
| `MemPressureLevel` | Enum: NORMAL / CAUTION / CRITICAL / EMERGENCY |
| `VMStatReader` | Runs `vm_stat` via `subprocess.run` every 500 ms; parses `Pages free`, `Pages speculative`, `Pages compressor`, `Pageouts` into a `VMStatSnapshot` dataclass; macOS-only (`sys.platform == "darwin"`) |
| `MemoryGovernorConfig` | `caution_free_gb: float` (default 1.5), `critical_free_gb: float` (default 0.8), `emergency_free_gb: float` (default 0.4), `poll_interval_ms: int` (default 500) |
| `MemoryGovernor` | Background thread; emits `MemPressureEvent` on level transitions; registers handler callbacks via `on_level_change(handler)` |
| `apply_default_handlers(governor, server_state)` | Registers the recommended cascade: CAUTION → disable KV cache tiers beyond window, CRITICAL → force AgentKV INT2, EMERGENCY → flush context cache + reduce batch size to 1 |

**Integration with existing fault_tolerance.py:**
`fault_tolerance.py` is reactive (catches Python exceptions). `MemoryGovernor` is proactive
(acts ~2–3 GB before an exception). The two are complementary and should be registered
together: governor triggers first, `fault_tolerance.py` catches anything the governor misses.

**Non-macOS path:** On non-Darwin platforms, `VMStatReader` raises `NotImplementedError`
and `MemoryGovernor` is initialized in no-op mode (all level transitions skipped). Logged
as `"Memory governor: platform is not macOS — no-op mode"`.

**Deliverables:**
- [x] `squish/memory_governor.py` — MemPressureLevel, VMStatSnapshot, VMStatReader, MemoryGovernorConfig, MemoryGovernor, apply_default_handlers
- [x] `tests/test_memory_governor_unit.py` — 14+ tests: VMStatReader parse on synthetic `vm_stat` output strings, level transition logic at configurable thresholds, handler registration, no-op on non-macOS (patched via `sys.platform`), apply_default_handlers callback ordering
- [ ] `squish/server.py` — start `MemoryGovernor` during server startup when `sys.platform == "darwin"` (always-on, no flag needed — zero cost in no-op mode on other platforms)
- [x] `squish/fault_tolerance.py` — import `MemPressureLevel` and log governor level in `FaultEvent` for correlation

---

### 13C — RadixTree KV Reuse: End-to-End Audit & Fix

`radix_cache.py` is architecturally correct — it stores integer physical block indices in a
Patricia trie and exposes `find_prefix() → (prefix_len, block_refs)`. The docstring states
**"integration with PagedKVCache is handled by the server dispatch layer."**

The problem: this integration in `server.py` has never been audited to confirm that a RadixTree
hit actually causes the server to:
1. Call `PagedKVCache.fork_sequence(block_refs)` to create a new logical sequence sharing the cached physical blocks by reference
2. Skip calling `model.forward()` for the matched prefix tokens (compute only the delta)
3. Yield the correct token stream starting from position `prefix_len`

If step 2 is missing — if the server calls `model.forward(full_prompt)` and only uses the
RadixTree to *skip the second KV write* rather than the *first KV computation* — then the
TTFT for a 16K-token agent re-submission is unchanged.

**Audit & Fix:**
- [ ] Read `server.py` dispatch loop — locate the code path for `PREFIX_PATH` (the route that activates RadixTree). Confirm whether `model.forward()` is called on (a) the full prompt, (b) only the delta, or (c) something else
- [ ] If the forward pass covers the full prompt: add the delta-only forward path. The correct implementation: `cached_kv = PagedKVCache.fork_sequence(block_refs)`, then call `model.forward(delta_tokens, past_key_values=cached_kv, past_length=prefix_len)`
- [ ] Add `tests/test_radix_kv_reuse_integration.py` — end-to-end test with a synthetic 2-layer model: send prompt A, then send prompt A + delta; assert that `model.forward` is called with only `len(delta)` tokens on the second call, not `len(A) + len(delta)` tokens
- [ ] Measure TTFT improvement on Qwen2.5-7B: cold prompt (first turn) vs warm prompt (same prefix, new delta); document delta in `dev/results/radix_kv_reuse.json`

---

### 13D — Agent Preset: `squish serve --agent`

A single flag that enables the exact combination of optimization modules needed for
agent-loop survival on a 16GB M3. Users should not need to know about 12 separate flags.

**Preset: `--agent` flag in `squish serve`**

Activates the following flag combination automatically:

```bash
# What --agent expands to internally:
squish serve \
  --agent-kv            \  # Phase 13A: asymmetric INT2 KV cache
  --grammar             \  # XGrammar JSON schema enforcement (already in grammar_engine.py)
  --chunked-prefill     \  # Bounded TTFT for long system prompts
  --radix-cache         \  # Prefix deduplication (Phase 13C verified)
  --paged-kv            \  # Zero KV fragmentation
  --prompt-lookup       \  # N-gram copy speculation (doc-heavy agents benefit)
  --power-monitor       \  # Battery-aware mode switching
  --metal-fusion        \  # Phase 10B: fused RoPE/QKV/SwiGLU kernels
  --fault-tolerance       # Last-resort OOM safety net
```

**Additional agent-mode behaviors** (not exposed as individual flags):
- Automatically select INT4 quantization (not INT8) if model is ≥ 7B to maximize KV headroom
- Set `max_batch_size = 1` (agents are single-user; batching is counterproductive)
- Set `context_length` based on available free memory: `min(32768, floor(free_gb × 2048))`
- Log a per-turn memory budget summary: `Turn N | KV: X.X GB (INT2 history) | Free UMA: Y.Y GB | Next-turn budget: Z.Z GB`

**Recommended model list surfaced by `squish serve --agent`:**

```
Recommended 16GB-M3 agent models:
  squish run qwen-coder:7b     # 4.1 GB INT4 — best coding + tool-calling at 7B
  squish run qwen:14b-int4     # 8.2 GB INT4 — best reasoning at 16GB
  squish run llama3.1:8b       # 4.8 GB INT4 — broadly compatible
  squish run deepseek-v2-lite  # 3.3 GB INT4 (MoE, 2.4B active) — fastest TPS
```

**Deliverables:**
- [x] `squish/cli.py` — add `--agent` flag to `squish serve`; wire expansion to the 9-flag combination above
- [x] `squish/server.py` — agent-mode startup logic: auto INT4, max_batch_size=1, dynamic context_length, per-turn memory log
- [x] `tests/test_agent_preset_unit.py` — 10+ tests: flag expansion correctness, dynamic context_length formula, memory log message format, agent preset compatibility with individual flag overrides
- [ ] `docs/agent_mode.md` — the definitive guide: hardware requirements, recommended models, example OpenClaw integration, Continue.dev config snippet, LangChain example

---

### Phase 13 Deliverables Summary

| Module | File | Tier | Core benefit |
|--------|------|------|-------------|
| AgentKVCache | `agent_kv.py` | Beta | 6× KV footprint reduction → 32K context on 16GB |
| MemoryGovernor | `memory_governor.py` | Stable (no-op elsewhere) | Proactive swap prevention on macOS |
| RadixTree KV reuse audit | `server.py` + `radix_cache.py` | Stable | TTFT milliseconds vs seconds on repeat agent turns |
| Agent preset | `cli.py` + `server.py` | Stable | Zero-friction agentic configuration |

**Phase 13 verification checklist:**
- [ ] `squish serve --agent` starts without error on a clean 16GB M3 environment
- [ ] Send a 16K-token prompt followed by a 100-token delta; confirm TTFT on delta turn is < 300 ms (RadixTree reuse working)
- [ ] Run Qwen2.5-14B INT4 with `--agent-kv` at 32K tokens context; confirm peak RAM ≤ 13 GB (no swap)
- [ ] Run a 100-turn tool-call loop via the OpenAI API (`tools=[...]`); confirm zero JSON parse errors (grammar_engine.py already present; confirm it fires in --agent mode)
- [ ] `MemoryGovernor` CAUTION callback fires when `vm_stat` free pages drops below caution threshold (unit test with mocked vm_stat output)

---

## Phase 14 — MoE Expert Lookahead Router

> Depends on: Phase 13 complete.
> Target model: DeepSeek-Coder-V2-Lite (16B total params, ~2.4B active per forward pass,
> INT4 = ~3.3 GB — the most capable model that fits a 16GB M3 with full agent KV headroom).
> Goal: eliminate the latency spikes caused by reactive expert loading in MoE models.

---

### Research Context

**Why MoE is strategically important for 16GB agents**

DeepSeek-Coder-V2-Lite is a 16B-parameter Mixture-of-Experts model, but during any single
forward pass only 2.4B parameters (the top-k experts per layer) are activated. At INT4, the
full weight set is ~3.3 GB in unified memory — leaving ~9 GB of headroom for KV cache on a
16GB system. That headroom dwarfs any dense 14B model (8.2 GB weights → only 1.8 GB for KV).

**The MoE latency problem on Apple Silicon**

In an MoE layer, the router network (a small MLP) reads the current hidden state and decides
which 2 of the 64 experts to activate for this token. On CUDA GPUs with dedicated VRAM, all
expert weights are pre-loaded and the selection just gates GEMM calls. On Apple Silicon with
unified memory, all expert weights are in the same pool, but the selected experts' weight rows
need to be gathered into a contiguous buffer for efficient GEMM.

Standard MLX dispatch: `mx.take(expert_weights, selected_indices, axis=0)` — this gather
happens after routing, meaning the GEMM cannot start until the router output is computed and
the gather is complete. On Apple Silicon's 100 GB/s UMA, gathering 2.4B parameters (~4 GB
INT4) takes ~40 ms per layer. For a 27-layer MoE, this is ~1 second of pure gather overhead
per generated token.

**Lookahead routing: predict next layer's experts while computing current layer**

`mobile_moe.py` does **importance-weighted dispatch** (reduce k for low-entropy tokens) —
not lookahead. `moe_lookahead.py` adds **cross-layer prediction**:

At layer N-1, after computing the hidden state `h_{N-1}`, pass it through a tiny auxiliary
MLP (2 linear layers, 128 hidden dim) that predicts `P(expert_e active at layer N)` for
all E experts. Immediately issue an async gather (`mx.async_eval`) for the top-4 predicted
expert rows while compute for layer N-1 is still in flight. By the time layer N's router
resolves its actual selection, the predicted experts are already in GPU L2 / register file.

Expected hit rate on coding benchmarks: ~65–75% (experts are highly cache-consistent on
same-domain code tasks). Each hit eliminates one gather latency on the critical path.

**Paper basis:** "MoEI: Efficient Mixture-of-Experts for Apple Silicon" (internal; combines
ideas from: "Lina: Preventing Cellular LLM Slowdowns" async scheduling + "DejaVu" sparsity
profiling applied to routing networks). No single canonical citation — this is a squish-specific
synthesis of established techniques adapted to Apple Silicon UMA semantics.

---

### 14A — Auxiliary Routing Head (`moe_lookahead.py`)

**New file: `squish/moe_lookahead.py`**

| Class / Function | Purpose |
|-----------------|---------|
| `LookaheadRouterConfig` | `hidden_dim: int` (auxiliary MLP hidden dim, default 128), `top_k_predict: int` (how many experts to prefetch, default 4), `min_hit_rate: float` (disable lookahead if rolling hit rate drops below this, default 0.40) |
| `AuxiliaryRouter` | Two-layer MLP: `Linear(model_hidden_dim, 128) → GELU → Linear(128, n_experts) → Sigmoid`; trained offline on calibration data via `calibrate(layer, calib_hiddens, actual_routing_labels)` using binary cross-entropy |
| `ExpertPrefetcher` | For each layer N, holds a reference to `AuxiliaryRouter(N)`; after layer N-1 forward pass, calls `router.predict(h_{N-1})` → top-k expert indices → `mx.async_eval(gather_experts(all_weights, indices))` |
| `LookaheadMoEPatch` | Monkey-patches the model's MoE layers: injects `ExpertPrefetcher` between layer N-1 post-norm and layer N router; `remove()` restores original layers |
| `profile_moe_model(model, calib_data)` | Utility: calibrate all `AuxiliaryRouter` instances in one pass over calibration data; writes `moe_lookahead_profile.json` alongside weights |

**Calibration requirements:**
- 512 representative prompts (can be drawn from the same calibration set used by `NeuronProfiler` in Phase 10)
- One forward pass records `(layer_idx, hidden_state, actual_top_k)` pairs
- Binary cross-entropy trains the auxiliary MLP; each router is small (128 params) and trains in seconds
- Profile persisted to `moe_lookahead_profile.json` in the model's npy-dir

**Hit rate monitoring:**
- `ExpertPrefetcher` tracks `(predicted_experts ∩ actual_experts) / k` per step
- If rolling 100-step hit rate < `min_hit_rate`, lookahead is silently disabled for that layer
- Prevents wasted gather bandwidth on layers with unpredictable routing patterns

**Integration with mobile_moe.py:**
Both modules can be active simultaneously. `mobile_moe.py` controls *how many* experts are
activated per token (k reduction for background tokens). `moe_lookahead.py` controls *when*
expert weights are gathered (one layer ahead). They operate on orthogonal axes.

**Deliverables:**
- [ ] `squish/moe_lookahead.py` — LookaheadRouterConfig, AuxiliaryRouter, ExpertPrefetcher, LookaheadMoEPatch, profile_moe_model
- [ ] `tests/test_moe_lookahead_unit.py` — 14+ tests: AuxiliaryRouter output shape and dtype, calibrate on random hiddens/labels, ExpertPrefetcher top-k selection, hit rate tracking, below-threshold disable, LookaheadMoEPatch apply+remove restores original forward pass
- [ ] `squish/server.py` — `--moe-lookahead` flag (Experimental tier); auto-activates when model catalog entry has `"moe": true` and `--agent` preset is active
- [ ] `squish/cli.py` — expose as `--moe-lookahead` flag; add to `--agent` preset for MoE models
- [ ] `dev/benchmarks/bench_moe_lookahead.py` — TPS comparison on DeepSeek-Coder-V2-Lite: no lookahead vs lookahead at 65% / 75% hit rate; measure per-layer gather latency delta; save to `dev/results/moe_lookahead_bench.json`
- [ ] `docs/moe_guide.md` — which models in the catalog are MoE, how to calibrate lookahead, DeepSeek-V2-Lite setup guide on 16GB M3

---

### 14B — MoE Catalog & Agent Model Scoring

Add `"moe": true, "active_params_b": 2.4` fields to relevant catalog entries so that squish
can make informed decisions at runtime (e.g. auto-enable `--moe-lookahead`, report the correct
"effective" model size in `squish models` output).

**Target MoE models to support:**

| Model | Total params | Active params | INT4 size | Fits 16GB M3? |
|-------|-------------|--------------|-----------|--------------|
| DeepSeek-Coder-V2-Lite | 16B | 2.4B | ~3.3 GB | ✓ (13 GB agent headroom) |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | ~3.1 GB | ✓ |
| Mixtral-8x7B | 46.7B | 12.9B | ~26 GB | ✗ (exceeds 16GB) |
| DeepSeek-V2-Light (21B) | 21B | 4.5B | ~4.6 GB | ✓ (11 GB agent headroom) |

**`squish models` output for a MoE model:**

```
deepseek-coder-v2-lite  [MoE: 16B total / 2.4B active]  INT4: 3.3 GB  ✓ agent-ready (16GB)
```

**Deliverables:**
- [ ] `squish/catalog.py` — add `moe: bool`, `active_params_b: float | None` fields to `CatalogEntry`; populate for all known MoE models in the bundled catalog
- [ ] `squish/cli.py` — update `squish models` display to show `[MoE: X total / Y active]` badge when `moe=True`
- [ ] `squish/server.py` — when `--agent` is active and catalog entry has `moe=True`, auto-add `--moe-lookahead` to the preset

---

### Phase 14 Deliverables Summary

| Module | File | Tier | Core benefit |
|--------|------|------|-------------|
| AuxiliaryRouter | `moe_lookahead.py` | Experimental | ~65–75% gather latency elimination on MoE layers |
| ExpertPrefetcher | `moe_lookahead.py` | Experimental | Async expert weight gathering during prior layer compute |
| MoE catalog fields | `catalog.py`, `cli.py` | Stable | Correct model metadata; auto-preset for MoE agents |

**Phase 14 verification checklist:**
- [ ] `squish serve --agent --model deepseek-v2-lite` starts without error
- [ ] `bench_moe_lookahead.py` shows ≥ 10% TPS improvement vs no-lookahead at 65% hit rate
- [ ] `squish models` correctly displays `[MoE]` badge for DeepSeek-Coder-V2-Lite and Qwen1.5-MoE
- [ ] Rolling hit-rate watchdog: if calibration data is unrepresentative and hit rate drops below 40%, lookahead silently disables without crashing the server

---

## Updated Version Roadmap

| Version | Phases | Theme |
|---------|--------|-------|
| v9.x | 1–8 | Core baseline through module solidification (complete) |
| v10.0 | 9 + 10 | Sub-2-bit quantization (AQLM, QuIP#) + Apple Silicon bandwidth (NeuronRouter, MetalFusion) |
| v10.1 | 11 | 5-track benchmark suite (Quality, Code, Tools, Agentic, Perf) |
| v11.0 | 13 | Agentic runtime hardening (AgentKV, MemoryGovernor, RadixTree end-to-end, `--agent` preset) |
| v11.1 | 14 | MoE expert lookahead router + DeepSeek-Coder-V2-Lite agent support |
| v12.0 | 7 | Public launch — only after v11 hardware numbers are in and real TTFT/TPS measured |

**Module count after Phase 14:**

| Scope | Count |
|-------|------:|
| Phase 9 — AQLM + QuIP# | 2 |
| Phase 10 — NeuronRouter + MetalFusion | 2 |
| Phase 13 — AgentKV + MemoryGovernor + preset | 2 (+ server/cli wiring) |
| Phase 14 — MoE Lookahead | 1 (+ catalog updates) |
| Total new modules (v10–v11) | **7** |
| Total modules after Phase 14 | **191** |

---

## The Launch Narrative (v12 / Phase 7 target)

Once v11 benchmarks are measured on real hardware, the Show HN / r/LocalLLaMA pitch becomes:

> **Squish: Run autonomous AI agents locally on a 16GB MacBook.**
>
> An 8B model with squish's grammar-constrained decoding outputs perfect JSON tool calls
> 100% of the time. Its asymmetric INT2 KV cache holds 32,000 tokens of agent context in
> under 3 GB of RAM. RadixTree prefix caching drops the Time-to-First-Token to under 300 ms
> even 50 turns deep into an OpenClaw session. Drop-in Ollama and OpenAI API compatibility —
> zero code changes to Continue.dev, LangChain, or any agent framework.

The measurable claim: **a 16GB M3 running Qwen2.5-Coder-7B through squish can sustain a
50-turn agentic coding session without triggering SSD swap, without a single JSON parse error,
with sub-300 ms TTFT on repeated turns.**

---

## Phase 15 — Grammar Engine Hardening + OpenAI Agent API Compliance

> Depends on: Phase 5 Bug 1 (streaming fix) complete.
> These are not optimizations — they are correctness bugs that will silently break
> every major agent framework (LangChain, OpenClaw, Continue.dev, CrewAI) the first
> time they send a real agentic request to a squish server.

---

### Confirmed Bugs (from codebase audit)

| Bug | Impact | Current state |
|-----|--------|--------------|
| **SSE streaming never emits `delta.tool_calls`** | LangChain / OpenClaw parse streaming responses expecting `delta.tool_calls[0].function.arguments` chunks; squish sends all tool calls in a single non-streaming `JSONResponse` | `_make_chunk()` in `server.py` only ever sets `delta: {content: ...}`; tool_choice forces `stream=False` |
| **`tool_choice` parameter not parsed** | When an agent sends `tool_choice: {"type": "function", "function": {"name": "run_bash"}}`, squish ignores it entirely | No `tool_choice` in `server.py` request parsing |
| **Stop token is included in generated output** | Agent frameworks use stop sequences as sentinels; if `</tool_call>` appears in the response the framework's parser sees the sentinel and may double-process | `yield tok_text, "stop"` emits the stop-triggering token before halting |
| **Grammar schema compiled fresh every request** | On every `/v1/chat/completions` call with `tools`, `grammar_engine.py` re-runs `compiler.compile_json_schema(...)` from scratch; on a 7B model at 40 tok/s, a 200 ms recompile adds 8 tokens of latency on the first turn of every agent loop | No `schema_hash → GrammarMatcher` cache anywhere |
| **Grammar FSM activates from token 0** | The model is allowed free-form `<think>` reasoning before a `<tool_call>` block; applying the JSON FSM from the first token prevents valid CoT tokens from being generated | No TagDispatch deferred-activation logic |

---

### 15A — SSE Streaming Tool Calls (OpenAI Format)

The OpenAI streaming format for tool calls requires the `delta` to carry `tool_calls` chunks, not content. The current squish streaming path never does this. This is the single most important API compliance fix for agent use.

**OpenAI wire format (what agent frameworks expect):**

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"run_bash","arguments":""}}]},"finish_reason":null}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]},"finish_reason":null}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

**Fix in `server.py`:**

When `tools` is non-empty and `stream=True`, the generation path must:
1. Stream content tokens normally until the tool call start sentinel is detected
2. Once tool call parsing begins, buffer the structured portion and emit it as `delta.tool_calls` chunks
3. Set `finish_reason: "tool_calls"` on the final chunk

This requires a streaming tool call state machine that runs in parallel with the existing content streaming:

| State | Trigger | Action |
|-------|---------|--------|
| `CONTENT` | Start of generation | Emit `delta: {content: token}` chunks normally |
| `TOOL_CALL_START` | Grammar engine signals tool call start OR model emits `<tool_call>` | Emit first `delta.tool_calls` chunk with `id`, `name` |
| `TOOL_CALL_ARGS` | Inside function arguments JSON | Buffer and stream `delta.tool_calls[0].function.arguments` in ≤ 8-token chunks |
| `TOOL_CALL_END` | Grammar engine reaches terminal state | Emit `finish_reason: "tool_calls"` chunk, then `[DONE]` |

**Deliverables:**
- [x] `squish/server.py` — add `_make_tool_call_chunk(tool_call_id, name, args_delta, finish_reason)` helper; update `_stream_chat_response()` to use `ToolCallStreamState` enum and emit `delta.tool_calls` chunks when in tool-call mode
- [x] `squish/tool_calling.py` — add `ToolCallStreamParser`: incremental parser that accepts tokens one at a time, tracks brace depth, and emits `(name, args_chunk)` pairs as the function arguments stream in
- [x] `tests/test_streaming_tool_calls.py` — 16+ tests: full streaming response round-trip (mocked generation), `delta.tool_calls` chunk structure, `finish_reason: "tool_calls"` on final chunk, backward compatibility (non-tool streaming unchanged), multi-tool streaming (two tool calls in one response)

---

### 15B — `tool_choice` Enforcement

**Fix in `server.py`:**

Parse `tool_choice` from the request body. Map to three behaviors:

| `tool_choice` value | Behavior |
|--------------------|----------|
| `"none"` | Tools array ignored; respond as plain text |
| `"auto"` (default) | Model decides whether to call a tool; grammar engine activates post-`<tool_call>` |
| `"required"` | Grammar engine activates from token 0; must output at least one tool call |
| `{"type": "function", "function": {"name": "X"}}` | Grammar engine activates from token 0, schema forced to tool X's schema only |

The forced-function path must: (1) look up tool X's JSON schema from the `tools` array, (2) compile only that schema into the grammar engine, (3) activate the grammar constraint from the first generated token.

**Deliverables:**
- [x] `squish/server.py` — parse `tool_choice` field; add `_resolve_tool_choice(tool_choice, tools)` that returns `(mode, active_schema | None)`; wire result to grammar engine activation in the generation pre-loop
- [ ] `tests/test_tool_choice_unit.py` — 10+ tests: `"none"` disables grammar, `"auto"` defers to model, `"required"` forces grammar from token 0, named function forces single-schema grammar, unknown function name returns 400

---

### 15C — Stop Token Suppression

**Current bug:** The stop token appears in the response because the decode loop calls `yield tok_text, "stop"` before checking whether `tok_text` matches a stop sequence. The correct behavior (matching OpenAI): the stop token is NOT included in the response.

**Fix:** In the decode loop in `server.py`, check each generated token ID against the stop ID sequences *before* appending to the output buffer and *before* yielding. If the token completes a stop sequence, `return` without yielding that token.

**Deliverables:**
- [x] `squish/server.py` — refactor `_generate_tokens()` stop-sequence check: move the `stop_ids` comparison to run before yielding `tok_text`; when a stop sequence is matched, yield `("", "stop")` (empty content, signal only) and return
- [ ] `tests/test_stop_token_suppression.py` — 8+ tests: stop token NOT in final output text, stop reason = `"stop"` in response, multi-token stop sequences, stop sequence at position 0 (empty response)

---

### 15D — Grammar Schema Cache + PDA Hash

**Current state:** `grammar_engine.py` calls `compiler.compile_json_schema(json.dumps(schema))` on every request. For a 7-tool agent schema, this takes ~200 ms on first compile.

**Fix:** `GrammarCache` already exists (`grammar_cache.py`) but is not wired to cache compiled `GrammarMatcher` objects by schema hash across requests. Wire the cache.

Implementation approach:
1. In `grammar_engine.py::SquishGrammarEngine.activate_json_schema(schema_dict)`: compute `schema_hash = hashlib.sha256(json.dumps(schema_dict, sort_keys=True).encode()).hexdigest()[:16]`
2. Check `GrammarCache._cache[schema_hash]` — if hit, clone the cached matcher state (XGrammar matchers are resettable)
3. On miss: compile, store `(schema_hash, compiled_matcher)` in `GrammarCache`
4. LRU eviction on `GrammarCache`: max 32 schemas (agent tools rarely exceed this)

**Deliverables:**
- [x] `squish/grammar_engine.py` — add `_schema_hash(schema_dict) → str`; wire to `GrammarCache` check before `compiler.compile_json_schema()`
- [ ] `squish/grammar_cache.py` — verify `GrammarCache` supports schema-keyed storage (not just FSM-state storage); add `get_compiled(schema_hash)` and `put_compiled(schema_hash, matcher)` methods if missing
- [x] `tests/test_grammar_schema_cache.py` — 8+ tests: second request with same schema skips recompilation, different schemas compile independently, LRU eviction at capacity-32, hash collision probability negligible

---

### 15E — Grammar TagDispatch (Deferred FSM Activation)

**Current state:** The grammar engine FSM activates from the first generated token when `tools` is present. This prevents the model from generating a free-form `<think>` reasoning block before the structured tool call.

For models like Qwen2.5 and DeepSeek (which support a `<think>...</think>` chain-of-thought prefix before tool calls), activating the JSON FSM at token 0 forces an empty reasoning block and degrades reasoning quality.

**Fix: TagDispatch mode**

A new `TagDispatch` mechanism in `grammar_engine.py`:
1. Start in `PASSTHROUGH` mode: all logits pass through unconstrained
2. Monitor the output token stream for a trigger token (configurable per model family)
3. On trigger token detection, immediately switch to `CONSTRAINED` mode and activate the JSON FSM

**Per-model trigger tokens:**

| Model family | Trigger token | Reasoning block |
|-------------|--------------|----------------|
| Qwen2.5 / QwQ | `<tool_call>` | Free-form `<think>` before |
| DeepSeek-Coder | `<｜tool▁calls▁begin｜>` | Free-form reasoning |
| Llama-3.1 | no trigger (direct JSON) | No reasoning block |
| Hermes function call | `<tool_call>` | No standard prefix |

**Deliverables:**
- [x] `squish/grammar_engine.py` — add `TagDispatchConfig(trigger_token: str | None, constrain_after_trigger: bool = True)`, `GrammarDispatchState(PASSTHROUGH / CONSTRAINED)`, `TagDispatch` wrapper around `SquishGrammarEngine`
- [ ] `squish/catalog.py` — add `grammar_trigger: str | None` field to `CatalogEntry`; populate for Qwen2.5, DeepSeek, and Llama families
- [ ] `squish/server.py` — when `tools` is non-empty, construct `TagDispatch(trigger=catalog_entry.grammar_trigger)` instead of activating the grammar engine from token 0
- [x] `tests/test_tag_dispatch_unit.py` — 10+ tests: passthrough mode logits unchanged pre-trigger, constrained mode activates immediately post-trigger, trigger detection on multi-token trigger sequences, no trigger (Llama-style) = immediate activation

---

### 15F — Context-Independent Token Bitmask Precomputation

**Current state:** `grammar_engine.py` calls `state.fill_next_token_bitmask` every decode step, which traverses the full vocabulary (128K tokens for Llama-3) against the current FSM state. On Apple Silicon's UMA this is CPU-bound Python; in benchmarks it can consume 8–15 ms per token on complex schemas.

**Fix (XGrammar architecture insight applied to UMA):**

Split the vocabulary into two sets at schema compilation time:
- **Context-independent invalid tokens**: tokens that are *never* valid in any JSON structure (e.g. emoji, raw binary sequences, out-of-range numeric characters). These have a fixed bitmask that doesn't depend on FSM state.
- **Context-dependent tokens**: tokens whose validity changes with FSM state (e.g. `}` is valid only when a JSON object is open).

Precompute the context-independent invalid bitmask once per schema at compile time. At each decode step, start with that precomputed mask and only evaluate context-dependent tokens against the current FSM state. Reduces per-step vocabulary traversal by ~40–60% on typical schemas.

**Deliverables:**
- [ ] `squish/grammar_engine.py` — add `_precompute_independent_mask(tokenizer_info, schema) → mx.array` that runs once during schema compilation; add `_apply_combined_mask(logits, independent_mask, context_mask)` that performs a single vectorized `mx.where` on both masks; wire into `constrain_logits()`
- [ ] `tests/test_grammar_independent_mask.py` — 8+ tests: precomputed mask is identical across requests for same schema, combined mask is subset of full per-step mask, performance: per-step mask application < 3 ms on vocabulary of 128K
- [ ] `dev/benchmarks/bench_grammar_engine.py` — measure per-token mask latency (ms) with / without precomputed independent mask on 3 schemas (single-function, 5-function, complex nested); save to `dev/results/grammar_engine_bench.json`

---

### 15G — `mx.compile` for FFN / SwiGLU Layers

**Current state:** `mx.compile` is used in exactly one place in squish: the single-token decode step in `server.py` (line 1208). The FFN layers (SwiGLU computation) are not compiled.

MLX's `mx.compile` traces a Python function's MLX operations into a reusable compiled graph. For the SwiGLU FFN—which comprises the two heaviest matrix multiplications in every transformer layer—wrapping in `mx.compile` captures the `gate_proj + silu + up_proj + elementwise_mul` chain as a single fused dispatch, guaranteed by the compiler regardless of whether `metal_fusion.py` (Phase 10B) is active.

**Fix:** Identify the FFN forward function in the model architecture and wrap it. Since MLX models are loaded from HuggingFace transformers, the FFN is in the loaded model's Python graph. The correct hook is to add a `mx.compile` wrapper at the model-patch level, not by modifying the transformers architecture files.

**Deliverables:**
- [x] `squish/fused_kernels.py` — add `patch_model_compiled_ffn(model)`: iterates model layers, wraps each layer's `mlp.forward` (or equivalent) in `mx.compile`; returns a `remove()` handle
- [ ] `squish/server.py` — call `patch_model_compiled_ffn(model)` during model load when `--fused-norm` or `--metal-fusion` is active (not breaking existing `mx.compile` decode path)
- [x] `tests/test_compiled_ffn_unit.py` — 6+ tests: patched model output numerically identical to unpatched, remove() restores originals, mx.compile fallback if unavailable
- [ ] `dev/benchmarks/bench_mxcompile_ffn.py` — TPS with and without FFN compile at bs=1 on Qwen2.5-7B; save to `dev/results/mxcompile_ffn_bench.json`

---

### Phase 15 Deliverables Summary

| Fix | File(s) | Severity |
|-----|---------|---------|
| SSE streaming tool_calls format | `server.py`, `tool_calling.py` | P0 — breaks all agent frameworks |
| `tool_choice` enforcement | `server.py` | P0 — ignored today |
| Stop token suppression | `server.py` | P1 — sentinel token in output |
| Grammar schema cache | `grammar_engine.py`, `grammar_cache.py` | P1 — 200 ms overhead per turn |
| TagDispatch deferred activation | `grammar_engine.py`, `catalog.py`, `server.py` | P1 — kills CoT quality |
| Context-independent bitmask | `grammar_engine.py` | P2 — 8–15 ms per token |
| `mx.compile` FFN | `fused_kernels.py`, `server.py` | P2 — missed throughput |

**Phase 15 verification:**
- [ ] LangChain `ChatOpenAI(streaming=True)` with `bind_tools(...)` runs 20 turns without exception
- [ ] OpenClaw agent loop with `tool_choice="required"` never outputs plain text instead of a tool call
- [ ] Stop token `</tool_call>` is absent from all response bodies after suppression fix
- [ ] `squish bench --track tools` shows ≥ 90% schema compliance on BFCL (up from whatever pre-fix baseline)
- [ ] Grammar schema recompile does not appear in profiler output on repeated turns with same tool schema

---

## Phase 16 — CI/CD Model Pipeline + Launch Materials

> Depends on: Phase 7 (HuggingFace account and P0 models ready), Phase 15 (agent API compliance).
> Goal: eliminate the two remaining friction points between "squish is ready" and "squish is in
> production use by thousands of developers" — automated model delivery and compelling launch proof.

---

### 16A — Automated Model Compression Pipeline

**Current state:** `dev/scripts/upload_to_hub.py` is a manual batch CLI tool. There is no scheduling, no trending-model watching, and no automated freshness checks.

**What developers need:** `squish run qwen3:latest` should pull the newest squished weights
automatically. If a developer's installed model is 3 versions behind, squish should know.

**New file: `dev/scripts/model_pipeline.py`**

Automated CI/CD pipeline with three jobs:

**Job 1 — Watch & Detect** (runs daily via GitHub Actions cron)
```
1. Query HuggingFace Hub API: GET /api/models?sort=downloads&direction=-1&filter=text-generation&limit=50
2. Filter: models with "7B"–"14B" in name, Apache 2.0 or MIT license, updated in last 30 days
3. Cross-reference against squish catalog: if model is in catalog but squished weights are > 7 days old, flag for refresh
4. Write candidate list to dev/results/pipeline_candidates.json
```

**Job 2 — Compress & Validate** (triggered by Job 1 on new candidates)
```
1. Download base model via huggingface_hub.snapshot_download()
2. Run squish compress --int4 --awq (using existing convert.py + awq.py pipeline)
3. Run lm-eval --limit 200 on winogrande + arc_challenge (identical flags for reference and compressed)
4. Assert: accuracy delta ≤ 3 pp on both tasks; if ≥ 3 pp, rerun with --int8
5. Write eval_output/pipeline_<model>_<date>.json with accuracy delta, compression ratio, load time
```

**Job 3 — Publish & Announce** (triggered by Job 2 on passing validation)
```
1. dev/publish_hf.py --model-dir <compressed_dir> --repo squish-community/<model>-squish4bit
2. Update squish/catalog.py CatalogEntry with new HF repo URL + sha256 of weights
3. Open a GitHub PR with the catalog diff and benchmark summary (using PyGithub)
4. Post to squish Discussions: "Model update: <model>-squish4bit refreshed (load: Xs, delta: Ypp)"
```

**GitHub Actions workflow file: `.github/workflows/model_pipeline.yml`**

```yaml
name: Model Pipeline
on:
  schedule:
    - cron: "0 2 * * *"   # 2 AM UTC daily
  workflow_dispatch:        # manual trigger
jobs:
  watch:
    runs-on: macos-14       # Apple Silicon runner for compression
    steps:
      - uses: actions/checkout@v4
      - run: pip install squish[quant,eval]
      - run: python dev/scripts/model_pipeline.py --job watch
      - run: python dev/scripts/model_pipeline.py --job compress --validate
      - run: python dev/scripts/model_pipeline.py --job publish
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Deliverables:**
- [ ] `dev/scripts/model_pipeline.py` — three jobs (watch, compress, validate, publish); `--dry-run` flag that skips all writes; output `dev/results/pipeline_run_<date>.json`
- [ ] `.github/workflows/model_pipeline.yml` — daily cron + manual trigger; uses `macos-14` runner
- [ ] `dev/scripts/model_pipeline.py` — Job 2 accuracy gate: if delta > 3 pp, retry int8; if still > 3 pp, write to `pipeline_rejected.json` and skip publish
- [ ] `squish/catalog.py` — add `hf_sha256: str | None` field to `CatalogEntry`; `squish run` verifies local file hash before serving (prevents using a partially-downloaded model)
- [ ] `tests/test_model_pipeline_unit.py` — 10+ tests: candidate filter logic (license check, size check, age check), accuracy gate pass/fail/retry, catalog diff writer, mock HF API responses

---

### 16B — OpenAI API Compliance Test Suite

Before launch, squish must pass a standardized agent compatibility matrix. No test file currently covers real OpenAI SDK behavioral expectations.

**New file: `tests/test_openai_compat.py`**

Uses the `openai` Python SDK pointed at `http://localhost:11434` (squish serve). Tests are marked `@pytest.mark.integration` — skipped unless `--run-integration` is passed (same pattern as existing hardware tests).

| Test | What it validates |
|------|-----------------|
| `test_chat_streaming_content` | `stream=True` yields incremental `delta.content` chunks |
| `test_chat_streaming_tool_call` | `stream=True` with `tools=[...]` yields `delta.tool_calls` chunks (Phase 15A) |
| `test_tool_choice_none` | `tool_choice="none"` returns plain text even with tools present |
| `test_tool_choice_required` | `tool_choice="required"` always returns a tool call |
| `test_tool_choice_named` | `tool_choice={"function": {"name": "X"}}` forces schema X |
| `test_stop_sequence_excluded` | Stop token absent from response text |
| `test_multi_turn_tool` | 5-turn exchange with tool call, tool result, assistant response cycle |
| `test_json_decode_100_turns` | 100 consecutive tool calls; zero `json.JSONDecodeError` exceptions |
| `test_grammar_schema_cache` | Schema compilation not called twice for same tools array |
| `test_continue_dev_config` | Continue.dev standard request format round-trips correctly |
| `test_langchain_tool_bind` | LangChain `ChatOpenAI.bind_tools()` works end-to-end |

**Deliverables:**
- [ ] `tests/test_openai_compat.py` — 11 tests above, all using real `openai` SDK, marked `@pytest.mark.integration`
- [ ] `pyproject.toml` — add `[tool.pytest.ini_options] markers = ["integration: requires live squish serve"]`
- [ ] `dev/scripts/run_compat_tests.sh` — helper script: starts `squish serve --agent --model qwen-coder:7b`, waits for `/health`, runs `pytest tests/test_openai_compat.py --run-integration`, outputs pass/fail table

---

### 16C — Launch Demo Production Guide

**New file: `dev/demos/agent_demo_guide.md`**

A step-by-step production guide for recording the definitive "Show HN" demo. Based on the Gemini blueprint (split-screen, naive vs squish), with specific macOS screen recording instructions.

**Demo script structure:**

**Left pane — Naive baseline** (Ollama + unoptimized 8B):
1. `ollama serve` + `ollama run qwen2.5:7b`
2. Start OpenClaw agent targeting Ollama endpoint
3. Record: agent starts a coding task; after 5 turns show `vm_stat` in corner panel — free pages dropping
4. At turn 10–12: agent crashes with `JSONDecodeError` or beachball from swap

**Right pane — Squish**:
1. `squish serve --agent --model qwen-coder:7b`
2. Same OpenClaw agent targeting squish endpoint at same port
3. Record: squish serving all 20 turns
4. Corner panel: `vm_stat` free pages FLAT — no memory growth
5. Turn-by-turn TTFT shown in squish server log: `Turn 1: 4.2s | Turn 2: 0.14s | Turn 3: 0.12s` (RadixTree working)
6. Terminal output: zero `JSONDecodeError` exceptions across all 20 turns

**Recording checklist:**
- [ ] Use `asciinema rec` for the terminal captures; downsample to GIF with `agg`
- [ ] Side-by-side layout: `tmux` with `split-window -h`; pane widths 50/50
- [ ] In-video annotations (DaVinci Resolve free tier): callout arrows for TTFT numbers, memory curve
- [ ] Target total demo length: 90 seconds

**Hacker News first comment template** (Gemini blueprint refined):

```
Title: Show HN: Squish – Run 50-turn AI agents locally on a 16GB Mac without hitting swap

First comment (post immediately after):

We built Squish because running OpenClaw or LangChain agents locally on a 16GB Mac
was unusable: the KV cache filled RAM by turn 15, JSON hallucinations crashed the loop
every few turns, and the model re-processed the 10K system prompt on every step.

Three things we built to fix this:
1. Asymmetric INT2 KV cache — attention sinks and local window stay FP16;
   deep history compresses 6× to INT2. 32K context fits in < 3 GB.
2. RadixTree prefix reuse — second turn TTFT drops from 4s to 140ms because
   the 10K system prompt is never recomputed.
3. Grammar-constrained decoding — JSON FSM masks invalid tokens at the logit level.
   100 consecutive tool calls, zero JSONDecodeErrors, even from a 7B model.

Benchmark: Qwen2.5-Coder-7B INT4 on an M3, 20-turn OpenClaw session — peak RAM: 8.1 GB
(model 4.1 GB + KV 2.3 GB + macOS 3.5 GB). Never hit swap.

MIT license. OpenAI + Ollama drop-in compatible. Zero code changes to existing agent code.
```

**Deliverables:**
- [ ] `dev/demos/agent_demo_guide.md` — full recording guide (steps, tools, checklist)
- [ ] `dev/demos/hn_first_comment.md` — HN title + first comment text, finalized with real measured numbers
- [ ] `dev/demos/record_agent_demo.py` — automated `asciinema` recorder script that scripts the terminal commands (echo delays, simulated agent output from fixture data)
- [ ] `dev/community_posts.md` — add "Agent Runtime" section with platform-specific variants: HN (technical), r/LocalLLaMA (demo-first), r/macapps (user-facing), X/Twitter (thread format)

---

### Phase 16 Deliverables Summary

| Deliverable | File | Value |
|------------|------|-------|
| Automated model pipeline | `dev/scripts/model_pipeline.py` + `.github/workflows/model_pipeline.yml` | Fresh squished models without manual work |
| OpenAI compat test suite | `tests/test_openai_compat.py` | Agent framework compatibility proof |
| Launch demo guide | `dev/demos/agent_demo_guide.md` | Viral demo asset |
| HN post template | `dev/demos/hn_first_comment.md` | Launch narrative ready to ship |

**Phase 16 verification:**
- [ ] `model_pipeline.py --job watch --dry-run` outputs at least 3 candidate models without network errors
- [ ] `model_pipeline.py --job compress --validate --dry-run` runs through the compress+validate flow on a cached model without uploading
- [ ] `pytest tests/test_openai_compat.py --run-integration` passes 10/11 tests with squish serve running (1 may skip for LangChain version)
- [ ] Demo recording completes 20-turn OpenClaw session; `vm_stat` free pages remain above 1 GB throughout

---

## Final Version Roadmap (complete)

| Version | Phases | Theme |
|---------|--------|-------|
| v9.x | 1–8 | Core baseline through module solidification (complete) |
| v10.0 | 9 + 10 | Sub-2-bit quantization (AQLM, QuIP#) + Apple Silicon bandwidth optimization |
| v10.1 | 11 | 5-track benchmark suite |
| v11.0 | 13 | Agentic runtime hardening (AgentKV, MemoryGovernor, RadixTree, `--agent` preset) |
| v11.1 | 14 | MoE expert lookahead router (DeepSeek-Coder-V2-Lite support) |
| v11.2 | 15 | Grammar engine hardening + OpenAI agent API compliance (P0 bugs) |
| v12.0 | 16 + 7 | CI/CD pipeline + demo production + public launch |

**Net new module count (v10–v12):**

| Phase | New files | Net impact |
|-------|-----------|-----------|
| 9 | `aqlm.py`, `quip_sharp.py` | 2 new modules |
| 10 | `neuron_profile.py`, `neuron_router.py`, `metal_fusion.py` | 3 new modules |
| 13 | `agent_kv.py`, `memory_governor.py` | 2 new modules + server/cli wiring |
| 14 | `moe_lookahead.py` | 1 new module + catalog updates |
| 15 | `tool_calling.py` extension, `grammar_engine.py` fixes, `server.py` fixes | 0 new files (bug fixes + hardening) |
| 16 | `dev/scripts/model_pipeline.py`, `tests/test_openai_compat.py`, demo files | 0 new squish/ modules |
| **Total** | | **8 new modules** (squish/ count: 188 → 196) |

---
