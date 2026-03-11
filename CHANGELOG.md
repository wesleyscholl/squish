# Changelog

All notable changes to Squish are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [3.0.0] — 2026-03-11

### Added — Wave 17: Attention Architecture

- **SageAttention2** (`squish/sage_attention2.py`) — INT4/INT8 warp-tile quantised attention via `SageAttention2Kernel.forward()` + `warp_quantize_int4()`. 672 µs forward (4 heads, seq=32, d=64); bandwidth-optimal for long sequences.
- **StreamingSink** (`squish/streaming_sink.py`) — Attention-sink KV eviction cache via `StreamingSinkCache`. Keeps `num_sinks` initial tokens + a sliding window; bounded memory at any context length.
- **KVSlab** (`squish/kv_slab.py`) — Pre-allocated slab page allocator for KV via `KVSlabAllocator`. 0.87 µs alloc+free round-trip; eliminates per-token malloc fragmentation.
- **SqueezeAttention** (`squish/squeeze_attention.py`) — Joint 2D KV budget allocation (token × layer axes) via `BudgetAllocator.allocate()` + `SqueezeKVCache`. Pareto-optimal vs. independent axis compression.
- **SmallKV** (`squish/smallkv.py`) — Saliency-compensated KV recall for small models via `SmallKVStore`. 39 µs ingest, 8 µs check-and-recall; protects quality under aggressive KV budgets.
- **SpeContext** (`squish/specontext.py`) — Speculative-decode context retrieval cache via `SpeContextCache`. Cosine-similarity top-k retrieve at 3.3 ms; eliminates context re-fetch per draft step.
- **SVDq** (`squish/svdq.py`) — Head-wise SVD low-rank K quantisation via `SVDqCalibrator.search()`. 62 ms one-time calibration; mixed-precision K across layers and heads.
- **CommVQ** (`squish/comm_vq.py`) — Communal vector-quantised KV codebook via `CommVQCodebook`. 55 µs encode, 68 µs decode; shared codebook eliminates per-layer redundancy.
- **ChunkedPrefill** (`squish/chunked_prefill.py`) — Interleaved chunked prefill iterator via `ChunkedPrefillIterator`. Bounded per-chunk latency; prevents decoding stalls during long prefills.
- **GemFilter** (`squish/gemfilter.py`) — Attention-score KV token selector via `GemSelector.select()` + `AttentionScoreBuffer`. 0.90× compression ratio, 50 µs selection for 512-token contexts.
- **MInferencePatch** (`squish/minference_patch.py`) — Dynamic sparse attention patcher via `patch_model_minference()`. Sub-quadratic attention for 1M+ token contexts via vertical/diagonal/slash patterns.
- **PromptCompressor** (`squish/prompt_compressor.py`) — TF-IDF sentence-level prompt compression via `PromptCompressor.compress()`. 686 µs for 50 sentences at ratio=0.3; preserves query-relevant content.
- **PromptLookup** (`squish/prompt_lookup.py`) — N-gram speculative draft generator via `PromptLookupBuffer`. 0.8 µs find, 3.3 µs push; zero-model spec-decode from prompt n-grams.
- **TRAIL** (`squish/trail.py`) — Output-length linear-probe predictor via `TrailLinearProbe.predict()` + `TrailPredictor.srpt_priority()`. 10 µs predict; feeds SRPT scheduling queue.

### Added — Wave 18: Adaptive Compute

- **VPTQ** (`squish/vptq.py`) — Vector-product tree quantisation via `VPTQCodebook` + `VPTQQuantizer`. 15 µs decode, 133 ms one-time compress (W=32×32); captures intra-vector correlations.
- **LayerSkip** (`squish/layer_skip.py`) — Confidence-gated early exit via `LayerSkipEstimator`. 266 µs estimate; exits before `lm_head` when token confidence exceeds threshold=0.85.
- **SWIFT** (`squish/swift.py`) — Weight-irrelevant FFN layer skip via `SWIFTCalibrator.calibrate()`. 162 µs calibrate; identifies and skips 34% of functionally redundant FFN layers.
- **SpecReason** (`squish/spec_reason.py`) — Speculative reasoning step orchestrator via `SpecReasonOrchestrator.generate_step()`. 6.6 µs per step; pipelines draft+target verification.
- **MirrorSD** (`squish/mirror_sd.py`) — Mirror speculative decode pipeline via `MirrorDraftPipeline.step()`. 867 µs step (vocab=32k); runs parallel draft branches to capture acceptance bursts.
- **SparseVerify** (`squish/sparse_verify.py`) — Inter-draft KV reuse cache via `InterDraftReuseCache`. 0.28 µs `query_reuse()`; near-zero overhead for skipping re-verified identical KV slices.
- **RobustScheduler** (`squish/robust_scheduler.py`) — A-balanced SRPT request scheduler via `RobustScheduler.schedule_batch()`. 3.7 µs schedule 32 requests; prevents priority inversions under bursty workloads.
- **BlockExpertArchive** (`squish/block_expert_archive.py`) — Block-expert weight archive and router via `ExpertRouter.route()`. 73 µs route 8 experts; enables offline expert delta caching.
- **DISCRouter** (`squish/disc_router.py`) — Decomposed inference sub-task planner via `DISCRouter.plan()` + `execute_plan()`. 22.9 µs plan, 3.1 µs execute; parallelises independent sub-tasks.
- **SelfLearning** (`squish/self_learning.py`) — LoRA-free online domain adaptation via `SelfLearner.learn_from_examples()`. 6 ms per 4-example step; absorbs domain examples without full fine-tuning.
- **SemanticCache** (`squish/semantic_cache.py`) — sqlite-vec semantic response cache via `SemanticCache`. Cosine-similarity hit short-circuits full inference for semantically equivalent queries.
- **IPW** (`squish/ipw.py`) — Inference performance-per-watt tracker via `IPWTracker`. 0.16 µs record, 4.6 ms `summary()`; tracks tokens/watt across workloads.
- **PowerMonitor** (`squish/power_monitor.py`) — Apple Silicon power source advisor via `PowerMonitor`. 0.5 µs `get_power_source()` + `get_recommended_mode()`; adjusts compute policy for battery vs. AC.
- **DiffusionDraft** (`squish/diffusion_draft.py`) — Diffusion-model draft head capability gate via `DiffusionDraftHead`. `is_available()` + `is_suitable_for_task()`; enables parallel diffusion-based speculation.

### Tests

- Added `tests/test_wave17_server_wiring.py` — 56 tests covering all 14 Wave 17 module import, instantiation, and core API paths.
- Added `tests/test_wave18_server_wiring.py` — 56 tests covering all 14 Wave 18 module import, instantiation, and core API paths.
- Total tests: **4 166 passing**, 16 skipped, 0 failures.

### Benchmarks

- Added `dev/benchmarks/bench_wave17_18.py` — micro-benchmark suite for all 28 Wave 17+18 modules.
- Added `dev/results/wave17_18_bench.json` — machine-readable benchmark output.
- Added `docs/benchmark_wave17_18.md` — human-readable results table.

### Docs

- Updated `README.md` with v5 section, Wave 17+18 module tables, and combined stack CLI examples.
- Updated `PLAN.md` to mark v5 complete and note v6 roadmap.
- Added `dev/demos/record_v5_demo.py` — v5 demo GIF generator.

---

## [2.0.0] — 2026-03-12

### Added — Wave 15: Serving Intelligence + KV Architecture Evolution

- **AdaServe** (`squish/ada_serve.py`) — SLO-aware speculative decode scheduling via `AdaServeScheduler`; `register_slo()` + `enqueue()` + `get_gamma()`. 30% P99 latency reduction · 1.5–2× throughput across mixed SLO workloads.
- **ConfSpec** (`squish/conf_spec.py`) — Confidence-gated verification routing with three paths (AUTO_ACCEPT / LIGHTWEIGHT / FULL_TARGET) via `ConfSpecVerifier.verify_step()`. 54% verification cost reduction.
- **SeqPacking** (`squish/seq_packing.py`) — Barrel-effect-free sequence packing via `SequencePacker.pack()`. +1.8× effective batch throughput.
- **MetaReasoner** (`squish/meta_reasoner.py`) — Dynamic per-token thinking budget via `MetaReasoner.step()` with entropy gates. 44–89% CoT energy saved on non-reasoning turns.
- **YOCO** (`squish/yoco.py`) — You Only Cache Once cross-decoder KV sharing via `YOCOKVStore`; self-attention layers cache normally, cross-decoder layers share. −50% KV memory.
- **CLA** (`squish/cla.py`) — Cross-Layer Attention sharing schedule via `CLASchedule.from_config()`; configurable sharing factor. 10–30% KV cache reduction.
- **KVSharer** (`squish/kvsharer.py`) — Data-driven cross-layer KV correlation calibration via `KVSharerCalibrator`; produces `KVShareMap`. ~30% KV ops saved.
- **DiffKV** (`squish/diffkv.py`) — Differentiated asymmetric K/V precision tiering (head-type-aware) via `DiffKVPolicyManager`. 2.7–5.7× KV compression · 1.9–5.4× decode throughput.
- **ParisKV** (`squish/paris_kv.py`) — Drift-robust online KV quantisation via `ParisKVCodebook`; calibrated VQ with continuous centroid adaptation. 4× KV compression.
- **KVTuner** (`squish/kvtuner.py`) — Sensitivity-aware mixed-precision KV search via `KVTunerCalibrator.search()`. 20–35% accuracy restored vs uniform quantisation.

### Added — Wave 16: Heterogeneous Compute + Advanced Spec-Decode

- **Dovetail** (`squish/dovetail.py`) — CPU+GPU concurrent speculative decode via `DovetailCPUVerifier` + `DovetailDecoder` + `DovetailDraftRunner`. 2× throughput via pipeline overlap.
- **PIPO** (`squish/pipo.py`) — Pipelined prefetch-offload INT4 matmul via `PIPOScheduler`; weight DMA overlapped with GPU compute. +1.7× throughput on offloaded models.
- **MobileMoE** (`squish/mobile_moe.py`) — MoE balanced layer-expert routing via `MoBiLERouter`. +1.4× throughput vs naïve expert dispatch.
- **OnlineSD** (`squish/online_sd.py`) — Continuous draft-head adaptation via `OnlineDraftUpdater`; updates draft weights from trace buffer without full retraining. +5–8 pp acceptance rate.
- **LookaheadReasoning** (`squish/lookahead_reasoning.py`) — Parallel step reasoning verification via `LookaheadReasoningEngine.run_cycle()`. +2.1× reasoning throughput.
- **SparseSpec** (`squish/sparse_spec.py`) — Dynamic sparse self-speculation with pillar-attention cache via `SparseSpecDecoder` + `PillarAttnCache`. +2.13× spec-decode throughput.
- **FRSpec** (`squish/fr_spec.py`) — Frequency-ranked vocab subset draft head via `FRSpecHead`; subset calibrated by `FRSpecCalibrator`. −13% draft latency.
- **LongSpec** (`squish/long_spec.py`) — Long-context shared-KV draft head via `LongSpecHead`; zero draft KV overhead at any context length.
- **ForeLen** (`squish/forelen.py`) — Entropy-guided output length prediction via `EGTPPredictor` (entropy histogram) + `PLPPredictor` (exponential decay). −29% MAE vs TRAIL.
- **RASD** (`squish/rasd.py`) — Retrieval-augmented speculative decode via `CorpusIndex` + `RASDBatcher.build_retrieval_tree()`. 40–60% corpus hit rate.

### Tests

- Added `tests/test_wave15_server_wiring.py` — 44 tests covering all Wave 15 module import, instantiation, and core API paths.
- Added `tests/test_wave16_server_wiring.py` — 45 tests covering all Wave 16 module import, instantiation, and core API paths.
- Total tests: **3 937 passing**, 0 failures.

### Benchmarks

- Added `dev/benchmarks/bench_wave15_16.py` — micro-benchmark suite for all 21 Wave 15+16 modules.
- Added `dev/results/wave15_16_bench.json` — machine-readable benchmark output.
- Added `docs/benchmark_wave15_16.md` — human-readable results table.

### Docs

- Updated `README.md` with v4 section, Wave 15+16 module tables, and combined stack CLI example.
- Added `PLAN.md` documenting v1–v4 release history and v5 roadmap.
- Added `dev/demos/record_v4_demo.py` — v4 demo GIF generator.
- Added `dev/demos/squish-v4-demo.cast` + `squish-v4-demo.gif`.

---

## [1.0.1] — 2026-03-04

### Fixed

- **`eval_output/eval_report.md`** — Replaced physically impossible benchmark numbers
  (+14.1% ARC, +15.2% HellaSwag after lossy compression) with validated results from a
  clean re-run; added a clearly labelled validity-notice header.
- **`KVLayerCache.update_and_fetch` / `.offset`** — Added the `update_and_fetch(keys, values)`
  method and read-only `offset` property required by the mlx_lm per-layer cache protocol.
  Without these, `--kv-cache-mode int8/snap` silently had no effect on generation.
- **`QuantizedKVCache.__getitem__`** — Now returns `self._layers[idx]` (a `KVLayerCache`
  with `update_and_fetch`) instead of a `_LayerCacheView` wrapper that lacked the protocol
  method.
- **`server.py` `_sample_mx()`** — Added module-level temperature + nucleus-sampling helper
  used by the quantized KV cache generation path.
- **`server.py` KV cache generation path** — Wired the quantized cache into `_stream_tokens`;
  `--kv-cache-mode int8/snap` now routes through `model(x, cache=layer_caches)` per decode
  step with graceful fallback to `mlx_lm.stream_generate` on error.
- **`server.py` `/v1/embeddings`** — Semantic embeddings now use `model.model(x)` (last
  hidden state) as the preferred path, falling back to `embed_tokens` then logits mean-pool.
  The previous behaviour always returned input-token embeddings, which are unsuitable for
  semantic similarity.
- **`server.py` `--log-level`** — Added argument to control uvicorn log verbosity
  (choices: `critical` / `error` / `warning` / `info` / `debug` / `trace`; default:
  `warning`).  Previously hardcoded.
- **`cli.py compress --awq / --awq-samples`** — AWQ activation-calibration pass now exposed
  on the `squish compress` subcommand.  Loads the full model, collects activation scales,
  and passes `--awq-scales` to the conversion subprocess automatically.
- **`cli.py run/serve --log-level`** — Log-level argument forwarded from `squish run` /
  `squish serve` to the server process.
- **`cli.py compress/pull --int4` help text** — Corrected disk-savings claim from “~50%” to
  “~44%” and replaced “Recommended for 1.5B models” with an explicit warning: INT4
  quantization produces degenerate output on models smaller than 3B parameters.
  Use INT8 (`--int8`, the default) for 1.5B models.

---

## [1.0.0] — 2026-03-03

**Initial public release**, accompanying the research paper.

### Added

- **Three-tier compressed weight loader** — INT8 Vectro → float16 npy → bf16 MLX safetensors
- **OpenAI-compatible API server** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`)
- **Ollama drop-in compatibility** (`/api/generate`, `/api/chat`, `/api/tags`, `/api/embeddings`)
- **Web chat UI** at `/chat` — dark-themed, streaming, multi-session history, offline
- **CLI** — `squish run` / `squish serve`, `squish chat`, `squish models`, `squish bench`, `squish info`, `squish rm`, `squish search`, `squish pull`, `squish --version`
- **Speculative decoding** — target + draft model acceleration
- **Batch scheduler** — dynamic batching with priority queues
- **KV cache quantisation** — KIVI INT8 + SnapKV compression
- **Prefix cache** — prompt prefix reuse across requests
- **Tool / function calling** — OpenAI-format `tools` → `tool_calls` round-trip
- **Rust/PyO3 INT8 quantiser** (`squish_quant_rs`) — ARM NEON SIMD vectorised
- **AWQ calibration** pass for activation-guided mixed-precision
- Integrations: Continue.dev, aider, LiteLLM (config templates in `configs/`)
- Evaluation harness wrapper (`squish[eval]`) — lm-evaluation-harness compatible

### Benchmark (Qwen2.5-1.5B-Instruct, Apple Silicon M-series)

| Metric | mlx_lm (cold) | Squish (cached) | Improvement |
|---|---:|---:|---:|
| Load time | 28.81 s | 0.53 s | **54×** |
| Peak load RAM | ~2600 MB | 402 MB | **6×** |
| Accuracy delta | — | ≤1.5% on all tasks | ✅ |
