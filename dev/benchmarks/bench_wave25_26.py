#!/usr/bin/env python3
"""
bench_wave25_26.py — Micro-benchmark suite for Squish Wave 25+26 modules.

Measures in-process CPU/numpy performance of all 28 Wave 25 and Wave 26 modules
and produces a structured JSON results file + human-readable summary table.

Wave 25 modules benchmarked (Cutting-Edge Attention Variants & Compute Fusion)
────────────────────────────────────────────────────────────────────────────────
  FlashMLA          DeepSeek-V2 multi-head latent attention       (append+attend)
  NativeSparseAttn  Block-sparse + sliding-window attention        (forward lat)
  FusedSampler      Fused temperature/top-k/top-p/rep-pen sampling (sample lat)
  KVDefrag          Online KV cache defragmentation                (alloc+defrag)
  DualChunkAttn     Intra+inter-chunk long-context attention       (forward lat)
  ActivationOffload CPU activation offloading                      (offload+fetch)
  MorphAttn         Per-layer full/sparse/linear attention morph   (select lat)
  HydraSpec         Multi-draft head speculative decoding          (draft lat)
  SeqCompact        In-place KV sequence compaction                (compact lat)
  LatencyPredictor  OLS latency forecasting for scheduler          (predict lat)
  ParallelSampler   Best-of-n + diversity-scored sampling          (sample lat)
  ContextSummarizer Importance/stride/recency context compression  (compress lat)
  TokenWatermark    Kirchenbauer green-list watermarking           (mark+detect)
  SchemaGen         FSM constrained JSON generation                (constrain lat)

Wave 26 modules benchmarked (Distributed Inference & Production Reliability)
────────────────────────────────────────────────────────────────────────────────
  TensorParallel    Row/column tensor shard + forward              (shard+fwd)
  SequenceParallel  Ulysses-style sequence scatter/gather          (scatter+gather)
  KVMigrate         Live KV state pack/unpack                      (pack+unpack)
  DisaggPrefill     Disaggregated prefill + decode step            (prefill lat)
  RequestPreempt    SRPT preemption swap/recompute                 (preempt lat)
  InferGateway      Smart request routing gateway                  (route lat)
  ModelVersionSwap  Canary→promote→rollback swap policy            (route lat)
  ProductionProfiler APM p50/p99/p999 windowed profiling          (record+stats)
  AdaptiveBatcher   Throughput/latency-objective dynamic batching  (next_batch)
  SafetyLayer       Inline token safety classifier                 (score lat)
  SemanticResponseCache Embedding-similarity LRU cache            (lookup+store)
  RateLimiter       Token-bucket per-tenant rate limiting          (consume lat)
  SchemaValidator   JSON schema validation                         (validate lat)
  AuditLogger       SHA-256 chained audit log                      (log lat)

Usage
─────
    python3 dev/benchmarks/bench_wave25_26.py
    python3 dev/benchmarks/bench_wave25_26.py --output dev/results/wave25_26_bench.json
    python3 dev/benchmarks/bench_wave25_26.py --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Colour helpers
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

RNG = np.random.default_rng(42)


def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 64}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 64}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<44} {G}{val:>14}{NC}  {D}{extra}{NC}")


def _skip(label: str, reason: str = "") -> None:
    print(f"  {Y}~ SKIP{NC}  {label:<44} {D}{reason}{NC}")


def _timeit(fn, n: int = 200, warmup: int = 10):
    """Return (mean_us, min_us, max_us) over *n* calls after *warmup* discards."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(times)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# ─────────────────────────────────────────────────────────────────────────────
# Wave 25 benchmarks — Cutting-Edge Attention Variants & Compute Fusion
# ─────────────────────────────────────────────────────────────────────────────

def bench_flash_mla(results: dict) -> None:
    _hdr("FlashMLA — DeepSeek-V2 Multi-Head Latent Attention")
    try:
        from squish.flash_mla import FlashMLACache, MLAConfig

        n_heads, head_dim, latent_dim = 8, 64, 128
        cfg   = MLAConfig(n_heads=n_heads, head_dim=head_dim, latent_dim=latent_dim)
        cache = FlashMLACache(cfg, max_seq_len=8192)

        x = RNG.random((latent_dim,)).astype(np.float32)

        mean_a, lo_a, hi_a = _timeit(
            lambda: cache.append(x), n=500
        )
        _row(f"append() latent_dim={latent_dim} (single token)",
             f"{mean_a:.2f} µs", f"min={lo_a:.2f} max={hi_a:.2f} µs")

        # Bench attend — needs projection matrices
        cache2 = FlashMLACache(cfg, max_seq_len=8192)
        for _ in range(16):
            cache2.append(RNG.random((latent_dim,)).astype(np.float32))

        q    = RNG.random((n_heads, head_dim)).astype(np.float32)
        W_uk = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)
        W_uv = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)

        mean_t, lo_t, hi_t = _timeit(
            lambda: cache2.attend(q, W_uk, W_uv), n=300
        )
        _row(f"attend() seq=16 h={n_heads} d={head_dim}",
             f"{mean_t:.2f} µs", f"min={lo_t:.2f} max={hi_t:.2f} µs")

        results["flash_mla"] = dict(
            append_mean_us=mean_a,
            append_min_us=lo_a,
            attend_mean_us=mean_t,
            attend_min_us=lo_t,
            compression_ratio=cache2.compression_ratio,
        )
    except Exception as e:
        _skip("FlashMLA", str(e))


def bench_native_sparse_attn(results: dict) -> None:
    _hdr("NativeSparseAttn — Block-Sparse + Sliding-Window Attention")
    try:
        from squish.native_sparse_attn import NSAConfig, NativeSparseAttention

        n_heads, seq, head_dim = 4, 256, 32
        cfg  = NSAConfig(n_heads=n_heads, head_dim=head_dim,
                         block_size=64, top_k_blocks=4, window_size=64)
        attn = NativeSparseAttention(cfg)

        q = RNG.random((n_heads, 16, head_dim)).astype(np.float32)
        k = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
        v = RNG.random((n_heads, seq, head_dim)).astype(np.float32)

        mean_f, lo_f, hi_f = _timeit(
            lambda: attn.forward(q, k, v), n=200
        )
        _row(f"forward() h={n_heads} q_seq=16 kv_seq={seq} d={head_dim}",
             f"{mean_f:.1f} µs", f"min={lo_f:.1f} max={hi_f:.1f} µs")

        results["native_sparse_attn"] = dict(
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
            forward_max_us=hi_f,
            sparsity=attn.sparsity,
        )
    except Exception as e:
        _skip("NativeSparseAttn", str(e))


def bench_fused_sampler(results: dict) -> None:
    _hdr("FusedSampler — Fused Temperature/Top-k/Top-p/Rep-Penalty Sampling")
    try:
        from squish.fused_sampler import FusedSampler, SamplerConfig

        vocab   = 32000
        cfg     = SamplerConfig(temperature=0.8, top_k=50, top_p=0.9,
                                min_p=0.05, repetition_penalty=1.1, seed=0)
        sampler = FusedSampler(cfg)

        logits    = RNG.standard_normal(vocab).astype(np.float32)
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        batch_logits = RNG.standard_normal((16, vocab)).astype(np.float32)

        mean_s, lo_s, hi_s = _timeit(
            lambda: sampler.sample(logits, input_ids), n=1000
        )
        _row(f"sample() vocab={vocab} with rep_penalty",
             f"{mean_s:.2f} µs", f"min={lo_s:.2f} max={hi_s:.2f} µs")

        mean_b, lo_b, hi_b = _timeit(
            lambda: sampler.sample_batch(batch_logits), n=500
        )
        _row(f"sample_batch() batch=16 vocab={vocab}",
             f"{mean_b:.1f} µs", f"min={lo_b:.1f} max={hi_b:.1f} µs")

        results["fused_sampler"] = dict(
            sample_mean_us=mean_s,
            sample_min_us=lo_s,
            sample_batch_16_mean_us=mean_b,
        )
    except Exception as e:
        _skip("FusedSampler", str(e))


def bench_kv_defrag(results: dict) -> None:
    _hdr("KVDefrag — Online KV Cache Defragmentation")
    try:
        from squish.kv_defrag import KVDefragmenter

        defrag = KVDefragmenter(page_size=16, n_heads=8, head_dim=64)

        # Pre-populate with many sequences
        for seq_id in range(60):
            defrag.allocate(seq_id, 32)

        # Free every other to create fragmentation
        for seq_id in range(0, 60, 2):
            defrag.free(seq_id)

        counter = [0]

        def _alloc():
            counter[0] += 1
            defrag.allocate(counter[0] + 1000, 16)
            defrag.free(counter[0] + 1000)

        mean_a, lo_a, hi_a = _timeit(_alloc, n=2000)
        _row("allocate()+free() 16-token seq (page_size=16)",
             f"{mean_a:.2f} µs", f"min={lo_a:.2f} max={hi_a:.2f} µs")

        mean_d, lo_d, hi_d = _timeit(lambda: defrag.defrag(), n=200)
        _row(f"defrag() frag_ratio={defrag.fragmentation_ratio:.2f}",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        results["kv_defrag"] = dict(
            alloc_free_mean_us=mean_a,
            defrag_mean_us=mean_d,
            defrag_min_us=lo_d,
            fragmentation_ratio=defrag.fragmentation_ratio,
        )
    except Exception as e:
        _skip("KVDefrag", str(e))


def bench_dual_chunk_attn(results: dict) -> None:
    _hdr("DualChunkAttn — Intra+Inter-Chunk Long-Context Attention")
    try:
        from squish.dual_chunk_attn import DCAConfig, DualChunkAttention

        n_heads, head_dim, chunk_size = 4, 32, 64
        cfg  = DCAConfig(n_heads=n_heads, head_dim=head_dim,
                         chunk_size=chunk_size, inter_chunk_top_k=4)
        dca  = DualChunkAttention(cfg)

        k_chunk = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
        v_chunk = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)

        mean_e, lo_e, hi_e = _timeit(
            lambda: dca.encode_chunk(k_chunk, v_chunk), n=500
        )
        _row(f"encode_chunk() h={n_heads} chunk_size={chunk_size} d={head_dim}",
             f"{mean_e:.2f} µs", f"min={lo_e:.2f} max={hi_e:.2f} µs")

        # Build past chunks
        past_chunks = [dca.encode_chunk(k_chunk, v_chunk) for _ in range(4)]
        q   = RNG.random((n_heads, 16, head_dim)).astype(np.float32)
        k   = RNG.random((n_heads, 16, head_dim)).astype(np.float32)
        v   = RNG.random((n_heads, 16, head_dim)).astype(np.float32)

        mean_f, lo_f, hi_f = _timeit(
            lambda: dca.forward(q, k, v, past_chunks), n=300
        )
        _row(f"forward() q_seq=16 kv_seq={chunk_size} past_chunks=4",
             f"{mean_f:.1f} µs", f"min={lo_f:.1f} max={hi_f:.1f} µs")

        results["dual_chunk_attn"] = dict(
            encode_chunk_mean_us=mean_e,
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
        )
    except Exception as e:
        _skip("DualChunkAttn", str(e))


def bench_activation_offload(results: dict) -> None:
    _hdr("ActivationOffload — CPU Activation Offloading")
    try:
        from squish.activation_offload import ActivationOffloader, OffloadPolicy

        offload_layers = list(range(0, 12, 2))  # every other layer
        policy   = OffloadPolicy(offload_layers=offload_layers, prefetch_ahead=2)
        offloader = ActivationOffloader(policy)

        tensor = RNG.random((512, 128)).astype(np.float32)

        mean_o, lo_o, hi_o = _timeit(
            lambda: offloader.offload(0, tensor), n=500
        )
        _row(f"offload() layer=0 tensor={tensor.shape} ({tensor.nbytes//1024} KB)",
             f"{mean_o:.2f} µs", f"min={lo_o:.2f} max={hi_o:.2f} µs")

        # Ensure something is present before fetch bench
        offloader.offload(2, tensor)

        mean_f, lo_f, hi_f = _timeit(
            lambda: offloader.fetch(2), n=500
        )
        _row(f"fetch() layer=2 tensor={tensor.shape}",
             f"{mean_f:.2f} µs", f"min={lo_f:.2f} max={hi_f:.2f} µs")

        results["activation_offload"] = dict(
            offload_mean_us=mean_o,
            fetch_mean_us=mean_f,
            buffer_bytes=offloader.buffer_bytes,
        )
    except Exception as e:
        _skip("ActivationOffload", str(e))


def bench_morph_attn(results: dict) -> None:
    _hdr("MorphAttn — Per-Layer Attention Pattern Morphing")
    try:
        from squish.morph_attn import AttentionMorpher, MorphConfig

        cfg    = MorphConfig(n_layers=24, seq_len_full_threshold=512,
                             seq_len_sparse_threshold=4096)
        morpher = AttentionMorpher(cfg)

        mean_s, lo_s, hi_s = _timeit(
            lambda: morpher.select_pattern(0, 1024), n=5000
        )
        _row("select_pattern() layer=0 seq=1024",
             f"{mean_s:.3f} µs", f"min={lo_s:.3f} max={hi_s:.3f} µs")

        mean_l, lo_l, hi_l = _timeit(
            lambda: morpher.layer_patterns(2048), n=2000
        )
        _row(f"layer_patterns() seq=2048 n_layers=24",
             f"{mean_l:.3f} µs", f"min={lo_l:.3f} max={hi_l:.3f} µs")

        mean_r, lo_r, hi_r = _timeit(
            lambda: morpher.estimate_flops_reduction(2048), n=5000
        )
        _row("estimate_flops_reduction() seq=2048",
             f"{mean_r:.3f} µs", f"min={lo_r:.3f} max={hi_r:.3f} µs")

        results["morph_attn"] = dict(
            select_pattern_mean_us=mean_s,
            layer_patterns_mean_us=mean_l,
            flops_reduction_seq2048=morpher.estimate_flops_reduction(2048),
        )
    except Exception as e:
        _skip("MorphAttn", str(e))


def bench_hydra_spec(results: dict) -> None:
    _hdr("HydraSpec — Multi-Draft Head Speculative Decoding")
    try:
        from squish.hydra_spec import HydraConfig, HydraSpecDecoder

        n_heads, n_draft, hidden_dim, vocab = 4, 5, 256, 8192
        cfg     = HydraConfig(n_heads=n_heads, n_draft=n_draft,
                              hidden_dim=hidden_dim, vocab_size=vocab)
        decoder = HydraSpecDecoder(cfg)

        hidden       = RNG.random((hidden_dim,)).astype(np.float32)
        target_logits = RNG.standard_normal((n_heads, n_draft, vocab)).astype(np.float32)

        mean_d, lo_d, hi_d = _timeit(
            lambda: decoder.draft(hidden), n=500
        )
        _row(f"draft() hidden={hidden_dim} heads={n_heads} n_draft={n_draft}",
             f"{mean_d:.1f} µs", f"min={lo_d:.1f} max={hi_d:.1f} µs")

        draft_out    = decoder.draft(hidden)
        mean_v, lo_v, hi_v = _timeit(
            lambda: decoder.verify(draft_out.draft_tokens, target_logits), n=1000
        )
        _row(f"verify() n_heads={n_heads} n_draft={n_draft} vocab={vocab}",
             f"{mean_v:.2f} µs", f"min={lo_v:.2f} max={hi_v:.2f} µs")

        results["hydra_spec"] = dict(
            draft_mean_us=mean_d,
            draft_min_us=lo_d,
            verify_mean_us=mean_v,
            verify_min_us=lo_v,
        )
    except Exception as e:
        _skip("HydraSpec", str(e))


def bench_seq_compact(results: dict) -> None:
    _hdr("SeqCompact — In-Place KV Sequence Compaction")
    try:
        from squish.seq_compact import SequenceCompactor

        n_heads, seq_len, head_dim = 8, 512, 64
        sc   = SequenceCompactor(n_heads=n_heads, head_dim=head_dim)

        keys   = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)
        values = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)

        # Keep ~half the tokens
        keep_mask = (RNG.random(seq_len) > 0.5).astype(bool)

        mean_c, lo_c, hi_c = _timeit(
            lambda: sc.compact(keys, values, keep_mask), n=200
        )
        _row(f"compact() h={n_heads} seq={seq_len} d={head_dim} keep≈50%",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        keep_idx = np.where(keep_mask)[0].astype(np.int64)
        mean_i, lo_i, hi_i = _timeit(
            lambda: sc.compact_indices(seq_len, keep_idx), n=2000
        )
        _row(f"compact_indices() seq={seq_len}",
             f"{mean_i:.2f} µs", f"min={lo_i:.2f} max={hi_i:.2f} µs")

        results["seq_compact"] = dict(
            compact_mean_us=mean_c,
            compact_min_us=lo_c,
            compact_indices_mean_us=mean_i,
        )
    except Exception as e:
        _skip("SeqCompact", str(e))


def bench_latency_predictor(results: dict) -> None:
    _hdr("LatencyPredictor — OLS Latency Forecasting")
    try:
        from squish.latency_predictor import LatencyPredictor

        lp = LatencyPredictor(n_heads=8, head_dim=64)

        # Populate enough samples for fit()
        for i in range(20):
            lp.record(n_prefill=i * 10 + 10,
                      n_decode=i * 2 + 2,
                      measured_ms=0.5 + i * 0.05)

        lp.fit()

        mean_p, lo_p, hi_p = _timeit(
            lambda: lp.predict(100, 10), n=5000
        )
        _row("predict() n_prefill=100 n_decode=10 (fitted model)",
             f"{mean_p:.3f} µs", f"min={lo_p:.3f} max={hi_p:.3f} µs")

        mean_r, lo_r, hi_r = _timeit(
            lambda: lp.record(100, 10, 5.5), n=5000
        )
        _row("record() single observation",
             f"{mean_r:.3f} µs", f"min={lo_r:.3f} max={hi_r:.3f} µs")

        results["latency_predictor"] = dict(
            predict_mean_us=mean_p,
            predict_min_us=lo_p,
            record_mean_us=mean_r,
            n_samples=lp.n_samples,
        )
    except Exception as e:
        _skip("LatencyPredictor", str(e))


def bench_parallel_sampler(results: dict) -> None:
    _hdr("ParallelSampler — Best-of-N + Diversity-Scored Sampling")
    try:
        from squish.parallel_sampler import DiversityConfig, ParallelSampler

        vocab  = 32000
        cfg    = DiversityConfig(n_samples=8, temperature=0.8,
                                 diversity_weight=0.1, seed=0)
        psampl = ParallelSampler(cfg)

        logits       = RNG.standard_normal(vocab).astype(np.float32)
        batch_logits = RNG.standard_normal((16, vocab)).astype(np.float32)

        mean_s, lo_s, hi_s = _timeit(
            lambda: psampl.sample(logits), n=500
        )
        _row(f"sample() vocab={vocab} n_samples=8",
             f"{mean_s:.2f} µs", f"min={lo_s:.2f} max={hi_s:.2f} µs")

        mean_b, lo_b, hi_b = _timeit(
            lambda: psampl.sample_batch(batch_logits), n=200
        )
        _row(f"sample_batch() batch=16 vocab={vocab}",
             f"{mean_b:.1f} µs", f"min={lo_b:.1f} max={hi_b:.1f} µs")

        results["parallel_sampler"] = dict(
            sample_mean_us=mean_s,
            sample_min_us=lo_s,
            sample_batch_16_mean_us=mean_b,
        )
    except Exception as e:
        _skip("ParallelSampler", str(e))


def bench_context_summarizer(results: dict) -> None:
    _hdr("ContextSummarizer — Context Compression (Importance / Recency)")
    try:
        from squish.context_summarizer import ContextSummarizer, SummaryConfig

        seq_len, embed_dim = 1024, 128
        cfg = SummaryConfig(method="importance", budget=256, min_keep_recent=64)
        cs  = ContextSummarizer(cfg)

        token_ids  = np.arange(seq_len, dtype=np.int32)
        embeddings = RNG.random((seq_len, embed_dim)).astype(np.float32)

        mean_c, lo_c, hi_c = _timeit(
            lambda: cs.summarize(token_ids, embeddings), n=100
        )
        _row(f"summarize() seq={seq_len} embed={embed_dim} budget=256",
             f"{mean_c:.1f} µs", f"min={lo_c:.1f} max={hi_c:.1f} µs")

        # Recency method — no embeddings needed
        cfg_r = SummaryConfig(method="recency", budget=256, min_keep_recent=64)
        cs_r  = ContextSummarizer(cfg_r)

        mean_r, lo_r, hi_r = _timeit(
            lambda: cs_r.summarize(token_ids), n=500
        )
        _row(f"summarize() recency seq={seq_len} budget=256",
             f"{mean_r:.1f} µs", f"min={lo_r:.1f} max={hi_r:.1f} µs")

        results["context_summarizer"] = dict(
            summarize_importance_mean_us=mean_c,
            summarize_recency_mean_us=mean_r,
        )
    except Exception as e:
        _skip("ContextSummarizer", str(e))


def bench_token_watermark(results: dict) -> None:
    _hdr("TokenWatermark — Kirchenbauer Green-List Watermarking")
    try:
        from squish.token_watermark import TokenWatermarker, WatermarkConfig

        vocab = 8192
        cfg   = WatermarkConfig(vocab_size=vocab, green_list_fraction=0.5,
                                delta=2.0, seed=42, z_threshold=4.0)
        wm    = TokenWatermarker(cfg)

        logits    = RNG.standard_normal(vocab).astype(np.float32)
        token_seq = RNG.integers(0, vocab, size=128, dtype=np.int32)

        mean_m, lo_m, hi_m = _timeit(
            lambda: wm.mark(logits, context_token=42), n=1000
        )
        _row(f"mark() vocab={vocab} with context_token",
             f"{mean_m:.2f} µs", f"min={lo_m:.2f} max={hi_m:.2f} µs")

        mean_d, lo_d, hi_d = _timeit(
            lambda: wm.detect(token_seq), n=1000
        )
        _row(f"detect() seq_len=128 vocab={vocab}",
             f"{mean_d:.2f} µs", f"min={lo_d:.2f} max={hi_d:.2f} µs")

        results["token_watermark"] = dict(
            mark_mean_us=mean_m,
            mark_min_us=lo_m,
            detect_128_mean_us=mean_d,
            detect_128_min_us=lo_d,
        )
    except Exception as e:
        _skip("TokenWatermark", str(e))


def bench_schema_gen(results: dict) -> None:
    _hdr("SchemaGen — FSM Constrained JSON Generation")
    try:
        from squish.schema_gen import SchemaGenEngine

        vocab  = 256  # character-level
        engine = SchemaGenEngine(vocab_size=vocab)

        state  = engine.reset()

        mean_c, lo_c, hi_c = _timeit(
            lambda: engine.constrain(
                RNG.standard_normal(vocab).astype(np.float32), state
            ),
            n=5000
        )
        _row(f"constrain() vocab={vocab} at initial state",
             f"{mean_c:.3f} µs", f"min={lo_c:.3f} max={hi_c:.3f} µs")

        # Step through a few valid JSON chars to advance state.
        # In _DEFAULT_SPECIAL: '{' → 0, '[' → 2, '"' → 4, digit → 10.
        # Advance with '{' (id=0) to move into object-parsing state.
        LBRACE_ID = 0
        new_state = engine.advance(LBRACE_ID, state)

        mean_a, lo_a, hi_a = _timeit(
            lambda: engine.advance(LBRACE_ID, state), n=5000
        )
        _row("advance() '{' token_id=0 (object open)",
             f"{mean_a:.3f} µs", f"min={lo_a:.3f} max={hi_a:.3f} µs")

        results["schema_gen"] = dict(
            constrain_mean_us=mean_c,
            advance_mean_us=mean_a,
        )

    except Exception as e:
        _skip("SchemaGen", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Wave 26 benchmarks — Distributed Inference & Production Reliability
# ─────────────────────────────────────────────────────────────────────────────

def bench_tensor_parallel(results: dict) -> None:
    _hdr("TensorParallel — Row/Column Tensor Shard + Forward Pass")
    try:
        from squish.tensor_parallel import TPConfig, TensorParallelShard

        n_devices, in_dim, out_dim = 4, 256, 512
        cfg  = TPConfig(n_devices=n_devices, mode="column")
        tp   = TensorParallelShard(cfg)

        W    = RNG.random((in_dim, out_dim)).astype(np.float32)
        x    = RNG.random((8, in_dim)).astype(np.float32)

        mean_sh, lo_sh, hi_sh = _timeit(
            lambda: tp.shard(W), n=500
        )
        _row(f"shard() W={in_dim}×{out_dim} n_devices={n_devices} column-mode",
             f"{mean_sh:.2f} µs", f"min={lo_sh:.2f} max={hi_sh:.2f} µs")

        shards = tp.shard(W)
        mean_f, lo_f, hi_f = _timeit(
            lambda: tp.forward(x, shards), n=500
        )
        _row(f"forward() batch=8 in={in_dim} out={out_dim}",
             f"{mean_f:.2f} µs", f"min={lo_f:.2f} max={hi_f:.2f} µs")

        results["tensor_parallel"] = dict(
            shard_mean_us=mean_sh,
            forward_mean_us=mean_f,
            forward_min_us=lo_f,
        )
    except Exception as e:
        _skip("TensorParallel", str(e))


def bench_sequence_parallel(results: dict) -> None:
    _hdr("SequenceParallel — Ulysses-Style Sequence Scatter/Gather")
    try:
        from squish.sequence_parallel import SPConfig, SequenceParallelScatter

        n_devices, n_heads, seq, head_dim = 4, 8, 256, 64
        cfg = SPConfig(n_devices=n_devices, n_heads=n_heads, head_dim=head_dim)
        sp  = SequenceParallelScatter(cfg)

        x = RNG.random((n_heads, seq, head_dim)).astype(np.float32)

        mean_sc, lo_sc, hi_sc = _timeit(
            lambda: sp.scatter(x), n=500
        )
        _row(f"scatter() h={n_heads} seq={seq} d={head_dim} → {n_devices} devices",
             f"{mean_sc:.2f} µs", f"min={lo_sc:.2f} max={hi_sc:.2f} µs")

        chunks = sp.scatter(x)
        mean_g, lo_g, hi_g = _timeit(
            lambda: sp.gather(chunks), n=500
        )
        _row(f"gather() {n_devices} chunks → h={n_heads} seq={seq} d={head_dim}",
             f"{mean_g:.2f} µs", f"min={lo_g:.2f} max={hi_g:.2f} µs")

        results["sequence_parallel"] = dict(
            scatter_mean_us=mean_sc,
            gather_mean_us=mean_g,
            communication_bytes=sp.communication_bytes,
        )
    except Exception as e:
        _skip("SequenceParallel", str(e))


def bench_kv_migrate(results: dict) -> None:
    _hdr("KVMigrate — Live KV State Pack/Unpack")
    try:
        from squish.kv_migrate import KVMigrator

        n_heads, seq_len, head_dim = 8, 128, 64
        migrator = KVMigrator(n_heads=n_heads, head_dim=head_dim)

        keys   = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)
        values = RNG.random((n_heads, seq_len, head_dim)).astype(np.float32)

        mean_p, lo_p, hi_p = _timeit(
            lambda: migrator.pack(42, keys, values), n=200
        )
        _row(f"pack() seq={seq_len} h={n_heads} d={head_dim}",
             f"{mean_p:.1f} µs", f"min={lo_p:.1f} max={hi_p:.1f} µs")

        packed_data, stats = migrator.pack(42, keys, values)
        mean_u, lo_u, hi_u = _timeit(
            lambda: migrator.unpack(packed_data, stats), n=200
        )
        _row(f"unpack() {len(packed_data)//1024} KB payload",
             f"{mean_u:.1f} µs", f"min={lo_u:.1f} max={hi_u:.1f} µs")

        results["kv_migrate"] = dict(
            pack_mean_us=mean_p,
            pack_min_us=lo_p,
            unpack_mean_us=mean_u,
            packed_bytes=stats.packed_bytes,
        )
    except Exception as e:
        _skip("KVMigrate", str(e))


def bench_disagg_prefill(results: dict) -> None:
    _hdr("DisaggPrefill — Disaggregated Prefill + Decode Step")
    try:
        from squish.disagg_prefill import DisaggConfig, DisaggDecodeNode, DisaggPrefillNode

        n_heads, head_dim, n_layers = 8, 64, 4
        cfg = DisaggConfig(n_heads=n_heads, head_dim=head_dim, n_layers=n_layers)

        prefill_node = DisaggPrefillNode(cfg)
        decode_node  = DisaggDecodeNode(cfg)

        token_ids = np.arange(64, dtype=np.int32)

        counter = [0]

        def _prefill():
            counter[0] += 1
            return prefill_node.prefill(counter[0], token_ids)

        mean_p, lo_p, hi_p = _timeit(_prefill, n=200)
        _row(f"prefill() seq=64 h={n_heads} d={head_dim} n_layers={n_layers}",
             f"{mean_p:.1f} µs", f"min={lo_p:.1f} max={hi_p:.1f} µs")

        # Load a payload and bench decode step
        payload = prefill_node.prefill(9999, token_ids)
        decode_node.load_payload(payload)

        mean_s, lo_s, hi_s = _timeit(
            lambda: decode_node.step(9999), n=1000
        )
        _row(f"step() decode token generation",
             f"{mean_s:.3f} µs", f"min={lo_s:.3f} max={hi_s:.3f} µs")

        results["disagg_prefill"] = dict(
            prefill_64_mean_us=mean_p,
            decode_step_mean_us=mean_s,
            decode_step_min_us=lo_s,
        )
    except Exception as e:
        _skip("DisaggPrefill", str(e))


def bench_request_preempt(results: dict) -> None:
    _hdr("RequestPreempt — SRPT Preemption Swap/Recompute")
    try:
        from squish.request_preempt import PreemptScheduler

        n_heads, head_dim, n_layers = 4, 32, 4
        sched = PreemptScheduler(n_heads=n_heads, head_dim=head_dim,
                                 n_layers=n_layers)

        kv = RNG.random((n_layers, 2, n_heads, 32, head_dim)).astype(np.float32)

        counter = [0]

        def _preempt_resume():
            counter[0] += 1
            seq_id = counter[0]
            sched.preempt(seq_id, current_kv=kv, strategy="swap")
            sched.resume(seq_id)

        mean_pr, lo_pr, hi_pr = _timeit(_preempt_resume, n=500)
        _row(f"preempt(swap)+resume() h={n_heads} d={head_dim} layers={n_layers}",
             f"{mean_pr:.2f} µs", f"min={lo_pr:.2f} max={hi_pr:.2f} µs")

        def _preempt_recompute():
            counter[0] += 1
            seq_id = counter[0]
            sched.preempt(seq_id, strategy="recompute")
            sched.resume(seq_id)

        mean_rc, lo_rc, hi_rc = _timeit(_preempt_recompute, n=500)
        _row("preempt(recompute)+resume()",
             f"{mean_rc:.2f} µs", f"min={lo_rc:.2f} max={hi_rc:.2f} µs")

        results["request_preempt"] = dict(
            preempt_swap_resume_mean_us=mean_pr,
            preempt_recompute_resume_mean_us=mean_rc,
        )
    except Exception as e:
        _skip("RequestPreempt", str(e))


def bench_infer_gateway(results: dict) -> None:
    _hdr("InferGateway — Smart Request Routing Gateway")
    try:
        from squish.infer_gateway import InferenceGateway

        gw = InferenceGateway()
        for i in range(8):
            gw.register(f"worker-{i}", capacity=16, model_version="v1")

        counter = [0]

        def _route_complete():
            counter[0] += 1
            result = gw.route(counter[0])
            gw.complete(result.worker_id)

        mean_r, lo_r, hi_r = _timeit(_route_complete, n=5000)
        _row("route()+complete() across 8 workers cap=16",
             f"{mean_r:.3f} µs", f"min={lo_r:.3f} max={hi_r:.3f} µs")

        results["infer_gateway"] = dict(
            route_complete_mean_us=mean_r,
            route_complete_min_us=lo_r,
            n_workers=len(gw.workers),
        )
    except Exception as e:
        _skip("InferGateway", str(e))


def bench_model_version_swap(results: dict) -> None:
    _hdr("ModelVersionSwap — Canary→Promote→Rollback Policy")
    try:
        from squish.model_version_swap import ModelVersionManager, SwapPolicy

        policy = SwapPolicy(canary_fraction=0.1, min_canary_requests=50)
        mgr    = ModelVersionManager(policy)

        # Bootstrap: register v1 and commit unconditionally
        mgr.register_version("v1")
        mgr.stage("v1")
        mgr.commit()

        # Register v2, stage as canary
        mgr.register_version("v2")
        mgr.stage("v2")

        mean_r, lo_r, hi_r = _timeit(
            lambda: mgr.route_request(), n=5000
        )
        _row("route_request() canary 10% fraction",
             f"{mean_r:.3f} µs", f"min={lo_r:.3f} max={hi_r:.3f} µs")

        results["model_version_swap"] = dict(
            route_request_mean_us=mean_r,
            route_request_min_us=lo_r,
            active_version=mgr.active_version,
        )
    except Exception as e:
        _skip("ModelVersionSwap", str(e))


def bench_production_profiler(results: dict) -> None:
    _hdr("ProductionProfiler — APM p50/p99/p999 Windowed Profiling")
    try:
        from squish.production_profiler import ProductionProfiler

        prof = ProductionProfiler()

        # Pre-populate with 500 samples
        for i in range(500):
            prof.record("prefill", float(1.0 + i * 0.01))
            prof.record("decode",  float(0.5 + i * 0.005))

        mean_rec, lo_rec, hi_rec = _timeit(
            lambda: prof.record("prefill", 5.0), n=5000
        )
        _row("record() single latency observation",
             f"{mean_rec:.3f} µs", f"min={lo_rec:.3f} max={hi_rec:.3f} µs")

        mean_st, lo_st, hi_st = _timeit(
            lambda: prof.stats("prefill"), n=2000
        )
        _row("stats() — p50/p99/p999 calculation",
             f"{mean_st:.3f} µs", f"min={lo_st:.3f} max={hi_st:.3f} µs")

        stats = prof.stats("prefill")
        results["production_profiler"] = dict(
            record_mean_us=mean_rec,
            stats_mean_us=mean_st,
            p99_ms=stats.p99_ms,
        )
    except Exception as e:
        _skip("ProductionProfiler", str(e))


def bench_adaptive_batcher(results: dict) -> None:
    _hdr("AdaptiveBatcher — Throughput/Latency-Objective Dynamic Batching")
    try:
        from squish.adaptive_batcher import AdaptiveBatchController, BatchObjective

        obj  = BatchObjective(mode="throughput", target_latency_ms=100.0,
                              max_batch_size=32, min_batch_size=1)
        ctrl = AdaptiveBatchController(obj)

        # Warm up with observations
        for bs in range(1, 17):
            ctrl.record_observation(bs, float(bs * 4.0))

        mean_n, lo_n, hi_n = _timeit(
            lambda: ctrl.next_batch(64), n=5000
        )
        _row("next_batch() queue_depth=64 throughput mode",
             f"{mean_n:.3f} µs", f"min={lo_n:.3f} max={hi_n:.3f} µs")

        mean_o, lo_o, hi_o = _timeit(
            lambda: ctrl.record_observation(8, 32.0), n=5000
        )
        _row("record_observation() EMA update",
             f"{mean_o:.3f} µs", f"min={lo_o:.3f} max={hi_o:.3f} µs")

        dec = ctrl.next_batch(64)
        results["adaptive_batcher"] = dict(
            next_batch_mean_us=mean_n,
            record_obs_mean_us=mean_o,
            recommended_batch_size=dec.batch_size,
        )
    except Exception as e:
        _skip("AdaptiveBatcher", str(e))


def bench_safety_layer(results: dict) -> None:
    _hdr("SafetyLayer — Inline Token Safety Classifier")
    try:
        from squish.safety_layer import SafetyClassifier, SafetyConfig

        vocab = 8192
        cfg   = SafetyConfig(vocab_size=vocab, n_categories=4,
                             threshold=0.5, seed=42)
        clf   = SafetyClassifier(cfg)

        token_ids   = RNG.integers(0, vocab, size=64, dtype=np.int32)
        logits_1d   = RNG.standard_normal(vocab).astype(np.float32)
        logits_2d   = RNG.standard_normal((32, vocab)).astype(np.float32)

        mean_s, lo_s, hi_s = _timeit(
            lambda: clf.score(token_ids), n=500
        )
        _row(f"score() seq=64 vocab={vocab}",
             f"{mean_s:.2f} µs", f"min={lo_s:.2f} max={hi_s:.2f} µs")

        mean_l, lo_l, hi_l = _timeit(
            lambda: clf.score_logits(logits_1d), n=500
        )
        _row(f"score_logits() 1D logits vocab={vocab}",
             f"{mean_l:.2f} µs", f"min={lo_l:.2f} max={hi_l:.2f} µs")

        results["safety_layer"] = dict(
            score_64_mean_us=mean_s,
            score_logits_mean_us=mean_l,
            score_logits_min_us=lo_l,
        )
    except Exception as e:
        _skip("SafetyLayer", str(e))


def bench_semantic_response_cache(results: dict) -> None:
    _hdr("SemanticResponseCache — Embedding-Similarity LRU Cache")
    try:
        from squish.semantic_response_cache import CacheConfig, SemanticResponseCache

        embed_dim = 128
        cfg   = CacheConfig(capacity=256, similarity_threshold=0.95,
                            embedding_dim=embed_dim)
        cache = SemanticResponseCache(cfg)

        # Pre-populate cache
        for i in range(64):
            emb = RNG.random(embed_dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            cache.store(emb, f"response-{i}")

        # Query with a fresh random vector (miss path)
        query = RNG.random(embed_dim).astype(np.float32)
        query /= np.linalg.norm(query)

        mean_miss, lo_miss, hi_miss = _timeit(
            lambda: cache.lookup(query), n=500
        )
        _row(f"lookup() miss path capacity=64 embed={embed_dim}",
             f"{mean_miss:.2f} µs", f"min={lo_miss:.2f} max={hi_miss:.2f} µs")

        # Store timing
        mean_st, lo_st, hi_st = _timeit(
            lambda: cache.store(query, "bench-response"), n=500
        )
        _row(f"store() embed={embed_dim}",
             f"{mean_st:.2f} µs", f"min={lo_st:.2f} max={hi_st:.2f} µs")

        results["semantic_response_cache"] = dict(
            lookup_miss_mean_us=mean_miss,
            store_mean_us=mean_st,
            cache_size=cache.size,
        )
    except Exception as e:
        _skip("SemanticResponseCache", str(e))


def bench_rate_limiter(results: dict) -> None:
    _hdr("RateLimiter — Token-Bucket Per-Tenant Rate Limiting")
    try:
        from squish.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

        default_cfg = RateLimitConfig(rate=1000.0, burst=1000)
        rl = TokenBucketRateLimiter(default_cfg)

        for i in range(8):
            rl.register_tenant(f"tenant-{i}", RateLimitConfig(rate=100.0, burst=100))

        counter = [0.0]

        def _consume():
            counter[0] += 0.001  # advance time slightly
            return rl.consume("tenant-0", n_tokens=1, now=counter[0])

        mean_c, lo_c, hi_c = _timeit(_consume, n=5000)
        _row("consume() 1 token tenant-0 rate=100/s burst=100",
             f"{mean_c:.3f} µs", f"min={lo_c:.3f} max={hi_c:.3f} µs")

        mean_r, lo_r, hi_r = _timeit(
            lambda: rl.refill("tenant-0", now=100.0), n=5000
        )
        _row("refill() tenant-0",
             f"{mean_r:.3f} µs", f"min={lo_r:.3f} max={hi_r:.3f} µs")

        results["rate_limiter"] = dict(
            consume_mean_us=mean_c,
            consume_min_us=lo_c,
            refill_mean_us=mean_r,
            n_tenants=len(rl.tenants),
        )
    except Exception as e:
        _skip("RateLimiter", str(e))


def bench_schema_validator(results: dict) -> None:
    _hdr("SchemaValidator — JSON Schema Validation")
    try:
        from squish.schema_validator import SchemaValidator

        validator = SchemaValidator()

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name":  {"type": "string", "minLength": 1, "maxLength": 64},
                "age":   {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "minLength": 3},
                "score": {"type": "number",  "minimum": 0.0, "maximum": 1.0},
                "tags":  {"type": "array", "items": {"type": "string"}},
            },
        }

        valid_json   = '{"name":"Alice","age":30,"email":"alice@example.com","score":0.95,"tags":["ai","ml"]}'
        invalid_json = '{"name":"","age":-1,"email":"x"}'

        mean_v, lo_v, hi_v = _timeit(
            lambda: validator.validate(valid_json, schema), n=2000
        )
        _row("validate() valid 5-field object",
             f"{mean_v:.2f} µs", f"min={lo_v:.2f} max={hi_v:.2f} µs")

        mean_i, lo_i, hi_i = _timeit(
            lambda: validator.validate(invalid_json, schema), n=2000
        )
        _row("validate() invalid object (3 errors)",
             f"{mean_i:.2f} µs", f"min={lo_i:.2f} max={hi_i:.2f} µs")

        results["schema_validator"] = dict(
            validate_valid_mean_us=mean_v,
            validate_invalid_mean_us=mean_i,
            validate_valid_min_us=lo_v,
        )
    except Exception as e:
        _skip("SchemaValidator", str(e))


def bench_audit_logger(results: dict) -> None:
    _hdr("AuditLogger — SHA-256 Chained Audit Log")
    try:
        from squish.audit_logger import AuditLogger

        logger = AuditLogger()

        mean_l, lo_l, hi_l = _timeit(
            lambda: logger.log(
                request_id="req-bench",
                tokens_in=128,
                tokens_out=64,
                model="squish-v9",
            ),
            n=2000
        )
        _row("log() SHA-256 chain append",
             f"{mean_l:.2f} µs", f"min={lo_l:.2f} max={hi_l:.2f} µs")

        # Bench verify over current chain
        mean_ver, lo_ver, hi_ver = _timeit(
            lambda: logger.verify(), n=200
        )
        chain_len = logger.chain_length
        _row(f"verify() chain_length={chain_len}",
             f"{mean_ver:.1f} µs", f"min={lo_ver:.1f} max={hi_ver:.1f} µs")

        results["audit_logger"] = dict(
            log_mean_us=mean_l,
            log_min_us=lo_l,
            verify_mean_us=mean_ver,
            chain_length=chain_len,
        )
    except Exception as e:
        _skip("AuditLogger", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    print(f"\n{W}{'═' * 64}{NC}")
    print(f"{B}  Squish — Wave 25+26 Benchmark Summary{NC}")
    print(f"{W}{'═' * 64}{NC}")

    rows_wave25 = [
        ("flash_mla",           "FlashMLA attend() seq=16",             "attend_mean_us"),
        ("native_sparse_attn",  "NativeSparseAttn forward() seq=256",   "forward_mean_us"),
        ("fused_sampler",       "FusedSampler sample() vocab=32k",      "sample_mean_us"),
        ("kv_defrag",           "KVDefrag defrag()",                    "defrag_mean_us"),
        ("dual_chunk_attn",     "DualChunkAttn forward() 4 chunks",     "forward_mean_us"),
        ("activation_offload",  "ActivationOffload offload() 256KB",    "offload_mean_us"),
        ("morph_attn",          "MorphAttn select_pattern()",           "select_pattern_mean_us"),
        ("hydra_spec",          "HydraSpec draft() h=4 x5",             "draft_mean_us"),
        ("seq_compact",         "SeqCompact compact() seq=512 keep50%", "compact_mean_us"),
        ("latency_predictor",   "LatencyPredictor predict()",           "predict_mean_us"),
        ("parallel_sampler",    "ParallelSampler sample() n=8",         "sample_mean_us"),
        ("context_summarizer",  "ContextSummarizer summarize() 1k",     "summarize_importance_mean_us"),
        ("token_watermark",     "TokenWatermark mark() vocab=8k",       "mark_mean_us"),
        ("schema_gen",          "SchemaGen constrain()",                "constrain_mean_us"),
    ]

    rows_wave26 = [
        ("tensor_parallel",          "TensorParallel forward() 256→512",   "forward_mean_us"),
        ("sequence_parallel",        "SequenceParallel scatter() h=8",      "scatter_mean_us"),
        ("kv_migrate",               "KVMigrate pack() seq=128 h=8",        "pack_mean_us"),
        ("disagg_prefill",           "DisaggPrefill prefill() seq=64",      "prefill_64_mean_us"),
        ("request_preempt",          "RequestPreempt preempt+resume swap",  "preempt_swap_resume_mean_us"),
        ("infer_gateway",            "InferGateway route()+complete()",     "route_complete_mean_us"),
        ("model_version_swap",       "ModelVersionSwap route_request()",    "route_request_mean_us"),
        ("production_profiler",      "ProductionProfiler record()",         "record_mean_us"),
        ("adaptive_batcher",         "AdaptiveBatcher next_batch()",        "next_batch_mean_us"),
        ("safety_layer",             "SafetyLayer score() seq=64",          "score_64_mean_us"),
        ("semantic_response_cache",  "SemanticRespCache lookup() miss",     "lookup_miss_mean_us"),
        ("rate_limiter",             "RateLimiter consume() 1 token",       "consume_mean_us"),
        ("schema_validator",         "SchemaValidator validate() 5-field",  "validate_valid_mean_us"),
        ("audit_logger",             "AuditLogger log() SHA-256",           "log_mean_us"),
    ]

    print(f"\n{B}Wave 25 — Cutting-Edge Attention Variants & Compute Fusion{NC}")
    for key, label, field in rows_wave25:
        if key in results and field in results[key]:
            _row(label, f"{results[key][field]:.2f} µs")

    print(f"\n{B}Wave 26 — Distributed Inference & Production Reliability{NC}")
    for key, label, field in rows_wave26:
        if key in results and field in results[key]:
            _row(label, f"{results[key][field]:.2f} µs")


def to_markdown(results: dict) -> str:
    lines = [
        "# Squish — Wave 25+26 Benchmark Results",
        "",
        "> CPU/numpy micro-benchmarks — pure Python, no GPU required.",
        "> Measured on Apple Silicon M-series (or equivalent CPU).",
        "",
        "---",
        "",
        "## Wave 25 — Cutting-Edge Attention Variants & Compute Fusion",
        "",
        "| Module | Operation | Latency (µs) | Notes |",
        "|--------|-----------|:------------:|-------|",
    ]

    if "flash_mla" in results:
        r = results["flash_mla"]
        lines += [
            f"| FlashMLA | `append()` latent_dim=128 single token | {r['append_mean_us']:.2f} | DeepSeek-V2 latent KV insert |",
            f"| FlashMLA | `attend()` seq=16 h=8 d=64 | {r['attend_mean_us']:.2f} | MLA projection + softmax |",
        ]
    if "native_sparse_attn" in results:
        r = results["native_sparse_attn"]
        lines += [f"| NativeSparseAttn | `forward()` h=4 q=16 kv=256 d=32 | {r['forward_mean_us']:.1f} | Block-sparse + window attention |"]
    if "fused_sampler" in results:
        r = results["fused_sampler"]
        lines += [
            f"| FusedSampler | `sample()` vocab=32k with rep_penalty | {r['sample_mean_us']:.2f} | Fused top-k/p/min-p |",
            f"| FusedSampler | `sample_batch()` batch=16 vocab=32k | {r['sample_batch_16_mean_us']:.1f} | Batched sampling |",
        ]
    if "kv_defrag" in results:
        r = results["kv_defrag"]
        lines += [
            f"| KVDefrag | `allocate()+free()` 16-token seq | {r['alloc_free_mean_us']:.2f} | Page-level KV allocation |",
            f"| KVDefrag | `defrag()` with ~50% fragmentation | {r['defrag_mean_us']:.1f} | Online compaction |",
        ]
    if "dual_chunk_attn" in results:
        r = results["dual_chunk_attn"]
        lines += [
            f"| DualChunkAttn | `encode_chunk()` h=4 chunk=64 d=32 | {r['encode_chunk_mean_us']:.2f} | Intra-chunk key compression |",
            f"| DualChunkAttn | `forward()` q=16 past_chunks=4 | {r['forward_mean_us']:.1f} | Intra+inter-chunk attention |",
        ]
    if "activation_offload" in results:
        r = results["activation_offload"]
        lines += [
            f"| ActivationOffload | `offload()` 512×128 tensor | {r['offload_mean_us']:.2f} | CPU copy + flag |",
            f"| ActivationOffload | `fetch()` 512×128 tensor | {r['fetch_mean_us']:.2f} | Returns stored activation |",
        ]
    if "morph_attn" in results:
        r = results["morph_attn"]
        lines += [f"| MorphAttn | `select_pattern()` layer=0 seq=1024 | {r['select_pattern_mean_us']:.3f} | Constant-time threshold lookup |"]
    if "hydra_spec" in results:
        r = results["hydra_spec"]
        lines += [
            f"| HydraSpec | `draft()` hidden=256 h=4 n_draft=5 | {r['draft_mean_us']:.1f} | Multi-head draft generation |",
            f"| HydraSpec | `verify()` h=4 n_draft=5 vocab=8k | {r['verify_mean_us']:.2f} | Best-head accept/reject |",
        ]
    if "seq_compact" in results:
        r = results["seq_compact"]
        lines += [
            f"| SeqCompact | `compact()` h=8 seq=512 keep≈50% | {r['compact_mean_us']:.1f} | Boolean-mask KV compaction |",
            f"| SeqCompact | `compact_indices()` seq=512 | {r['compact_indices_mean_us']:.2f} | Index mapping with -1 fills |",
        ]
    if "latency_predictor" in results:
        r = results["latency_predictor"]
        lines += [f"| LatencyPredictor | `predict()` n_prefill=100 n_decode=10 | {r['predict_mean_us']:.3f} | OLS dot-product prediction |"]
    if "parallel_sampler" in results:
        r = results["parallel_sampler"]
        lines += [
            f"| ParallelSampler | `sample()` vocab=32k n=8 | {r['sample_mean_us']:.2f} | Best-of-N with diversity score |",
            f"| ParallelSampler | `sample_batch()` batch=16 | {r['sample_batch_16_mean_us']:.1f} | Batched best-of-N |",
        ]
    if "context_summarizer" in results:
        r = results["context_summarizer"]
        lines += [
            f"| ContextSummarizer | `summarize()` importance seq=1024 | {r['summarize_importance_mean_us']:.1f} | Embedding-importance ranking |",
            f"| ContextSummarizer | `summarize()` recency seq=1024 | {r['summarize_recency_mean_us']:.1f} | Keep-last-N compaction |",
        ]
    if "token_watermark" in results:
        r = results["token_watermark"]
        lines += [
            f"| TokenWatermark | `mark()` vocab=8k with context | {r['mark_mean_us']:.2f} | Green-list logit boost |",
            f"| TokenWatermark | `detect()` seq=128 | {r['detect_128_mean_us']:.2f} | Z-score detection |",
        ]
    if "schema_gen" in results:
        r = results["schema_gen"]
        lines += [f"| SchemaGen | `constrain()` vocab=256 initial | {r['constrain_mean_us']:.3f} | FSM-based logit masking |"]

    lines += [
        "",
        "## Wave 26 — Distributed Inference & Production Reliability",
        "",
        "| Module | Operation | Latency (µs) | Notes |",
        "|--------|-----------|:------------:|-------|",
    ]

    if "tensor_parallel" in results:
        r = results["tensor_parallel"]
        lines += [
            f"| TensorParallel | `shard()` 256×512 column-mode | {r['shard_mean_us']:.2f} | Column-split sharding |",
            f"| TensorParallel | `forward()` batch=8 in=256 out=512 | {r['forward_mean_us']:.2f} | Shard matmul + all-reduce |",
        ]
    if "sequence_parallel" in results:
        r = results["sequence_parallel"]
        lines += [
            f"| SequenceParallel | `scatter()` h=8 seq=256 d=64 | {r['scatter_mean_us']:.2f} | Ulysses sequence split |",
            f"| SequenceParallel | `gather()` 4 devices → full | {r['gather_mean_us']:.2f} | Reconstruct full sequence |",
        ]
    if "kv_migrate" in results:
        r = results["kv_migrate"]
        lines += [
            f"| KVMigrate | `pack()` seq=128 h=8 d=64 | {r['pack_mean_us']:.1f} | Serialize KV to bytes |",
            f"| KVMigrate | `unpack()` {r['packed_bytes']//1024} KB payload | {r['unpack_mean_us']:.1f} | Deserialize + verify |",
        ]
    if "disagg_prefill" in results:
        r = results["disagg_prefill"]
        lines += [
            f"| DisaggPrefill | `prefill()` seq=64 h=8 layers=4 | {r['prefill_64_mean_us']:.1f} | Prefill node KV build |",
            f"| DisaggPrefill | `step()` decode token | {r['decode_step_mean_us']:.3f} | Greedy decode from payload |",
        ]
    if "request_preempt" in results:
        r = results["request_preempt"]
        lines += [
            f"| RequestPreempt | `preempt(swap)+resume()` | {r['preempt_swap_resume_mean_us']:.2f} | KV swap to CPU + restore |",
            f"| RequestPreempt | `preempt(recompute)+resume()` | {r['preempt_recompute_resume_mean_us']:.2f} | Marker-only + no-op restore |",
        ]
    if "infer_gateway" in results:
        r = results["infer_gateway"]
        lines += [f"| InferGateway | `route()+complete()` 8 workers | {r['route_complete_mean_us']:.3f} | Least-loaded dispatch |"]
    if "model_version_swap" in results:
        r = results["model_version_swap"]
        lines += [f"| ModelVersionSwap | `route_request()` canary 10% | {r['route_request_mean_us']:.3f} | Fraction-based canary split |"]
    if "production_profiler" in results:
        r = results["production_profiler"]
        lines += [
            f"| ProductionProfiler | `record()` single observation | {r['record_mean_us']:.3f} | Ring-buffer insert |",
            f"| ProductionProfiler | `stats()` p50/p99/p999 | {r['stats_mean_us']:.3f} | Sorted-copy percentile |",
        ]
    if "adaptive_batcher" in results:
        r = results["adaptive_batcher"]
        lines += [
            f"| AdaptiveBatcher | `next_batch()` queue_depth=64 | {r['next_batch_mean_us']:.3f} | EMA-model batch selection |",
            f"| AdaptiveBatcher | `record_observation()` EMA update | {r['record_obs_mean_us']:.3f} | Exponential moving average |",
        ]
    if "safety_layer" in results:
        r = results["safety_layer"]
        lines += [
            f"| SafetyLayer | `score()` seq=64 vocab=8k | {r['score_64_mean_us']:.2f} | Token-frequency scoring |",
            f"| SafetyLayer | `score_logits()` 1D vocab=8k | {r['score_logits_mean_us']:.2f} | Next-token safety pre-check |",
        ]
    if "semantic_response_cache" in results:
        r = results["semantic_response_cache"]
        lines += [
            f"| SemanticResponseCache | `lookup()` miss path cap=64 | {r['lookup_miss_mean_us']:.2f} | Cosine scan + miss |",
            f"| SemanticResponseCache | `store()` embed=128 | {r['store_mean_us']:.2f} | LRU insert |",
        ]
    if "rate_limiter" in results:
        r = results["rate_limiter"]
        lines += [f"| RateLimiter | `consume()` 1 token tenant=0 | {r['consume_mean_us']:.3f} | Token-bucket debit |"]
    if "schema_validator" in results:
        r = results["schema_validator"]
        lines += [
            f"| SchemaValidator | `validate()` valid 5-field object | {r['validate_valid_mean_us']:.2f} | Type+constraint checking |",
            f"| SchemaValidator | `validate()` invalid 3-error | {r['validate_invalid_mean_us']:.2f} | Error accumulation |",
        ]
    if "audit_logger" in results:
        r = results["audit_logger"]
        lines += [
            f"| AuditLogger | `log()` SHA-256 chain append | {r['log_mean_us']:.2f} | Hash-chained entry |",
            f"| AuditLogger | `verify()` chain_length={r['chain_length']} | {r['verify_mean_us']:.1f} | Full chain rehash |",
        ]

    lines += ["", "---", "", "_Generated by `bench_wave25_26.py`_"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Squish Wave 25+26 micro-benchmark suite"
    )
    parser.add_argument(
        "--output", "-o",
        default="dev/results/wave25_26_bench.json",
        help="Path for JSON results file (default: dev/results/wave25_26_bench.json)",
    )
    parser.add_argument(
        "--markdown", "-m",
        action="store_true",
        help="Print Markdown results table to stdout",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the JSON output file",
    )
    args = parser.parse_args()

    print(f"{W}{'═' * 64}{NC}")
    print(f"{C}  Squish — Wave 25+26 Micro-Benchmark Suite{NC}")
    print(f"{D}  CPU/numpy · pure Python · no GPU required{NC}")
    print(f"{W}{'═' * 64}{NC}")

    # Add repo root to sys.path so squish package is importable when run from
    # dev/benchmarks/ or from the project root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    results: dict = {}

    # ── Wave 25 ───────────────────────────────────────────────────────────────
    bench_flash_mla(results)
    bench_native_sparse_attn(results)
    bench_fused_sampler(results)
    bench_kv_defrag(results)
    bench_dual_chunk_attn(results)
    bench_activation_offload(results)
    bench_morph_attn(results)
    bench_hydra_spec(results)
    bench_seq_compact(results)
    bench_latency_predictor(results)
    bench_parallel_sampler(results)
    bench_context_summarizer(results)
    bench_token_watermark(results)
    bench_schema_gen(results)

    # ── Wave 26 ───────────────────────────────────────────────────────────────
    bench_tensor_parallel(results)
    bench_sequence_parallel(results)
    bench_kv_migrate(results)
    bench_disagg_prefill(results)
    bench_request_preempt(results)
    bench_infer_gateway(results)
    bench_model_version_swap(results)
    bench_production_profiler(results)
    bench_adaptive_batcher(results)
    bench_safety_layer(results)
    bench_semantic_response_cache(results)
    bench_rate_limiter(results)
    bench_schema_validator(results)
    bench_audit_logger(results)

    _print_summary(results)

    if args.markdown:
        print("\n\n" + to_markdown(results))

    if not args.no_save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n{G}  ✓ Results saved →{NC} {out_path}")

    n_ok   = len(results)
    n_skip = 28 - n_ok
    print(f"\n{B}  {n_ok}/28 modules benchmarked{NC}"
          + (f"  {Y}({n_skip} skipped){NC}" if n_skip else f"  {G}(0 skipped){NC}"))


if __name__ == "__main__":
    main()
