"""
test_wave25_server_wiring.py — Wave 25 server-wiring tests.

4 tests per module × 14 modules = 56 tests.
Each test covers: import, instantiation, core method invocation, and stats/properties.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

RNG = np.random.default_rng(0xDEAD_BEEF)


# ── FlashMLA ──────────────────────────────────────────────────────────────────

def test_flash_mla_import():
    from squish.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=4, head_dim=32, latent_dim=16)
    assert cfg.n_heads == 4
    assert cfg.latent_dim == 16


def test_flash_mla_append_and_seq_len():
    from squish.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=2, head_dim=8, latent_dim=4)
    cache = FlashMLACache(cfg, max_seq_len=32)
    assert cache.seq_len == 0
    x = RNG.random((4,)).astype(np.float32)
    cache.append(x)
    cache.append(x)
    assert cache.seq_len == 2


def test_flash_mla_attend():
    from squish.flash_mla import FlashMLACache, MLAConfig
    n_heads, head_dim, latent_dim = 2, 8, 4
    cfg = MLAConfig(n_heads=n_heads, head_dim=head_dim, latent_dim=latent_dim)
    cache = FlashMLACache(cfg, max_seq_len=16)
    for _ in range(3):
        cache.append(RNG.random((latent_dim,)).astype(np.float32))
    q = RNG.random((n_heads, head_dim)).astype(np.float32)
    W_uk = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)
    W_uv = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)
    out = cache.attend(q, W_uk, W_uv)
    assert out.shape == (n_heads, head_dim)
    assert out.dtype == np.float32


def test_flash_mla_compression_ratio():
    from squish.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=4, head_dim=32, latent_dim=8)
    cache = FlashMLACache(cfg)
    assert cache.compression_ratio == pytest.approx(4 * 32 / 8)


# ── NativeSparseAttn ──────────────────────────────────────────────────────────

def test_native_sparse_attn_import():
    from squish.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=4, head_dim=16, block_size=8, top_k_blocks=2, window_size=16)
    assert cfg.block_size == 8


def test_native_sparse_attn_forward_shape():
    from squish.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8, block_size=4, top_k_blocks=2, window_size=8)
    attn = NativeSparseAttention(cfg)
    seq = 16
    q = RNG.random((2, seq, 8)).astype(np.float32)
    k = RNG.random((2, seq, 8)).astype(np.float32)
    v = RNG.random((2, seq, 8)).astype(np.float32)
    out = attn.forward(q, k, v)
    assert out.shape == (2, seq, 8)
    assert out.dtype == np.float32


def test_native_sparse_attn_sparsity():
    from squish.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8, block_size=4, top_k_blocks=1, window_size=4)
    attn = NativeSparseAttention(cfg)
    assert attn.sparsity == pytest.approx(0.0)
    seq = 32
    q = RNG.random((2, seq, 8)).astype(np.float32)
    k = RNG.random((2, seq, 8)).astype(np.float32)
    v = RNG.random((2, seq, 8)).astype(np.float32)
    attn.forward(q, k, v)
    assert 0.0 <= attn.sparsity <= 1.0


def test_native_sparse_attn_bad_shape():
    from squish.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8)
    attn = NativeSparseAttention(cfg)
    bad = RNG.random((4, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        attn.forward(bad, bad, bad)


# ── FusedSampler ─────────────────────────────────────────────────────────────

def test_fused_sampler_import():
    from squish.fused_sampler import FusedSampler, SamplerConfig
    cfg = SamplerConfig(temperature=0.9, top_k=50, top_p=0.9, seed=42)
    s = FusedSampler(cfg)
    assert s is not None


def test_fused_sampler_sample_returns_valid_token():
    from squish.fused_sampler import FusedSampler, SamplerConfig
    vocab = 100
    cfg = SamplerConfig(temperature=1.0, seed=7)
    s = FusedSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    tok = s.sample(logits)
    assert isinstance(tok, int)
    assert 0 <= tok < vocab


def test_fused_sampler_repetition_penalty():
    from squish.fused_sampler import FusedSampler, SamplerConfig
    vocab = 50
    cfg = SamplerConfig(temperature=1.0, repetition_penalty=1.5, seed=0)
    s = FusedSampler(cfg)
    logits = np.zeros(vocab, dtype=np.float32)
    input_ids = np.array([5, 10, 15], dtype=np.int32)
    tok = s.sample(logits, input_ids)
    assert 0 <= tok < vocab


def test_fused_sampler_batch():
    from squish.fused_sampler import FusedSampler, SamplerConfig
    vocab, batch = 200, 4
    cfg = SamplerConfig(temperature=0.7, top_k=20, seed=1)
    s = FusedSampler(cfg)
    logits = RNG.random((batch, vocab)).astype(np.float32)
    toks = s.sample_batch(logits)
    assert toks.shape == (batch,)
    assert all(0 <= t < vocab for t in toks)


# ── KVDefrag ──────────────────────────────────────────────────────────────────

def test_kv_defrag_import():
    from squish.kv_defrag import KVDefragmenter, DefragStats
    d = KVDefragmenter(page_size=8, n_heads=2, head_dim=4)
    assert d is not None


def test_kv_defrag_allocate_free():
    from squish.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    pages = d.allocate(1, 12)
    assert len(pages) == 3   # ceil(12/4)
    assert d.utilization > 0.0
    d.free(1)
    assert d.utilization == pytest.approx(0.0)


def test_kv_defrag_defrag():
    from squish.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    d.allocate(1, 8)
    d.allocate(2, 4)
    d.free(1)
    stats = d.defrag()
    assert stats.n_pages_before == stats.n_pages_after
    assert 0.0 <= stats.fragmentation_after <= stats.fragmentation_before


def test_kv_defrag_fragmentation_ratio():
    from squish.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    assert d.fragmentation_ratio == pytest.approx(0.0)
    d.allocate(10, 4)
    assert d.utilization > 0.0


# ── DualChunkAttn ─────────────────────────────────────────────────────────────

def test_dual_chunk_attn_import():
    from squish.dual_chunk_attn import DualChunkAttention, DCAConfig
    cfg = DCAConfig(n_heads=2, head_dim=8, chunk_size=16, inter_chunk_top_k=2)
    assert cfg.chunk_size == 16


def test_dual_chunk_attn_encode_chunk():
    from squish.dual_chunk_attn import DualChunkAttention, DCAConfig
    n_heads, head_dim, chunk_size = 2, 8, 16
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=chunk_size)
    attn = DualChunkAttention(cfg)
    k = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    summary = attn.encode_chunk(k, v)
    assert summary.shape == (n_heads, head_dim)
    assert summary.dtype == np.float32


def test_dual_chunk_attn_forward_intra_only():
    from squish.dual_chunk_attn import DualChunkAttention, DCAConfig
    n_heads, head_dim = 2, 8
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=16)
    attn = DualChunkAttention(cfg)
    seq = 8
    q = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    k = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    out = attn.forward(q, k, v)
    assert out.shape == (n_heads, seq, head_dim)


def test_dual_chunk_attn_forward_with_past():
    from squish.dual_chunk_attn import DualChunkAttention, DCAConfig
    n_heads, head_dim, chunk_size = 2, 8, 16
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=chunk_size, inter_chunk_top_k=2)
    attn = DualChunkAttention(cfg)
    k_full = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    v_full = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    summary = attn.encode_chunk(k_full, v_full)
    past = [summary, summary]
    seq = 8
    q = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    k = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    out = attn.forward(q, k, v, past_chunks=past)
    assert out.shape == (n_heads, seq, head_dim)


# ── ActivationOffload ─────────────────────────────────────────────────────────

def test_activation_offload_import():
    from squish.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[0, 2, 4])
    loader = ActivationOffloader(policy)
    assert loader is not None


def test_activation_offload_offload_and_fetch():
    from squish.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[1, 3])
    loader = ActivationOffloader(policy)
    tensor = RNG.random((16, 32)).astype(np.float32)
    loader.offload(1, tensor)
    fetched = loader.fetch(1)
    np.testing.assert_array_equal(fetched, tensor)
    assert loader.stats.n_offloaded == 1
    assert loader.stats.n_fetched == 1


def test_activation_offload_evict():
    from squish.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[0])
    loader = ActivationOffloader(policy)
    t = RNG.random((8,)).astype(np.float32)
    loader.offload(0, t)
    loader.evict(0)
    assert loader.buffer_bytes == 0
    with pytest.raises(KeyError):
        loader.fetch(0)


def test_activation_offload_should_offload():
    from squish.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[2, 5])
    loader = ActivationOffloader(policy)
    assert loader.should_offload(2) is True
    assert loader.should_offload(5) is True
    assert loader.should_offload(3) is False


# ── MorphAttn ─────────────────────────────────────────────────────────────────

def test_morph_attn_import():
    from squish.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=12, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    assert m is not None


def test_morph_attn_select_pattern_full():
    from squish.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    for layer in range(8):
        assert m.select_pattern(layer, 256) == "full"


def test_morph_attn_layer_patterns():
    from squish.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    patterns = m.layer_patterns(1000)
    assert len(patterns) == 8
    assert all(p in {"full", "sparse", "linear"} for p in patterns)


def test_morph_attn_flops_reduction():
    from squish.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    r_short = m.estimate_flops_reduction(256)
    r_long = m.estimate_flops_reduction(8192)
    assert 0.0 <= r_short <= 1.0
    assert 0.0 <= r_long <= 1.0
    assert r_long >= r_short   # longer context → more savings


# ── HydraSpec ─────────────────────────────────────────────────────────────────

def test_hydra_spec_import():
    from squish.hydra_spec import HydraSpecDecoder, HydraConfig
    cfg = HydraConfig(n_heads=3, n_draft=4, hidden_dim=32, vocab_size=100)
    decoder = HydraSpecDecoder(cfg)
    assert decoder is not None


def test_hydra_spec_draft_shape():
    from squish.hydra_spec import HydraSpecDecoder, HydraConfig
    n_heads, n_draft, hidden_dim, vocab = 3, 4, 32, 100
    cfg = HydraConfig(n_heads=n_heads, n_draft=n_draft, hidden_dim=hidden_dim, vocab_size=vocab)
    decoder = HydraSpecDecoder(cfg)
    hidden = RNG.random((hidden_dim,)).astype(np.float32)
    out = decoder.draft(hidden)
    assert out.draft_tokens.shape == (n_heads, n_draft)
    assert out.draft_logits.shape == (n_heads, n_draft, vocab)
    assert out.draft_tokens.dtype == np.int32


def test_hydra_spec_verify():
    from squish.hydra_spec import HydraSpecDecoder, HydraConfig
    n_heads, n_draft, hidden_dim, vocab = 2, 3, 16, 50
    cfg = HydraConfig(n_heads=n_heads, n_draft=n_draft, hidden_dim=hidden_dim, vocab_size=vocab)
    decoder = HydraSpecDecoder(cfg)
    hidden = RNG.random((hidden_dim,)).astype(np.float32)
    out = decoder.draft(hidden)
    target_logits = RNG.random((n_heads, n_draft, vocab)).astype(np.float32)
    accepted = decoder.verify(out.draft_tokens, target_logits)
    assert accepted.ndim == 1
    assert accepted.dtype == np.int32
    assert len(accepted) <= n_draft


def test_hydra_spec_acceptance_rate():
    from squish.hydra_spec import HydraSpecDecoder, HydraConfig
    cfg = HydraConfig(n_heads=2, n_draft=3, hidden_dim=16, vocab_size=50)
    decoder = HydraSpecDecoder(cfg)
    history = np.array([True, False, True, True, False], dtype=bool)
    rate = decoder.acceptance_rate(history)
    assert rate == pytest.approx(0.6)


# ── SeqCompact ────────────────────────────────────────────────────────────────

def test_seq_compact_import():
    from squish.seq_compact import SequenceCompactor, CompactStats
    sc = SequenceCompactor(n_heads=2, head_dim=8)
    assert sc is not None


def test_seq_compact_compact():
    from squish.seq_compact import SequenceCompactor
    n_heads, seq, head_dim = 2, 8, 4
    sc = SequenceCompactor(n_heads=n_heads, head_dim=head_dim)
    keys = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    vals = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    keep_mask = np.array([True, False, True, True, False, True, False, True], dtype=bool)
    ck, cv, stats = sc.compact(keys, vals, keep_mask)
    n_kept = int(keep_mask.sum())
    assert ck.shape == (n_heads, n_kept, head_dim)
    assert cv.shape == (n_heads, n_kept, head_dim)
    assert stats.n_tokens_after == n_kept
    assert stats.n_tokens_before == seq


def test_seq_compact_compaction_ratio():
    from squish.seq_compact import SequenceCompactor
    sc = SequenceCompactor(n_heads=2, head_dim=4)
    keys = RNG.random((2, 10, 4)).astype(np.float32)
    vals = RNG.random((2, 10, 4)).astype(np.float32)
    keep_mask = np.ones(10, dtype=bool)
    keep_mask[5:] = False  # keep 5
    _, _, stats = sc.compact(keys, vals, keep_mask)
    assert stats.compaction_ratio == pytest.approx(0.5)


def test_seq_compact_compact_indices():
    from squish.seq_compact import SequenceCompactor
    sc = SequenceCompactor(n_heads=2, head_dim=4)
    keep = np.array([0, 2, 4], dtype=np.int64)
    mapping = sc.compact_indices(6, keep)
    assert mapping.shape == (6,)
    assert mapping[0] == 0
    assert mapping[2] == 1
    assert mapping[4] == 2
    assert mapping[1] == -1
    assert mapping[3] == -1
    assert mapping[5] == -1


# ── LatencyPredictor ──────────────────────────────────────────────────────────

def test_latency_predictor_import():
    from squish.latency_predictor import LatencyPredictor, LatencyModel
    lp = LatencyPredictor(n_heads=4, head_dim=32)
    assert lp.n_samples == 0


def test_latency_predictor_record_and_fit():
    from squish.latency_predictor import LatencyPredictor
    lp = LatencyPredictor(n_heads=4, head_dim=32)
    for n_p in range(10, 60, 10):
        lp.record(n_p, 20, n_p * 0.05 + 5.0)
    model = lp.fit()
    assert lp.n_samples == 5
    assert model.base_latency != 0.0 or model.prefill_coeff != 0.0


def test_latency_predictor_predict():
    from squish.latency_predictor import LatencyPredictor
    lp = LatencyPredictor(n_heads=2, head_dim=16)
    for i in range(5):
        lp.record(i * 10 + 10, 5, i * 0.5 + 2.0)
    lp.fit()
    pred = lp.predict(50, 10)
    assert pred.total_ms >= 0.0
    assert 0.0 <= pred.confidence <= 1.0


def test_latency_predictor_low_confidence():
    from squish.latency_predictor import LatencyPredictor
    lp = LatencyPredictor(n_heads=2, head_dim=16)
    # fewer than 3 samples → confidence 0.0
    lp.record(10, 5, 2.0)
    lp.record(20, 5, 3.0)
    pred = lp.predict(15, 5)
    assert pred.confidence == pytest.approx(0.0)


# ── ParallelSampler ───────────────────────────────────────────────────────────

def test_parallel_sampler_import():
    from squish.parallel_sampler import ParallelSampler, DiversityConfig
    cfg = DiversityConfig(n_samples=4, temperature=0.8, seed=0)
    ps = ParallelSampler(cfg)
    assert ps is not None


def test_parallel_sampler_sample_result():
    from squish.parallel_sampler import ParallelSampler, DiversityConfig, SampleResult
    vocab = 100
    cfg = DiversityConfig(n_samples=8, temperature=1.0, seed=42)
    ps = ParallelSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    result = ps.sample(logits)
    assert isinstance(result, SampleResult)
    assert 0 <= result.best_token < vocab
    assert result.all_tokens.shape == (8,)
    assert result.all_probs.shape == (8,)
    assert 0.0 <= result.diversity_score <= 1.0


def test_parallel_sampler_batch():
    from squish.parallel_sampler import ParallelSampler, DiversityConfig
    vocab, batch = 50, 4
    cfg = DiversityConfig(n_samples=4, seed=1)
    ps = ParallelSampler(cfg)
    logits = RNG.random((batch, vocab)).astype(np.float32)
    toks = ps.sample_batch(logits)
    assert toks.shape == (batch,)
    assert all(0 <= t < vocab for t in toks)


def test_parallel_sampler_best_in_candidates():
    from squish.parallel_sampler import ParallelSampler, DiversityConfig
    vocab = 100
    cfg = DiversityConfig(n_samples=16, seed=3)
    ps = ParallelSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    result = ps.sample(logits)
    # best_token is always one of the candidates
    assert result.best_token in result.all_tokens


# ── ContextSummarizer ─────────────────────────────────────────────────────────

def test_context_summarizer_import():
    from squish.context_summarizer import ContextSummarizer, SummaryConfig
    cfg = SummaryConfig(method="importance", budget=64)
    cs = ContextSummarizer(cfg)
    assert cs is not None


def test_context_summarizer_needs_compression():
    from squish.context_summarizer import ContextSummarizer, SummaryConfig
    cfg = SummaryConfig(budget=128)
    cs = ContextSummarizer(cfg)
    assert cs.needs_compression(200) is True
    assert cs.needs_compression(64) is False


def test_context_summarizer_importance():
    from squish.context_summarizer import ContextSummarizer, SummaryConfig
    seq, dim, budget = 200, 32, 64
    cfg = SummaryConfig(method="importance", budget=budget, min_keep_recent=16)
    cs = ContextSummarizer(cfg)
    tokens = np.arange(seq, dtype=np.int32)
    embs = RNG.random((seq, dim)).astype(np.float32)
    compressed, stats = cs.summarize(tokens, embs)
    assert len(compressed) <= budget
    assert stats.n_tokens_in == seq
    assert stats.compression_ratio <= 1.0


def test_context_summarizer_recency():
    from squish.context_summarizer import ContextSummarizer, SummaryConfig
    seq, budget = 300, 100
    cfg = SummaryConfig(method="recency", budget=budget)
    cs = ContextSummarizer(cfg)
    tokens = np.arange(seq, dtype=np.int32)
    compressed, stats = cs.summarize(tokens)
    assert len(compressed) == budget
    # recency keeps the most recent tokens
    np.testing.assert_array_equal(compressed, tokens[seq - budget:])


# ── TokenWatermark ────────────────────────────────────────────────────────────

def test_token_watermark_import():
    from squish.token_watermark import TokenWatermarker, WatermarkConfig
    cfg = WatermarkConfig(vocab_size=1000, delta=2.0, seed=42)
    wm = TokenWatermarker(cfg)
    assert wm is not None


def test_token_watermark_mark():
    from squish.token_watermark import TokenWatermarker, WatermarkConfig
    vocab = 500
    cfg = WatermarkConfig(vocab_size=vocab, delta=3.0, seed=7)
    wm = TokenWatermarker(cfg)
    logits = np.zeros(vocab, dtype=np.float32)
    marked = wm.mark(logits)
    assert marked.shape == (vocab,)
    assert marked.dtype == np.float32
    assert marked.max() == pytest.approx(3.0)


def test_token_watermark_green_list():
    from squish.token_watermark import TokenWatermarker, WatermarkConfig
    vocab = 200
    cfg = WatermarkConfig(vocab_size=vocab, green_list_fraction=0.5, seed=0)
    wm = TokenWatermarker(cfg)
    gl = wm.green_list()
    assert gl.shape == (vocab,)
    assert gl.dtype == bool
    # Approximately half should be green
    n_green = gl.sum()
    assert abs(n_green - vocab // 2) <= 2


def test_token_watermark_detect():
    from squish.token_watermark import TokenWatermarker, WatermarkConfig
    vocab = 1000
    cfg = WatermarkConfig(vocab_size=vocab, delta=5.0, green_list_fraction=0.5, seed=1)
    wm = TokenWatermarker(cfg)
    # No tokens → safe
    result = wm.detect(np.zeros(0, dtype=np.int32))
    assert result.z_score == pytest.approx(0.0)
    # Some tokens
    tokens = np.arange(10, dtype=np.int32)
    result2 = wm.detect(tokens)
    assert isinstance(result2.is_watermarked, bool)
    assert 0 <= result2.n_green_tokens <= 10


# ── SchemaGen ─────────────────────────────────────────────────────────────────

def test_schema_gen_import():
    from squish.schema_gen import SchemaGenEngine, SchemaState
    engine = SchemaGenEngine(vocab_size=50)
    assert engine is not None


def test_schema_gen_reset():
    from squish.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    assert not state.is_complete
    assert len(state.stack) > 0


def test_schema_gen_constrain():
    from squish.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    logits = np.zeros(50, dtype=np.float32)
    constrained = engine.constrain(logits, state)
    assert constrained.shape == (50,)
    # Some positions should be -inf (invalid tokens masked)
    assert (constrained == -np.inf).any()


def test_schema_gen_valid_next_chars():
    from squish.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    chars = engine.valid_next_chars(state)
    assert isinstance(chars, list)
    assert len(chars) > 0
