"""tests/test_wave17_server_wiring.py

Verifies that all Wave 17 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 17 modules (Attention Architecture + Memory Management):
  sage_attention2, streaming_sink, kv_slab, squeeze_attention, smallkv,
  specontext, svdq, comm_vq, chunked_prefill, gemfilter, minference_patch,
  prompt_compressor, prompt_lookup, trail
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SageAttention2
# ---------------------------------------------------------------------------

class TestSageAttention2Wiring:
    def test_import(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        cfg    = SageAttention2Config(head_dim=64, n_heads=8, block_size=64)
        kernel = SageAttention2Kernel(cfg)
        assert kernel is not None

    def test_config_defaults(self):
        from squish.sage_attention2 import SageAttention2Config
        cfg = SageAttention2Config()
        assert cfg.head_dim > 0
        assert cfg.n_heads > 0
        assert cfg.warp_size > 0
        assert cfg.block_size >= cfg.warp_size
        assert 0.0 < cfg.smooth_alpha < 1.0
        assert isinstance(cfg.use_int4, bool)
        assert isinstance(cfg.use_fp8_pv, bool)

    def test_kernel_forward_shape(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        rng    = np.random.default_rng(0)
        head_dim, n_heads, seq_len = 64, 4, 32
        cfg    = SageAttention2Config(head_dim=head_dim, n_heads=n_heads,
                                       block_size=32, warp_size=32)
        kernel = SageAttention2Kernel(cfg)
        # simulate_sage2_attention expects (n_heads, seq_len, head_dim)
        q = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        k = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        out, stats = kernel.forward(q, k, v)
        assert out.shape == (n_heads, seq_len, head_dim)
        assert stats is not None

    def test_cumulative_stats_and_reset(self):
        from squish.sage_attention2 import SageAttention2Config, SageAttention2Kernel
        rng    = np.random.default_rng(1)
        cfg    = SageAttention2Config(head_dim=64, n_heads=4, block_size=32, warp_size=32)
        kernel = SageAttention2Kernel(cfg)
        # simulate_sage2_attention expects (n_heads, seq_len, head_dim)
        q = rng.standard_normal((4, 16, 64)).astype(np.float32)
        k = rng.standard_normal((4, 16, 64)).astype(np.float32)
        v = rng.standard_normal((4, 16, 64)).astype(np.float32)
        kernel.forward(q, k, v)
        s = kernel.cumulative_stats
        assert s.total_blocks >= 1
        kernel.reset()
        s2 = kernel.cumulative_stats
        assert s2.total_blocks == 0

    def test_warp_quantize_function(self):
        from squish.sage_attention2 import warp_quantize_int4
        rng = np.random.default_rng(2)
        x   = rng.standard_normal(64).astype(np.float32)
        result = warp_quantize_int4(x, warp_size=32, fallback_threshold=6.5)
        assert result is not None
        assert result.warp_scales is not None
        assert isinstance(result.used_int4, bool)


# ---------------------------------------------------------------------------
# StreamingSink (SinkKVCache)
# ---------------------------------------------------------------------------

class TestStreamingSinkWiring:
    def test_import(self):
        from squish.streaming_sink import SinkConfig, SinkKVCache
        cfg   = SinkConfig(num_sinks=4, window_size=64, head_dim=32)
        cache = SinkKVCache(cfg)
        assert cache is not None

    def test_config_capacity(self):
        from squish.streaming_sink import SinkConfig
        cfg = SinkConfig(num_sinks=4, window_size=64, head_dim=32)
        assert cfg.capacity == cfg.num_sinks + cfg.window_size

    def test_append_and_get_kv(self):
        from squish.streaming_sink import SinkConfig, SinkKVCache
        rng   = np.random.default_rng(0)
        cfg   = SinkConfig(num_sinks=2, window_size=8, head_dim=16)
        cache = SinkKVCache(cfg)
        assert cache.size == 0
        for _ in range(4):
            k = rng.standard_normal(16).astype(np.float32)
            v = rng.standard_normal(16).astype(np.float32)
            cache.append(k, v)
        assert cache.size == 4
        keys, values, positions = cache.get_kv()
        assert keys.shape[0] == values.shape[0]

    def test_eviction_and_stats(self):
        from squish.streaming_sink import SinkConfig, SinkKVCache
        rng   = np.random.default_rng(1)
        cfg   = SinkConfig(num_sinks=2, window_size=4, head_dim=8)
        cache = SinkKVCache(cfg)
        for _ in range(20):
            k = rng.standard_normal(8).astype(np.float32)
            v = rng.standard_normal(8).astype(np.float32)
            cache.append(k, v)
        s = cache.stats
        assert s.total_appended == 20
        assert s.evictions >= 0
        assert cache.is_full

    def test_reset_clears_cache(self):
        from squish.streaming_sink import SinkConfig, SinkKVCache
        rng   = np.random.default_rng(2)
        cfg   = SinkConfig(num_sinks=2, window_size=8, head_dim=8)
        cache = SinkKVCache(cfg)
        for _ in range(5):
            cache.append(rng.standard_normal(8).astype(np.float32),
                         rng.standard_normal(8).astype(np.float32))
        cache.reset()
        assert cache.size == 0


# ---------------------------------------------------------------------------
# KVSlab (KVSlabAllocator)
# ---------------------------------------------------------------------------

class TestKVSlabWiring:
    def test_import(self):
        from squish.kv_slab import KVPage, KVSlabAllocator
        alloc = KVSlabAllocator(n_pages=32, page_size=8, n_layers=4,
                                 n_heads=2, head_dim=16)
        assert alloc is not None

    def test_config_defaults(self):
        from squish.kv_slab import KVSlabAllocator
        alloc = KVSlabAllocator()
        assert alloc.n_free() > 0
        assert alloc.n_used() == 0
        assert alloc.memory_bytes() > 0

    def test_alloc_and_free(self):
        from squish.kv_slab import KVPage, KVSlabAllocator
        alloc = KVSlabAllocator(n_pages=8, page_size=4, n_layers=2,
                                 n_heads=2, head_dim=8)
        initial_free = alloc.n_free()
        page = alloc.alloc()
        assert isinstance(page, KVPage)
        assert alloc.n_free() == initial_free - 1
        assert alloc.n_used() == 1
        alloc.free(page)
        assert alloc.n_free() == initial_free

    def test_page_fill_and_remaining(self):
        from squish.kv_slab import KVSlabAllocator
        alloc = KVSlabAllocator(n_pages=4, page_size=8, n_layers=2,
                                 n_heads=2, head_dim=8)
        page = alloc.alloc()
        assert page.remaining() == page.page_size
        assert not page.is_full()

    def test_free_many_and_stats(self):
        from squish.kv_slab import KVSlabAllocator
        alloc  = KVSlabAllocator(n_pages=16, page_size=4, n_layers=2,
                                  n_heads=2, head_dim=8)
        pages  = [alloc.alloc() for _ in range(4)]
        assert alloc.n_used() == 4
        alloc.free_many(pages)
        assert alloc.n_used() == 0
        st = alloc.stats()
        assert isinstance(st, dict)


# ---------------------------------------------------------------------------
# SqueezeAttention (SqueezeKVCache)
# ---------------------------------------------------------------------------

class TestSqueezeAttentionWiring:
    def test_import(self):
        from squish.squeeze_attention import BudgetAllocator, SqueezeConfig, SqueezeKVCache
        cfg   = SqueezeConfig(n_layers=4, total_kv_budget=1024,
                               min_tokens_per_layer=16, max_tokens_per_layer=512)
        alloc = BudgetAllocator(cfg)
        budgets = alloc.allocate()
        cache   = SqueezeKVCache(budgets, cfg)
        assert cache is not None

    def test_config_defaults(self):
        from squish.squeeze_attention import SqueezeConfig
        cfg = SqueezeConfig()
        assert cfg.n_layers >= 1
        assert cfg.total_kv_budget > 0
        assert cfg.min_tokens_per_layer <= cfg.avg_tokens_per_layer
        assert cfg.avg_tokens_per_layer <= cfg.max_tokens_per_layer

    def test_append_and_get_kv(self):
        from squish.squeeze_attention import BudgetAllocator, SqueezeConfig, SqueezeKVCache
        rng     = np.random.default_rng(0)
        cfg     = SqueezeConfig(n_layers=2, total_kv_budget=128,
                                 min_tokens_per_layer=8, max_tokens_per_layer=64)
        budgets = BudgetAllocator(cfg).allocate()
        cache   = SqueezeKVCache(budgets, cfg)
        k = rng.standard_normal(16).astype(np.float32)
        v = rng.standard_normal(16).astype(np.float32)
        cache.append(0, k, v, attn_score=1.0)
        assert cache.size(0) == 1
        k_out, v_out = cache.get_kv(0)
        assert k_out.shape[0] >= 1

    def test_total_size_and_stats(self):
        from squish.squeeze_attention import BudgetAllocator, SqueezeConfig, SqueezeKVCache
        rng     = np.random.default_rng(1)
        n       = 4
        cfg     = SqueezeConfig(n_layers=n, total_kv_budget=64,
                                 min_tokens_per_layer=4, max_tokens_per_layer=32)
        budgets = BudgetAllocator(cfg).allocate()
        cache   = SqueezeKVCache(budgets, cfg)
        for layer in range(n):
            for _ in range(3):
                cache.append(layer,
                             rng.standard_normal(8).astype(np.float32),
                             rng.standard_normal(8).astype(np.float32))
        assert cache.total_size() >= 1
        s = cache.stats
        assert s.total_appended >= 1


# ---------------------------------------------------------------------------
# SmallKV (SmallKVCache)
# ---------------------------------------------------------------------------

class TestSmallKVWiring:
    def test_import(self):
        from squish.smallkv import SmallKVCache, SmallKVConfig
        cfg   = SmallKVConfig(n_layers=4, kv_budget_fraction=0.2)
        cache = SmallKVCache(cfg)
        assert cache is not None

    def test_config_defaults(self):
        from squish.smallkv import SmallKVConfig
        cfg = SmallKVConfig()
        assert cfg.n_layers >= 1
        assert 0.0 < cfg.kv_budget_fraction <= 1.0
        assert 0.0 < cfg.score_ema_alpha < 1.0
        assert cfg.recall_top_k >= 1

    def test_ingest_and_get_kv(self):
        from squish.smallkv import SmallKVCache, SmallKVConfig
        rng     = np.random.default_rng(0)
        n_tok   = 8
        dim     = 16
        cfg     = SmallKVConfig(n_layers=2, kv_budget_fraction=0.5)
        cache   = SmallKVCache(cfg)
        indices   = np.arange(n_tok, dtype=np.int64)
        keys      = rng.standard_normal((n_tok, dim)).astype(np.float32)
        values    = rng.standard_normal((n_tok, dim)).astype(np.float32)
        scores    = rng.random(n_tok).astype(np.float32)
        cache.ingest(0, indices, keys, values, scores)
        k_out, v_out = cache.get_kv(0, 0)
        # May return None if token was evicted — either is valid
        assert k_out is None or k_out.shape[-1] == dim

    def test_stats_and_reset(self):
        from squish.smallkv import SmallKVCache, SmallKVConfig
        rng   = np.random.default_rng(1)
        cfg   = SmallKVConfig(n_layers=2, kv_budget_fraction=0.5)
        cache = SmallKVCache(cfg)
        for layer in range(2):
            idx    = np.arange(4, dtype=np.int64)
            keys   = rng.standard_normal((4, 8)).astype(np.float32)
            values = rng.standard_normal((4, 8)).astype(np.float32)
            scores = rng.random(4).astype(np.float32)
            cache.ingest(layer, idx, keys, values, scores)
        s = cache.stats
        assert s.total_tokens >= 0
        cache.reset()


# ---------------------------------------------------------------------------
# SpeContext (SpeContextCache)
# ---------------------------------------------------------------------------

class TestSpeContextWiring:
    def test_import(self):
        from squish.specontext import DistilledRetrievalHead, SpeContextCache, SpeContextConfig
        cfg    = SpeContextConfig(retrieval_topk=8, head_dim=16, n_retrieval_heads=2)
        head   = DistilledRetrievalHead(cfg)
        cache  = SpeContextCache(head, cfg)
        assert cache is not None

    def test_config_defaults(self):
        from squish.specontext import SpeContextConfig
        cfg = SpeContextConfig()
        assert cfg.retrieval_topk >= 1
        assert cfg.head_dim > 0
        assert cfg.n_retrieval_heads >= 1

    def test_append_and_retrieve(self):
        from squish.specontext import DistilledRetrievalHead, SpeContextCache, SpeContextConfig
        rng   = np.random.default_rng(0)
        dim   = 16
        t     = 32
        cfg   = SpeContextConfig(retrieval_topk=4, head_dim=dim, n_retrieval_heads=1)
        head  = DistilledRetrievalHead(cfg)
        cache = SpeContextCache(head, cfg)
        for _ in range(t):
            k = rng.standard_normal(dim).astype(np.float32)
            v = rng.standard_normal(dim).astype(np.float32)
            cache.append(k, v)
        assert cache.size == t
        q     = rng.standard_normal(dim).astype(np.float32)
        k_ret, v_ret, pos = cache.retrieve(q)
        assert k_ret.ndim >= 1

    def test_retrieval_head_top_k(self):
        from squish.specontext import DistilledRetrievalHead, SpeContextConfig
        rng  = np.random.default_rng(1)
        dim  = 16
        t    = 20
        cfg  = SpeContextConfig(retrieval_topk=5, head_dim=dim)
        head = DistilledRetrievalHead(cfg)
        q    = rng.standard_normal(dim).astype(np.float32)
        keys = rng.standard_normal((t, dim)).astype(np.float32)
        idx  = head.top_k_indices(q, keys, k=5)
        assert len(idx) == 5
        assert len(set(idx.tolist())) == 5  # unique


# ---------------------------------------------------------------------------
# SVDq (SVDqCalibrator)
# ---------------------------------------------------------------------------

class TestSVDqWiring:
    def test_import(self):
        from squish.svdq import SVDqCalibrator, SVDqConfig
        cfg = SVDqConfig(n_layers=4, n_heads=4, head_dim=32,
                          candidate_bits=(2, 4, 8), target_avg_bits=4.0)
        cal = SVDqCalibrator(cfg)
        assert cal is not None

    def test_config_defaults(self):
        from squish.svdq import SVDqConfig
        cfg = SVDqConfig()
        assert cfg.n_layers >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim > 0
        assert len(cfg.candidate_bits) >= 2
        assert 0.0 < cfg.energy_threshold <= 1.0
        assert cfg.min_rank >= 1

    def test_record_and_search(self):
        from squish.svdq import SVDqCalibrator, SVDqConfig, SVDqPrecisionMap
        rng = np.random.default_rng(0)
        n_layers, n_heads, head_dim = 2, 2, 16
        cfg = SVDqConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
                          candidate_bits=(2, 4, 8), target_avg_bits=4.0)
        cal = SVDqCalibrator(cfg)
        for layer in range(n_layers):
            for head in range(n_heads):
                keys = rng.standard_normal((32, head_dim)).astype(np.float32)
                cal.record_head_keys(layer, head, keys)
        precision_map = cal.search()
        assert isinstance(precision_map, SVDqPrecisionMap)
        assert 1.0 <= precision_map.avg_bits <= 32.0

    def test_head_svd_profile(self):
        from squish.svdq import HeadSVDProfile
        rng = np.random.default_rng(2)
        sv  = np.sort(rng.random(16).astype(np.float32))[::-1] + 0.1
        profile = HeadSVDProfile(layer_idx=0, head_idx=0, singular_values=sv)
        assert profile.total_energy > 0.0
        rank = profile.effective_rank(energy_threshold=0.95)
        assert 1 <= rank <= len(sv)
        assert 0.0 <= profile.compressibility(0.95) <= 1.0


# ---------------------------------------------------------------------------
# CommVQ (CommVQCodebook)
# ---------------------------------------------------------------------------

class TestCommVQWiring:
    def test_import(self):
        from squish.comm_vq import CommVQCodebook
        cb = CommVQCodebook(dim=16, n_codes=8)
        assert cb is not None

    def test_fit_encode_decode_roundtrip(self):
        from squish.comm_vq import CommVQCodebook
        rng  = np.random.default_rng(0)
        dim  = 16
        cb   = CommVQCodebook(dim=dim, n_codes=8)
        data = rng.standard_normal((64, dim)).astype(np.float32)
        cb.fit(data)
        indices = cb.encode(data[:8])
        assert indices.shape == (8,)
        decoded = cb.decode(indices)
        assert decoded.shape == (8, dim)
        err = cb.quantization_error(data[:8])
        assert err >= 0.0

    def test_multi_codebook_vq(self):
        from squish.comm_vq import MultiCodebookVQ
        rng  = np.random.default_rng(1)
        dim  = 16
        mvq  = MultiCodebookVQ(dim=dim, n_subvectors=4, n_codes=8)
        assert not mvq.is_fitted
        data = rng.standard_normal((64, dim)).astype(np.float32)
        mvq.fit(data)
        assert mvq.is_fitted
        indices = mvq.encode(data[:4])
        assert indices.shape == (4, 4)
        decoded = mvq.decode(indices)
        assert decoded.shape == (4, dim)

    def test_fit_kv_codebooks_function(self):
        from squish.comm_vq import CommVQCodebook, fit_kv_codebooks
        rng  = np.random.default_rng(2)
        k_vecs = rng.standard_normal((32, 8)).astype(np.float32)
        v_vecs = rng.standard_normal((32, 8)).astype(np.float32)
        k_cb, v_cb = fit_kv_codebooks(k_vecs, v_vecs, n_codes=4, n_iters=5)
        assert isinstance(k_cb, CommVQCodebook)
        assert isinstance(v_cb, CommVQCodebook)


# ---------------------------------------------------------------------------
# ChunkedPrefill
# ---------------------------------------------------------------------------

class TestChunkedPrefillWiring:
    def test_import(self):
        from squish.chunked_prefill import ChunkedPrefillConfig
        cfg = ChunkedPrefillConfig(chunk_size=256, interleave_decode=True)
        assert cfg is not None

    def test_config_defaults(self):
        from squish.chunked_prefill import ChunkedPrefillConfig
        cfg = ChunkedPrefillConfig()
        assert cfg.chunk_size >= 1
        assert isinstance(cfg.interleave_decode, bool)

    def test_config_custom_values(self):
        from squish.chunked_prefill import ChunkedPrefillConfig
        cfg = ChunkedPrefillConfig(chunk_size=128, interleave_decode=False)
        assert cfg.chunk_size == 128
        assert cfg.interleave_decode is False

    def test_chunk_prefill_function_importable(self):
        from squish.chunked_prefill import chunk_prefill
        assert callable(chunk_prefill)


# ---------------------------------------------------------------------------
# GemFilter (GemSelector)
# ---------------------------------------------------------------------------

class TestGemFilterWiring:
    def test_import(self):
        from squish.gemfilter import GemFilterConfig, GemSelector
        cfg = GemFilterConfig(filter_layer=5, top_k_fraction=0.20,
                               always_keep_first=True, always_keep_last=True)
        sel = GemSelector(cfg)
        assert sel is not None

    def test_config_defaults(self):
        from squish.gemfilter import GemFilterConfig
        cfg = GemFilterConfig()
        assert cfg.filter_layer >= 0
        assert 0.0 < cfg.top_k_fraction <= 1.0
        assert cfg.keep_prefix_tokens >= 0
        assert cfg.keep_suffix_tokens >= 0
        assert cfg.aggregation in ("mean", "max", "sum")

    def test_selector_select(self):
        from squish.gemfilter import GemFilterConfig, GemSelector
        rng    = np.random.default_rng(0)
        cfg    = GemFilterConfig(top_k_fraction=0.25, keep_prefix_tokens=2,
                                  keep_suffix_tokens=2)
        sel    = GemSelector(cfg)
        scores = rng.random(32).astype(np.float32)
        idx    = sel.select(scores, seq_len=32)
        assert len(idx) >= 1
        assert len(idx) <= 32
        ratio  = sel.compression_ratio(n_original=32, n_kept=len(idx))
        assert ratio >= 0.0

    def test_attention_score_buffer(self):
        from squish.gemfilter import AttentionScoreBuffer, GemFilterConfig
        rng    = np.random.default_rng(1)
        cfg    = GemFilterConfig(filter_layer=2)
        buf    = AttentionScoreBuffer(cfg)
        attn   = rng.random((8, 8)).astype(np.float32)
        attn  /= attn.sum(axis=-1, keepdims=True)
        buf.record(layer_idx=2, attn_map=attn)
        scores = buf.get_scores()
        assert scores is not None
        buf.reset()
        assert buf.get_scores() is None


# ---------------------------------------------------------------------------
# MInference (monkey-patch module)
# ---------------------------------------------------------------------------

class TestMInferencePatchWiring:
    def test_import(self):
        from squish.minference_patch import (
            patch_model_minference,
            select_pattern_for_sequence,
            unpatch_model_minference,
        )
        assert callable(patch_model_minference)
        assert callable(unpatch_model_minference)
        assert callable(select_pattern_for_sequence)

    def test_select_pattern_short_sequence(self):
        from squish.minference_patch import select_pattern_for_sequence
        pattern = select_pattern_for_sequence(seq_len=512)
        assert isinstance(pattern, str)

    def test_select_pattern_long_sequence(self):
        from squish.minference_patch import select_pattern_for_sequence
        pattern = select_pattern_for_sequence(seq_len=100_000)
        assert pattern in ("a-shape", "vertical-slash", "streaming",
                           "block-sparse", "dense")

    def test_select_pattern_returns_string(self):
        from squish.minference_patch import select_pattern_for_sequence
        for seq_len in [64, 512, 1024, 8192, 131072, 1_000_000]:
            result = select_pattern_for_sequence(seq_len)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# PromptCompressor
# ---------------------------------------------------------------------------

class TestPromptCompressorWiring:
    def test_import(self):
        from squish.prompt_compressor import compress
        assert callable(compress)

    def test_compress_short_text(self):
        from squish.prompt_compressor import compress
        text      = "The quick brown fox jumped over the lazy dog. " * 20
        compressed = compress(text, ratio=0.5)
        assert isinstance(compressed, str)
        assert len(compressed) > 0

    def test_compress_respects_ratio(self):
        from squish.prompt_compressor import compress
        # Multi-sentence corpus so sentence-level compression can actually drop sentences
        text      = " ".join(f"Sentence {i} contains term{i} as its key word." for i in range(50))
        compressed = compress(text, ratio=0.3)
        # 50 sentences → keep 15 → output must be shorter
        assert len(compressed) < len(text)

    def test_compress_with_question(self):
        from squish.prompt_compressor import compress
        text      = "Context: " + "some details here. " * 50
        question  = "What details are mentioned?"
        compressed = compress(text, ratio=0.5, question=question)
        assert isinstance(compressed, str)


# ---------------------------------------------------------------------------
# PromptLookup (PromptLookupDecoder)
# ---------------------------------------------------------------------------

class TestPromptLookupWiring:
    def test_import(self):
        from squish.prompt_lookup import NGramIndex, PromptLookupConfig, PromptLookupDecoder
        cfg   = PromptLookupConfig(ngram_min=2, ngram_max=4, max_speculative=4)
        rng   = np.random.default_rng(0)
        index = NGramIndex(ngram_min=2, ngram_max=4)
        assert index is not None

        def full_forward(token_ids):
            probs = rng.dirichlet(np.ones(32)).astype(np.float32)
            return probs

        dec = PromptLookupDecoder(full_forward, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.prompt_lookup import PromptLookupConfig
        cfg = PromptLookupConfig()
        assert cfg.ngram_min >= 1
        assert cfg.ngram_max >= cfg.ngram_min
        assert cfg.max_speculative >= 1

    def test_ngram_index_build_and_find(self):
        from squish.prompt_lookup import NGramIndex
        index = NGramIndex(ngram_min=2, ngram_max=3)
        seq   = [1, 2, 3, 4, 5, 1, 2, 6, 7]
        index.build(seq)
        # Query [1, 2] — should find continuations
        results = index.find([1, 2])
        assert isinstance(results, list)

    def test_decoder_generate(self):
        from squish.prompt_lookup import PromptLookupConfig, PromptLookupDecoder, PromptLookupStats
        rng   = np.random.default_rng(0)
        vocab = 32
        cfg   = PromptLookupConfig(ngram_min=2, ngram_max=3, max_speculative=3)

        def full_forward(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        dec    = PromptLookupDecoder(full_forward, cfg)
        tokens, stats = dec.generate([1, 2, 3, 4, 1, 2], max_new_tokens=8)
        assert len(tokens) > 0
        assert isinstance(stats, PromptLookupStats)
        assert 0.0 <= stats.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# TRAIL (TrailPredictor)
# ---------------------------------------------------------------------------

class TestTrailWiring:
    def test_import(self):
        from squish.trail import TrailConfig, TrailLinearProbe, TrailPredictor
        cfg   = TrailConfig(probe_layer=4, hidden_dim=32, max_length=512, n_buckets=4)
        probe = TrailLinearProbe(cfg)
        pred  = TrailPredictor(cfg)
        assert probe is not None
        assert pred is not None

    def test_config_defaults(self):
        from squish.trail import TrailConfig
        cfg = TrailConfig()
        assert cfg.probe_layer >= 0
        assert cfg.hidden_dim > 0
        assert cfg.max_length >= 1
        assert cfg.n_buckets >= 2

    def test_probe_fit_and_predict(self):
        from squish.trail import TrailConfig, TrailLinearProbe
        rng    = np.random.default_rng(0)
        hdim   = 32
        cfg    = TrailConfig(hidden_dim=hdim, max_length=256, n_buckets=4)
        probe  = TrailLinearProbe(cfg)
        assert not probe.is_fitted
        embeddings = rng.standard_normal((64, hdim)).astype(np.float32)
        lengths    = rng.integers(1, 256, size=64).astype(np.int32)
        probe.fit(embeddings, lengths)
        assert probe.is_fitted
        q          = rng.standard_normal(hdim).astype(np.float32)
        prediction = probe.predict(q)
        assert isinstance(prediction, (int, np.integer))
        assert 0 < prediction <= cfg.max_length

    def test_predictor_srpt_priority(self):
        from squish.trail import TrailConfig, TrailPredictor
        rng  = np.random.default_rng(1)
        cfg  = TrailConfig(hidden_dim=32, max_length=256, n_buckets=4)
        pred = TrailPredictor(cfg)
        q    = rng.standard_normal(32).astype(np.float32)
        prio = pred.srpt_priority(q, current_tokens=10)
        assert isinstance(prio, float)

    def test_trail_stats_record(self):
        from squish.trail import TrailStats
        stats = TrailStats()
        stats.record(predicted=100, actual=110,
                     predicted_bucket=2, actual_bucket=2)
        assert stats.prediction_count == 1
        assert stats.mae >= 0.0
        assert 0.0 <= stats.bucket_accuracy <= 1.0
