"""tests/test_wave13_server_wiring.py

Verifies that all Wave 13 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 13 modules:
  duo_attention, shadow_kv, pq_cache, spe_cache, duo_decoding,
  knapspec, token_merging, token_swift, c2t, clasp
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DuoAttention
# ---------------------------------------------------------------------------

class TestDuoAttentionWiring:
    def test_import(self):
        from squish.duo_attention import DuoAttentionConfig, DuoKVManager
        cfg = DuoAttentionConfig(num_layers=4, num_heads=8, head_dim=64)
        labels = {(0, 0): "retrieval", (0, 1): "streaming"}
        mgr = DuoKVManager(cfg, labels)
        assert mgr is not None

    def test_config_defaults(self):
        from squish.duo_attention import DuoAttentionConfig
        cfg = DuoAttentionConfig()
        assert cfg.num_layers > 0
        assert cfg.num_heads > 0
        assert cfg.head_dim > 0
        assert cfg.sink_tokens >= 0
        assert cfg.local_window >= 0

    def test_head_classifier_import(self):
        from squish.duo_attention import DuoAttentionConfig, HeadClassifier
        cfg = DuoAttentionConfig(num_layers=2, num_heads=4, head_dim=32)
        clf = HeadClassifier(cfg)
        assert clf is not None

    def test_store_and_load_kv(self):
        from squish.duo_attention import DuoAttentionConfig, DuoKVManager
        rng = np.random.default_rng(42)
        cfg = DuoAttentionConfig(num_layers=2, num_heads=2, head_dim=16)
        labels = {(0, 0): "retrieval", (0, 1): "streaming"}
        mgr = DuoKVManager(cfg, labels)
        k = rng.standard_normal(16).astype(np.float32)
        v = rng.standard_normal(16).astype(np.float32)
        mgr.store_kv(layer=0, head=0, pos=0, key=k, value=v)
        keys, vals = mgr.load_kv(layer=0, head=0)
        assert keys.shape[-1] == 16


# ---------------------------------------------------------------------------
# ShadowKV
# ---------------------------------------------------------------------------

class TestShadowKVWiring:
    def test_import(self):
        from squish.shadow_kv import ShadowKVCache, ShadowKVConfig
        cfg   = ShadowKVConfig(svd_rank=8, n_landmarks=16)
        cache = ShadowKVCache(n_layers=2, n_heads=4, head_dim=16, config=cfg)
        assert cache is not None

    def test_config_defaults(self):
        from squish.shadow_kv import ShadowKVConfig
        cfg = ShadowKVConfig()
        assert cfg.svd_rank >= 1
        assert cfg.n_landmarks >= 1
        assert cfg.min_calibration_tokens >= 1

    def test_config_rejects_bad_rank(self):
        from squish.shadow_kv import ShadowKVConfig
        with pytest.raises(ValueError):
            ShadowKVConfig(svd_rank=0)

    def test_store_and_recall(self):
        from squish.shadow_kv import ShadowKVCache, ShadowKVConfig
        rng = np.random.default_rng(0)
        cfg = ShadowKVConfig(svd_rank=4, n_landmarks=8, min_calibration_tokens=4)
        cache = ShadowKVCache(n_layers=1, n_heads=2, head_dim=8, config=cfg)
        k = rng.standard_normal((8, 2, 8)).astype(np.float32)
        v = rng.standard_normal((8, 2, 8)).astype(np.float32)
        cache.store(layer_id=0, keys=k, values=v)
        q = rng.standard_normal((2, 8)).astype(np.float32)
        k_ret, v_ret = cache.recall(layer_id=0, query=q, top_k=4)
        assert k_ret.ndim == 3


# ---------------------------------------------------------------------------
# PQCache
# ---------------------------------------------------------------------------

class TestPQCacheWiring:
    def test_import(self):
        from squish.pq_cache import PQCacheConfig, PQKeyIndex
        cfg = PQCacheConfig(n_subvectors=4, n_codes=8)
        idx = PQKeyIndex(dim=32, config=cfg)
        assert idx is not None

    def test_config_defaults(self):
        from squish.pq_cache import PQCacheConfig
        cfg = PQCacheConfig()
        assert cfg.n_subvectors >= 1
        assert cfg.n_codes >= 2
        assert cfg.train_iters >= 1

    def test_add_and_retrieve(self):
        from squish.pq_cache import PQCacheConfig, PQKeyIndex, PQValueStore, retrieve
        rng = np.random.default_rng(1)
        cfg = PQCacheConfig(n_subvectors=2, n_codes=4, train_iters=5)
        idx = PQKeyIndex(dim=8, config=cfg)
        st  = PQValueStore()
        keys = rng.standard_normal((20, 8)).astype(np.float32)
        vals = rng.standard_normal((20, 8)).astype(np.float32)
        idx.fit(keys)
        for i, (k, v) in enumerate(zip(keys, vals, strict=True)):
            idx.add(k, i)
            st.add(i, v)
        q = rng.standard_normal(8).astype(np.float32)
        top_keys, top_vals = retrieve(q, idx, st, top_k=4)
        assert len(top_keys) > 0


# ---------------------------------------------------------------------------
# SpeCache
# ---------------------------------------------------------------------------

class TestSpeCacheWiring:
    def test_import(self):
        from squish.spe_cache import InMemoryBlockStore, SpeCacheConfig, SpeCachePrefetcher
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=2)
        store = InMemoryBlockStore(block_size=4)
        pf    = SpeCachePrefetcher(cfg, store)
        assert pf is not None

    def test_config_defaults(self):
        from squish.spe_cache import SpeCacheConfig
        cfg = SpeCacheConfig()
        assert cfg.block_size >= 1
        assert cfg.prefetch_budget >= 1
        assert cfg.sink_blocks >= 0

    def test_record_and_plan(self):
        from squish.spe_cache import InMemoryBlockStore, SpeCacheConfig, SpeCachePrefetcher
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=2, sink_blocks=0)
        store = InMemoryBlockStore(block_size=4)
        pf    = SpeCachePrefetcher(cfg, store)
        scores = np.ones(20, dtype=np.float32)
        pf.record_attention(scores)
        plan = pf.predict_next_turn_blocks(total_blocks=10)
        assert isinstance(plan, list)

    def test_predict_returns_list(self):
        from squish.spe_cache import InMemoryBlockStore, SpeCacheConfig, SpeCachePrefetcher
        cfg   = SpeCacheConfig(block_size=4, prefetch_budget=2)
        store = InMemoryBlockStore(block_size=4)
        pf    = SpeCachePrefetcher(cfg, store)
        plan  = pf.predict_next_turn_blocks(total_blocks=8)
        assert isinstance(plan, list)


# ---------------------------------------------------------------------------
# DuoDecoding
# ---------------------------------------------------------------------------

class TestDuoDecodingWiring:
    def test_import(self):
        from squish.duo_decoding import (
            DuoCPUVerifier,
            DuoDecodingConfig,
            DuoDecodingDecoder,
            DuoScheduler,
        )
        cfg   = DuoDecodingConfig(gamma=2, k_max=2)
        rng   = np.random.default_rng(0)
        vocab = 32

        def draft_fn(tokens):
            return rng.standard_normal(vocab).astype(np.float32)

        def target_fn(tokens):
            return rng.standard_normal(vocab).astype(np.float32)

        sched  = DuoScheduler(draft_fn, cfg)
        verif  = DuoCPUVerifier(target_fn, cfg)
        dec    = DuoDecodingDecoder(sched, verif, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.duo_decoding import DuoDecodingConfig
        cfg = DuoDecodingConfig()
        assert cfg.k_max >= 1
        assert cfg.gamma >= 1

    def test_generate_returns_tokens(self):
        from squish.duo_decoding import (
            DuoCPUVerifier,
            DuoDecodingConfig,
            DuoDecodingDecoder,
            DuoScheduler,
        )
        rng   = np.random.default_rng(7)
        vocab = 16
        cfg   = DuoDecodingConfig(gamma=2, k_max=2)

        def draft_fn(tokens):
            return rng.standard_normal(vocab).astype(np.float32)

        def target_fn(tokens):
            return rng.standard_normal(vocab).astype(np.float32)

        sched  = DuoScheduler(draft_fn, cfg)
        verif  = DuoCPUVerifier(target_fn, cfg)
        dec    = DuoDecodingDecoder(sched, verif, cfg)
        tokens, stats = dec.generate(input_ids=[1, 2, 3], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# KnapSpec
# ---------------------------------------------------------------------------

class TestKnapSpecWiring:
    def test_import(self):
        from squish.knapspec import KnapSpecConfig, KnapSpecSelector
        cfg = KnapSpecConfig(num_layers=4, budget_fraction=0.5)
        sel = KnapSpecSelector(cfg)
        assert sel is not None

    def test_config_defaults(self):
        from squish.knapspec import KnapSpecConfig
        cfg = KnapSpecConfig()
        assert cfg.num_layers >= 1
        assert 0.0 < cfg.budget_fraction <= 1.0
        assert cfg.dp_resolution >= 1

    def test_select_returns_tuple(self):
        from squish.knapspec import KnapSpecConfig, KnapSpecSelector
        cfg = KnapSpecConfig(num_layers=4, budget_fraction=0.6, dp_resolution=20)
        sel = KnapSpecSelector(cfg)
        attn_keep, mlp_keep = sel.select(context_len=128)
        assert isinstance(attn_keep, list)
        assert isinstance(mlp_keep, list)

    def test_invalid_budget_raises(self):
        from squish.knapspec import KnapSpecConfig
        with pytest.raises(ValueError):
            KnapSpecConfig(budget_fraction=0.0)


# ---------------------------------------------------------------------------
# TokenMerging
# ---------------------------------------------------------------------------

class TestTokenMergingWiring:
    def test_import(self):
        from squish.token_merging import TokenMergingConfig, TokenMergingState, bipartite_merge
        cfg   = TokenMergingConfig(r=4)
        state = TokenMergingState()
        assert cfg is not None
        assert state is not None

    def test_config_defaults(self):
        from squish.token_merging import TokenMergingConfig
        cfg = TokenMergingConfig()
        assert cfg.r >= 0
        assert cfg.start_layer >= 0

    def test_merge_reduces_length(self):
        from squish.token_merging import TokenMergingConfig, bipartite_merge
        rng    = np.random.default_rng(0)
        tokens = rng.standard_normal((32, 16)).astype(np.float32)
        merged, src_idx, dst_idx = bipartite_merge(tokens, r=4, similarity_threshold=-1.0)
        assert merged.shape[0] <= tokens.shape[0]
        assert merged.shape[1] == tokens.shape[1]

    def test_unmerge_restores_length(self):
        from squish.token_merging import bipartite_merge, unmerge_tokens
        rng    = np.random.default_rng(1)
        tokens = rng.standard_normal((16, 8)).astype(np.float32)
        merged, src_idx, dst_idx = bipartite_merge(tokens, r=2, similarity_threshold=-1.0)
        restored = unmerge_tokens(merged, src_idx, dst_idx, t_original=len(tokens))
        assert restored.shape[0] == tokens.shape[0]


# ---------------------------------------------------------------------------
# TokenSwift
# ---------------------------------------------------------------------------

class TestTokenSwiftWiring:
    def test_import(self):
        from squish.token_swift import TokenSwiftConfig, TokenSwiftDecoder
        rng   = np.random.default_rng(0)
        vocab = 16
        cfg   = TokenSwiftConfig(n_heads=2, window_size=8)

        def model_fn(token_ids):
            return rng.standard_normal(vocab).astype(np.float32)  # (vocab,)

        dec = TokenSwiftDecoder(model_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.token_swift import TokenSwiftConfig
        cfg = TokenSwiftConfig()
        assert cfg.n_heads >= 1
        assert cfg.window_size >= 1

    def test_generate_returns_tokens(self):
        from squish.token_swift import TokenSwiftConfig, TokenSwiftDecoder
        rng   = np.random.default_rng(5)
        vocab = 16
        cfg   = TokenSwiftConfig(n_heads=2, window_size=4)

        def model_fn(token_ids):
            return rng.standard_normal(vocab).astype(np.float32)  # (vocab,)

        dec           = TokenSwiftDecoder(model_fn, cfg)
        tokens, stats = dec.generate(input_ids=[1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# C2T (Confidence-to-Tree)
# ---------------------------------------------------------------------------

class TestC2TWiring:
    def test_import(self):
        from squish.c2t import AdaptiveTreeBuilder, C2TConfig
        cfg     = C2TConfig(tree_depth=3)
        builder = AdaptiveTreeBuilder(cfg)
        assert builder is not None

    def test_config_defaults(self):
        from squish.c2t import C2TConfig
        cfg = C2TConfig()
        assert cfg.tree_depth >= 1
        assert cfg.wide_branches >= 1
        assert cfg.narrow_branches >= 1

    def test_build_returns_tree(self):
        from squish.c2t import AdaptiveTreeBuilder, C2TConfig
        rng    = np.random.default_rng(0)
        vocab  = 32
        hidden = 8
        cfg    = C2TConfig(tree_depth=2, wide_branches=2, narrow_branches=1)
        builder = AdaptiveTreeBuilder(cfg)

        def draft_fn(h):
            logits     = rng.standard_normal(vocab).astype(np.float32)
            next_hidden = rng.standard_normal(hidden).astype(np.float32)
            return logits, next_hidden

        root_h = rng.standard_normal(hidden).astype(np.float32)
        tree   = builder.build(draft_fn=draft_fn, root_hidden=root_h)
        assert tree is not None

    def test_features_compute_shape(self):
        from squish.c2t import C2TFeatures
        rng    = np.random.default_rng(2)
        logits = rng.standard_normal(64).astype(np.float32)
        feats  = C2TFeatures.compute(logits)
        assert feats.shape == (3,)


# ---------------------------------------------------------------------------
# CLaSP
# ---------------------------------------------------------------------------

class TestCLaSPWiring:
    def test_import(self):
        from squish.clasp import CLaSPConfig, CLaSPDecoder
        rng    = np.random.default_rng(0)
        vocab  = 16
        n_lay  = 4
        cfg    = CLaSPConfig(num_layers=n_lay, max_skip_layers=2, draft_gamma=2)

        def model_fn(token_ids, skip_mask=None):
            logits  = rng.standard_normal(vocab).astype(np.float32)
            hiddens = [rng.standard_normal(8).astype(np.float32) for _ in range(n_lay)]
            return logits, hiddens

        dec = CLaSPDecoder(model_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.clasp import CLaSPConfig
        cfg = CLaSPConfig()
        assert cfg.num_layers >= 2
        assert cfg.max_skip_layers >= 0
        assert cfg.max_skip_layers < cfg.num_layers
        assert cfg.draft_gamma >= 1
        assert 0.0 < cfg.similarity_threshold <= 1.0

    def test_config_rejects_bad_skip(self):
        from squish.clasp import CLaSPConfig
        with pytest.raises(ValueError):
            CLaSPConfig(num_layers=4, max_skip_layers=4)  # must be < num_layers

    def test_generate_returns_tokens(self):
        from squish.clasp import CLaSPConfig, CLaSPDecoder
        rng    = np.random.default_rng(3)
        vocab  = 16
        n_lay  = 4
        cfg    = CLaSPConfig(num_layers=n_lay, max_skip_layers=2, draft_gamma=2)

        def model_fn(token_ids, skip_mask=None):
            logits  = rng.standard_normal(vocab).astype(np.float32)
            hiddens = [rng.standard_normal(8).astype(np.float32) for _ in range(n_lay)]
            return logits, hiddens

        dec           = CLaSPDecoder(model_fn, cfg)
        tokens, stats = dec.generate(input_ids=[1, 2, 3], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Integration — all Wave 13 modules are importable
# ---------------------------------------------------------------------------

class TestWave13AllImportable:
    """Smoke test: verify every Wave 13 module can be imported."""

    @pytest.mark.parametrize("module,symbols", [
        ("duo_attention",  ["DuoAttentionConfig", "DuoKVManager"]),
        ("shadow_kv",      ["ShadowKVConfig", "ShadowKVCache"]),
        ("pq_cache",       ["PQCacheConfig", "PQKeyIndex", "PQValueStore"]),
        ("spe_cache",      ["SpeCacheConfig", "SpeCachePrefetcher"]),
        ("duo_decoding",   ["DuoDecodingConfig", "DuoDecodingDecoder"]),
        ("knapspec",       ["KnapSpecConfig", "KnapSpecSelector"]),
        ("token_merging",  ["TokenMergingConfig", "TokenMergingState", "bipartite_merge"]),
        ("token_swift",    ["TokenSwiftConfig", "TokenSwiftDecoder"]),
        ("c2t",            ["C2TConfig", "AdaptiveTreeBuilder"]),
        ("clasp",          ["CLaSPConfig", "CLaSPDecoder"]),
    ])
    def test_module_importable(self, module, symbols):
        import importlib
        mod = importlib.import_module(f"squish.{module}")
        for sym in symbols:
            assert hasattr(mod, sym), f"squish.{module} missing {sym!r}"
