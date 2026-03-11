"""tests/test_wave16_server_wiring.py

Verifies that all Wave 16 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 16 modules (Heterogeneous Compute + Advanced Spec-Decode):
  dovetail, swiftspec, pipo, mobile_moe, online_sd, lookahead_reasoning,
  sparse_spec, fr_spec, long_spec, forelen, rasd
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dovetail (CPU+GPU heterogeneous spec decode)
# ---------------------------------------------------------------------------

class TestDovetailWiring:
    def test_import(self):
        from squish.dovetail import DovetailConfig, DovetailCPUVerifier
        cfg      = DovetailConfig(gamma=4, temperature=1.0, top_p=1.0)
        rng      = np.random.default_rng(0)
        vocab    = 32

        def target_fn(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        verifier = DovetailCPUVerifier(target_fn, cfg)
        assert verifier is not None

    def test_config_defaults(self):
        from squish.dovetail import DovetailConfig
        cfg = DovetailConfig()
        assert cfg.gamma >= 1
        assert 0.0 < cfg.temperature <= 2.0
        assert 0.0 < cfg.top_p <= 1.0

    def test_cpu_verifier_returns_token_and_probs(self):
        from squish.dovetail import DovetailConfig, DovetailCPUVerifier
        rng   = np.random.default_rng(0)
        vocab = 16
        cfg   = DovetailConfig(gamma=2)

        def target_fn(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        verifier = DovetailCPUVerifier(target_fn, cfg)
        token_id, probs = verifier.verify_one([1, 2, 3])
        assert isinstance(token_id, int)
        assert probs.shape == (vocab,)
        assert abs(probs.sum() - 1.0) < 0.01

    def test_decoder_generate_returns_tokens_and_stats(self):
        from squish.dovetail import (
            DovetailConfig,
            DovetailCPUVerifier,
            DovetailDecoder,
            DovetailDraftRunner,
        )
        rng   = np.random.default_rng(1)
        vocab = 16
        cfg   = DovetailConfig(gamma=2)

        def target_fn(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        def draft_fn(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        verifier = DovetailCPUVerifier(target_fn, cfg)
        runner   = DovetailDraftRunner(draft_fn, cfg, rng_seed=0)
        decoder  = DovetailDecoder(runner, verifier, cfg, rng_seed=0)
        tokens, stats = decoder.generate([1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# SwiftSpec (async disaggregated speculative decode)
# ---------------------------------------------------------------------------

class TestSwiftSpecWiring:
    def test_import(self):
        from squish.swiftspec import SwiftSpecConfig, SwiftSpecDecoder
        cfg = SwiftSpecConfig(gamma=4, max_workers=2)
        rng = np.random.default_rng(0)
        vocab = 32

        def draft_fn(prefix, n):
            return [int(rng.integers(0, vocab)) for _ in range(n)]

        def verify_fn(prefix, draft_tokens):
            accepted = draft_tokens[:2]
            return accepted, None

        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.swiftspec import SwiftSpecConfig
        cfg = SwiftSpecConfig()
        assert cfg.gamma >= 1
        assert cfg.max_workers >= 1

    def test_generate_returns_tokens_and_stats(self):
        from squish.swiftspec import SwiftSpecConfig, SwiftSpecDecoder
        rng   = np.random.default_rng(2)
        vocab = 16
        cfg   = SwiftSpecConfig(gamma=3, max_workers=1)

        def draft_fn(prefix, n):
            return [int(rng.integers(0, vocab)) for _ in range(n)]

        def verify_fn(prefix, draft_tokens):
            return draft_tokens[:1], None

        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        tokens, stats = dec.generate([1, 2, 3], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# PIPO (Pipelined Inference with Prefetch Offloading)
# ---------------------------------------------------------------------------

class TestPIPOWiring:
    def test_import(self):
        from squish.pipo import PIPOConfig, PIPOScheduler
        rng  = np.random.default_rng(0)
        n    = 4
        cfg  = PIPOConfig(n_prefetch_layers=1, bypass_batch_threshold=16, group_size=64)

        def weight_loader(layer_idx):
            W = rng.standard_normal((8, 8)).astype(np.float32)
            b = rng.standard_normal(8).astype(np.float32)
            return W, b

        sched = PIPOScheduler(cfg, weight_loader, n_layers=n)
        assert sched is not None

    def test_config_defaults(self):
        from squish.pipo import PIPOConfig
        cfg = PIPOConfig()
        assert cfg.n_prefetch_layers >= 1
        assert cfg.bypass_batch_threshold >= 1
        assert cfg.dequant_cache_size >= 1
        assert cfg.group_size >= 1

    def test_run_layer_returns_array(self):
        from squish.pipo import PIPOConfig, PIPOScheduler
        rng  = np.random.default_rng(0)
        # group_size=64: in_features must be divisible by group_size
        # weight_int4: uint8 shape (out, in//2); scale: float32 shape (n_groups,)
        group_size = 64
        in_features = 128   # 2 groups of 64
        out_features = 64
        cfg  = PIPOConfig(n_prefetch_layers=1, group_size=group_size)
        n    = 4

        def weight_loader(layer_idx):
            # Pack random INT4 values into uint8 (two nibbles per byte)
            n_groups = in_features // group_size
            w_int4 = rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
            scale  = rng.random(n_groups).astype(np.float32) * 0.1 + 1e-3
            return w_int4, scale

        sched  = PIPOScheduler(cfg, weight_loader, n_layers=n)
        x      = rng.standard_normal((1, in_features)).astype(np.float32)
        output = sched.run_layer(0, x)
        assert output.shape[-1] == out_features


# ---------------------------------------------------------------------------
# MobileMoE (MoBiLE — Balanced Layer-Expert skip)
# ---------------------------------------------------------------------------

class TestMobileMoEWiring:
    def test_import(self):
        from squish.mobile_moe import MoBiLEConfig, MoBiLERouter
        cfg    = MoBiLEConfig(n_experts_total=8, n_experts_active=2,
                               n_experts_min=1, importance_threshold=0.3)
        router = MoBiLERouter(config=cfg)
        assert router is not None

    def test_config_defaults(self):
        from squish.mobile_moe import MoBiLEConfig
        cfg = MoBiLEConfig()
        assert cfg.n_experts_min >= 1
        assert cfg.n_experts_active >= cfg.n_experts_min
        assert cfg.n_experts_total >= cfg.n_experts_active
        assert 0.0 <= cfg.importance_threshold <= 1.0

    def test_route_returns_int(self):
        from squish.mobile_moe import MoBiLEConfig, MoBiLERouter
        rng    = np.random.default_rng(0)
        cfg    = MoBiLEConfig(n_experts_total=8, n_experts_active=2)
        router = MoBiLERouter(config=cfg)
        weights = rng.random(cfg.n_experts_total).astype(np.float32)
        weights /= weights.sum()
        chosen  = router.route(weights)
        assert isinstance(chosen, int)
        assert 0 <= chosen < cfg.n_experts_total

    def test_route_batch_returns_list(self):
        from squish.mobile_moe import MoBiLEConfig, MoBiLERouter
        rng  = np.random.default_rng(1)
        cfg  = MoBiLEConfig(n_experts_total=4, n_experts_active=2)
        router = MoBiLERouter(config=cfg)
        batch  = rng.random((8, cfg.n_experts_total)).astype(np.float32)
        batch /= batch.sum(axis=1, keepdims=True)
        chosen = router.route_batch(batch)
        assert len(chosen) == 8


# ---------------------------------------------------------------------------
# OnlineSD (Continuous draft-head adaptation)
# ---------------------------------------------------------------------------

class TestOnlineSDWiring:
    def test_import(self):
        from squish.online_sd import OnlineDraftUpdater, OnlineSDConfig
        cfg     = OnlineSDConfig(buffer_capacity=128, update_every=32)
        updater = OnlineDraftUpdater(cfg)
        assert updater is not None

    def test_config_defaults(self):
        from squish.online_sd import OnlineSDConfig
        cfg = OnlineSDConfig()
        assert cfg.buffer_capacity >= 1
        assert cfg.update_every >= 1
        assert 0.0 < cfg.learning_rate < 1.0
        assert cfg.lora_rank >= 1

    def test_record_and_should_update(self):
        from squish.online_sd import OnlineDraftUpdater, OnlineSDConfig
        rng     = np.random.default_rng(0)
        # hidden_dim and vocab_size are args of OnlineDraftUpdater, not OnlineSDConfig
        cfg     = OnlineSDConfig(buffer_capacity=16, update_every=4)
        updater = OnlineDraftUpdater(cfg, hidden_dim=8, vocab_size=16)
        for _ in range(3):
            hidden = rng.standard_normal(8).astype(np.float32)
            updater.record(hidden, accepted_token=int(rng.integers(0, 16)))
        # should_update should return False until enough records
        flag = updater.should_update()
        assert isinstance(flag, bool)

    def test_apply_update_preserves_shape(self):
        from squish.online_sd import OnlineDraftUpdater, OnlineSDConfig
        rng     = np.random.default_rng(1)
        hdim, vsize = 8, 16
        cfg     = OnlineSDConfig(buffer_capacity=8, update_every=4)
        updater = OnlineDraftUpdater(cfg, hidden_dim=hdim, vocab_size=vsize)
        # Must record enough samples before apply_update can work
        for _ in range(8):
            hidden = rng.standard_normal(hdim).astype(np.float32)
            updater.record(hidden, accepted_token=int(rng.integers(0, vsize)))
        W       = rng.standard_normal((vsize, hdim)).astype(np.float32)
        W_new   = updater.apply_update(W)
        assert W_new.shape == W.shape


# ---------------------------------------------------------------------------
# LookaheadReasoning
# ---------------------------------------------------------------------------

class TestLookaheadReasoningWiring:
    def test_import(self):
        from squish.lookahead_reasoning import LookaheadConfig, LookaheadReasoningEngine
        cfg = LookaheadConfig(lookahead_k=4, min_acceptance_score=0.7)

        def draft_fn(context):
            return "step token"

        engine = LookaheadReasoningEngine(cfg, draft_fn=draft_fn)
        assert engine is not None

    def test_config_defaults(self):
        from squish.lookahead_reasoning import LookaheadConfig
        cfg = LookaheadConfig()
        assert cfg.lookahead_k >= 1
        assert 0.0 < cfg.min_acceptance_score <= 1.0
        assert cfg.max_step_tokens >= 1
        assert isinstance(cfg.greedy_prefix_accept, bool)

    def test_run_cycle_returns_batch(self):
        from squish.lookahead_reasoning import (
            LookaheadConfig,
            LookaheadReasoningEngine,
            LookaheadStep,
        )
        cfg = LookaheadConfig(lookahead_k=2, max_step_tokens=8)
        call_count = [0]

        # draft_fn must return a LookaheadStep, not a plain string
        def draft_fn(context):
            call_count[0] += 1
            return LookaheadStep(
                text=f"token_{call_count[0]}",
                source="draft",
                confidence=0.8,
                tokens_used=1,
            )

        engine = LookaheadReasoningEngine(cfg, draft_fn=draft_fn)
        batch  = engine.run_cycle("What is 2+2?")
        assert batch is not None

    def test_stats_and_reset(self):
        from squish.lookahead_reasoning import (
            LookaheadConfig,
            LookaheadReasoningEngine,
            LookaheadStep,
        )
        cfg = LookaheadConfig(lookahead_k=2)

        def draft_fn(context):
            return LookaheadStep(
                text="a", source="draft", confidence=0.9, tokens_used=1
            )

        engine = LookaheadReasoningEngine(cfg, draft_fn=draft_fn)
        engine.run_cycle("hello")
        s = engine.stats  # LookaheadStats is a property, not a method
        assert s is not None
        engine.reset()


# ---------------------------------------------------------------------------
# SparseSpec (Dynamic sparse self-speculation)
# ---------------------------------------------------------------------------

class TestSparseSpecWiring:
    def test_import(self):
        from squish.sparse_spec import PillarAttnCache, SparseSpecConfig
        cfg   = SparseSpecConfig(gamma=4, top_k_ratio=0.1)
        cache = PillarAttnCache(capacity=256)
        assert cfg is not None
        assert cache is not None

    def test_config_defaults(self):
        from squish.sparse_spec import SparseSpecConfig
        cfg = SparseSpecConfig()
        assert cfg.gamma >= 1
        assert 0.0 < cfg.top_k_ratio <= 1.0

    def test_pillar_cache_update_and_topk(self):
        from squish.sparse_spec import PillarAttnCache
        rng   = np.random.default_rng(0)
        cache = PillarAttnCache(capacity=32)
        scores = rng.random(32).astype(np.float32)
        cache.update(scores)
        assert cache.n_positions == 32
        k  = 8
        idx = cache.top_k_indices(k)
        assert len(idx) == k
        assert len(set(idx.tolist())) == k  # unique indices

    def test_pillar_cache_reset(self):
        from squish.sparse_spec import PillarAttnCache
        rng   = np.random.default_rng(1)
        cache = PillarAttnCache(capacity=16)
        cache.update(rng.random(16).astype(np.float32))
        cache.reset()
        assert cache.n_positions == 0

    def test_decoder_generate(self):
        from squish.sparse_spec import (
            PillarAttnCache,
            SparseSpecConfig,
            SparseSpecDecoder,
            SparseSpecDrafter,
        )
        rng   = np.random.default_rng(2)
        vocab = 16
        cfg   = SparseSpecConfig(gamma=3, top_k_ratio=0.5)

        # SparseSpecDrafter requires a callable and a PillarAttnCache
        cache  = PillarAttnCache(capacity=64)

        def raw_draft_fn(token_ids):
            chosen = int(rng.integers(0, vocab))
            probs  = rng.dirichlet(np.ones(vocab)).astype(np.float32)
            return chosen, probs

        drafter = SparseSpecDrafter(raw_draft_fn, cache, cfg)

        def target_fn(prefix):
            chosen = int(rng.integers(0, vocab))
            probs  = np.zeros(vocab, dtype=np.float32)
            probs[chosen] = 1.0
            return chosen, probs

        dec = SparseSpecDecoder(drafter, target_fn, cfg)
        tokens, stats = dec.generate([1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# FRSpec (Frequency-ranked vocab compression)
# ---------------------------------------------------------------------------

class TestFRSpecWiring:
    def test_import(self):
        from squish.fr_spec import FRSpecCalibrator, FRSpecConfig
        cfg = FRSpecConfig(vocab_size=1000, top_k_fraction=0.25)
        cal = FRSpecCalibrator(cfg)
        assert cal is not None

    def test_config_defaults(self):
        from squish.fr_spec import FRSpecConfig
        cfg = FRSpecConfig()
        assert cfg.vocab_size > 0
        assert 0.0 < cfg.top_k_fraction <= 1.0
        assert cfg.min_frequent_tokens >= 1

    def test_calibrate_and_build_subset(self):
        from squish.fr_spec import FreqTokenSubset, FRSpecCalibrator, FRSpecConfig
        rng = np.random.default_rng(0)
        # min_frequent_tokens must not exceed vocab_size
        cfg = FRSpecConfig(vocab_size=100, top_k_fraction=0.3, min_frequent_tokens=10)
        cal = FRSpecCalibrator(cfg)
        for _ in range(20):
            tokens = rng.integers(0, 100, size=32).tolist()
            cal.record(tokens)
        subset = cal.build_subset()
        assert isinstance(subset, FreqTokenSubset)
        assert len(subset.indices) >= cfg.min_frequent_tokens

    def test_freq_subset_coverage(self):
        from squish.fr_spec import FRSpecCalibrator, FRSpecConfig
        cfg = FRSpecConfig(vocab_size=64, top_k_fraction=0.5)
        cal = FRSpecCalibrator(cfg)
        tokens = list(range(64)) * 4
        cal.record(tokens)
        subset = cal.build_subset()
        cov = subset.coverage(tokens)
        assert 0.0 <= cov <= 1.0

    def test_frspec_head_compress_expand(self):
        from squish.fr_spec import FRSpecCalibrator, FRSpecConfig, FRSpecHead
        rng   = np.random.default_rng(2)
        vocab = 32
        hdim  = 8
        cfg   = FRSpecConfig(vocab_size=vocab, top_k_fraction=0.5)
        cal   = FRSpecCalibrator(cfg)
        cal.record(list(range(vocab)) * 4)
        subset = cal.build_subset()
        full_W = rng.standard_normal((vocab, hdim)).astype(np.float32)
        head   = FRSpecHead(full_W, subset)
        hidden = rng.standard_normal(hdim).astype(np.float32)
        compressed = head.forward(hidden)
        assert compressed.shape == (len(subset.indices),)
        expanded = head.expand_logits(compressed)
        assert expanded.shape == (vocab,)


# ---------------------------------------------------------------------------
# LongSpec (Long-context speculative decode with shared KV)
# ---------------------------------------------------------------------------

class TestLongSpecWiring:
    def test_import(self):
        from squish.long_spec import LongSpecConfig, LongSpecHead
        cfg  = LongSpecConfig(gamma=4, hidden_size=64, vocab_size=128)
        head = LongSpecHead(vocab_size=128, hidden_size=64, rng_seed=0)
        assert cfg is not None
        assert head is not None

    def test_config_defaults(self):
        from squish.long_spec import LongSpecConfig
        cfg = LongSpecConfig()
        assert cfg.gamma >= 1
        assert cfg.hidden_size >= 1
        assert cfg.vocab_size >= 2
        assert 0.0 < cfg.temperature <= 2.0

    def test_head_forward(self):
        from squish.long_spec import LongSpecConfig, LongSpecHead
        rng    = np.random.default_rng(0)
        vocab  = 64
        hdim   = 16
        head   = LongSpecHead(vocab_size=vocab, hidden_size=hdim, rng_seed=0)
        hidden = rng.standard_normal(hdim).astype(np.float32)
        logits = head.forward(hidden)
        assert logits.shape == (vocab,)

    def test_decoder_generate(self):
        from squish.long_spec import LongSpecConfig, LongSpecDecoder, LongSpecHead
        rng   = np.random.default_rng(1)
        vocab = 32
        hdim  = 8
        cfg   = LongSpecConfig(gamma=3, hidden_size=hdim, vocab_size=vocab)
        head  = LongSpecHead(vocab_size=vocab, hidden_size=hdim, rng_seed=0)

        def target_fn(token_ids):
            return rng.dirichlet(np.ones(vocab)).astype(np.float32)

        def hidden_fn(token_ids):
            return rng.standard_normal(hdim).astype(np.float32)

        dec = LongSpecDecoder(target_fn, hidden_fn, head, cfg, rng_seed=0)
        tokens, stats = dec.generate([1, 2, 3], max_new_tokens=6)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# ForeLen (Entropy-guided output length prediction)
# ---------------------------------------------------------------------------

class TestForelenWiring:
    def test_import(self):
        from squish.forelen import EGTPPredictor, ForelenConfig, PLPPredictor
        cfg  = ForelenConfig(entropy_bins=16, n_length_buckets=8, max_length=1024)
        egtp = EGTPPredictor(cfg)
        plp  = PLPPredictor(initial_prediction=256, config=cfg)
        assert egtp is not None
        assert plp is not None

    def test_config_defaults(self):
        from squish.forelen import ForelenConfig
        cfg = ForelenConfig()
        assert cfg.entropy_bins >= 2
        assert cfg.n_length_buckets >= 2
        assert cfg.max_length >= 1
        assert 0.0 < cfg.plp_decay <= 1.0

    def test_egtp_fit_and_predict(self):
        from squish.forelen import EGTPPredictor, ForelenConfig
        rng  = np.random.default_rng(0)
        cfg  = ForelenConfig(entropy_bins=8, n_length_buckets=4, max_length=256)
        egtp = EGTPPredictor(cfg)
        assert not egtp.is_fitted
        # histograms shape: (n_samples, entropy_bins); output_lengths: (n_samples,)
        hists   = rng.random((32, cfg.entropy_bins)).astype(np.float32)
        hists  /= hists.sum(axis=1, keepdims=True)
        lengths = rng.integers(1, 256, size=32).astype(np.int32)
        egtp.fit(hists, lengths)
        assert egtp.is_fitted
        query       = rng.random(cfg.entropy_bins).astype(np.float32)
        query      /= query.sum()
        prediction  = egtp.predict(query)
        assert isinstance(prediction, (int, np.integer))
        assert 0 < prediction <= cfg.max_length

    def test_plp_update_and_estimate(self):
        from squish.forelen import ForelenConfig, PLPPredictor
        cfg = ForelenConfig(plp_decay=0.9, plp_update_every=4)
        plp = PLPPredictor(initial_prediction=128, config=cfg)
        assert plp.current_estimate == 128
        for step in range(8):
            plp.update(current_len=step * 16, step_entropy=0.5)
        assert plp.n_updates >= 1
        assert plp.current_estimate > 0


# ---------------------------------------------------------------------------
# RASD (Retrieval-Augmented Speculative Decode)
# ---------------------------------------------------------------------------

class TestRASDWiring:
    def test_import(self):
        from squish.rasd import CorpusIndex, DraftTree, RASDBatcher, RASDConfig
        cfg    = RASDConfig(beam_width=4, max_retrieval_candidates=8, min_prefix_len=2)
        corpus = CorpusIndex(min_prefix_len=2)
        tree   = DraftTree(max_depth=6)
        batcher = RASDBatcher(cfg)
        assert corpus is not None
        assert tree is not None
        assert batcher is not None

    def test_config_defaults(self):
        from squish.rasd import RASDConfig
        cfg = RASDConfig()
        assert cfg.beam_width >= 1
        assert cfg.max_retrieval_candidates >= 1
        assert cfg.min_prefix_len >= 1
        assert cfg.max_tree_depth >= 1

    def test_corpus_index_add_and_search(self):
        from squish.rasd import CorpusIndex
        corpus = CorpusIndex(min_prefix_len=2)
        corpus.add_sequence([1, 2, 3, 4, 5])
        corpus.add_sequence([1, 2, 6, 7, 8])
        assert corpus.n_sequences == 2
        results = corpus.search([1, 2], top_k=4)
        assert len(results) >= 1

    def test_draft_tree_add_and_traverse(self):
        from squish.rasd import DraftTree
        tree = DraftTree(max_depth=4)
        assert tree.is_empty()
        tree.add_path([10, 20, 30], probs=[0.8, 0.7, 0.6])
        tree.add_path([10, 20, 40], probs=[0.8, 0.7, 0.4])
        assert not tree.is_empty()
        assert tree.n_nodes() >= 3
        paths = tree.all_paths()
        assert len(paths) >= 2

    def test_batcher_build_and_prune_tree(self):
        from squish.rasd import CorpusIndex, RASDBatcher, RASDConfig
        cfg    = RASDConfig(beam_width=2, min_prefix_len=2, max_tree_depth=4)
        corpus = CorpusIndex(min_prefix_len=2)
        for seq in [[1, 2, 3, 4], [1, 2, 5, 6], [7, 8, 9]]:
            corpus.add_sequence(seq)
        batcher = RASDBatcher(cfg)
        tree    = batcher.build_retrieval_tree([1, 2], corpus)
        assert tree is not None
        pruned  = batcher.prune_tree(tree, beam_width=2)
        assert pruned is not None
