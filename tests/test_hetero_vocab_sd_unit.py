"""tests/test_hetero_vocab_sd_unit.py — unit tests for squish/hetero_vocab_sd.py"""
import numpy as np
import pytest

from squish.hetero_vocab_sd import (
    HeteroVocabConfig,
    VocabMapper,
    HeteroVocabDrafter,
    HeteroVocabStats,
    HeteroVocabDecoder,
)

DRAFT_VOCAB = 12
TARGET_VOCAB = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_draft_fn(agree_tok: int, vocab: int = DRAFT_VOCAB):
    """Draft model always returns max logit on agree_tok (in draft vocab)."""
    def fn(ids):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        logits[agree_tok % vocab] = 10.0
        return logits
    return fn


def _fixed_target_fn(agree_tok: int, vocab: int = TARGET_VOCAB):
    """Target model always returns max logit on agree_tok (in target vocab)."""
    def fn(ids):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        logits[agree_tok % vocab] = 10.0
        return logits
    return fn


def _identity_mapper():
    """Direct-map the first min(DRAFT,TARGET) token IDs 1-to-1."""
    return VocabMapper(
        draft_vocab_size=DRAFT_VOCAB,
        target_vocab_size=TARGET_VOCAB,
    )


def _make_drafter(draft_tok: int = 5, mapper: VocabMapper = None):
    if mapper is None:
        mapper = _identity_mapper()
    cfg = HeteroVocabConfig(
        draft_vocab_size=DRAFT_VOCAB,
        target_vocab_size=TARGET_VOCAB,
    )
    return HeteroVocabDrafter(_fixed_draft_fn(draft_tok), mapper, cfg, rng_seed=0)


def _make_decoder(draft_tok: int = 5, target_tok: int = 5, rng_seed: int = 0):
    mapper = _identity_mapper()
    cfg = HeteroVocabConfig(
        draft_vocab_size=DRAFT_VOCAB,
        target_vocab_size=TARGET_VOCAB,
    )
    drafter = HeteroVocabDrafter(
        _fixed_draft_fn(draft_tok), mapper, cfg, rng_seed=rng_seed
    )
    return HeteroVocabDecoder(
        drafter, _fixed_target_fn(target_tok), cfg, rng_seed=rng_seed + 1
    )


# ---------------------------------------------------------------------------
# HeteroVocabConfig
# ---------------------------------------------------------------------------

class TestHeteroVocabConfig:
    def test_defaults(self):
        cfg = HeteroVocabConfig()
        assert cfg.gamma == 4
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.unmapped_prob == pytest.approx(1e-6)
        assert cfg.draft_vocab_size == 32000
        assert cfg.target_vocab_size == 32000

    def test_custom(self):
        cfg = HeteroVocabConfig(
            gamma=3, draft_vocab_size=10, target_vocab_size=20
        )
        assert cfg.draft_vocab_size == 10
        assert cfg.target_vocab_size == 20

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},                    "gamma"),
        ({"temperature": 0},              "temperature"),
        ({"top_p": 0.0},                  "top_p"),
        ({"top_p": 1.1},                  "top_p"),
        ({"unmapped_prob": 0.0},          "unmapped_prob"),
        ({"unmapped_prob": 1.0},          "unmapped_prob"),
        ({"draft_vocab_size": 1},         "draft_vocab_size"),
        ({"target_vocab_size": 1},        "target_vocab_size"),
    ])
    def test_validation(self, kwargs, match):
        base = {"draft_vocab_size": 8, "target_vocab_size": 16}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            HeteroVocabConfig(**base)


# ---------------------------------------------------------------------------
# VocabMapper
# ---------------------------------------------------------------------------

class TestVocabMapper:
    def test_default_identity_map(self):
        mapper = VocabMapper(
            draft_vocab_size=DRAFT_VOCAB,
            target_vocab_size=TARGET_VOCAB,
        )
        # Map token 0 in draft → token 0 in target
        draft_logits = np.full(DRAFT_VOCAB, -10.0, dtype=np.float32)
        draft_logits[0] = 10.0
        target_probs = mapper.map_logits(draft_logits)
        assert target_probs.shape == (TARGET_VOCAB,)
        assert abs(target_probs.sum() - 1.0) < 1e-5
        assert target_probs[0] > target_probs[1]  # token 0 most probable

    def test_dense_matrix_mode(self):
        # Use identity matrix for equal-vocab sizes
        W = np.eye(TARGET_VOCAB, DRAFT_VOCAB, dtype=np.float32)
        mapper = VocabMapper(DRAFT_VOCAB, TARGET_VOCAB, weight_matrix=W)
        draft_logits = np.zeros(DRAFT_VOCAB, dtype=np.float32)
        draft_logits[3] = 5.0
        target_probs = mapper.map_logits(draft_logits)
        assert target_probs.shape == (TARGET_VOCAB,)
        assert abs(target_probs.sum() - 1.0) < 1e-5

    def test_dense_matrix_wrong_shape(self):
        W = np.zeros((8, 8), dtype=np.float32)  # wrong dims
        with pytest.raises(ValueError):
            VocabMapper(DRAFT_VOCAB, TARGET_VOCAB, weight_matrix=W)

    def test_sparse_map(self):
        token_map = {0: 0, 1: 2, 2: 4}  # draft → target
        mapper = VocabMapper(
            DRAFT_VOCAB, TARGET_VOCAB, token_map=token_map, unmapped_prob=0.01
        )
        draft_logits = np.full(DRAFT_VOCAB, -10.0, dtype=np.float32)
        draft_logits[1] = 10.0  # draft token 1 → target token 2
        target_probs = mapper.map_logits(draft_logits)
        assert target_probs.shape == (TARGET_VOCAB,)
        assert target_probs[2] > target_probs[0]  # target 2 gets the mass

    def test_map_logits_wrong_vocab_size(self):
        mapper = VocabMapper(DRAFT_VOCAB, TARGET_VOCAB)
        with pytest.raises(ValueError):
            mapper.map_logits(np.zeros(DRAFT_VOCAB + 1, dtype=np.float32))

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="draft_vocab_size"):
            VocabMapper(1, TARGET_VOCAB)
        with pytest.raises(ValueError, match="target_vocab_size"):
            VocabMapper(DRAFT_VOCAB, 1)

    def test_sample_target_token_returns_target_space(self):
        mapper = _identity_mapper()
        rng = np.random.default_rng(0)
        draft_logits = np.full(DRAFT_VOCAB, -10.0, dtype=np.float32)
        draft_logits[3] = 10.0
        tok, probs = mapper.sample_target_token(draft_logits, rng)
        assert 0 <= tok < TARGET_VOCAB
        assert probs.shape == (TARGET_VOCAB,)

    def test_mapped_probs_sum_to_one(self):
        mapper = _identity_mapper()
        draft_logits = np.random.randn(DRAFT_VOCAB).astype(np.float32)
        probs = mapper.map_logits(draft_logits)
        assert abs(probs.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# HeteroVocabDrafter
# ---------------------------------------------------------------------------

class TestHeteroVocabDrafter:
    def test_draft_sequence_length(self):
        drafter = _make_drafter()
        tokens, probs = drafter.draft_sequence([0], gamma=3)
        assert len(tokens) == 3
        assert len(probs) == 3

    def test_tokens_in_target_vocab(self):
        drafter = _make_drafter()
        tokens, _ = drafter.draft_sequence([0], gamma=5)
        for tok in tokens:
            assert 0 <= tok < TARGET_VOCAB

    def test_probs_shape_and_sum(self):
        drafter = _make_drafter()
        _, probs = drafter.draft_sequence([0], gamma=4)
        for p in probs:
            assert p.shape == (TARGET_VOCAB,)
            assert abs(p.sum() - 1.0) < 1e-5

    def test_dense_mapper_works(self):
        W = np.eye(TARGET_VOCAB, DRAFT_VOCAB, dtype=np.float32)
        mapper = VocabMapper(DRAFT_VOCAB, TARGET_VOCAB, weight_matrix=W)
        drafter = _make_drafter(mapper=mapper)
        tokens, _ = drafter.draft_sequence([0], gamma=3)
        assert len(tokens) == 3


# ---------------------------------------------------------------------------
# HeteroVocabStats
# ---------------------------------------------------------------------------

class TestHeteroVocabStats:
    def test_defaults(self):
        s = HeteroVocabStats()
        assert s.total_tokens == 0
        assert s.draft_steps == 0

    def test_acceptance_rate_zero(self):
        assert HeteroVocabStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = HeteroVocabStats(accepted_total=6, rejected_total=2)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero(self):
        assert HeteroVocabStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = HeteroVocabStats(accepted_total=10, draft_steps=5)
        assert s.mean_accepted_per_step == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# HeteroVocabDecoder
# ---------------------------------------------------------------------------

class TestHeteroVocabDecoderGenerate:
    def test_generates_token_count(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6

    def test_respects_max_new_tokens(self):
        dec = _make_decoder(rng_seed=42)
        _, stats = dec.generate([0, 1], max_new_tokens=7)
        assert stats.total_tokens == 7

    def test_output_ids_length(self):
        dec = _make_decoder()
        out, stats = dec.generate([0, 1, 2], max_new_tokens=5)
        assert len(out) == 3 + 5

    def test_draft_steps_tracked(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_agreement_high_acceptance(self):
        """Draft always maps token 5→5, target always selects 5."""
        mapper = VocabMapper(
            draft_vocab_size=DRAFT_VOCAB,
            target_vocab_size=TARGET_VOCAB,
            token_map={5: 5},
            unmapped_prob=1e-9,
        )
        cfg = HeteroVocabConfig(
            gamma=4, temperature=0.01,
            draft_vocab_size=DRAFT_VOCAB, target_vocab_size=TARGET_VOCAB,
            unmapped_prob=1e-9,
        )
        drafter = HeteroVocabDrafter(
            _fixed_draft_fn(5), mapper, cfg, rng_seed=0
        )
        dec = HeteroVocabDecoder(
            drafter, _fixed_target_fn(5), cfg, rng_seed=1
        )
        _, stats = dec.generate([0], max_new_tokens=20)
        assert stats.acceptance_rate > 0.0

    def test_disagreement_causes_rejections(self):
        """Draft consistently maps to token 1, target always selects token 2."""
        mapper = VocabMapper(
            draft_vocab_size=DRAFT_VOCAB,
            target_vocab_size=TARGET_VOCAB,
            token_map={1: 1},
            unmapped_prob=1e-9,
        )
        cfg = HeteroVocabConfig(
            gamma=2,
            draft_vocab_size=DRAFT_VOCAB, target_vocab_size=TARGET_VOCAB,
            unmapped_prob=1e-9,
        )
        drafter = HeteroVocabDrafter(
            _fixed_draft_fn(1), mapper, cfg, rng_seed=0
        )
        dec = HeteroVocabDecoder(
            drafter, _fixed_target_fn(2), cfg, rng_seed=1
        )
        _, stats = dec.generate([0], max_new_tokens=10)
        assert stats.rejected_total > 0

    def test_empty_prompt(self):
        dec = _make_decoder()
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4
        assert len(out) == 4

    def test_default_config_in_decoder(self):
        mapper = _identity_mapper()
        cfg = HeteroVocabConfig(
            draft_vocab_size=DRAFT_VOCAB, target_vocab_size=TARGET_VOCAB
        )
        drafter = HeteroVocabDrafter(
            _fixed_draft_fn(3), mapper, cfg, rng_seed=0
        )
        dec = HeteroVocabDecoder(drafter, _fixed_target_fn(3))  # no explicit config
        _, stats = dec.generate([0], max_new_tokens=3)
        assert stats.total_tokens == 3

    def test_dense_matrix_mapper_end_to_end(self):
        W = np.eye(TARGET_VOCAB, DRAFT_VOCAB, dtype=np.float32)
        mapper = VocabMapper(DRAFT_VOCAB, TARGET_VOCAB, weight_matrix=W)
        cfg = HeteroVocabConfig(
            draft_vocab_size=DRAFT_VOCAB, target_vocab_size=TARGET_VOCAB
        )
        drafter = HeteroVocabDrafter(_fixed_draft_fn(4), mapper, cfg, rng_seed=0)
        dec = HeteroVocabDecoder(drafter, _fixed_target_fn(4), cfg, rng_seed=1)
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6
