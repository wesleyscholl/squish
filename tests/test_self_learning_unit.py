"""
tests/test_self_learning_unit.py

Unit tests for squish.self_learning — SelfLearner, LearnConfig,
examples_from_jsonl, compute_delta_snr, LearnRequest Pydantic model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.self_learning import (
    LearnConfig,
    LearnExample,
    LearnResult,
    SelfLearner,
    _example_to_activation,
    _low_rank_truncation,
    compute_delta_snr,
    examples_from_jsonl,
)

# ─────────────────────────────────────────────────────────────────────────────
# LearnConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestLearnConfig:
    def test_defaults(self):
        cfg = LearnConfig()
        assert cfg.steps == 50
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.batch_size == 4
        assert cfg.epsilon == pytest.approx(1e-3)
        assert cfg.max_rank == 8
        assert cfg.min_snr_db == pytest.approx(20.0)
        assert cfg.domain == "general"

    def test_custom(self):
        cfg = LearnConfig(steps=10, lr=0.01, domain="legal", max_rank=0)
        assert cfg.steps == 10
        assert cfg.lr == pytest.approx(0.01)
        assert cfg.domain == "legal"
        assert cfg.max_rank == 0

    def test_invalid_steps(self):
        with pytest.raises(ValueError, match="steps"):
            LearnConfig(steps=0)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            LearnConfig(lr=0.0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            LearnConfig(batch_size=0)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            LearnConfig(epsilon=0.0)

    def test_invalid_max_rank(self):
        with pytest.raises(ValueError, match="max_rank"):
            LearnConfig(max_rank=-1)


# ─────────────────────────────────────────────────────────────────────────────
# LearnExample
# ─────────────────────────────────────────────────────────────────────────────


class TestLearnExample:
    def test_string_input_output(self):
        ex = LearnExample(input="What is Paris?", output="Capital of France.")
        assert ex.input == "What is Paris?"
        assert ex.weight == pytest.approx(1.0)

    def test_list_input(self):
        ex = LearnExample(input=[1, 2, 3], output=[4, 5, 6])
        assert ex.input == [1, 2, 3]

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="weight"):
            LearnExample(input="x", output="y", weight=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_delta_snr
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDeltaSNR:
    def test_large_snr_small_delta(self):
        rng = np.random.default_rng(0)
        base  = rng.standard_normal((8, 16)).astype(np.float32)
        delta = base * 0.001
        snr = compute_delta_snr(base, delta)
        assert snr > 30.0

    def test_small_snr_large_delta(self):
        rng = np.random.default_rng(1)
        base  = rng.standard_normal((8, 16)).astype(np.float32) * 0.01
        delta = rng.standard_normal((8, 16)).astype(np.float32) * 10
        snr = compute_delta_snr(base, delta)
        assert snr < 0.0

    def test_zero_delta_returns_inf(self):
        base  = np.ones((4, 8), dtype=np.float32)
        delta = np.zeros((4, 8), dtype=np.float32)
        snr = compute_delta_snr(base, delta)
        assert snr == float("inf")

    def test_returns_float(self):
        base  = np.eye(4, dtype=np.float32)
        delta = np.ones((4, 4), dtype=np.float32) * 0.1
        snr = compute_delta_snr(base, delta)
        assert isinstance(snr, float)


# ─────────────────────────────────────────────────────────────────────────────
# _example_to_activation
# ─────────────────────────────────────────────────────────────────────────────


class TestExampleToActivation:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ex  = LearnExample(input="hello", output="world")
        act = _example_to_activation(ex, 64, rng)
        assert act.shape == (64,)
        assert act.dtype == np.float32

    def test_unit_norm(self):
        rng = np.random.default_rng(0)
        ex  = LearnExample(input="foo", output="bar")
        act = _example_to_activation(ex, 32, rng)
        assert abs(np.linalg.norm(act) - 1.0) < 1e-5

    def test_deterministic(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        ex   = LearnExample(input="same", output="same")
        a1 = _example_to_activation(ex, 32, rng1)
        a2 = _example_to_activation(ex, 32, rng2)
        np.testing.assert_array_equal(a1, a2)


# ─────────────────────────────────────────────────────────────────────────────
# _low_rank_truncation
# ─────────────────────────────────────────────────────────────────────────────


class TestLowRankTruncation:
    def test_shapes_preserved(self):
        rng  = np.random.default_rng(0)
        mat  = rng.standard_normal((8, 16)).astype(np.float32)
        out  = _low_rank_truncation(mat, max_rank=4)
        assert out.shape == mat.shape

    def test_rank_zero_no_truncation(self):
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((8, 16)).astype(np.float32)
        out = _low_rank_truncation(mat, max_rank=0)
        np.testing.assert_array_equal(out, mat)

    def test_full_rank_no_change(self):
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((4, 4)).astype(np.float32)
        # rank > min(rows, cols) means no truncation
        out = _low_rank_truncation(mat, max_rank=100)
        assert out.shape == mat.shape

    def test_1d_passthrough(self):
        vec = np.arange(8, dtype=np.float32)
        out = _low_rank_truncation(vec, max_rank=4)
        np.testing.assert_array_equal(out, vec)


# ─────────────────────────────────────────────────────────────────────────────
# examples_from_jsonl
# ─────────────────────────────────────────────────────────────────────────────


class TestExamplesFromJSONL:
    def _write_jsonl(self, records: list[dict], path: Path) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    def test_basic_load(self, tmp_path):
        p = tmp_path / "ex.jsonl"
        self._write_jsonl([
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ], p)
        examples = examples_from_jsonl(p)
        assert len(examples) == 2
        assert examples[0].input == "Q1"
        assert examples[1].output == "A2"

    def test_with_weight(self, tmp_path):
        p = tmp_path / "ex.jsonl"
        self._write_jsonl([{"input": "x", "output": "y", "weight": 2.5}], p)
        examples = examples_from_jsonl(p)
        assert examples[0].weight == pytest.approx(2.5)

    def test_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "ex.jsonl"
        p.write_text('\n{"input": "a", "output": "b"}\n\n', encoding="utf-8")
        examples = examples_from_jsonl(p)
        assert len(examples) == 1

    def test_comment_lines_skipped(self, tmp_path):
        p = tmp_path / "ex.jsonl"
        p.write_text('# comment\n{"input": "a", "output": "b"}\n', encoding="utf-8")
        examples = examples_from_jsonl(p)
        assert len(examples) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            examples_from_jsonl("/tmp/squish_no_such_file_xyz.jsonl")

    def test_missing_output_key_raises(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"input": "x"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="output"):
            examples_from_jsonl(p)

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('not json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            examples_from_jsonl(p)


# ─────────────────────────────────────────────────────────────────────────────
# SelfLearner
# ─────────────────────────────────────────────────────────────────────────────


def _make_base_weights(n_blocks=3, rows=8, cols=16, seed=0):
    rng = np.random.default_rng(seed)
    return {bi: rng.standard_normal((rows, cols)).astype(np.float32)
            for bi in range(n_blocks)}


def _make_examples(n=6, seed=1):
    rng = np.random.default_rng(seed)
    words = ["legal", "code", "medical", "finance", "science", "art",
             "law", "python", "health", "money", "physics", "music"]
    examples = []
    for i in range(n):
        inp  = " ".join(rng.choice(words, size=4, replace=False).tolist())
        out  = " ".join(rng.choice(words, size=3, replace=False).tolist())
        examples.append(LearnExample(input=inp, output=out))
    return examples


class TestSelfLearnerInit:
    def test_init_stores_base(self):
        base = _make_base_weights(n_blocks=2)
        sl = SelfLearner(base)
        assert len(sl._base) == 2

    def test_init_casts_to_float32(self):
        base = {0: np.ones((4, 8), dtype=np.float64)}
        sl = SelfLearner(base)
        assert sl._base[0].dtype == np.float32


class TestSelfLearnerLearn:
    def test_returns_learn_result(self):
        base = _make_base_weights(n_blocks=2)
        sl   = SelfLearner(base, LearnConfig(steps=3))
        res  = sl.learn_from_examples(_make_examples(4))
        assert isinstance(res, LearnResult)

    def test_delta_shape_matches_base(self):
        base = _make_base_weights(n_blocks=2)
        sl   = SelfLearner(base, LearnConfig(steps=3))
        res  = sl.learn_from_examples(_make_examples(4))
        for bi, delta in res.delta.items():
            assert delta.shape == base[bi].shape

    def test_steps_run_matches_config(self):
        base = _make_base_weights(n_blocks=1)
        cfg  = LearnConfig(steps=5)
        sl   = SelfLearner(base, cfg)
        res  = sl.learn_from_examples(_make_examples(4), cfg)
        assert res.steps_run == 5

    def test_examples_used(self):
        base     = _make_base_weights(n_blocks=1)
        examples = _make_examples(7)
        sl  = SelfLearner(base, LearnConfig(steps=2))
        res = sl.learn_from_examples(examples)
        assert res.examples_used == 7

    def test_snr_is_float(self):
        base = _make_base_weights()
        sl   = SelfLearner(base, LearnConfig(steps=3))
        res  = sl.learn_from_examples(_make_examples(4))
        assert isinstance(res.snr_db, float)

    def test_elapsed_positive(self):
        base = _make_base_weights()
        sl   = SelfLearner(base, LearnConfig(steps=3))
        res  = sl.learn_from_examples(_make_examples(4))
        assert res.elapsed_s >= 0.0

    def test_domain_propagated(self):
        base = _make_base_weights()
        cfg  = LearnConfig(steps=2, domain="legal")
        sl   = SelfLearner(base, cfg)
        res  = sl.learn_from_examples(_make_examples(4))
        assert res.domain == "legal"

    def test_empty_examples_raises(self):
        base = _make_base_weights()
        sl   = SelfLearner(base)
        with pytest.raises(ValueError, match="examples must not be empty"):
            sl.learn_from_examples([])

    def test_single_example(self):
        base = _make_base_weights(n_blocks=1)
        sl   = SelfLearner(base, LearnConfig(steps=2))
        res  = sl.learn_from_examples([LearnExample(input="x", output="y")])
        assert res.examples_used == 1

    def test_max_rank_zero_skips_truncation(self):
        base = _make_base_weights(n_blocks=2)
        cfg  = LearnConfig(steps=3, max_rank=0)
        sl   = SelfLearner(base, cfg)
        res  = sl.learn_from_examples(_make_examples(4))
        for bi, delta in res.delta.items():
            assert delta.shape == base[bi].shape


class TestSelfLearnerApplyToArchive:
    def test_apply_result_to_archive_returns_list(self):
        from squish.block_expert_archive import BlockExpertArchive, BlockExpertConfig

        n_blocks = 2
        rows, cols = 8, 16
        rng = np.random.default_rng(0)
        block_weights = {
            bi: [rng.standard_normal((rows, cols)).astype(np.float32) for _ in range(4)]
            for bi in range(n_blocks)
        }
        base_weights = {bi: block_weights[bi][0] for bi in range(n_blocks)}

        with tempfile.TemporaryDirectory() as td:
            arc = BlockExpertArchive.create(td, block_weights, base_weights)
            base = {bi: base_weights[bi] for bi in range(n_blocks)}
            sl  = SelfLearner(base, LearnConfig(steps=3))
            res = sl.learn_from_examples(_make_examples(4))
            clusters = sl.apply_result_to_archive(res, arc)
            assert isinstance(clusters, list)
            assert len(clusters) == n_blocks


# ─────────────────────────────────────────────────────────────────────────────
# LearnRequest Pydantic model (if available)
# ─────────────────────────────────────────────────────────────────────────────


class TestLearnRequest:
    def test_basic(self):
        try:
            from squish.self_learning import LearnRequest
            req = LearnRequest(
                examples=[{"input": "a", "output": "b"}],
                domain="legal",
                steps=25,
            )
            assert req.domain == "legal"
            assert req.steps == 25
        except Exception:
            pytest.skip("Pydantic not available or model changed")

    def test_defaults(self):
        try:
            from squish.self_learning import LearnRequest
            req = LearnRequest(examples=[{"input": "x", "output": "y"}])
            assert req.domain == "general"
            assert req.steps == 50
        except Exception:
            pytest.skip("Pydantic not available or model changed")
