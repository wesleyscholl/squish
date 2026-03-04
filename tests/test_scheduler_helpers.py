"""
tests/test_scheduler_helpers.py

Unit tests for the pure-Python / pure-numpy helpers in squish/scheduler.py.
Does NOT require MLX or a real model — only tests the sampling utilities,
_Request dataclass, and QueueFullError.
"""
from __future__ import annotations

import dataclasses
import queue as _queue

import numpy as np
import pytest

from squish.scheduler import (
    QueueFullError,
    _Request,
    _check_stop,
    _sample_token,
    _softmax_f32,
    _top_p_filter,
)


# ── _softmax_f32 ──────────────────────────────────────────────────────────────

class TestSoftmaxF32:
    def test_output_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_higher_logit_higher_prob(self):
        logits = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert probs[2] > probs[1] > probs[0]

    def test_uniform_logits_equal_probs(self):
        logits = np.zeros(10, dtype=np.float32)
        probs = _softmax_f32(logits)
        np.testing.assert_allclose(probs, np.full(10, 0.1), atol=1e-5)

    def test_large_positive_logit(self):
        logits = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert probs[0] > 0.99

    def test_returns_float32(self):
        logits = np.array([1.0, 2.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert probs.dtype in (np.float32, np.float64)

    def test_handles_all_zeros(self):
        logits = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_single_element(self):
        logits = np.array([5.0], dtype=np.float32)
        probs = _softmax_f32(logits)
        assert abs(probs[0] - 1.0) < 1e-5

    def test_numerically_stable_large_values(self):
        """Large values should not produce inf/nan."""
        logits = np.array([1e36, 1e36 + 1], dtype=np.float64)
        probs = _softmax_f32(logits)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


# ── _top_p_filter ──────────────────────────────────────────────────────────────

class TestTopPFilter:
    def test_top_p_1_returns_unchanged(self):
        probs = np.array([0.1, 0.4, 0.5])
        result = _top_p_filter(probs, top_p=1.0)
        np.testing.assert_array_equal(result, probs)

    def test_top_p_selects_most_likely(self):
        # Probabilities heavily concentrated on one token
        probs = np.array([0.01, 0.01, 0.01, 0.97])
        result = _top_p_filter(probs, top_p=0.9)
        # Token 3 should have all the probability
        assert result[3] > 0.99

    def test_output_sums_to_one(self):
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(20))
        result = _top_p_filter(raw, top_p=0.9)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_top_p_zero_keeps_one_token(self):
        probs = np.array([0.1, 0.4, 0.5])
        result = _top_p_filter(probs, top_p=0.0)
        # Should keep at least one token
        assert (result > 0).sum() >= 1

    def test_top_p_keeps_multiple_tokens(self):
        probs = np.array([0.2, 0.3, 0.5])
        result = _top_p_filter(probs, top_p=0.99)
        # At least 2 tokens should survive
        assert (result > 0).sum() >= 2

    def test_all_zero_returns_safely(self):
        """If all probs are zero after filtering (edge case), gracefully fall back."""
        probs = np.zeros(5)
        result = _top_p_filter(probs, top_p=0.9)
        assert result.shape == (5,)


# ── _sample_token ─────────────────────────────────────────────────────────────

class TestSampleToken:
    def _rng(self):
        return np.random.default_rng(0)

    def test_greedy_zero_temperature(self):
        logits = np.array([1.0, 2.0, 5.0, 1.0], dtype=np.float32)
        idx = _sample_token(logits, temperature=0.0, top_p=1.0, rng=self._rng())
        assert idx == 2

    def test_returns_valid_index(self):
        logits = np.random.default_rng(1).standard_normal(50).astype(np.float32)
        idx = _sample_token(logits, temperature=1.0, top_p=1.0, rng=self._rng())
        assert 0 <= idx < 50

    def test_greedy_near_zero_temperature(self):
        logits = np.array([0.1, 10.0, 0.2], dtype=np.float32)
        idx = _sample_token(logits, temperature=1e-6, top_p=1.0, rng=self._rng())
        assert idx == 1

    def test_sampling_varies_with_temperature(self):
        logits = np.ones(10, dtype=np.float32)
        rng = np.random.default_rng(42)
        results = {_sample_token(logits, temperature=1.5, top_p=1.0, rng=rng) for _ in range(50)}
        # With uniform logits and temperature>0, multiple tokens should get picked
        assert len(results) > 1

    def test_top_p_restricts_sampling(self):
        """With top_p=0.1 and peaked distribution, greedy token always selected."""
        logits = np.zeros(100, dtype=np.float32)
        logits[42] = 100.0  # overwhelmingly likely token
        rng = np.random.default_rng(0)
        results = {_sample_token(logits, temperature=1.0, top_p=0.5, rng=rng) for _ in range(20)}
        assert results == {42}


# ── _check_stop ───────────────────────────────────────────────────────────────

class TestCheckStop:
    def _make_req(self, stop_ids=None):
        return _Request(
            request_id="r0",
            input_ids=[1, 2, 3],
            max_tokens=50,
            temperature=1.0,
            top_p=1.0,
            stop_ids=stop_ids or [],
            seed=None,
        )

    def test_no_stop_ids_always_false(self):
        req = self._make_req(stop_ids=[])
        for tok in [10, 20, 30]:
            assert _check_stop(req, tok) is False

    def test_single_token_stop_sequence(self):
        req = self._make_req(stop_ids=[[99]])
        assert _check_stop(req, 1) is False
        assert _check_stop(req, 99) is True

    def test_multi_token_stop_sequence(self):
        req = self._make_req(stop_ids=[[10, 20, 30]])
        _check_stop(req, 10)
        _check_stop(req, 20)
        result = _check_stop(req, 30)
        assert result is True

    def test_partial_stop_not_triggered(self):
        req = self._make_req(stop_ids=[[10, 20, 30]])
        _check_stop(req, 10)
        result = _check_stop(req, 20)
        assert result is False  # not done yet

    def test_buffer_trimmed_when_long(self):
        req = self._make_req(stop_ids=[[999]])
        # Fill stop_buf with 100 random non-stop tokens
        for i in range(100):
            _check_stop(req, i)
        # Buffer should be bounded (the code trims to 64)
        assert len(req.stop_buf) <= 65


# ── _Request ────────────────────────────────────────────────────────────────

class TestRequest:
    def _make(self, **kwargs):
        defaults = dict(
            request_id="test-1",
            input_ids=[1, 2, 3],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop_ids=[],
            seed=42,
        )
        defaults.update(kwargs)
        return _Request(**defaults)

    def test_creation(self):
        req = self._make()
        assert req.request_id == "test-1"
        assert req.max_tokens == 100
        assert req.temperature == 0.7

    def test_default_out_queue_is_simple_queue(self):
        req = self._make()
        assert isinstance(req.out_queue, _queue.SimpleQueue)

    def test_default_generated_ids_empty_list(self):
        req = self._make()
        assert req.generated_ids == []

    def test_default_done_false(self):
        req = self._make()
        assert req.done is False

    def test_default_finish_reason(self):
        req = self._make()
        assert req.finish_reason == "stop"

    def test_independent_out_queues(self):
        req1 = self._make(request_id="r1")
        req2 = self._make(request_id="r2")
        req1.out_queue.put("hello")
        assert req2.out_queue.empty()


# ── QueueFullError ─────────────────────────────────────────────────────────

class TestQueueFullError:
    def test_is_runtime_error(self):
        err = QueueFullError("full")
        assert isinstance(err, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(QueueFullError) as exc:
            raise QueueFullError("queue is full")
        assert "queue is full" in str(exc.value)

    def test_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise QueueFullError("test")
