"""tests/test_meta_reasoner_unit.py — 100% coverage for squish/meta_reasoner.py"""
import numpy as np
import pytest

from squish.meta_reasoner import MetaReasoner, MetaReasonerConfig


# ---------------------------------------------------------------------------
# MetaReasonerConfig tests
# ---------------------------------------------------------------------------

class TestMetaReasonerConfig:
    def test_defaults(self):
        cfg = MetaReasonerConfig()
        assert cfg.think_start_token_id == 151667
        assert cfg.think_end_token_id   == 151668
        assert cfg.entropy_threshold    == 1.5
        assert cfg.entropy_high_threshold == 4.0
        assert cfg.patience              == 3
        assert cfg.min_think_tokens      == 5
        assert cfg.max_think_tokens      == 512

    def test_custom_values(self):
        cfg = MetaReasonerConfig(
            think_start_token_id=10,
            think_end_token_id=11,
            entropy_threshold=1.0,
            entropy_high_threshold=5.0,
            patience=2,
            min_think_tokens=0,
            max_think_tokens=100,
        )
        assert cfg.think_start_token_id == 10
        assert cfg.patience == 2

    def test_invalid_entropy_threshold(self):
        with pytest.raises(ValueError, match="entropy_threshold"):
            MetaReasonerConfig(entropy_threshold=0.0)

    def test_invalid_entropy_high_threshold(self):
        with pytest.raises(ValueError, match="entropy_high_threshold"):
            MetaReasonerConfig(entropy_threshold=3.0, entropy_high_threshold=2.0)

    def test_invalid_patience(self):
        with pytest.raises(ValueError, match="patience"):
            MetaReasonerConfig(patience=0)

    def test_invalid_min_think_tokens(self):
        with pytest.raises(ValueError, match="min_think_tokens"):
            MetaReasonerConfig(min_think_tokens=-1)

    def test_invalid_max_think_tokens(self):
        with pytest.raises(ValueError, match="max_think_tokens"):
            MetaReasonerConfig(min_think_tokens=10, max_think_tokens=5)


# ---------------------------------------------------------------------------
# MetaReasoner.compute_entropy
# ---------------------------------------------------------------------------

class TestComputeEntropy:
    def test_uniform_distribution_max_entropy(self):
        # Uniform over 4 tokens → H = log(4)
        logits = np.zeros(4, dtype=np.float32)
        H = MetaReasoner.compute_entropy(logits)
        assert abs(H - np.log(4)) < 1e-4

    def test_peaked_distribution_low_entropy(self):
        # One token dominates → near-zero entropy
        logits = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
        H = MetaReasoner.compute_entropy(logits)
        assert H < 0.01

    def test_returns_float(self):
        logits = np.random.rand(1000).astype(np.float32)
        result = MetaReasoner.compute_entropy(logits)
        assert isinstance(result, float)

    def test_handles_single_token(self):
        logits = np.array([5.0], dtype=np.float32)
        H = MetaReasoner.compute_entropy(logits)
        assert H < 1e-6   # only one token, 0 entropy


# ---------------------------------------------------------------------------
# MetaReasoner.think_end_probability
# ---------------------------------------------------------------------------

class TestThinkEndProbability:
    def test_high_prob_when_logit_dominates(self):
        logits = np.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        p = MetaReasoner.think_end_probability(logits, think_end_id=2)
        assert p > 0.99

    def test_low_prob_when_other_dominates(self):
        logits = np.array([100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        p = MetaReasoner.think_end_probability(logits, think_end_id=2)
        assert p < 0.01

    def test_invalid_token_id_returns_zero(self):
        logits = np.zeros(10, dtype=np.float32)
        assert MetaReasoner.think_end_probability(logits, think_end_id=100) == 0.0
        assert MetaReasoner.think_end_probability(logits, think_end_id=-1)  == 0.0


# ---------------------------------------------------------------------------
# MetaReasoner phase tracking + step logic
# ---------------------------------------------------------------------------

class TestMetaReasonerStep:
    def _make(self, **kw):
        defaults = dict(
            think_start_token_id=1,
            think_end_token_id=2,
            entropy_threshold=2.0,
            entropy_high_threshold=4.5,
            patience=2,
            min_think_tokens=2,
            max_think_tokens=20,
        )
        defaults.update(kw)
        cfg = MetaReasonerConfig(**defaults)
        return MetaReasoner(cfg)

    def test_not_in_thinking_before_start_token(self):
        m = self._make()
        assert not m.in_thinking_phase

    def test_enter_thinking_on_start_token(self):
        m = self._make()
        m.advance(1)   # <think>
        assert m.in_thinking_phase

    def test_exit_thinking_on_end_token(self):
        m = self._make()
        m.advance(1)
        m.advance(2)   # </think>
        assert not m.in_thinking_phase

    def test_step_returns_false_outside_thinking(self):
        m = self._make()
        logits = np.zeros(100)
        assert m.step(logits) is False

    def test_step_returns_false_before_min_tokens(self):
        m = self._make(min_think_tokens=5)
        m.advance(1)   # <think>
        # Generate only 3 tokens before calling step
        for _ in range(3):
            m.advance(99)
        logits = np.zeros(100)   # uniform → max entropy → won't converge anyway
        assert m.step(logits) is False

    def test_step_forces_end_on_hard_cap(self):
        m = self._make(max_think_tokens=3, min_think_tokens=0)
        m.advance(1)   # <think>
        for _ in range(3):
            m.advance(99)
        # At max_think_tokens, step should return True
        logits = np.zeros(100)
        assert m.step(logits) is True

    def test_step_forces_end_after_patience_low_entropy(self):
        m = self._make(patience=2, min_think_tokens=0,
                       entropy_threshold=10.0, entropy_high_threshold=12.0)
        m.advance(1)   # <think>
        # Two consecutive low-entropy steps → should force
        peaked = np.array([100.0] + [0.0] * 99, dtype=np.float32)
        assert m.step(peaked) is False   # first time: counter=1 < patience
        assert m.step(peaked) is True    # second time: counter=2 >= patience

    def test_step_resets_counter_on_medium_entropy(self):
        m = self._make(patience=2, min_think_tokens=0,
                       entropy_threshold=2.0, entropy_high_threshold=4.5)
        m.advance(1)
        peaked = np.array([100.0] + [0.0] * 99, dtype=np.float32)
        m.step(peaked)   # counter_low = 1
        uniform = np.zeros(100, dtype=np.float32)  # medium entropy
        m.step(uniform)  # resets counter_low = 0
        assert m.consecutive_converged_steps == 0

    def test_step_no_double_force(self):
        m = self._make(max_think_tokens=2, min_think_tokens=0)
        m.advance(1)
        for _ in range(2):
            m.advance(99)
        logits = np.zeros(100)
        assert m.step(logits) is True    # first force
        assert m.step(logits) is False   # already forced

    def test_properties(self):
        m = self._make()
        m.advance(1)
        for _ in range(4):
            m.advance(99)
        assert m.think_tokens_generated == 4

    def test_reset(self):
        m = self._make()
        m.advance(1)
        m.advance(99)
        m.reset()
        assert not m.in_thinking_phase
        assert m.think_tokens_generated == 0
        assert m.consecutive_converged_steps == 0


# ---------------------------------------------------------------------------
# force_think_end
# ---------------------------------------------------------------------------

class TestForceThinkEnd:
    def test_spikes_think_end_token(self):
        cfg = MetaReasonerConfig(think_start_token_id=1, think_end_token_id=2,
                                 entropy_threshold=1.0, entropy_high_threshold=5.0)
        m = MetaReasoner(cfg)
        logits = np.zeros(10, dtype=np.float32)
        modified = m.force_think_end(logits)
        assert modified[2] > 99.0    # should be max + 100
        assert modified[2] > modified[0]

    def test_highest_after_force(self):
        cfg = MetaReasonerConfig(think_start_token_id=1, think_end_token_id=3,
                                 entropy_threshold=1.0, entropy_high_threshold=5.0)
        m   = MetaReasoner(cfg)
        logits = np.array([50.0, 40.0, 30.0, -1.0, 20.0], dtype=np.float32)
        mod = m.force_think_end(logits)
        assert mod.argmax() == 3


# ---------------------------------------------------------------------------
# high-entropy exploration detection
# ---------------------------------------------------------------------------

class TestHighEntropyTracking:
    def test_consecutive_high_resets_low_counter(self):
        # entropy_threshold=0.5 < entropy_high_threshold=2.0 (valid config).
        # peaked logits → H ≈ 0 < 0.5 → consecutive_low += 1
        # uniform logits → H ≈ log(100) ≈ 4.6 > 2.0 → resets consecutive_low to 0
        cfg = MetaReasonerConfig(
            think_start_token_id=1, think_end_token_id=2,
            entropy_threshold=0.5, entropy_high_threshold=2.0,
            patience=3, min_think_tokens=0, max_think_tokens=100,
        )
        m = MetaReasoner(cfg)
        m.advance(1)   # enter thinking phase
        peaked  = np.array([100.0] + [0.0] * 99, dtype=np.float32)
        uniform = np.zeros(100, dtype=np.float32)
        m.step(peaked)   # H ≈ 0 < threshold → consecutive_low = 1
        assert m.consecutive_converged_steps == 1
        m.step(uniform)  # H ≈ 4.6 > high_threshold → consecutive_low reset to 0
        assert m.consecutive_converged_steps == 0
        # Config validation: entropy_high_threshold ≤ entropy_threshold must raise
        with pytest.raises(ValueError):
            MetaReasonerConfig(entropy_threshold=3.0, entropy_high_threshold=2.9)


class TestObserveAndStepEdgeCases:
    def _make_thinking_reasoner(self):
        cfg = MetaReasonerConfig(
            think_start_token_id=1, think_end_token_id=2,
            entropy_threshold=0.5, entropy_high_threshold=2.0,
            patience=10, min_think_tokens=0, max_think_tokens=50,
        )
        m = MetaReasoner(cfg)
        m.advance(1)   # enter thinking phase
        return m

    def test_observe_non_special_token_outside_thinking_no_increment(self):
        """Branch 152→exit: observe() with non-think token when not in thinking
        phase → elif self._in_thinking is False → think_tokens stays at 0."""
        cfg = MetaReasonerConfig(think_start_token_id=1, think_end_token_id=2)
        m   = MetaReasoner(cfg)
        # Not in thinking yet; send a plain token (id=99)
        m.advance(99)
        assert m._think_tokens == 0   # the elif branch body was skipped

    def test_step_neutral_entropy_resets_both_counters(self):
        """Lines 188-189: H between threshold and high_threshold → else branch
        sets both consecutive_low and consecutive_high to 0."""
        m = self._make_thinking_reasoner()
        # First, build up consecutive_low > 0
        peaked = np.array([100.0] + [0.0] * 99, dtype=np.float32)
        m.step(peaked)   # H ≈ 0 < 0.5 → consecutive_low = 1
        assert m.consecutive_converged_steps == 1
        # Now pass logits with neutral entropy between 0.5 and 2.0.
        # Use a 3-token softmax: [2.0, 1.0, 0.0] → H ≈ 0.988, in [0.5, 2.0].
        # Pad to 100 elements with a very large negative value so softmax is ~0.
        # Easiest: just use 3-element logits array directly.
        neutral = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        m.step(neutral)  # else branch → both counters reset to 0
        assert m.consecutive_converged_steps == 0
        assert m._consecutive_high == 0
