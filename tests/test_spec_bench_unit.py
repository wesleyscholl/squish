"""
tests/test_spec_bench_unit.py

Unit tests for squish/spec_bench.py — 100% coverage.
"""

import pytest

from squish.spec_bench import (
    SpecBenchResult,
    SpecBenchRunner,
    SpecBenchStats,
    SpecBenchTask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_accept(gamma):
    """Returns draft and target functions that always accept all tokens."""
    def draft_fn(prompt):
        return list(range(gamma))

    def target_fn(prompt, draft_tokens):
        return [True] * len(draft_tokens)

    return draft_fn, target_fn


def _never_accept(gamma):
    """Returns draft and target functions that never accept any token."""
    def draft_fn(prompt):
        return list(range(gamma))

    def target_fn(prompt, draft_tokens):
        return [False] * len(draft_tokens)

    return draft_fn, target_fn


def _half_accept(gamma):
    """Accepts first half of draft tokens."""
    def draft_fn(prompt):
        return list(range(gamma))

    def target_fn(prompt, draft_tokens):
        n = len(draft_tokens)
        return [True] * (n // 2) + [False] * (n - n // 2)

    return draft_fn, target_fn


# ---------------------------------------------------------------------------
# SpecBenchTask
# ---------------------------------------------------------------------------


class TestSpecBenchTask:
    def test_defaults(self):
        t = SpecBenchTask("math", ["p1", "p2"])
        assert t.category == "general"
        assert t.n_prompts == 2

    def test_custom_category(self):
        t = SpecBenchTask("code", ["p1"], category="code")
        assert t.category == "code"

    def test_empty_task_name_raises(self):
        with pytest.raises(ValueError, match="task_name"):
            SpecBenchTask("", ["p1"])

    def test_empty_prompts_raises(self):
        with pytest.raises(ValueError, match="prompts"):
            SpecBenchTask("math", [])

    def test_n_prompts(self):
        t = SpecBenchTask("t", ["a", "b", "c"])
        assert t.n_prompts == 3


# ---------------------------------------------------------------------------
# SpecBenchResult
# ---------------------------------------------------------------------------


class TestSpecBenchResult:
    def _result(self, total=8, accepted=4, steps=2, latency=100.0):
        return SpecBenchResult("t", 2, total, accepted, steps, latency)

    def test_acceptance_rate(self):
        r = self._result(total=8, accepted=6)
        assert abs(r.acceptance_rate - 0.75) < 1e-9

    def test_acceptance_rate_zero_tokens(self):
        r = SpecBenchResult("t", 0, 0, 0, 0, 0.0)
        assert r.acceptance_rate == 0.0

    def test_mean_accepted_per_step(self):
        r = self._result(accepted=6, steps=3)
        assert abs(r.mean_accepted_per_step - 2.0) < 1e-9

    def test_mean_accepted_per_step_zero_steps(self):
        r = SpecBenchResult("t", 0, 4, 4, 0, 100.0)
        assert r.mean_accepted_per_step == 0.0

    def test_tokens_per_second(self):
        # 8 tokens / 0.5 s = 16 tok/s
        r = self._result(total=8, latency=500.0)
        assert abs(r.tokens_per_second - 16.0) < 1e-6

    def test_tokens_per_second_zero_latency(self):
        r = SpecBenchResult("t", 1, 4, 4, 1, 0.0)
        assert r.tokens_per_second == 0.0


# ---------------------------------------------------------------------------
# SpecBenchStats
# ---------------------------------------------------------------------------


class TestSpecBenchStats:
    @pytest.mark.parametrize(
        "rate, grade",
        [(0.8, "A"), (0.9, "A"), (0.6, "B"), (0.79, "B"), (0.4, "C"), (0.59, "C"), (0.39, "D"), (0.0, "D")],
    )
    def test_grade(self, rate, grade):
        s = SpecBenchStats(1, 1, rate)
        assert s.grade == grade


# ---------------------------------------------------------------------------
# SpecBenchRunner — construction
# ---------------------------------------------------------------------------


class TestSpecBenchRunnerConstruction:
    def test_defaults(self):
        r = SpecBenchRunner()
        assert r._gamma == 4
        assert r._temperature == 1.0

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            SpecBenchRunner(gamma=0)

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SpecBenchRunner(temperature=0.0)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SpecBenchRunner(temperature=-1.0)


# ---------------------------------------------------------------------------
# SpecBenchRunner — run_task
# ---------------------------------------------------------------------------


class TestSpecBenchRunnerRunTask:
    def test_run_task_all_accepted(self):
        runner = SpecBenchRunner(gamma=4)
        task = SpecBenchTask("test", ["p1", "p2"])
        draft_fn, target_fn = _always_accept(4)
        result = runner.run_task(task, draft_fn, target_fn)
        assert result.task_name == "test"
        assert result.n_prompts == 2
        assert result.total_tokens == 8     # 4 * 2
        assert result.total_accepted == 8
        assert result.total_steps == 2
        assert abs(result.acceptance_rate - 1.0) < 1e-9

    def test_run_task_none_accepted(self):
        runner = SpecBenchRunner(gamma=4)
        task = SpecBenchTask("test", ["p1"])
        draft_fn, target_fn = _never_accept(4)
        result = runner.run_task(task, draft_fn, target_fn)
        assert result.total_accepted == 0
        assert result.acceptance_rate == 0.0

    def test_run_task_half_accepted(self):
        runner = SpecBenchRunner(gamma=4)
        task = SpecBenchTask("test", ["p1"])
        draft_fn, target_fn = _half_accept(4)
        result = runner.run_task(task, draft_fn, target_fn)
        assert abs(result.acceptance_rate - 0.5) < 1e-9

    def test_run_task_latency_positive(self):
        runner = SpecBenchRunner(gamma=2)
        task = SpecBenchTask("t", ["p"])
        d, t = _always_accept(2)
        result = runner.run_task(task, d, t)
        assert result.latency_ms_total >= 0.0

    def test_run_task_single_prompt(self):
        runner = SpecBenchRunner(gamma=3)
        task = SpecBenchTask("t", ["only one"])
        draft_fn, target_fn = _always_accept(3)
        result = runner.run_task(task, draft_fn, target_fn)
        assert result.total_steps == 1
        assert result.total_tokens == 3


# ---------------------------------------------------------------------------
# SpecBenchRunner — default_tasks
# ---------------------------------------------------------------------------


class TestSpecBenchRunnerDefaultTasks:
    def test_six_tasks(self):
        tasks = SpecBenchRunner.default_tasks()
        assert len(tasks) == 6

    def test_task_names(self):
        names = {t.task_name for t in SpecBenchRunner.default_tasks()}
        assert names == {"translation", "summarization", "qa", "math", "rag", "code"}

    def test_each_task_has_two_prompts(self):
        for task in SpecBenchRunner.default_tasks():
            assert task.n_prompts == 2, f"{task.task_name} has {task.n_prompts} prompts"

    def test_tasks_have_categories(self):
        for task in SpecBenchRunner.default_tasks():
            assert task.category != ""


# ---------------------------------------------------------------------------
# SpecBenchRunner — run_suite and overall stats
# ---------------------------------------------------------------------------


class TestSpecBenchRunnerSuite:
    def test_run_suite_returns_all_tasks(self):
        runner = SpecBenchRunner(gamma=2)
        d, t = _always_accept(2)
        results = runner.run_suite(d, t)
        assert len(results) == 6
        assert set(results.keys()) == {"translation", "summarization", "qa", "math", "rag", "code"}

    def test_run_suite_all_accepted(self):
        runner = SpecBenchRunner(gamma=4)
        d, t = _always_accept(4)
        results = runner.run_suite(d, t)
        oar = runner.overall_acceptance_rate(results)
        assert abs(oar - 1.0) < 1e-9

    def test_run_suite_none_accepted(self):
        runner = SpecBenchRunner(gamma=4)
        d, t = _never_accept(4)
        results = runner.run_suite(d, t)
        oar = runner.overall_acceptance_rate(results)
        assert oar == 0.0

    def test_overall_acceptance_rate_empty(self):
        oar = SpecBenchRunner.overall_acceptance_rate({})
        assert oar == 0.0

    def test_suite_stats(self):
        runner = SpecBenchRunner(gamma=2)
        d, t = _always_accept(2)
        results = runner.run_suite(d, t)
        ss = runner.suite_stats(results)
        assert ss.tasks_run == 6
        assert ss.total_prompts == 12  # 2 prompts × 6 tasks
        assert abs(ss.overall_acceptance_rate - 1.0) < 1e-9
        assert ss.grade == "A"

    def test_suite_stats_grade_d(self):
        runner = SpecBenchRunner(gamma=4)
        d, t = _never_accept(4)
        results = runner.run_suite(d, t)
        ss = runner.suite_stats(results)
        assert ss.grade == "D"
