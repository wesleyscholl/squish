"""tests/test_disc_router_unit.py — 100% coverage for squish/disc_router.py"""
import pytest
from squish.disc_router import (
    TaskType,
    SubTask,
    DISCPlan,
    DISCRouterConfig,
    DISCRouter,
    _DEFAULT_SYSTEM_PROMPTS,
)


# ---------------------------------------------------------------------------
# TaskType
# ---------------------------------------------------------------------------

class TestTaskType:
    def test_all_values(self):
        expected = {"summarize", "compare", "retrieve", "generate",
                    "qa", "code", "aggregate", "unknown"}
        assert {t.value for t in TaskType} == expected

    def test_string_equality(self):
        assert TaskType.SUMMARIZE == "summarize"


# ---------------------------------------------------------------------------
# SubTask
# ---------------------------------------------------------------------------

class TestSubTask:
    def test_auto_output_var(self):
        t = SubTask(task_id="t1", task_type=TaskType.GENERATE, prompt="go")
        assert t.output_var == "t1_out"

    def test_explicit_output_var_preserved(self):
        t = SubTask(task_id="t1", task_type=TaskType.GENERATE, prompt="go",
                    output_var="my_out")
        assert t.output_var == "my_out"

    def test_empty_task_id_raises(self):
        with pytest.raises(ValueError):
            SubTask(task_id="", task_type=TaskType.GENERATE, prompt="x")

    def test_defaults(self):
        t = SubTask(task_id="x", task_type=TaskType.QA, prompt="q")
        assert t.inputs == []
        assert t.depends_on == []
        assert t.context_key is None
        assert t.metadata == {}


# ---------------------------------------------------------------------------
# DISCPlan
# ---------------------------------------------------------------------------

class TestDISCPlan:
    def test_add_and_len(self):
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "p1"))
        assert len(plan) == 1

    def test_topological_order_no_deps(self):
        plan = DISCPlan()
        plan.add(SubTask("t2", TaskType.GENERATE, "p2"))
        plan.add(SubTask("t1", TaskType.GENERATE, "p1"))
        order = plan.topological_order()
        assert {t.task_id for t in order} == {"t1", "t2"}

    def test_topological_order_with_deps(self):
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.SUMMARIZE, "p1"))
        plan.add(SubTask("t2", TaskType.AGGREGATE, "p2", depends_on=["t1"]))
        order = plan.topological_order()
        ids = [t.task_id for t in order]
        assert ids.index("t1") < ids.index("t2")

    def test_cycle_raises(self):
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "p1", depends_on=["t2"]))
        plan.add(SubTask("t2", TaskType.GENERATE, "p2", depends_on=["t1"]))
        with pytest.raises(ValueError, match="cycle"):
            plan.topological_order()

    def test_empty_plan(self):
        plan = DISCPlan()
        assert plan.topological_order() == []


# ---------------------------------------------------------------------------
# DISCRouterConfig
# ---------------------------------------------------------------------------

class TestDISCRouterConfig:
    def test_defaults(self):
        cfg = DISCRouterConfig()
        assert cfg.max_subtasks == 12
        assert cfg.parallel_execution is False

    def test_invalid_max_subtasks(self):
        with pytest.raises(ValueError, match="max_subtasks"):
            DISCRouterConfig(max_subtasks=0)

    def test_get_system_prompt_default(self):
        cfg = DISCRouterConfig()
        prompt = cfg.get_system_prompt(TaskType.SUMMARIZE)
        assert "summariz" in prompt.lower()

    def test_get_system_prompt_custom(self):
        cfg = DISCRouterConfig(task_prompt_templates={"summarize": "Custom!"})
        assert cfg.get_system_prompt(TaskType.SUMMARIZE) == "Custom!"

    def test_get_system_prompt_unknown(self):
        cfg = DISCRouterConfig()
        prompt = cfg.get_system_prompt(TaskType.UNKNOWN)
        assert isinstance(prompt, str)

    def test_all_task_types_have_defaults(self):
        cfg = DISCRouterConfig()
        for tt in TaskType:
            prompt = cfg.get_system_prompt(tt)
            assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# DISCRouter — _build_planner_prompt
# ---------------------------------------------------------------------------

class TestBuildPlannerPrompt:
    def test_contains_user_request(self):
        prompt = DISCRouter._build_planner_prompt("summarise this", "ctx")
        assert "summarise this" in prompt

    def test_contains_context_snippet(self):
        prompt = DISCRouter._build_planner_prompt("q", "my context")
        assert "my context" in prompt

    def test_no_context(self):
        prompt = DISCRouter._build_planner_prompt("q", "")
        assert "(none)" in prompt

    def test_context_truncated_to_500(self):
        long_ctx = "x" * 600
        prompt = DISCRouter._build_planner_prompt("q", long_ctx)
        assert "x" * 500 in prompt
        assert "x" * 501 not in prompt


# ---------------------------------------------------------------------------
# DISCRouter — _parse_plan
# ---------------------------------------------------------------------------

class TestParsePlan:
    def test_valid_task_line(self):
        raw = "TASK|t1|summarize||summary_out|Do it."
        plan = DISCRouter._parse_plan(raw, "request")
        assert len(plan) == 1
        t = plan.tasks[0]
        assert t.task_id == "t1"
        assert t.task_type == TaskType.SUMMARIZE
        assert t.output_var == "summary_out"
        assert t.prompt == "Do it."

    def test_depends_on_parsed(self):
        raw = "TASK|t2|compare|t1||Compare them."
        plan = DISCRouter._parse_plan(raw, "r")
        assert plan.tasks[0].depends_on == ["t1"]

    def test_fallback_single_task(self):
        plan = DISCRouter._parse_plan("no task lines here", "my request")
        assert len(plan) == 1
        assert plan.tasks[0].task_type == TaskType.GENERATE
        assert plan.tasks[0].prompt == "my request"

    def test_unknown_task_type(self):
        raw = "TASK|t1|bogustype||out|Something."
        plan = DISCRouter._parse_plan(raw, "r")
        assert plan.tasks[0].task_type == TaskType.UNKNOWN

    def test_short_line_ignored(self):
        raw = "TASK|t1|summarize"
        plan = DISCRouter._parse_plan(raw, "r")
        # Should fall back to single task
        assert len(plan) == 1
        assert plan.tasks[0].task_type == TaskType.GENERATE

    def test_prompt_with_pipe_preserved(self):
        raw = "TASK|t1|generate||out|Do this | and that."
        plan = DISCRouter._parse_plan(raw, "r")
        assert "and that" in plan.tasks[0].prompt

    def test_empty_output_var_auto_set(self):
        raw = "TASK|t1|qa|||Ask this."
        plan = DISCRouter._parse_plan(raw, "r")
        assert plan.tasks[0].output_var == "t1_out"


# ---------------------------------------------------------------------------
# DISCRouter — plan / execute / execute_plan
# ---------------------------------------------------------------------------

def _make_llm(responses=None):
    """Returns an llm_fn that cycles through fixed responses."""
    if responses is None:
        responses = ["RESULT"]
    state = {"idx": 0}

    def fn(prompt, system, context):
        r = responses[state["idx"] % len(responses)]
        state["idx"] += 1
        return r

    return fn


class TestDISCRouterExecute:
    def test_plan_returns_disc_plan(self):
        llm = _make_llm(["TASK|t1|generate||out|Do it."])
        router = DISCRouter(llm)
        plan = router.plan("user request", "ctx")
        assert isinstance(plan, DISCPlan)

    def test_execute_returns_string(self):
        task_line = "TASK|t1|generate||out|Do it."
        llm = _make_llm([task_line, "FINAL"])
        router = DISCRouter(llm)
        result = router.execute("request", "ctx")
        assert isinstance(result, str)

    def test_execute_with_pre_built_plan(self):
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "prompt"))
        llm = _make_llm(["answer"])
        router = DISCRouter(llm)
        result = router.execute("user", "ctx", plan=plan)
        assert result == "answer"

    def test_execute_plan_returns_dict(self):
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "p1", output_var="out1"))
        plan.add(SubTask("t2", TaskType.AGGREGATE, "p2", depends_on=["t1"],
                          inputs=["out1"], output_var="out2"))
        llm = _make_llm(["a1", "a2"])
        router = DISCRouter(llm)
        result = router.execute_plan(plan, "ctx")
        assert "out1" in result
        assert "out2" in result
        assert result["out1"] == "a1"
        assert result["out2"] == "a2"

    def test_empty_plan_execute_returns_empty(self):
        llm = _make_llm(["TASK|t1|generate||out|x.", "res"])
        router = DISCRouter(llm)
        result = router.execute("", "")
        assert isinstance(result, str)

    def test_max_subtasks_truncated(self):
        lines = "\n".join(
            f"TASK|t{i}|generate||out{i}|Go." for i in range(1, 20)
        )
        llm = _make_llm([lines] + ["resp"] * 20)
        router = DISCRouter(llm, DISCRouterConfig(max_subtasks=3))
        plan = router.plan("r")
        assert len(plan) <= 3

    def test_context_key_used(self):
        captured = []

        def llm(prompt, system, context):
            captured.append(context)
            return "ok"

        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.QA, "ask?", context_key=None,
                          output_var="out1"))
        router = DISCRouter(llm)
        router.execute_plan(plan, "MY_CONTEXT")
        assert "MY_CONTEXT" in captured

    def test_prior_inputs_injected_in_prompt(self):
        prompts_seen = []

        def llm(prompt, system, context):
            prompts_seen.append(prompt)
            return "response"

        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "first", output_var="result1"))
        plan.add(SubTask("t2", TaskType.AGGREGATE, "second",
                          inputs=["result1"], depends_on=["t1"], output_var="out2"))
        router = DISCRouter(llm)
        router.execute_plan(plan, "")
        # The second prompt should include the first result
        assert any("result1" in p for p in prompts_seen)


class TestTopologicalEdgeCases:
    def test_external_dependency_ignored_in_in_deg(self):
        """Branch 145→144: dep not in in_deg → skipped (no KeyError)."""
        plan = DISCPlan()
        plan.add(SubTask("t1", TaskType.GENERATE, "task",
                         depends_on=["ghost_external"], output_var="out1"))
        # "ghost_external" is not in task_map → branch 145 False: in_deg unchanged
        order = plan.topological_order()
        assert len(order) == 1

    def test_multi_dep_task_not_queued_until_all_done(self):
        """Branch 157→154: in_deg[candidate] > 0 after decrement → not appended."""
        plan = DISCPlan()
        plan.add(SubTask("a", TaskType.GENERATE, "A", output_var="ra"))
        plan.add(SubTask("b", TaskType.GENERATE, "B", output_var="rb"))
        plan.add(SubTask("c", TaskType.AGGREGATE, "C",
                         depends_on=["a", "b"], output_var="rc"))
        order = plan.topological_order()
        task_ids = [t.task_id for t in order]
        assert task_ids.index("c") > task_ids.index("a")
        assert task_ids.index("c") > task_ids.index("b")

    def test_execute_empty_plan_returns_empty_string(self):
        """Line 301: ordered is empty → execute returns ''."""
        llm    = lambda prompt, system, context: "x"
        plan   = DISCPlan()          # no tasks
        router = DISCRouter(llm)
        result = router.execute("", plan=plan, context="anything")
        assert result == ""
