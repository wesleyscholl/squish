#!/usr/bin/env python3
"""
tests/test_fault_tolerance_unit.py

Unit tests for squish/fault_tolerance.py.

Coverage targets
────────────────
mem_pressure_fraction
  - LEVEL_NORMAL (0) → 0.0
  - LEVEL_WARNING (1) → < evict_kv_at default (0.85)
  - LEVEL_URGENT (2) → between disable_draft_at and reduce_batch_at defaults
  - LEVEL_CRITICAL (4) → 1.0
  - unknown level falls back to 1.0

FaultPolicy
  - valid policy constructs without error
  - evict_kv_at out of range raises
  - threshold ordering violated raises
  - min_batch_size < 1 raises

FaultHandler.evaluate
  - pressure below all thresholds → []
  - pressure >= evict_kv_at → ["evict_kv"]
  - pressure >= disable_draft_at → ["evict_kv", "disable_draft"]
  - pressure >= reduce_batch_at → [..., "reduce_batch"]
  - pressure == 1.0 → all four actions including "renegotiate_slo"
  - pressure out of range raises
  - current_batch_size < 1 raises
  - stats.total_evaluations increments on each call
  - stats.draft_disables increments only when disable_draft triggered
  - stats.batch_reductions increments only when batch reduced

FaultHandler.apply_evict_kv
  - returns n_to_evict
  - adds to stats.kv_evictions
  - negative raises

FaultHandler.evaluate_from_governor
  - NORMAL governor → [] actions, last_governor_level = 0
  - WARNING governor → actions and level recorded
  - URGENT governor → triggers evict + disable_draft
  - CRITICAL governor → full cascade
  - last_governor_level updates each call
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from squish.fault_tolerance import (
    FaultAction,
    FaultHandler,
    FaultPolicy,
    FaultStats,
    mem_pressure_fraction,
)
from squish.memory_governor import (
    LEVEL_CRITICAL,
    LEVEL_NORMAL,
    LEVEL_URGENT,
    LEVEL_WARNING,
)


# ---------------------------------------------------------------------------
# mem_pressure_fraction
# ---------------------------------------------------------------------------

class TestMemPressureFraction:
    def test_normal_is_zero(self):
        assert mem_pressure_fraction(LEVEL_NORMAL) == 0.0

    def test_warning_below_evict_threshold(self):
        # Default evict_kv_at = 0.85; WARNING should not trigger it by default
        frac = mem_pressure_fraction(LEVEL_WARNING)
        assert frac < FaultPolicy().evict_kv_at

    def test_urgent_triggers_evict_and_disable_draft(self):
        frac = mem_pressure_fraction(LEVEL_URGENT)
        p = FaultPolicy()
        assert frac >= p.evict_kv_at
        assert frac >= p.disable_draft_at
        assert frac < p.reduce_batch_at

    def test_critical_is_one(self):
        assert mem_pressure_fraction(LEVEL_CRITICAL) == 1.0

    def test_unknown_level_falls_back_to_one(self):
        assert mem_pressure_fraction(99) == 1.0

    def test_all_values_in_unit_range(self):
        for level in (0, 1, 2, 4):
            frac = mem_pressure_fraction(level)
            assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# FaultPolicy
# ---------------------------------------------------------------------------

class TestFaultPolicy:
    def test_valid_defaults(self):
        p = FaultPolicy()
        assert p.evict_kv_at == 0.85
        assert p.disable_draft_at == 0.90
        assert p.reduce_batch_at == 0.95
        assert p.min_batch_size == 1

    def test_evict_kv_at_zero_raises(self):
        with pytest.raises(ValueError, match="evict_kv_at"):
            FaultPolicy(evict_kv_at=0.0)

    def test_evict_kv_at_above_one_raises(self):
        with pytest.raises(ValueError, match="evict_kv_at"):
            FaultPolicy(evict_kv_at=1.1)

    def test_threshold_order_violated_raises(self):
        with pytest.raises(ValueError, match="evict_kv_at"):
            FaultPolicy(evict_kv_at=0.95, disable_draft_at=0.90, reduce_batch_at=0.85)

    def test_min_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="min_batch_size"):
            FaultPolicy(min_batch_size=0)

    def test_min_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="min_batch_size"):
            FaultPolicy(min_batch_size=-1)

    def test_valid_custom_config(self):
        p = FaultPolicy(evict_kv_at=0.70, disable_draft_at=0.80,
                        reduce_batch_at=0.90, min_batch_size=2)
        assert p.min_batch_size == 2


# ---------------------------------------------------------------------------
# FaultHandler.evaluate
# ---------------------------------------------------------------------------

class TestFaultHandlerEvaluate:
    @pytest.fixture()
    def handler(self):
        return FaultHandler(FaultPolicy())

    def test_low_pressure_no_actions(self, handler):
        assert handler.evaluate(pressure=0.5, current_batch_size=4) == []

    def test_evict_kv_triggered(self, handler):
        actions = handler.evaluate(pressure=0.86, current_batch_size=4)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT not in actions

    def test_disable_draft_triggered(self, handler):
        actions = handler.evaluate(pressure=0.91, current_batch_size=4)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT in actions
        assert FaultAction.REDUCE_BATCH not in actions

    def test_reduce_batch_triggered(self, handler):
        actions = handler.evaluate(pressure=0.96, current_batch_size=4)
        assert FaultAction.REDUCE_BATCH in actions

    def test_full_cascade_at_one(self, handler):
        actions = handler.evaluate(pressure=1.0, current_batch_size=4)
        assert FaultAction.RENEGOTIATE_SLO in actions
        assert len(actions) == 4

    def test_pressure_above_one_raises(self, handler):
        with pytest.raises(ValueError, match="pressure"):
            handler.evaluate(pressure=1.01, current_batch_size=1)

    def test_pressure_below_zero_raises(self, handler):
        with pytest.raises(ValueError, match="pressure"):
            handler.evaluate(pressure=-0.1, current_batch_size=1)

    def test_batch_size_zero_raises(self, handler):
        with pytest.raises(ValueError, match="current_batch_size"):
            handler.evaluate(pressure=0.5, current_batch_size=0)

    def test_stats_total_evaluations_increments(self, handler):
        handler.evaluate(0.5, 1)
        handler.evaluate(0.5, 1)
        assert handler.stats.total_evaluations == 2

    def test_stats_draft_disables_increments(self, handler):
        handler.evaluate(0.91, 4)
        assert handler.stats.draft_disables == 1
        handler.evaluate(0.5, 4)  # below threshold
        assert handler.stats.draft_disables == 1  # unchanged

    def test_stats_batch_reductions_increments_only_when_above_min(self, handler):
        handler.evaluate(0.96, 4)  # above min (1)
        assert handler.stats.batch_reductions == 1
        # batch_size == min — should not increment again
        handler.evaluate(0.96, 1)
        assert handler.stats.batch_reductions == 1


# ---------------------------------------------------------------------------
# FaultHandler.apply_evict_kv
# ---------------------------------------------------------------------------

class TestFaultHandlerApplyEvictKv:
    def test_returns_n_to_evict(self):
        handler = FaultHandler(FaultPolicy())
        assert handler.apply_evict_kv(10) == 10

    def test_accumulates_in_stats(self):
        handler = FaultHandler(FaultPolicy())
        handler.apply_evict_kv(5)
        handler.apply_evict_kv(3)
        assert handler.stats.kv_evictions == 8

    def test_zero_evictions_ok(self):
        handler = FaultHandler(FaultPolicy())
        assert handler.apply_evict_kv(0) == 0

    def test_negative_raises(self):
        handler = FaultHandler(FaultPolicy())
        with pytest.raises(ValueError, match="n_to_evict"):
            handler.apply_evict_kv(-1)


# ---------------------------------------------------------------------------
# FaultHandler.evaluate_from_governor
# ---------------------------------------------------------------------------

def _mock_governor(pressure_level: int) -> MagicMock:
    """Return a MagicMock that mimics MemoryGovernor.pressure_level."""
    gov = MagicMock()
    gov.pressure_level = pressure_level
    return gov


class TestEvaluateFromGovernor:
    def test_normal_level_no_actions(self):
        handler = FaultHandler(FaultPolicy())
        actions = handler.evaluate_from_governor(_mock_governor(LEVEL_NORMAL), 4)
        assert actions == []
        assert handler.stats.last_governor_level == LEVEL_NORMAL

    def test_warning_level_no_actions_with_defaults(self):
        handler = FaultHandler(FaultPolicy())
        actions = handler.evaluate_from_governor(_mock_governor(LEVEL_WARNING), 4)
        # WARNING maps to 0.75 < default 0.85 evict_kv_at → no actions
        assert actions == []
        assert handler.stats.last_governor_level == LEVEL_WARNING

    def test_urgent_level_triggers_evict_and_disable(self):
        handler = FaultHandler(FaultPolicy())
        actions = handler.evaluate_from_governor(_mock_governor(LEVEL_URGENT), 4)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT in actions
        assert FaultAction.REDUCE_BATCH not in actions

    def test_critical_level_full_cascade(self):
        handler = FaultHandler(FaultPolicy())
        actions = handler.evaluate_from_governor(_mock_governor(LEVEL_CRITICAL), 4)
        assert FaultAction.RENEGOTIATE_SLO in actions
        assert len(actions) == 4

    def test_last_governor_level_updates(self):
        handler = FaultHandler(FaultPolicy())
        handler.evaluate_from_governor(_mock_governor(LEVEL_NORMAL), 4)
        assert handler.stats.last_governor_level == LEVEL_NORMAL
        handler.evaluate_from_governor(_mock_governor(LEVEL_URGENT), 4)
        assert handler.stats.last_governor_level == LEVEL_URGENT

    def test_stats_total_evaluations_increments(self):
        handler = FaultHandler(FaultPolicy())
        handler.evaluate_from_governor(_mock_governor(LEVEL_NORMAL), 4)
        handler.evaluate_from_governor(_mock_governor(LEVEL_NORMAL), 4)
        assert handler.stats.total_evaluations == 2

    def test_last_governor_level_none_before_first_call(self):
        handler = FaultHandler(FaultPolicy())
        assert handler.stats.last_governor_level is None
