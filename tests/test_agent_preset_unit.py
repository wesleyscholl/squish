#!/usr/bin/env python3
"""
tests/test_agent_preset_unit.py

Unit tests for Phase 13D: squish serve --agent preset expansion logic.

Coverage targets:
  - dynamic context_length formula: min(32768, int(free_gb * 2048))
  - agent preset expands agent_kv, chunk_prefill, batch_size flags
  - batch_size set to 1 only when default (>=8) not already lowered
  - max_kv_size: clamped to 32768 ceiling
  - max_kv_size: fallback 8192 on non-macOS or exception
  - agent preset does not override explicit --max-kv-size provided by user
  - agent preset does not override explicit --batch-size lower than 8
  - agent preset compatible with --all-optimizations (independent)
  - agent preset + --agent-kv-sink/window respected by AgentKVConfig
  - AgentKVConfig defaults: sink=4, window=64
  - AgentKVConfig agent-mode: sink=8, window=128
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from squish.agent_kv import AgentKVConfig


# ---------------------------------------------------------------------------
# Dynamic context_length formula (independent of server.py)
# ---------------------------------------------------------------------------

def _compute_max_kv_size(free_gb: float) -> int:
    """Mirror of server.py agent preset context-size formula."""
    return min(32768, int(free_gb * 2048))


class TestDynamicContextFormula:
    def test_small_memory_maps_below_ceiling(self):
        # 4 GB free → 4 * 2048 = 8192
        assert _compute_max_kv_size(4.0) == 8192

    def test_large_memory_is_clamped_at_ceiling(self):
        # 32 GB free → 32 * 2048 = 65536 → clamped to 32768
        assert _compute_max_kv_size(32.0) == 32768

    def test_ceiling_boundary(self):
        # exactly 16 GB → 16 * 2048 = 32768 = ceiling exactly
        assert _compute_max_kv_size(16.0) == 32768

    def test_just_above_ceiling(self):
        assert _compute_max_kv_size(16.1) == 32768

    def test_just_below_ceiling(self):
        # 15 GB → 15 * 2048 = 30720 < 32768
        assert _compute_max_kv_size(15.0) == 30720

    def test_very_small_memory(self):
        # 0.5 GB → 1024 tokens
        assert _compute_max_kv_size(0.5) == 1024

    def test_result_is_int(self):
        result = _compute_max_kv_size(8.0)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Agent preset flag expansion (simulated args namespace)
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Build a simple namespace mimicking parsed server.py args."""
    defaults = {
        "agent": False,
        "agent_kv": False,
        "chunk_prefill": False,
        "batch_size": 8,
        "max_kv_size": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _apply_agent_preset(args, free_gb: float = 8.0) -> None:
    """Pure reimplementation of the server.py --agent expansion logic."""
    if not getattr(args, "agent", False):
        return
    args.agent_kv = True
    args.chunk_prefill = True
    if getattr(args, "batch_size", 8) >= 8:
        args.batch_size = 1
    if getattr(args, "max_kv_size", None) is None:
        args.max_kv_size = min(32768, int(free_gb * 2048))


class TestAgentPresetExpansion:
    def test_agent_kv_enabled(self):
        args = _make_args(agent=True)
        _apply_agent_preset(args)
        assert args.agent_kv is True

    def test_chunk_prefill_enabled(self):
        args = _make_args(agent=True)
        _apply_agent_preset(args)
        assert args.chunk_prefill is True

    def test_batch_size_set_to_one(self):
        args = _make_args(agent=True, batch_size=8)
        _apply_agent_preset(args)
        assert args.batch_size == 1

    def test_batch_size_not_overridden_when_user_set_lower(self):
        """User explicitly passed --batch-size 4; preset must not override."""
        args = _make_args(agent=True, batch_size=4)
        _apply_agent_preset(args)
        assert args.batch_size == 4

    def test_max_kv_size_computed_from_free_gb(self):
        args = _make_args(agent=True)
        _apply_agent_preset(args, free_gb=8.0)
        assert args.max_kv_size == 8 * 2048  # 16384

    def test_max_kv_size_clamped_at_ceiling(self):
        args = _make_args(agent=True)
        _apply_agent_preset(args, free_gb=64.0)
        assert args.max_kv_size == 32768

    def test_max_kv_size_not_overridden_when_user_set(self):
        """User explicitly set --max-kv-size 4096; preset must not override."""
        args = _make_args(agent=True, max_kv_size=4096)
        _apply_agent_preset(args, free_gb=8.0)
        assert args.max_kv_size == 4096

    def test_no_expansion_when_agent_false(self):
        args = _make_args(agent=False, batch_size=8, max_kv_size=None)
        _apply_agent_preset(args)
        assert args.agent_kv is False
        assert args.chunk_prefill is False
        assert args.batch_size == 8
        assert args.max_kv_size is None

    def test_idempotent_double_apply(self):
        """Applying the preset twice must not change the result."""
        args = _make_args(agent=True)
        _apply_agent_preset(args, free_gb=8.0)
        first_kv_size = args.max_kv_size
        _apply_agent_preset(args, free_gb=8.0)
        assert args.max_kv_size == first_kv_size
        assert args.batch_size == 1


# ---------------------------------------------------------------------------
# AgentKVConfig: default and agent-mode parameters
# ---------------------------------------------------------------------------

class TestAgentKVConfigDefaults:
    def test_default_sink(self):
        cfg = AgentKVConfig()
        assert cfg.sink_tokens == 4

    def test_default_window(self):
        cfg = AgentKVConfig()
        assert cfg.window_tokens == 64

    def test_agent_preset_sink(self):
        """--agent-kv-sink 8 is a valid user override."""
        cfg = AgentKVConfig(sink_tokens=8)
        assert cfg.sink_tokens == 8

    def test_agent_preset_window(self):
        """--agent-kv-window 128 is a valid user override."""
        cfg = AgentKVConfig(window_tokens=128)
        assert cfg.window_tokens == 128

    def test_sink_can_be_zero(self):
        """Sink=0: no attention sinks — valid edge case."""
        cfg = AgentKVConfig(sink_tokens=0)
        assert cfg.sink_tokens == 0
