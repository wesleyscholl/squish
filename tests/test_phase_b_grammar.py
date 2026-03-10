#!/usr/bin/env python3
"""
tests/test_phase_b_grammar.py

Phase B tests: GrammarEngine (squish/grammar_engine.py) and
parse_tool_calls_with_grammar (squish/tool_calling.py).

Coverage targets
────────────────
grammar_engine.py — 100%
  GrammarEngine.__init__
    - xgrammar import succeeds + init succeeds           → _available = True
    - xgrammar import fails (ImportError)                → _available = False
    - xgrammar import succeeds but TokenizerInfo raises  → _available = False
  GrammarEngine.is_available
    - xgrammar importable                                 → True
    - xgrammar not importable                             → False
  GrammarEngine.json_schema_grammar
    - engine not available                                → None
    - engine available                                    → GrammarMatcher
  GrammarEngine.json_object_grammar
    - engine not available                                → None
    - engine available                                    → GrammarMatcher
  GrammarEngine.regex_grammar
    - engine not available                                → None
    - engine available                                    → GrammarMatcher
  GrammarEngine.constrain_logits
    - engine not available                                → logits unchanged
    - state is None                                       → logits unchanged
    - engine available + valid state                      → constrained logits
    - exception during constraint                         → logits unchanged
  GrammarEngine.advance
    - engine not available                                → state unchanged
    - state is None                                       → None returned
    - engine available + valid state                      → state returned
    - accept_token raises                                 → state returned (no crash)
  GrammarEngine.jump_forward_tokens
    - engine not available                                → []
    - state is None                                       → []
    - engine available, fwd_str is empty                  → []
    - engine available, fwd_str non-empty                 → list of token IDs
    - exception during jump                               → []

tool_calling.py — parse_tool_calls_with_grammar
  - grammar_engine is None                               → heuristic parse_tool_calls
  - grammar_engine available, valid JSON tool call       → direct parse
  - grammar_engine available, invalid JSON               → heuristic fallback
  - grammar_engine available, valid JSON but not a tool  → heuristic fallback
  - grammar_engine not available (is_available=False)    → heuristic fallback
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not available (requires Apple Silicon)")

from squish.grammar_engine import GrammarEngine
from squish.tool_calling import parse_tool_calls_with_grammar

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_tokenizer(vocab=None):
    """Return a minimal HuggingFace-like tokenizer mock."""
    if vocab is None:
        vocab = ["a", "b", "c", "{", "}"]
    tok = MagicMock()
    tok.encode.return_value = [0, 1, 2]
    tok.get_vocab.return_value = {t: i for i, t in enumerate(vocab)}
    return tok


def _make_xgrammar_mock():
    """
    Build a comprehensive mock xgrammar module injected into sys.modules.

    Returns
    -------
    (mock_module, mock_tok_info, mock_compiler, mock_compiled, mock_matcher)
    """
    xgr = types.ModuleType("xgrammar")

    # TokenizerInfo
    mock_tok_info = MagicMock()
    mock_tok_info.vocab_size = 32000
    xgr.TokenizerInfo = MagicMock()
    xgr.TokenizerInfo.from_huggingface = MagicMock(return_value=mock_tok_info)

    # GrammarCompiler
    mock_compiled = MagicMock()
    mock_compiler = MagicMock()
    mock_compiler.compile_json_schema.return_value = mock_compiled
    mock_compiler.compile_builtin_json_grammar.return_value = mock_compiled
    mock_compiler.compile_regex.return_value = mock_compiled
    xgr.GrammarCompiler = MagicMock(return_value=mock_compiler)

    # GrammarMatcher
    mock_matcher = MagicMock()
    mock_matcher.find_jump_forward_string.return_value = ""
    xgr.GrammarMatcher = MagicMock(return_value=mock_matcher)

    # Bitmask functions
    mock_bitmask = np.ones(32000, dtype=np.float32)  # allow everything
    xgr.allocate_token_bitmask = MagicMock(return_value=mock_bitmask)
    xgr.apply_token_bitmask_inplace = MagicMock()

    return xgr, mock_tok_info, mock_compiler, mock_compiled, mock_matcher


# ══════════════════════════════════════════════════════════════════════════════
# GrammarEngine.__init__
# ══════════════════════════════════════════════════════════════════════════════

class TestGrammarEngineInit:
    def test_init_fallback_when_xgrammar_not_installed(self):
        """No xgrammar → _available=False, no attributes set."""
        # xgrammar is NOT installed in this environment; just instantiate.
        tok = _make_mock_tokenizer()
        engine = GrammarEngine(tok)
        assert engine._available is False
        assert engine._xgr is None
        assert engine._compiler is None
        assert engine._tok_info is None

    def test_init_available_when_xgrammar_importable(self):
        """When xgrammar is present, _available=True and objects built."""
        xgr, tok_info, compiler, compiled, matcher = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        assert engine._available is True
        assert engine._xgr is xgr
        assert engine._tok_info is tok_info
        assert engine._compiler is compiler

    def test_init_fallback_when_tokenizer_info_raises(self):
        """TokenizerInfo.from_huggingface raises → _available=False."""
        xgr, *_ = _make_xgrammar_mock()
        xgr.TokenizerInfo.from_huggingface.side_effect = RuntimeError("bad tokenizer")
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        assert engine._available is False


# ══════════════════════════════════════════════════════════════════════════════
# GrammarEngine.is_available
# ══════════════════════════════════════════════════════════════════════════════

class TestGrammarEngineIsAvailable:
    def test_is_available_false_when_not_installed(self):
        """xgrammar absent from sys.modules → False."""
        # Ensure xgrammar is absent (it's not installed, but be explicit)
        modules = {k: v for k, v in sys.modules.items() if k != "xgrammar"}
        with patch.dict(sys.modules, modules, clear=True):
            # Remove xgrammar if it somehow made it in
            sys.modules.pop("xgrammar", None)
            result = GrammarEngine.is_available()
        assert result is False

    def test_is_available_true_when_installed(self):
        """xgrammar present in sys.modules → True."""
        xgr = types.ModuleType("xgrammar")
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            result = GrammarEngine.is_available()
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# Grammar construction methods
# ══════════════════════════════════════════════════════════════════════════════

class TestGrammarConstruction:
    """json_schema_grammar, json_object_grammar, regex_grammar."""

    # ── json_schema_grammar ──────────────────────────────────────────────────

    def test_json_schema_grammar_fallback_returns_none(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        assert engine._available is False
        result = engine.json_schema_grammar({"type": "object"})
        assert result is None

    def test_json_schema_grammar_available_returns_matcher(self):
        xgr, _, compiler, compiled, matcher = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        result = engine.json_schema_grammar(schema)
        compiler.compile_json_schema.assert_called_once_with(json.dumps(schema))
        assert result is matcher

    # ── json_object_grammar ─────────────────────────────────────────────────

    def test_json_object_grammar_fallback_returns_none(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        result = engine.json_object_grammar()
        assert result is None

    def test_json_object_grammar_available_returns_matcher(self):
        xgr, _, compiler, compiled, matcher = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        result = engine.json_object_grammar()
        compiler.compile_builtin_json_grammar.assert_called_once()
        assert result is matcher

    # ── regex_grammar ────────────────────────────────────────────────────────

    def test_regex_grammar_fallback_returns_none(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        result = engine.regex_grammar(r"\d+")
        assert result is None

    def test_regex_grammar_available_returns_matcher(self):
        xgr, _, compiler, compiled, matcher = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        result = engine.regex_grammar(r"\d+")
        compiler.compile_regex.assert_called_once_with(r"\d+")
        assert result is matcher


# ══════════════════════════════════════════════════════════════════════════════
# GrammarEngine.constrain_logits
# ══════════════════════════════════════════════════════════════════════════════

class TestConstrainLogits:
    def _logits(self, n=32000):
        return mx.zeros([n])

    def test_fallback_engine_returns_logits_unchanged(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        logits = self._logits()
        result = engine.constrain_logits(logits, state=MagicMock())
        assert result is logits

    def test_none_state_returns_logits_unchanged(self):
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        logits = self._logits()
        result = engine.constrain_logits(logits, state=None)
        assert result is logits

    def test_available_engine_calls_bitmask_functions(self):
        """With valid state, fills + applies bitmask and returns new mx.array."""
        xgr, tok_info, compiler, compiled, matcher = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        # Real bitmask that apply_token_bitmask_inplace doesn't mutate (mock)
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        logits = self._logits()
        state = MagicMock()
        result = engine.constrain_logits(logits, state=state)
        # Should have called the xgrammar bitmask pipeline
        xgr.allocate_token_bitmask.assert_called_once_with(1, tok_info.vocab_size)
        state.fill_next_token_bitmask.assert_called_once()
        xgr.apply_token_bitmask_inplace.assert_called_once()
        # Result should be an mx.array (conversion back from numpy)
        assert isinstance(result, mx.array)

    def test_exception_returns_logits_unchanged(self):
        """If bitmask machinery raises, return original logits unmodified."""
        xgr, _, _, _, _ = _make_xgrammar_mock()
        xgr.allocate_token_bitmask.side_effect = RuntimeError("gpu error")
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        logits = self._logits()
        state = MagicMock()
        result = engine.constrain_logits(logits, state=state)
        # Must return the original mx.array on failure
        assert result is logits


# ══════════════════════════════════════════════════════════════════════════════
# GrammarEngine.advance
# ══════════════════════════════════════════════════════════════════════════════

class TestAdvance:
    def test_fallback_engine_returns_state_unchanged(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        state = MagicMock()
        result = engine.advance(state, 42)
        assert result is state

    def test_none_state_returns_none(self):
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        result = engine.advance(None, 42)
        assert result is None

    def test_available_engine_calls_accept_token(self):
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        state = MagicMock()
        result = engine.advance(state, 99)
        state.accept_token.assert_called_once_with(99)
        assert result is state

    def test_accept_token_exception_returns_state(self):
        """accept_token raises → state returned anyway, no crash."""
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        state = MagicMock()
        state.accept_token.side_effect = RuntimeError("invalid token")
        result = engine.advance(state, 7)
        assert result is state


# ══════════════════════════════════════════════════════════════════════════════
# GrammarEngine.jump_forward_tokens
# ══════════════════════════════════════════════════════════════════════════════

class TestJumpForwardTokens:
    def test_fallback_engine_returns_empty_list(self):
        engine = GrammarEngine(_make_mock_tokenizer())
        assert engine.jump_forward_tokens(MagicMock()) == []

    def test_none_state_returns_empty_list(self):
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        assert engine.jump_forward_tokens(None) == []

    def test_empty_jump_string_returns_empty_list(self):
        """find_jump_forward_string returns '' → []."""
        xgr, _, _, _, matcher = _make_xgrammar_mock()
        matcher.find_jump_forward_string.return_value = ""
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        state = MagicMock()
        state.find_jump_forward_string.return_value = ""
        # Use the real matcher (wired from _make_xgrammar_mock)
        result = engine.jump_forward_tokens(matcher)
        assert result == []

    def test_nonempty_jump_string_returns_token_ids(self):
        """find_jump_forward_string returns '{' → tokenizer.encode called, IDs returned."""
        xgr, _, _, _, matcher = _make_xgrammar_mock()
        matcher.find_jump_forward_string.return_value = '{"'
        tok = _make_mock_tokenizer()
        tok.encode.return_value = [4, 5, 6]
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        result = engine.jump_forward_tokens(matcher)
        tok.encode.assert_called_once_with('{"', add_special_tokens=False)
        assert result == [4, 5, 6]

    def test_exception_returns_empty_list(self):
        """find_jump_forward_string raises → [] returned, no crash."""
        xgr, _, _, _, matcher = _make_xgrammar_mock()
        matcher.find_jump_forward_string.side_effect = RuntimeError("fsm error")
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
        result = engine.jump_forward_tokens(matcher)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# parse_tool_calls_with_grammar (tool_calling.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseToolCallsWithGrammar:
    """Tests for the grammar-assisted tool-call parser."""

    _VALID_TOOL_JSON = '{"name": "get_weather", "arguments": {"city": "NYC"}}'

    def test_none_grammar_engine_falls_back_to_heuristic(self):
        """grammar_engine=None → behaves exactly like parse_tool_calls."""
        result = parse_tool_calls_with_grammar(self._VALID_TOOL_JSON, grammar_engine=None)
        assert result is not None
        assert result[0]["name"] == "get_weather"

    def test_none_grammar_engine_returns_none_for_plain_text(self):
        result = parse_tool_calls_with_grammar("Hello world", grammar_engine=None)
        assert result is None

    def test_grammar_engine_not_available_falls_back(self):
        """is_available() returns False → heuristic path used."""
        engine = GrammarEngine(_make_mock_tokenizer())
        # xgrammar not installed so engine is in fallback mode
        assert not engine.is_available()
        result = parse_tool_calls_with_grammar(self._VALID_TOOL_JSON, grammar_engine=engine)
        assert result is not None
        assert result[0]["name"] == "get_weather"

    def test_grammar_engine_available_direct_parse_succeeds(self):
        """is_available() True + valid JSON tool call → direct json.loads parse."""
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
            assert engine.is_available()
            result = parse_tool_calls_with_grammar(self._VALID_TOOL_JSON, grammar_engine=engine)
        assert result is not None
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["city"] == "NYC"

    def test_grammar_engine_available_invalid_json_heuristic_fallback(self):
        """Grammar available but text is invalid JSON → heuristic fallback, still finds call."""
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
            # Embed valid tool JSON in surrounding prose (heuristic path needed)
            text = "Here is the result: " + self._VALID_TOOL_JSON + " done."
            result = parse_tool_calls_with_grammar(text, grammar_engine=engine)
        assert result is not None
        assert result[0]["name"] == "get_weather"

    def test_grammar_engine_available_valid_json_not_tool_call(self):
        """Grammar available, JSON parses but doesn't look like tool call → heuristic."""
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
            # Valid JSON but missing 'name'/'arguments' keys
            text = '{"result": "ok", "value": 42}'
            result = parse_tool_calls_with_grammar(text, grammar_engine=engine)
        assert result is None

    def test_grammar_engine_available_array_of_tool_calls(self):
        """Grammar available + JSON array of tool calls → direct parse returns list."""
        xgr, *_ = _make_xgrammar_mock()
        tok = _make_mock_tokenizer()
        with patch.dict(sys.modules, {"xgrammar": xgr}):
            engine = GrammarEngine(tok)
            text = json.dumps([
                {"name": "func_a", "arguments": {}},
                {"name": "func_b", "arguments": {"x": 1}},
            ])
            result = parse_tool_calls_with_grammar(text, grammar_engine=engine)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "func_a"
        assert result[1]["name"] == "func_b"
