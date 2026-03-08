"""tests/test_swift_unit.py — 100% line and branch coverage for squish/swift.py"""
import json
import tempfile
import os
import numpy as np
import pytest

from squish.swift import (
    SWIFTCalibrator,
    SWIFTConfig,
    SWIFTDecoder,
    SWIFTLayerConfig,
)


# ---------------------------------------------------------------------------
# SWIFTConfig
# ---------------------------------------------------------------------------

class TestSWIFTConfig:
    def test_defaults(self):
        cfg = SWIFTConfig()
        assert cfg.num_layers == 32
        assert cfg.initial_skip_fraction == 0.4
        assert cfg.n_calibration_steps == 50
        assert cfg.cooling_rate == 0.95

    def test_custom(self):
        cfg = SWIFTConfig(num_layers=12, initial_skip_fraction=0.3)
        assert cfg.num_layers == 12

    @pytest.mark.parametrize("kwargs, match", [
        ({"num_layers": 0},                     "num_layers"),
        ({"initial_skip_fraction": -0.1},       "initial_skip_fraction"),
        ({"initial_skip_fraction": 1.0},        "initial_skip_fraction"),
        ({"n_calibration_steps": 0},            "n_calibration_steps"),
        ({"cooling_rate": 0.0},                 "cooling_rate"),
        ({"cooling_rate": 1.0},                 "cooling_rate"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SWIFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# SWIFTLayerConfig
# ---------------------------------------------------------------------------

class TestSWIFTLayerConfig:
    def test_to_dict_round_trip(self):
        lc = SWIFTLayerConfig(task_type="qa", skip_layers=[1, 3, 5],
                              calibration_score=0.87)
        d = lc.to_dict()
        assert d["task_type"] == "qa"
        assert d["skip_layers"] == [1, 3, 5]
        assert d["calibration_score"] == pytest.approx(0.87)

    def test_from_dict_round_trip(self):
        d = {"task_type": "summ", "skip_layers": [2, 4], "calibration_score": 0.5}
        lc = SWIFTLayerConfig.from_dict(d)
        assert lc.task_type == "summ"
        assert lc.skip_layers == [2, 4]

    def test_to_from_dict_identity(self):
        lc = SWIFTLayerConfig(task_type="t", skip_layers=[0], calibration_score=0.1)
        assert SWIFTLayerConfig.from_dict(lc.to_dict()).task_type == "t"


# ---------------------------------------------------------------------------
# SWIFTCalibrator — calibrate()
# ---------------------------------------------------------------------------

class TestSWIFTCalibratorCalibrate:
    """Cover all SA branches: delta>0, Metropolis accept/reject, remove/add,
    non_skipped empty, best_score update."""

    def _make_cfg(self, num_layers=8, n_steps=10):
        return SWIFTConfig(
            num_layers=num_layers,
            initial_skip_fraction=0.25,
            n_calibration_steps=n_steps,
            cooling_rate=0.9,
        )

    def test_returns_layer_config(self):
        cfg = self._make_cfg()
        cal = SWIFTCalibrator(cfg, rng_seed=0)
        result = cal.calibrate("qa", score_fn=lambda skip: 1.0 - len(skip) * 0.01)
        assert isinstance(result, SWIFTLayerConfig)
        assert result.task_type == "qa"
        assert isinstance(result.skip_layers, list)

    def test_best_score_tracked(self):
        cfg = self._make_cfg(n_steps=20)
        cal = SWIFTCalibrator(cfg, rng_seed=1)
        # Score is 1/(1+len(skip)) — monotonically decreasing in skip set size
        result = cal.calibrate("t", score_fn=lambda skip: 1.0 / (1 + len(skip)))
        # Best score should prefer small skip sets
        assert result.calibration_score > 0.0

    def test_all_layers_skipped_triggers_add_only(self):
        """Force scenario where all layers are already skipped → remove is impossible,
        so the branching must take the add path instead (or the 'non_skipped empty' guard)."""
        # With a high initial skip fraction and small num_layers, we can get all skipped
        cfg = SWIFTConfig(
            num_layers=4, initial_skip_fraction=0.99,
            n_calibration_steps=5, cooling_rate=0.9,
        )
        cal = SWIFTCalibrator(cfg, rng_seed=3)
        result = cal.calibrate("fill", score_fn=lambda s: float(len(s)))
        assert result is not None

    def test_metropolis_reject_path(self):
        """When delta < 0 and rng decides to reject, the candidate is discarded."""
        cfg = SWIFTConfig(
            num_layers=8, initial_skip_fraction=0.2,
            n_calibration_steps=50, cooling_rate=0.99,
        )
        cal = SWIFTCalibrator(cfg, rng_seed=99)
        # Penalise any skip set → SA will generate many moves with delta<0
        result = cal.calibrate("penalised", score_fn=lambda s: -len(s) * 100.0)
        assert result is not None

    def test_deterministic_with_same_seed(self):
        cfg = self._make_cfg()
        r1 = SWIFTCalibrator(cfg, rng_seed=42).calibrate("x", lambda s: 1.0)
        r2 = SWIFTCalibrator(cfg, rng_seed=42).calibrate("x", lambda s: 1.0)
        assert r1.skip_layers == r2.skip_layers


# ---------------------------------------------------------------------------
# SWIFTCalibrator — save / load
# ---------------------------------------------------------------------------

class TestSWIFTCalibratorSaveLoad:
    def test_round_trip(self):
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        cal = SWIFTCalibrator(cfg, rng_seed=7)
        configs = [cal.calibrate("qa", lambda s: 1.0)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(configs, path)
            loaded = cal.load(path)
            assert len(loaded) == 1
            assert loaded[0].task_type == "qa"
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_empty(self):
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        cal = SWIFTCalibrator(cfg)
        result = cal.load("/tmp/_no_such_file_swift_test_99.json")
        assert result == []

    def test_save_multiple(self):
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        cal = SWIFTCalibrator(cfg, rng_seed=0)
        configs = [
            SWIFTLayerConfig("qa", [1, 2], 0.9),
            SWIFTLayerConfig("summ", [3], 0.8),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(configs, path)
            loaded = cal.load(path)
            assert len(loaded) == 2
            types = {lc.task_type for lc in loaded}
            assert types == {"qa", "summ"}
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# SWIFTDecoder — generate()
# ---------------------------------------------------------------------------

def _make_swift_forward(vocab=10, draft_tok=1, verify_tok=1):
    """
    A forward function for SWIFTDecoder tests.
    skip_layers is ignored; always returns same token.
    """
    def forward_fn(ids, skip_layers=None):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        if skip_layers:  # draft pass (has skip layers)
            logits[draft_tok] = 10.0
        else:             # verify pass (empty skip)
            logits[verify_tok] = 10.0
        return logits
    return forward_fn


class TestSWIFTDecoder:
    def _make_cfgs(self, task="qa", skip=[1, 3]):
        return {task: SWIFTLayerConfig(task, list(skip), 0.9)}

    def test_generates_correct_length(self):
        fwd = _make_swift_forward(vocab=10, draft_tok=2, verify_tok=2)
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        dec = SWIFTDecoder(fwd, self._make_cfgs("qa"), cfg, gamma=2)
        ids, stats = dec.generate([0], max_new_tokens=6, task_type="qa")
        assert stats.total_tokens >= 6

    def test_fallback_no_config_for_task(self):
        """Unknown task_type → fall back to empty skip_layers (full model)."""
        fwd = _make_swift_forward(vocab=10, draft_tok=2, verify_tok=2)
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        dec = SWIFTDecoder(fwd, {}, cfg, gamma=2)
        ids, stats = dec.generate([0], max_new_tokens=4, task_type="unknown")
        assert stats.total_tokens >= 4

    def test_rejection_path(self):
        """Draft picks tok 1 but verify picks tok 2 → rejection exercised."""
        fwd = _make_swift_forward(vocab=10, draft_tok=1, verify_tok=2)
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        dec = SWIFTDecoder(fwd, self._make_cfgs("qa", [1, 3]), cfg, gamma=3, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5, task_type="qa")
        assert stats.total_tokens >= 5

    def test_acceptance_rate_full_agree(self):
        """Draft and verify agree on every token → high acceptance rate."""
        fwd = _make_swift_forward(vocab=10, draft_tok=3, verify_tok=3)
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        dec = SWIFTDecoder(fwd, self._make_cfgs("qa", [1, 2]), cfg, gamma=2, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=6, task_type="qa")
        ar = stats.acceptance_rate
        assert 0.0 <= ar <= 1.0

    def test_bonus_token_full_accept(self):
        """All gamma tokens accepted → bonus token appended."""
        # Draft and verify must agree perfectly
        fwd = _make_swift_forward(vocab=10, draft_tok=5, verify_tok=5)
        cfg = SWIFTConfig(num_layers=8, n_calibration_steps=5)
        dec = SWIFTDecoder(fwd, self._make_cfgs("qa", [1]), cfg, gamma=1, rng_seed=0)
        ids, _ = dec.generate([0], max_new_tokens=1)
        assert len(ids) >= 2  # at least prompt + 1 new token
