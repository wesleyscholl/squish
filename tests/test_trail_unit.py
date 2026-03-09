"""tests/test_trail_unit.py — unit tests for squish.trail"""

import numpy as np
import pytest
import tempfile
import os

from squish.trail import (
    TrailConfig,
    TrailLinearProbe,
    TrailPredictor,
    TrailStats,
)

HIDDEN = 8  # small dim for tests
RNG = np.random.default_rng(7)


def _make_probe(n: int = 20, hidden: int = HIDDEN, max_length: int = 256):
    cfg = TrailConfig(probe_layer=3, hidden_dim=hidden, max_length=max_length)
    probe = TrailLinearProbe(cfg)
    X = RNG.standard_normal((n, hidden))
    y = RNG.integers(1, max_length, size=n).astype(float)
    probe.fit(X, y)
    return probe, cfg


# ---------------------------------------------------------------------------
# TrailConfig
# ---------------------------------------------------------------------------

class TestTrailConfig:
    def test_defaults(self):
        cfg = TrailConfig()
        assert cfg.probe_layer == 11
        assert cfg.hidden_dim == 4096
        assert cfg.max_length == 8192
        assert cfg.n_buckets == 8

    def test_custom(self):
        cfg = TrailConfig(probe_layer=5, hidden_dim=512)
        assert cfg.probe_layer == 5

    @pytest.mark.parametrize("field,val", [
        ("probe_layer", -1),
        ("hidden_dim", 0),
        ("max_length", 0),
        ("n_buckets", 1),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            TrailConfig(**{field: val})


# ---------------------------------------------------------------------------
# TrailLinearProbe
# ---------------------------------------------------------------------------

class TestTrailLinearProbe:
    def test_not_fitted_initially(self):
        probe = TrailLinearProbe()
        assert not probe.is_fitted

    def test_heuristic_before_fit(self):
        cfg = TrailConfig(hidden_dim=HIDDEN, max_length=128)
        probe = TrailLinearProbe(cfg)
        emb = RNG.standard_normal(HIDDEN)
        result = probe.predict(emb)
        assert 1 <= result <= 128

    def test_fit_is_fitted(self):
        probe, _ = _make_probe()
        assert probe.is_fitted

    def test_predict_range(self):
        probe, cfg = _make_probe()
        for _ in range(10):
            emb = RNG.standard_normal(HIDDEN)
            result = probe.predict(emb)
            assert 1 <= result <= cfg.max_length

    def test_fit_row_mismatch(self):
        cfg = TrailConfig(hidden_dim=HIDDEN)
        probe = TrailLinearProbe(cfg)
        X = RNG.standard_normal((10, HIDDEN))
        y = np.ones(8)
        with pytest.raises(ValueError):
            probe.fit(X, y)

    def test_fit_dim_mismatch(self):
        cfg = TrailConfig(hidden_dim=HIDDEN)
        probe = TrailLinearProbe(cfg)
        X = RNG.standard_normal((10, HIDDEN + 2))
        y = np.ones(10)
        with pytest.raises(ValueError):
            probe.fit(X, y)

    def test_predict_dim_mismatch(self):
        probe, cfg = _make_probe()
        wrong_emb = RNG.standard_normal(HIDDEN + 1)
        with pytest.raises(ValueError):
            probe.predict(wrong_emb)

    def test_save_load(self):
        probe, _ = _make_probe()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            probe.save(path)
            cfg = TrailConfig(hidden_dim=HIDDEN, max_length=256)
            probe2 = TrailLinearProbe(cfg)
            probe2.load(path)
            emb = RNG.standard_normal(HIDDEN)
            assert probe.predict(emb) == probe2.predict(emb)
        finally:
            os.unlink(path)

    def test_save_unfitted_raises(self):
        probe = TrailLinearProbe()
        with pytest.raises(RuntimeError):
            probe.save("/tmp/squish_test_trail.npz")


# ---------------------------------------------------------------------------
# TrailPredictor
# ---------------------------------------------------------------------------

class TestTrailPredictor:
    def test_predict_range(self):
        cfg = TrailConfig(hidden_dim=HIDDEN, max_length=128)
        pred = TrailPredictor(cfg)
        X = RNG.standard_normal((20, HIDDEN))
        y = RNG.integers(1, 128, size=20).astype(float)
        pred.probe.fit(X, y)
        emb = RNG.standard_normal(HIDDEN)
        result = pred.predict(emb)
        assert 1 <= result <= 128

    def test_bucket_range(self):
        cfg = TrailConfig(hidden_dim=HIDDEN, max_length=128, n_buckets=4)
        pred = TrailPredictor(cfg)
        X = RNG.standard_normal((20, HIDDEN))
        y = RNG.integers(1, 128, size=20).astype(float)
        pred.probe.fit(X, y)
        for _ in range(8):
            emb = RNG.standard_normal(HIDDEN)
            b = pred.predict_bucket(emb)
            assert 0 <= b < 4

    def test_srpt_priority_non_negative(self):
        cfg = TrailConfig(hidden_dim=HIDDEN, max_length=64)
        pred = TrailPredictor(cfg)
        emb = RNG.standard_normal(HIDDEN)
        priority = pred.srpt_priority(emb, current_tokens=0)
        assert priority >= 0.0

    def test_srpt_priority_decreases_with_progress(self):
        cfg = TrailConfig(hidden_dim=HIDDEN, max_length=256)
        pred = TrailPredictor(cfg)
        X = RNG.standard_normal((20, HIDDEN))
        y = np.full(20, 200.0)
        pred.probe.fit(X, y)
        emb = RNG.standard_normal(HIDDEN)
        p0 = pred.srpt_priority(emb, 0)
        p50 = pred.srpt_priority(emb, 50)
        assert p50 <= p0


# ---------------------------------------------------------------------------
# TrailStats
# ---------------------------------------------------------------------------

class TestTrailStats:
    def test_defaults(self):
        s = TrailStats()
        assert s.mae == 0.0
        assert s.bucket_accuracy == 0.0

    def test_record(self):
        s = TrailStats()
        s.record(predicted=100, actual=80, predicted_bucket=2, actual_bucket=2)
        assert s.prediction_count == 1
        assert s.bucket_hits == 1
        assert abs(s.mae - 20.0) < 1e-6

    def test_bucket_miss(self):
        s = TrailStats()
        s.record(50, 50, 1, 2)
        assert s.bucket_hits == 0
        assert s.bucket_accuracy == 0.0

    def test_multiple(self):
        s = TrailStats()
        s.record(100, 100, 1, 1)
        s.record(200, 100, 2, 1)
        assert s.prediction_count == 2
        assert abs(s.mae - 50.0) < 1e-6
        assert s.bucket_accuracy == 0.5
