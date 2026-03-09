"""tests/test_forelen_unit.py — unit tests for squish.forelen"""

import numpy as np
import pytest

from squish.forelen import (
    ForelenConfig,
    EGTPPredictor,
    PLPPredictor,
    ForelenStats,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# ForelenConfig
# ---------------------------------------------------------------------------

class TestForelenConfig:
    def test_defaults(self):
        cfg = ForelenConfig()
        assert cfg.entropy_bins == 16
        assert cfg.n_length_buckets == 8
        assert cfg.max_length == 8192
        assert 0.0 <= cfg.plp_decay < 1.0

    def test_custom(self):
        cfg = ForelenConfig(entropy_bins=8, max_length=4096)
        assert cfg.entropy_bins == 8

    @pytest.mark.parametrize("field,val", [
        ("entropy_bins", 1),
        ("n_length_buckets", 1),
        ("max_length", 0),
        ("plp_decay", -0.1),
        ("plp_decay", 1.0),
        ("plp_update_every", 0),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            ForelenConfig(**{field: val})


# ---------------------------------------------------------------------------
# EGTPPredictor
# ---------------------------------------------------------------------------

class TestEGTPPredictor:
    def test_heuristic_prediction_before_fit(self):
        pred = EGTPPredictor()
        entropies = RNG.uniform(0.5, 2.0, size=16)
        result = pred.predict(entropies)
        assert 1 <= result <= pred._cfg.max_length

    def test_not_fitted_initially(self):
        pred = EGTPPredictor()
        assert not pred.is_fitted

    def test_fit_makes_fitted(self):
        cfg = ForelenConfig(entropy_bins=4, max_length=128)
        pred = EGTPPredictor(cfg)
        X = RNG.uniform(0, 1, size=(20, 4))
        X = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        y = RNG.integers(10, 128, size=20).astype(float)
        pred.fit(X, y)
        assert pred.is_fitted

    def test_fit_predict_range(self):
        cfg = ForelenConfig(entropy_bins=4, max_length=256)
        pred = EGTPPredictor(cfg)
        X = RNG.uniform(0, 1, size=(50, 4))
        X = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        y = RNG.integers(1, 256, size=50).astype(float)
        pred.fit(X, y)
        for _ in range(10):
            entropies = RNG.uniform(0, 2, size=20)
            result = pred.predict(entropies)
            assert 1 <= result <= 256

    def test_fit_row_mismatch(self):
        cfg = ForelenConfig(entropy_bins=4)
        pred = EGTPPredictor(cfg)
        X = RNG.uniform(0, 1, size=(10, 4))
        y = np.ones(8)
        with pytest.raises(ValueError):
            pred.fit(X, y)

    def test_fit_col_mismatch(self):
        cfg = ForelenConfig(entropy_bins=4)
        pred = EGTPPredictor(cfg)
        X = RNG.uniform(0, 1, size=(10, 6))  # wrong cols
        y = np.ones(10)
        with pytest.raises(ValueError):
            pred.fit(X, y)

    def test_empty_entropies(self):
        pred = EGTPPredictor()
        result = pred.predict(np.array([]))
        assert result >= 1


# ---------------------------------------------------------------------------
# PLPPredictor
# ---------------------------------------------------------------------------

class TestPLPPredictor:
    def test_initial_estimate(self):
        plp = PLPPredictor(initial_prediction=256)
        assert plp.current_estimate > 0

    def test_update_returns_positive(self):
        plp = PLPPredictor(initial_prediction=512)
        rem = plp.update(current_len=100, step_entropy=1.5)
        assert rem >= 0

    def test_update_decreases_remaining_near_end(self):
        # Use a small max_length so the entropy-based correction term is
        # proportionate: correction = entropy_ratio * (max_length - current_len)
        cfg = ForelenConfig(plp_decay=0.0, plp_update_every=1, max_length=110)
        plp = PLPPredictor(initial_prediction=100, config=cfg)
        rem = plp.update(current_len=99, step_entropy=0.01)
        # With max_length=110 and entropy≈0, correction≈0 → estimate≈99 → remaining=1
        assert rem <= 5  # nearly done; remaining should be very small

    def test_n_updates_increments(self):
        cfg = ForelenConfig(plp_update_every=2)
        plp = PLPPredictor(20, config=cfg)
        plp.update(1, 1.0)
        plp.update(2, 1.0)
        assert plp.n_updates == 1  # fired once at step 2


# ---------------------------------------------------------------------------
# ForelenStats
# ---------------------------------------------------------------------------

class TestForelenStats:
    def test_zero_predictions(self):
        s = ForelenStats()
        assert s.mae == 0.0
        assert s.bucket_accuracy == 0.0

    def test_record(self):
        s = ForelenStats()
        s.record(predicted=100, actual=120)
        assert s.predictions_made == 1
        assert abs(s.total_abs_error - 20) < 1e-6
        assert abs(s.mae - 20.0) < 1e-6

    def test_multiple_records(self):
        s = ForelenStats()
        errors = [10, 20, 30]
        for e in errors:
            s.record(predicted=e, actual=0)
        assert abs(s.mae - 20.0) < 1e-6

    def test_bucket_accuracy(self):
        s = ForelenStats(bucket_hits=7, bucket_total=10)
        assert abs(s.bucket_accuracy - 0.7) < 1e-6
