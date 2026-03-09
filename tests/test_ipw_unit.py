"""tests/test_ipw_unit.py — unit tests for squish.ipw"""

import numpy as np
import pytest

from squish.ipw import (
    IPWConfig,
    IPWMeasurement,
    IPWSummary,
    IPWTracker,
)

# ---------------------------------------------------------------------------
# IPWConfig
# ---------------------------------------------------------------------------

class TestIPWConfig:
    def test_defaults(self):
        cfg = IPWConfig()
        assert cfg.energy_unit == "mJ"
        assert cfg.quality_weight == 1.0
        assert cfg.min_energy_mj > 0

    def test_custom(self):
        cfg = IPWConfig(energy_unit="J", quality_weight=2.0)
        assert cfg.energy_unit == "J"

    @pytest.mark.parametrize("field,val", [
        ("energy_unit", "kJ"),
        ("quality_weight", 0.0),
        ("min_energy_mj", -1.0),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            IPWConfig(**{field: val})


# ---------------------------------------------------------------------------
# IPWMeasurement
# ---------------------------------------------------------------------------

class TestIPWMeasurement:
    def _m(self, quality=0.8, energy=10.0, time=200.0, tokens=50,
           task="code", config="q8b"):
        return IPWMeasurement(
            quality_score=quality,
            energy_mj=energy,
            time_ms=time,
            tokens_generated=tokens,
            task_type=task,
            config_label=config,
        )

    def test_ipw_property(self):
        m = self._m(quality=0.4, energy=4.0)
        assert abs(m.ipw - 0.1) < 1e-6

    def test_tokens_per_second(self):
        m = self._m(tokens=100, time=1000.0)
        assert abs(m.tokens_per_second - 100.0) < 1e-4

    def test_tokens_per_joule(self):
        m = self._m(tokens=100, energy=1000.0)  # 1000mJ = 1J
        assert abs(m.tokens_per_joule - 100.0) < 1e-4

    def test_zero_time_tps(self):
        m = self._m(time=0.0)
        assert m.tokens_per_second == 0.0

    def test_zero_energy_tpj(self):
        m = self._m(energy=0.0)
        assert m.tokens_per_joule == 0.0

    @pytest.mark.parametrize("field,val", [
        ("quality_score", -0.1),
        ("quality_score", 1.1),
        ("energy_mj", -1.0),
        ("time_ms", -1.0),
        ("tokens_generated", -1),
    ])
    def test_invalid(self, field, val):
        kw = dict(quality_score=0.5, energy_mj=5.0, time_ms=100.0, tokens_generated=20)
        kw[field] = val
        with pytest.raises(ValueError):
            IPWMeasurement(**kw)


# ---------------------------------------------------------------------------
# IPWTracker
# ---------------------------------------------------------------------------

class TestIPWTracker:
    def _tracker(self):
        return IPWTracker(IPWConfig(min_energy_mj=0.1))

    def test_empty_summary(self):
        t = self._tracker()
        s = t.summary()
        assert s.count == 0
        assert s.mean_ipw == 0.0

    def test_record_and_count(self):
        t = self._tracker()
        m = IPWMeasurement(0.9, 5.0, 100.0, 50)
        t.record(m)
        assert t.total_measurements == 1

    def test_record_values(self):
        t = self._tracker()
        m = t.record_values(0.8, 10.0, 200.0, 80, task_type="git")
        assert isinstance(m, IPWMeasurement)
        assert t.total_measurements == 1

    def test_summary_mean_ipw(self):
        t = self._tracker()
        t.record_values(1.0, 10.0, 100.0, 50)
        t.record_values(1.0, 20.0, 100.0, 50)
        s = t.summary()
        assert s.mean_ipw > 0.0

    def test_summary_by_task(self):
        t = self._tracker()
        t.record_values(0.9, 5.0, 100.0, 30, task_type="commit")
        t.record_values(0.7, 15.0, 500.0, 200, task_type="plan")
        per_task = t.summary_by_task()
        assert "commit" in per_task
        assert "plan" in per_task

    def test_reset(self):
        t = self._tracker()
        t.record_values(0.5, 1.0, 50.0, 10)
        t.reset()
        assert t.total_measurements == 0


# ---------------------------------------------------------------------------
# IPWSummary
# ---------------------------------------------------------------------------

class TestIPWSummary:
    def _measurements(self, n=5):
        rng = np.random.default_rng(3)
        ms = []
        for _ in range(n):
            ms.append(IPWMeasurement(
                quality_score=float(rng.uniform(0.5, 1.0)),
                energy_mj=float(rng.uniform(1.0, 20.0)),
                time_ms=float(rng.uniform(50.0, 500.0)),
                tokens_generated=int(rng.integers(10, 200)),
            ))
        return ms

    def test_from_empty(self):
        s = IPWSummary.from_measurements([])
        assert s.count == 0
        assert s.mean_ipw == 0.0

    def test_count(self):
        ms = self._measurements(6)
        s = IPWSummary.from_measurements(ms)
        assert s.count == 6

    def test_mean_ipw_positive(self):
        ms = self._measurements(5)
        s = IPWSummary.from_measurements(ms)
        assert s.mean_ipw > 0.0

    def test_median_le_p90(self):
        ms = self._measurements(20)
        s = IPWSummary.from_measurements(ms)
        assert s.median_ipw <= s.p90_ipw + 1e-6

    def test_best_config_selected(self):
        ms = [
            IPWMeasurement(0.9, 1.0, 100.0, 50, config_label="fast"),
            IPWMeasurement(0.1, 10.0, 100.0, 50, config_label="slow"),
        ]
        s = IPWSummary.from_measurements(ms)
        assert s.best_config == "fast"
