#!/usr/bin/env python3
"""
tests/test_neuron_profile_unit.py

Unit tests for squish/neuron_profile.py — Phase 10A offline profiling.

Coverage targets
────────────────
NeuronProfileConfig
  - valid defaults
  - n_calib_samples < 1 raises
  - hot_fraction <= 0 raises
  - hot_fraction >= 1 raises

NeuronProfiler
  - calibrate returns NeuronProfile with correct layer count
  - hot indices are top hot_fraction by activation count
  - cold indices are the rest
  - hot + cold indices cover all neurons (no overlap, no gaps)
  - hot indices sorted by descending activation frequency
  - empty act_counts list raises ValueError
  - single-layer profile: hot_fraction=0.5 gives ~half each
  - very small layer (2 neurons) still produces at least 1 hot neuron

NeuronProfile
  - layer_count property
  - n_hot / n_cold for valid layer
  - _check_layer raises IndexError on out-of-bounds
  - to_dict / from_dict round-trip
  - save / load round-trip (using tmp_path fixture)
  - __repr__ contains expected fields

load_profile
  - load_profile delegates to NeuronProfile.load
  - missing file raises FileNotFoundError
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from squish.neuron_profile import (
    NeuronProfile,
    NeuronProfileConfig,
    NeuronProfiler,
    load_profile,
)

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# NeuronProfileConfig
# ---------------------------------------------------------------------------

class TestNeuronProfileConfig:
    def test_valid_defaults(self):
        cfg = NeuronProfileConfig()
        assert cfg.n_calib_samples == 512
        assert cfg.hot_fraction == pytest.approx(0.20)
        assert cfg.save_path == ""

    def test_zero_calib_samples_raises(self):
        with pytest.raises(ValueError, match="n_calib_samples"):
            NeuronProfileConfig(n_calib_samples=0)

    def test_negative_calib_samples_raises(self):
        with pytest.raises(ValueError, match="n_calib_samples"):
            NeuronProfileConfig(n_calib_samples=-1)

    def test_hot_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="hot_fraction"):
            NeuronProfileConfig(hot_fraction=0.0)

    def test_hot_fraction_one_raises(self):
        with pytest.raises(ValueError, match="hot_fraction"):
            NeuronProfileConfig(hot_fraction=1.0)

    def test_hot_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="hot_fraction"):
            NeuronProfileConfig(hot_fraction=1.5)

    def test_custom_valid_config(self):
        cfg = NeuronProfileConfig(n_calib_samples=64, hot_fraction=0.30, save_path="/tmp/p.json")
        assert cfg.n_calib_samples == 64
        assert cfg.hot_fraction == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# NeuronProfiler + NeuronProfile creation
# ---------------------------------------------------------------------------

def _make_counts(n_layers=3, ffn_dim=16):
    """Return random per-neuron activation counts for each layer."""
    return [RNG.integers(0, 100, size=ffn_dim).astype(np.float32) for _ in range(n_layers)]


class TestNeuronProfiler:
    def test_calibrate_layer_count(self):
        profiler = NeuronProfiler()
        counts = _make_counts(n_layers=4)
        profile = profiler.calibrate(counts)
        assert profile.layer_count == 4

    def test_hot_fraction_correct(self):
        profiler = NeuronProfiler(NeuronProfileConfig(hot_fraction=0.25))
        counts = _make_counts(n_layers=1, ffn_dim=100)
        profile = profiler.calibrate(counts)
        expected_hot = max(1, round(100 * 0.25))
        assert profile.n_hot(0) == expected_hot

    def test_cold_count_is_complement(self):
        profiler = NeuronProfiler()
        counts = _make_counts(n_layers=1, ffn_dim=64)
        profile = profiler.calibrate(counts)
        assert profile.n_hot(0) + profile.n_cold(0) == 64

    def test_no_overlap_between_hot_and_cold(self):
        profiler = NeuronProfiler()
        counts = _make_counts(n_layers=1, ffn_dim=32)
        profile = profiler.calibrate(counts)
        hot_set  = set(profile.hot_indices[0].tolist())
        cold_set = set(profile.cold_indices[0].tolist())
        assert hot_set.isdisjoint(cold_set)

    def test_hot_and_cold_cover_all_indices(self):
        ffn_dim = 40
        profiler = NeuronProfiler()
        counts = _make_counts(n_layers=1, ffn_dim=ffn_dim)
        profile = profiler.calibrate(counts)
        all_idx = sorted(
            profile.hot_indices[0].tolist() + profile.cold_indices[0].tolist()
        )
        assert all_idx == list(range(ffn_dim))

    def test_hot_indices_have_highest_activation_counts(self):
        ffn_dim = 20
        counts_arr = np.arange(ffn_dim, dtype=np.float32)  # index 19 = highest
        profiler = NeuronProfiler(NeuronProfileConfig(hot_fraction=0.20))
        profile = profiler.calibrate([counts_arr])
        n_hot = profile.n_hot(0)
        # hot indices should correspond to the n_hot largest values (16-19)
        expected_hot = set(range(ffn_dim - n_hot, ffn_dim))
        assert set(profile.hot_indices[0].tolist()) == expected_hot

    def test_empty_act_counts_raises(self):
        profiler = NeuronProfiler()
        with pytest.raises(ValueError):
            profiler.calibrate([])

    def test_hot_fraction_half(self):
        profiler = NeuronProfiler(NeuronProfileConfig(hot_fraction=0.50))
        counts = _make_counts(n_layers=1, ffn_dim=10)
        profile = profiler.calibrate(counts)
        assert profile.n_hot(0) == 5
        assert profile.n_cold(0) == 5

    def test_small_layer_at_least_one_hot(self):
        profiler = NeuronProfiler(NeuronProfileConfig(hot_fraction=0.10))
        counts = [np.array([5.0, 1.0], dtype=np.float32)]  # 2 neurons
        profile = profiler.calibrate(counts)
        assert profile.n_hot(0) >= 1

    def test_indices_dtype_int64(self):
        profiler = NeuronProfiler()
        counts = _make_counts(n_layers=1, ffn_dim=8)
        profile = profiler.calibrate(counts)
        assert profile.hot_indices[0].dtype == np.int64
        assert profile.cold_indices[0].dtype == np.int64


# ---------------------------------------------------------------------------
# NeuronProfile properties and validation
# ---------------------------------------------------------------------------

class TestNeuronProfile:
    def _make_profile(self, n_layers=2, ffn_dim=16):
        profiler = NeuronProfiler()
        return profiler.calibrate(_make_counts(n_layers=n_layers, ffn_dim=ffn_dim))

    def test_layer_count(self):
        p = self._make_profile(n_layers=5)
        assert p.layer_count == 5

    def test_n_hot_n_cold(self):
        p = self._make_profile(n_layers=1, ffn_dim=20)
        assert p.n_hot(0) + p.n_cold(0) == 20

    def test_check_layer_negative_raises(self):
        p = self._make_profile()
        with pytest.raises(IndexError):
            p._check_layer(-1)

    def test_check_layer_too_large_raises(self):
        p = self._make_profile(n_layers=2)
        with pytest.raises(IndexError):
            p._check_layer(2)

    def test_to_dict_from_dict_round_trip(self):
        p = self._make_profile(n_layers=3, ffn_dim=12)
        d = p.to_dict()
        p2 = NeuronProfile.from_dict(d)
        assert p2.layer_count == 3
        assert p2.hot_fraction == pytest.approx(p.hot_fraction)
        for i in range(3):
            assert np.array_equal(p2.hot_indices[i], p.hot_indices[i])
            assert np.array_equal(p2.cold_indices[i], p.cold_indices[i])

    def test_save_load_round_trip(self, tmp_path):
        p = self._make_profile(n_layers=2, ffn_dim=8)
        path = str(tmp_path / "sub" / "profile.json")
        p.save(path)
        p2 = NeuronProfile.load(path)
        assert p2.layer_count == 2
        for i in range(2):
            assert np.array_equal(p2.hot_indices[i], p.hot_indices[i])

    def test_repr_contains_key_fields(self):
        p = self._make_profile()
        r = repr(p)
        assert "NeuronProfile" in r
        assert "layers=" in r

    def test_empty_profile_repr(self):
        p = NeuronProfile()
        assert "layers=0" in repr(p)


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_load_profile_returns_correct_type(self, tmp_path):
        profiler = NeuronProfiler()
        p = profiler.calibrate(_make_counts(n_layers=1, ffn_dim=8))
        path = str(tmp_path / "p.json")
        p.save(path)
        p2 = load_profile(path)
        assert isinstance(p2, NeuronProfile)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_profile("/nonexistent/path/to/profile.json")
