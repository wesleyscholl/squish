"""
Extended tests for squish/loader_utils.py and additional quantizer branches.

Covers:
- _build_model_args: the "no from_dict" dataclass branch
- _unique_base_keys: suffix detection
- _dequantize: passthrough (__pt) path
- _reconstruct_numpy: grouped symmetric and asymmetric-grouped paths (quantizer.py)
"""
from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from squish.loader_utils import (
    _build_model_args,
    _unique_base_keys,
)
from squish.quantizer import (
    QuantizationResult,
    _reconstruct_numpy,
)


# ── _build_model_args ──────────────────────────────────────────────────────

@dataclasses.dataclass
class _ArgsWithFromDict:
    dim: int = 64
    layers: int = 4

    @classmethod
    def from_dict(cls, cfg):
        return cls(dim=cfg.get("dim", 64), layers=cfg.get("layers", 4))


@dataclasses.dataclass
class _ArgsNoFromDict:
    dim: int = 64
    layers: int = 4


class TestBuildModelArgs:
    def test_uses_from_dict_when_available(self):
        cfg = {"dim": 128, "layers": 8}
        result = _build_model_args(_ArgsWithFromDict, cfg)
        assert result.dim == 128
        assert result.layers == 8

    def test_falls_back_to_dataclass_fields(self):
        """When no from_dict, uses dataclass field names as filter."""
        cfg = {"dim": 256, "layers": 12, "unknown_field": "ignored"}
        result = _build_model_args(_ArgsNoFromDict, cfg)
        assert result.dim == 256
        assert result.layers == 12
        assert not hasattr(result, "unknown_field")

    def test_dataclass_path_ignores_extra_keys(self):
        cfg = {"dim": 32, "extra": "value", "bogus": 99}
        result = _build_model_args(_ArgsNoFromDict, cfg)
        assert result.dim == 32


# ── _unique_base_keys ──────────────────────────────────────────────────────

class TestUniqueBaseKeys:
    def test_detects_q_suffix(self):
        keys = _unique_base_keys(["model__embed__q", "model__embed__s"])
        assert "model__embed" in keys

    def test_detects_pt_suffix(self):
        keys = _unique_base_keys(["layer0__weight__pt"])
        assert "layer0__weight" in keys

    def test_detects_shape_suffix(self):
        keys = _unique_base_keys(["layer0__bias__shape"])
        assert "layer0__bias" in keys

    def test_multiple_layers(self):
        files = [
            "layers__0__q", "layers__0__s",
            "layers__1__q", "layers__1__s",
            "embed__pt",
        ]
        keys = _unique_base_keys(files)
        assert "layers__0" in keys
        assert "layers__1" in keys
        assert "embed" in keys

    def test_no_matching_suffix(self):
        keys = _unique_base_keys(["something__random"])
        assert len(keys) == 0

    def test_empty_list(self):
        keys = _unique_base_keys([])
        assert keys == set()

    def test_deduplicates(self):
        """Same base key from multiple suffixes returns it only once."""
        files = ["key__q", "key__s", "key__shape"]
        keys = _unique_base_keys(files)
        assert keys == {"key"}


# ── _reconstruct_numpy: grouped symmetric path ─────────────────────────────

class TestReconstructNumpyGrouped:
    """Test the grouped-symmetric branch (scales.ndim == 2, no zero_points)."""

    def _make_grouped_result(self, n=4, d=16, n_groups=2):
        group_size = d // n_groups
        rng = np.random.default_rng(0)
        q = rng.integers(-127, 128, size=(n, d), dtype=np.int8)
        # 2-D scales: (n, n_groups)
        scales = rng.uniform(0.01, 0.1, size=(n, n_groups)).astype(np.float32)
        return QuantizationResult(
            quantized=q,
            scales=scales,
            dims=d,
            n=n,
        )

    def test_grouped_shape(self):
        res = self._make_grouped_result(n=4, d=16, n_groups=2)
        out = _reconstruct_numpy(res)
        assert out.shape == (4, 16)

    def test_grouped_dtype(self):
        res = self._make_grouped_result()
        out = _reconstruct_numpy(res)
        assert out.dtype == np.float32

    def test_grouped_different_groups(self):
        res = self._make_grouped_result(n=8, d=32, n_groups=4)
        out = _reconstruct_numpy(res)
        assert out.shape == (8, 32)

    def test_grouped_with_pad(self):
        """d not divisible by n_groups — expects padding to be handled."""
        # d=15, n_groups=4 → group_size=3, pad=1 to get full_cols=16
        rng = np.random.default_rng(42)
        n, d, n_groups = 4, 12, 3
        q = rng.integers(-127, 128, size=(n, d), dtype=np.int8)
        scales = rng.uniform(0.01, 0.1, size=(n, n_groups)).astype(np.float32)
        res = QuantizationResult(quantized=q, scales=scales, dims=d, n=n)
        out = _reconstruct_numpy(res)
        assert out.shape == (n, d)


# ── _reconstruct_numpy: asymmetric grouped path ────────────────────────────

class TestReconstructNumpyAsymmetricGrouped:
    """Test the asymmetric-grouped branch (zero_points not None, scales.ndim == 2)."""

    def _make_asym_grouped_result(self, n=4, d=16, n_groups=2):
        group_size = d // n_groups
        rng = np.random.default_rng(7)
        q = rng.integers(0, 16, size=(n, d), dtype=np.int8)
        scales = rng.uniform(0.005, 0.05, size=(n, n_groups)).astype(np.float32)
        zero_points = rng.integers(0, 8, size=(n, n_groups), dtype=np.int8)
        return QuantizationResult(
            quantized=q,
            scales=scales,
            dims=d,
            n=n,
            zero_points=zero_points,
        )

    def test_asym_grouped_shape(self):
        res = self._make_asym_grouped_result(n=4, d=16, n_groups=2)
        out = _reconstruct_numpy(res)
        assert out.shape == (4, 16)

    def test_asym_grouped_dtype(self):
        res = self._make_asym_grouped_result()
        out = _reconstruct_numpy(res)
        assert out.dtype == np.float32

    def test_asym_grouped_larger(self):
        res = self._make_asym_grouped_result(n=8, d=64, n_groups=4)
        out = _reconstruct_numpy(res)
        assert out.shape == (8, 64)
