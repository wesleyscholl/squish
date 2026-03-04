"""
tests/test_split_loader_unit.py

Unit tests for pure helpers in squish/split_loader.py.
Covers _total_ram_bytes (hardware query, pure Python — no MLX required).
"""
from __future__ import annotations


from squish.split_loader import _total_ram_bytes


class TestTotalRamBytes:
    def test_returns_positive_int(self):
        result = _total_ram_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_at_least_one_gb(self):
        result = _total_ram_bytes()
        assert result >= 1 * 1024 ** 3  # at least 1 GB RAM

    def test_reasonable_upper_bound(self):
        result = _total_ram_bytes()
        # No machine has more than 16 TB RAM in 2025
        assert result < 16 * 1024 ** 4
