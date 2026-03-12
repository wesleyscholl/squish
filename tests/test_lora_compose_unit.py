"""
tests/test_lora_compose_unit.py

Unit tests for squish/lora_compose.py — 100% coverage.
"""

import numpy as np
import pytest

from squish.lora_compose import AdapterConfig, AdapterStack, CompositionStats, LoRAComposer


# ---------------------------------------------------------------------------
# AdapterConfig
# ---------------------------------------------------------------------------


class TestAdapterConfig:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.hidden_dim == 4096
        assert cfg.out_dim == 4096
        assert cfg.scaling == 2.0

    def test_custom(self):
        cfg = AdapterConfig(rank=8, alpha=16.0, hidden_dim=64, out_dim=32)
        assert cfg.out_dim == 32
        assert cfg.scaling == 2.0

    def test_out_dim_defaults_to_hidden_dim(self):
        cfg = AdapterConfig(hidden_dim=128)
        assert cfg.out_dim == 128

    def test_scaling_computed(self):
        cfg = AdapterConfig(rank=4, alpha=8.0)
        assert cfg.scaling == 2.0
        cfg2 = AdapterConfig(rank=32, alpha=32.0)
        assert cfg2.scaling == 1.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rank": 0}, "rank"),
            ({"rank": -1}, "rank"),
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": -1.0}, "alpha"),
            ({"hidden_dim": 0}, "hidden_dim"),
            ({"hidden_dim": -1}, "hidden_dim"),
            ({"out_dim": 0}, "out_dim"),
            ({"out_dim": -5}, "out_dim"),
        ],
    )
    def test_validation_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            AdapterConfig(**kwargs)

    def test_frozen(self):
        cfg = AdapterConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.rank = 32  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AdapterStack
# ---------------------------------------------------------------------------


class TestAdapterStack:
    def _make_stack(self, hidden=8, rank=2, out=8, scale=1.0):
        A = np.random.randn(hidden, rank).astype(np.float32)
        B = np.random.randn(rank, out).astype(np.float32)
        return AdapterStack(name="test", A=A, B=B, scaling=scale)

    def test_forward_1d(self):
        stack = self._make_stack(hidden=8, rank=2, out=8)
        x = np.random.randn(8).astype(np.float32)
        out = stack.forward(x)
        assert out.shape == (8,)

    def test_forward_2d_batch(self):
        stack = self._make_stack(hidden=8, rank=2, out=8)
        x = np.random.randn(4, 8).astype(np.float32)
        out = stack.forward(x)
        assert out.shape == (4, 8)

    def test_forward_scaling_applied(self):
        stack = self._make_stack(scale=2.0)
        stack2 = self._make_stack(scale=1.0)
        # replace matrices to be identical
        A = np.eye(2, dtype=np.float32).reshape(2, 2)
        B = np.eye(2, dtype=np.float32).reshape(2, 2)
        # Manually set
        stack.A = A
        stack.B = B
        stack2.A = A
        stack2.B = B
        x = np.ones((3, 2), dtype=np.float32)
        out1 = stack.forward(x)
        out2 = stack2.forward(x)
        assert np.allclose(out1, out2 * 2.0, atol=1e-5)

    def test_n_params(self):
        stack = self._make_stack(hidden=8, rank=2, out=4)
        assert stack.n_params == 8 * 2 + 2 * 4  # 16 + 8 = 24

    def test_forward_output_dtype_float32(self):
        stack = self._make_stack()
        x = np.random.randn(3, 8).astype(np.float64)
        out = stack.forward(x)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# CompositionStats
# ---------------------------------------------------------------------------


class TestCompositionStats:
    def test_avg_adapters_per_call_normal(self):
        s = CompositionStats(n_forward_calls=4, adapters_used_total=12)
        assert s.avg_adapters_per_call == 3.0

    def test_avg_adapters_per_call_zero_calls(self):
        s = CompositionStats(n_forward_calls=0, adapters_used_total=0)
        assert s.avg_adapters_per_call == 0.0


# ---------------------------------------------------------------------------
# LoRAComposer
# ---------------------------------------------------------------------------


class TestLoRAComposer:
    def _make_AB(self, hidden=16, rank=4, out=16):
        A = np.random.randn(hidden, rank).astype(np.float32)
        B = np.random.randn(rank, out).astype(np.float32)
        return A, B

    def test_init_invalid_hidden_dim(self):
        with pytest.raises(ValueError):
            LoRAComposer(hidden_dim=0)

    def test_add_and_list_adapters(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B)
        assert c.adapter_names == ["a"]

    def test_add_duplicate_raises(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("x", A, B)
        with pytest.raises(ValueError, match="already registered"):
            c.add_adapter("x", A, B)

    def test_add_wrong_A_shape_raises(self):
        c = LoRAComposer(hidden_dim=16)
        A = np.zeros((8, 4), dtype=np.float32)  # wrong hidden_dim
        B = np.zeros((4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="A must have shape"):
            c.add_adapter("bad", A, B)

    def test_add_wrong_B_shape_raises(self):
        c = LoRAComposer(hidden_dim=16)
        A = np.zeros((16, 4), dtype=np.float32)
        B = np.zeros((4, 8), dtype=np.float32)  # wrong out_dim
        with pytest.raises(ValueError, match="B must have shape"):
            c.add_adapter("bad", A, B)

    def test_add_rank_mismatch_raises(self):
        c = LoRAComposer(hidden_dim=16)
        A = np.zeros((16, 4), dtype=np.float32)
        B = np.zeros((8, 16), dtype=np.float32)  # rank mismatch
        with pytest.raises(ValueError, match="rank mismatch"):
            c.add_adapter("bad", A, B)

    def test_remove_adapter(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("x", A, B)
        c.remove_adapter("x")
        assert "x" not in c.adapter_names

    def test_remove_unknown_raises(self):
        c = LoRAComposer(hidden_dim=16)
        with pytest.raises(KeyError):
            c.remove_adapter("ghost")

    def test_forward_no_adapters_returns_zeros(self):
        c = LoRAComposer(hidden_dim=8)
        x = np.ones((3, 8), dtype=np.float32)
        out = c.forward(x)
        assert np.allclose(out, 0.0)
        assert out.shape == (3, 8)

    def test_forward_single_adapter(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B, scale=1.0)
        x = np.random.randn(2, 16).astype(np.float32)
        out = c.forward(x)
        assert out.shape == (2, 16)

    def test_forward_equal_weights_default(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B, scale=1.0)
        c.add_adapter("b", A, B, scale=1.0)
        x = np.random.randn(2, 16).astype(np.float32)
        # equal weight → same as each adapter at 0.5
        out = c.forward(x)
        assert out.shape == (2, 16)

    def test_forward_explicit_weights(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B, scale=1.0)
        x = np.random.randn(2, 16).astype(np.float32)
        out = c.forward(x, weights={"a": 2.0})
        # 2x the scale of the adapter
        ref = c._adapters["a"].forward(x) * 2.0
        assert np.allclose(out, ref, atol=1e-5)

    def test_forward_unknown_adapter_raises(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B)
        x = np.random.randn(2, 16).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown adapter"):
            c.forward(x, weights={"ghost": 1.0})

    def test_total_params(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB(hidden=16, rank=4, out=16)
        c.add_adapter("a", A, B)
        c.add_adapter("b", A, B)
        # Each: 16*4 + 4*16 = 128; two → 256
        assert c.total_params == 256

    def test_composition_stats_tracking(self):
        c = LoRAComposer(hidden_dim=16)
        A, B = self._make_AB()
        c.add_adapter("a", A, B)
        c.add_adapter("b", A, B)
        x = np.random.randn(1, 16).astype(np.float32)
        c.forward(x)  # uses 2 adapters (equal weight)
        c.forward(x, weights={"a": 1.0})  # uses 1
        s = c.composition_stats()
        assert s.n_forward_calls == 2
        assert s.adapters_used_total == 3  # 2 + 1
        assert s.avg_adapters_per_call == 1.5

    def test_custom_out_dim(self):
        c = LoRAComposer(hidden_dim=16, out_dim=8)
        A = np.random.randn(16, 4).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32)
        c.add_adapter("a", A, B)
        x = np.random.randn(3, 16).astype(np.float32)
        out = c.forward(x)
        assert out.shape == (3, 8)
