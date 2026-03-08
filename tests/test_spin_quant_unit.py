"""tests/test_spin_quant_unit.py — 100% coverage for squish/spin_quant.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from squish.spin_quant import (
    _random_orthogonal,
    _quantize_fake_int8,
    _quantize_fake_int4,
    _quant_error,
    _riemannian_grad,
    _cayley_update,
    _load_safetensors_numpy,
    _save_safetensors_or_npz,
    _build_rotation_matrices,
    _apply_rotations,
    run_rotation,
)


# ---------------------------------------------------------------------------
# _random_orthogonal
# ---------------------------------------------------------------------------

class TestRandomOrthogonal:
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q = _random_orthogonal(4, rng)
        assert Q.shape == (4, 4)

    def test_is_orthogonal(self):
        rng = np.random.default_rng(42)
        Q = _random_orthogonal(8, rng)
        assert np.allclose(Q @ Q.T, np.eye(8), atol=1e-5)

    def test_dtype_float32(self):
        rng = np.random.default_rng(1)
        Q = _random_orthogonal(3, rng)
        assert Q.dtype == np.float32

    def test_zero_diagonal_fixed(self, monkeypatch):
        """Force a zero in diag(R) to cover the diag[diag == 0] = 1.0 line."""
        rng = np.random.default_rng(0)
        real_qr = np.linalg.qr

        def mock_qr(A):
            Q_real, R_real = real_qr(A)
            R_real = R_real.copy()
            R_real[0, 0] = 0.0  # force zero diagonal element
            return Q_real, R_real

        monkeypatch.setattr(np.linalg, "qr", mock_qr)
        result = _random_orthogonal(3, rng)
        # sign of 0 → fixed to 1; result should not crash and be correct shape
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# _quantize_fake_int8
# ---------------------------------------------------------------------------

class TestQuantizeFakeInt8:
    def test_shape_preserved(self):
        W = np.random.randn(4, 8).astype(np.float32)
        W_dq, scale = _quantize_fake_int8(W)
        assert W_dq.shape == W.shape
        assert scale.shape == (4,)

    def test_zero_row_uses_epsilon(self):
        """Zero row → scale clamped to 1e-8 (avoids division-by-zero)."""
        W = np.zeros((2, 4), dtype=np.float32)
        W_dq, scale = _quantize_fake_int8(W)
        assert np.all(scale > 0)

    def test_reconstruction_close(self):
        rng = np.random.default_rng(7)
        W = rng.uniform(-1.0, 1.0, (4, 16)).astype(np.float32)
        W_dq, _ = _quantize_fake_int8(W)
        max_err = np.max(np.abs(W)) / 127.0
        assert np.max(np.abs(W - W_dq)) <= max_err + 1e-6


# ---------------------------------------------------------------------------
# _quantize_fake_int4
# ---------------------------------------------------------------------------

class TestQuantizeFakeInt4:
    def test_shape_preserved_2d(self):
        W = np.random.randn(4, 32).astype(np.float32)
        result = _quantize_fake_int4(W, group_size=32)
        assert result.shape == W.shape

    def test_zero_group_uses_epsilon(self):
        W = np.zeros((1, 32), dtype=np.float32)
        result = _quantize_fake_int4(W, group_size=32)
        assert result.shape == W.shape

    def test_1d_input_else_branch(self):
        """1D array hits the `else W.shape[0]` branch of the ternary."""
        W = np.ones(32, dtype=np.float32)
        result = _quantize_fake_int4(W, group_size=32)
        assert result.shape == (32,)


# ---------------------------------------------------------------------------
# _quant_error
# ---------------------------------------------------------------------------

class TestQuantError:
    def _small_W_R(self, seed=0, dim=4):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((8, dim)).astype(np.float32)
        R = _random_orthogonal(dim, rng)
        return W, R

    def test_bits8_returns_nonneg_float(self):
        W, R = self._small_W_R()
        err = _quant_error(W, R, bits=8)
        assert isinstance(err, float)
        assert err >= 0.0

    def test_bits4_returns_nonneg_float(self):
        """bits=4 path — W must have n*dim divisible by group_size=32."""
        rng = np.random.default_rng(5)
        W = rng.standard_normal((8, 32)).astype(np.float32)
        R = _random_orthogonal(32, rng)
        err = _quant_error(W, R, bits=4)
        assert err >= 0.0

    def test_identity_rotation_gives_nonneg(self):
        rng = np.random.default_rng(9)
        W = rng.uniform(-2.0, 2.0, (4, 4)).astype(np.float32)
        R = np.eye(4, dtype=np.float32)
        err = _quant_error(W, R, bits=8)
        assert err >= 0.0


# ---------------------------------------------------------------------------
# _riemannian_grad
# ---------------------------------------------------------------------------

class TestRiemannianGrad:
    def test_shape_and_bits8(self):
        rng = np.random.default_rng(11)
        W = rng.standard_normal((4, 2)).astype(np.float32)
        R = _random_orthogonal(2, rng)
        G = _riemannian_grad(W, R, bits=8)
        assert G.shape == R.shape

    def test_bits4_path(self):
        """bits=4 path in _riemannian_grad; uses dim=4, W rows ensure divisibility."""
        rng = np.random.default_rng(13)
        # 8 rows * 4 cols = 32 elements → divisible by group_size=32
        W = rng.standard_normal((8, 4)).astype(np.float32)
        R = _random_orthogonal(4, rng)
        G = _riemannian_grad(W, R, bits=4)
        assert G.shape == (4, 4)


# ---------------------------------------------------------------------------
# _cayley_update
# ---------------------------------------------------------------------------

class TestCayleyUpdate:
    def test_returns_orthogonal(self):
        rng = np.random.default_rng(17)
        R = _random_orthogonal(4, rng)
        G = rng.standard_normal((4, 4)).astype(np.float32) * 0.01
        R_new = _cayley_update(R, G, lr=0.01)
        assert np.allclose(R_new @ R_new.T, np.eye(4), atol=1e-4)

    def test_singular_L_uses_pinv(self, monkeypatch):
        """np.linalg.solve raises LinAlgError → fallback to pinv."""
        rng = np.random.default_rng(19)
        R = _random_orthogonal(3, rng)
        G = np.eye(3, dtype=np.float32) * 0.01

        def mock_solve(L, b):
            raise np.linalg.LinAlgError("forced singular")

        monkeypatch.setattr(np.linalg, "solve", mock_solve)
        R_new = _cayley_update(R, G, lr=0.01)
        assert R_new.shape == (3, 3)


# ---------------------------------------------------------------------------
# _load_safetensors_numpy
# ---------------------------------------------------------------------------

class TestLoadSafetensorsNumpy:
    def test_loads_safetensors_when_available(self, tmp_path):
        """safetensors available + .safetensors file → returns loaded dict."""
        try:
            from safetensors.numpy import save_file  # noqa: PLC0415
        except ImportError:
            pytest.skip("safetensors not installed")
        w = {"layer.weight": np.ones((2, 2), dtype=np.float32)}
        save_file(w, str(tmp_path / "model.safetensors"))
        result = _load_safetensors_numpy(tmp_path)
        assert "layer.weight" in result

    def test_no_safetensors_files_falls_through_to_npz(self, tmp_path):
        """safetensors available but no .safetensors files → falls through to npz."""
        w = {"x": np.zeros((3,), dtype=np.float32)}
        np.savez(str(tmp_path / "data.npz"), **w)
        result = _load_safetensors_numpy(tmp_path)
        assert "x" in result

    def test_import_error_falls_back_to_npz(self, tmp_path, monkeypatch):
        """safetensors not importable → catches ImportError → loads npz instead."""
        monkeypatch.setitem(sys.modules, "safetensors", None)
        w = {"layer.weight": np.ones((2, 2), dtype=np.float32)}
        np.savez(str(tmp_path / "model.npz"), **w)
        result = _load_safetensors_numpy(tmp_path)
        assert "layer.weight" in result
        assert result["layer.weight"].shape == (2, 2)

    def test_returns_empty_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "safetensors", None)
        result = _load_safetensors_numpy(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# _save_safetensors_or_npz
# ---------------------------------------------------------------------------

class TestSaveSafetensorsOrNpz:
    def test_saves_safetensors_when_available(self, tmp_path):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")
        weights = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        _save_safetensors_or_npz(weights, tmp_path)
        assert (tmp_path / "model.safetensors").exists()

    def test_saves_npz_when_safetensors_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "safetensors", None)
        monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
        weights = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        _save_safetensors_or_npz(weights, tmp_path)
        assert (tmp_path / "model.npz").exists()
        loaded = np.load(str(tmp_path / "model.npz"))
        np.testing.assert_array_equal(loaded["w"], weights["w"])

    def test_creates_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "safetensors", None)
        monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
        new_dir = tmp_path / "sub" / "deep"
        _save_safetensors_or_npz({"x": np.zeros((2,), dtype=np.float32)}, new_dir)
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# _build_rotation_matrices
# ---------------------------------------------------------------------------

class TestBuildRotationMatrices:
    def _rotatable_weights(self, dim=4, n_rows=8, seed=23):
        rng = np.random.default_rng(seed)
        key = "model.layers.0.self_attn.q_proj.weight"
        return {
            key: rng.standard_normal((n_rows, dim)).astype(np.float32),
            "model.embed_tokens.weight": rng.standard_normal((10, dim)).astype(np.float32),
        }

    def test_returns_rotation_for_dim(self):
        weights = self._rotatable_weights(dim=4)
        rng = np.random.default_rng(25)
        rotations = _build_rotation_matrices(
            weights, steps=2, lr=0.01, bits=8, rng=rng, verbose=False
        )
        assert 4 in rotations
        assert rotations[4].shape == (4, 4)

    def test_rotation_is_orthogonal(self):
        weights = self._rotatable_weights(dim=4)
        rng = np.random.default_rng(27)
        rotations = _build_rotation_matrices(
            weights, steps=2, lr=0.01, bits=8, rng=rng, verbose=False
        )
        R = rotations[4]
        assert np.allclose(R @ R.T, np.eye(4), atol=1e-4)

    def test_empty_weights_returns_empty(self):
        rng = np.random.default_rng(29)
        rotations = _build_rotation_matrices(
            {}, steps=1, lr=0.01, bits=8, rng=rng, verbose=False
        )
        assert rotations == {}

    def test_no_sample_weights_for_dim(self):
        """Force 'if not sample_weights' branch via a dict that returns
        different items on successive .items() calls."""
        rng_build = np.random.default_rng(31)
        dim = 4
        key = "model.layers.0.self_attn.q_proj.weight"
        wt = np.ones((4, dim), dtype=np.float32)

        class FlippingItems(dict):
            """Returns real items on first call, empty on subsequent calls."""
            def __init__(self, data):
                super().__init__(data)
                self._call = 0

            def items(self):
                self._call += 1
                if self._call == 1:
                    return super().items()
                return {}.items()

        weights = FlippingItems({key: wt})
        rotations = _build_rotation_matrices(
            weights, steps=1, lr=0.01, bits=8, rng=rng_build, verbose=False
        )
        # A random R is stored for dim even when sample_weights is empty
        assert dim in rotations

    def test_verbose_output(self, capsys):
        weights = self._rotatable_weights(dim=4)
        rng = np.random.default_rng(33)
        _build_rotation_matrices(
            weights, steps=2, lr=0.01, bits=8, rng=rng, verbose=True
        )
        captured = capsys.readouterr()
        assert "SpinQuant" in captured.out

    def test_bits4_path(self):
        """bits=4 triggers _quantize_fake_int4 inside the step loop."""
        rng = np.random.default_rng(35)
        key = "model.layers.0.self_attn.q_proj.weight"
        # n_rows=8; dim=32 → 8*32=256 elements; reshape(-1,32) → (8,32) ✓
        weights = {key: rng.standard_normal((8, 32)).astype(np.float32)}
        rotations = _build_rotation_matrices(
            weights, steps=1, lr=0.01, bits=4, rng=rng, verbose=False
        )
        assert 32 in rotations

    def test_1d_matching_key_skipped_in_dim_set(self):
        """Key matches a rotatable suffix but weight is 1D → ndim != 2 → branch 269→267."""
        rng = np.random.default_rng(45)
        key = "model.layers.0.self_attn.q_proj.weight"
        # 1D bias-like tensor — matches key suffix but ndim == 1
        weights = {key: np.ones((4,), dtype=np.float32)}
        rotations = _build_rotation_matrices(
            weights, steps=1, lr=0.01, bits=8, rng=rng, verbose=False
        )
        # dim_set is empty (no 2D weight found) → rotations is {}
        assert rotations == {}


# ---------------------------------------------------------------------------
# _apply_rotations
# ---------------------------------------------------------------------------

class TestApplyRotations:
    def test_rotates_eligible_2d_weight(self):
        rng = np.random.default_rng(37)
        key = "model.layers.0.self_attn.q_proj.weight"
        W = rng.standard_normal((4, 4)).astype(np.float32)
        R = _random_orthogonal(4, rng)
        result = _apply_rotations({key: W}, {4: R})
        np.testing.assert_allclose(result[key], W @ R.T, atol=1e-5)

    def test_passes_through_non_eligible_key(self):
        W = np.ones((4, 4), dtype=np.float32)
        result = _apply_rotations(
            {"embed.weight": W}, {4: np.eye(4, dtype=np.float32)}
        )
        np.testing.assert_array_equal(result["embed.weight"], W)

    def test_eligible_key_dim_not_in_rotations(self):
        """Rotatable key but its dim is absent from rotations dict → pass-through."""
        rng = np.random.default_rng(39)
        key = "model.layers.0.self_attn.q_proj.weight"
        W = rng.standard_normal((4, 8)).astype(np.float32)
        # rotations only has dim=4, not dim=8
        result = _apply_rotations({key: W}, {4: np.eye(4, dtype=np.float32)})
        np.testing.assert_array_equal(result[key], W)

    def test_1d_weight_is_passed_through(self):
        """1D tensor (e.g. bias) has ndim!=2 → not rotated."""
        key = "model.layers.0.self_attn.q_proj.weight"
        W = np.ones((4,), dtype=np.float32)
        result = _apply_rotations({key: W}, {4: np.eye(4, dtype=np.float32)})
        np.testing.assert_array_equal(result[key], W)


# ---------------------------------------------------------------------------
# run_rotation
# ---------------------------------------------------------------------------

class TestRunRotation:
    def _make_model(self, tmp_path, rotatable=True, seed=41):
        src = tmp_path / "model"
        src.mkdir()
        rng = np.random.default_rng(seed)
        if rotatable:
            weights = {
                "model.layers.0.self_attn.q_proj.weight":
                    rng.standard_normal((4, 4)).astype(np.float32)
            }
        else:
            weights = {
                "embed.weight": rng.standard_normal((10, 4)).astype(np.float32)
            }
        np.savez(str(src / "model.npz"), **weights)
        (src / "config.json").write_text(json.dumps({"hidden_size": 4}))
        (src / "tokenizer.json").write_text("{}")
        (src / "README.md").write_text("# Model")   # misc file
        return src

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_rotation(
                str(tmp_path / "nonexistent"), str(tmp_path / "out"),
                steps=1, verbose=False,
            )

    def test_no_weights_raises(self, tmp_path):
        src = tmp_path / "empty_model"
        src.mkdir()
        with pytest.raises(RuntimeError, match="No weight files"):
            run_rotation(str(src), str(tmp_path / "out"), steps=1, verbose=False)

    def test_no_rotatable_dims_returns_early(self, tmp_path):
        """No rotatable dims → function returns before creating output dir."""
        src = self._make_model(tmp_path, rotatable=False)
        dst = tmp_path / "out"
        run_rotation(str(src), str(dst), steps=1, verbose=False)
        assert not dst.exists()

    def test_no_rotatable_dims_verbose_message(self, tmp_path, capsys):
        """verbose=True with no rotatable dims → prints 'nothing to do' message."""
        src = self._make_model(tmp_path, rotatable=False)
        run_rotation(str(src), str(tmp_path / "out_v"), steps=1, verbose=True)
        captured = capsys.readouterr()
        assert "nothing to do" in captured.out

    def test_successful_rotation_creates_output(self, tmp_path):
        """Full pass: rotatable weights → output dir with model file + copied files."""
        src = self._make_model(tmp_path, rotatable=True)
        dst = tmp_path / "out"
        run_rotation(str(src), str(dst), steps=1, bits=8, verbose=False)
        assert dst.exists()
        assert (dst / "model.npz").exists() or (dst / "model.safetensors").exists()
        # Named files should be copied
        assert (dst / "config.json").exists()
        assert (dst / "tokenizer.json").exists()
        # Misc file (README.md) should also be copied
        assert (dst / "README.md").exists()

    def test_existing_dst_is_removed_before_write(self, tmp_path):
        """Pre-existing dst (dst != src) is removed before rotation output."""
        src = self._make_model(tmp_path, rotatable=True)
        dst = tmp_path / "out"
        dst.mkdir()
        (dst / "stale_file.txt").write_text("old content")
        run_rotation(str(src), str(dst), steps=1, verbose=False)
        assert not (dst / "stale_file.txt").exists()

    def test_verbose_mode_prints_progress(self, tmp_path, capsys):
        src = self._make_model(tmp_path, rotatable=True)
        dst = tmp_path / "out_verbose"
        run_rotation(str(src), str(dst), steps=1, verbose=True)
        captured = capsys.readouterr()
        assert "SpinQuant" in captured.out

    def test_bits4_completes(self, tmp_path):
        """bits=4 end-to-end with dim divisible by group_size=32."""
        src = tmp_path / "model4"
        src.mkdir()
        rng = np.random.default_rng(43)
        w = rng.standard_normal((4, 32)).astype(np.float32)
        np.savez(
            str(src / "model.npz"),
            **{"model.layers.0.self_attn.q_proj.weight": w},
        )
        dst = tmp_path / "out4"
        run_rotation(str(src), str(dst), steps=1, bits=4, verbose=False)
        assert dst.exists()
