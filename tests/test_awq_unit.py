"""
tests/test_awq_unit.py

Unit tests for squish/awq.py pure-Python / pure-numpy functions:
    save_awq_scales, load_awq_scales, apply_awq_to_weights, _preceding_norm_name

Does NOT require MLX, activation hooks, or a real model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squish.awq import (
    _preceding_norm_name,
    apply_awq_to_weights,
    load_awq_scales,
    save_awq_scales,
)


# ── save_awq_scales / load_awq_scales roundtrip ───────────────────────────────

class TestSaveAndLoadAwqScales:
    def _make_scales(self, n: int = 3):
        rng = np.random.default_rng(0)
        return {
            f"model.layers.{i}.self_attn.q_proj": rng.random(64).astype(np.float32)
            for i in range(n)
        }

    def test_roundtrip(self, tmp_path):
        scales = self._make_scales(3)
        save_awq_scales(scales, tmp_path, verbose=False)
        loaded = load_awq_scales(tmp_path)
        assert set(loaded.keys()) == set(scales.keys())
        for k in scales:
            np.testing.assert_allclose(loaded[k], scales[k], rtol=1e-5)

    def test_creates_awq_ready_sentinel(self, tmp_path):
        save_awq_scales(self._make_scales(1), tmp_path, verbose=False)
        assert (tmp_path / ".awq_ready").exists()

    def test_creates_index_file(self, tmp_path):
        save_awq_scales(self._make_scales(2), tmp_path, verbose=False)
        assert (tmp_path / "awq_index.json").exists()

    def test_verbose_output(self, tmp_path, capsys):
        scales = self._make_scales(1)
        save_awq_scales(scales, tmp_path, verbose=True)
        captured = capsys.readouterr()
        assert "Saved" in captured.out

    def test_load_missing_dir_returns_empty(self, tmp_path):
        result = load_awq_scales(tmp_path / "nonexistent")
        assert result == {}

    def test_load_empty_dir_returns_empty(self, tmp_path):
        result = load_awq_scales(tmp_path)
        assert result == {}

    def test_layer_names_with_dots_encoded(self, tmp_path):
        scales = {"model.layers.0.self_attn.q_proj": np.ones(8, dtype=np.float32)}
        save_awq_scales(scales, tmp_path, verbose=False)
        # Check .npy file exists (dots become __)
        npy_files = list(tmp_path.glob("*.awq.npy"))
        assert len(npy_files) == 1

    def test_fallback_glob_without_index(self, tmp_path):
        """load_awq_scales falls back to .awq.npy glob when no index file."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.save(str(tmp_path / "layer__0.awq.npy"), arr)
        result = load_awq_scales(tmp_path)
        assert len(result) == 1

    def test_partial_load_if_file_missing(self, tmp_path):
        """Index references a file that's been deleted → it's silently skipped."""
        scales = self._make_scales(2)
        save_awq_scales(scales, tmp_path, verbose=False)
        # Delete one .npy file
        npy_files = sorted(tmp_path.glob("*.awq.npy"))
        npy_files[0].unlink()
        loaded = load_awq_scales(tmp_path)
        assert len(loaded) == 1  # only the surviving file

    def test_creates_output_dir(self, tmp_path):
        target = tmp_path / "deep" / "nested"
        scales = self._make_scales(1)
        save_awq_scales(scales, target, verbose=False)
        assert target.is_dir()


# ── apply_awq_to_weights ──────────────────────────────────────────────────────

class TestApplyAwqToWeights:
    def _linear_weight(self, out=16, in_=8):
        rng = np.random.default_rng(1)
        return rng.standard_normal((out, in_)).astype(np.float32)

    def test_returns_same_dict(self):
        W = self._linear_weight()
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": np.ones(8, dtype=np.float32)}
        result  = apply_awq_to_weights(weights, scales, verbose=False)
        assert result is weights

    def test_scale_ones_leaves_weights_unchanged(self):
        W = self._linear_weight(16, 8)
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": np.ones(8, dtype=np.float32)}
        apply_awq_to_weights(weights, scales, verbose=False)
        np.testing.assert_allclose(
            weights["model.layers.0.self_attn.q_proj.weight"], W, rtol=1e-5
        )

    def test_scale_applied_column_wise(self):
        W = np.ones((4, 8), dtype=np.float32)
        s = np.full(8, 2.0, dtype=np.float32)
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": s}
        apply_awq_to_weights(weights, scales, verbose=False)
        # Each column should be halved: 1.0 / 2.0 = 0.5
        np.testing.assert_allclose(
            weights["model.layers.0.self_attn.q_proj.weight"],
            np.full((4, 8), 0.5),
            rtol=1e-5,
        )

    def test_absorbs_scale_into_preceding_norm(self):
        W = np.ones((4, 8), dtype=np.float32)
        gamma = np.ones(8, dtype=np.float32)
        s = np.full(8, 2.0, dtype=np.float32)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": W.copy(),
            "model.layers.0.input_layernorm.weight": gamma.copy(),
        }
        scales = {"model.layers.0.self_attn.q_proj": s}
        apply_awq_to_weights(weights, scales, verbose=False)
        # Norm should be multiplied by scale: 1.0 * 2.0 = 2.0
        np.testing.assert_allclose(
            weights["model.layers.0.input_layernorm.weight"],
            np.full(8, 2.0),
            rtol=1e-5,
        )

    def test_shape_mismatch_is_skipped(self):
        # scale has wrong size → should be silently skipped
        W = np.ones((4, 8), dtype=np.float32)
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": np.ones(16)}  # wrong size
        apply_awq_to_weights(weights, scales, verbose=False)
        # Weight should be unchanged
        np.testing.assert_array_equal(
            weights["model.layers.0.self_attn.q_proj.weight"], W
        )

    def test_empty_weights_returns_empty(self):
        result = apply_awq_to_weights({}, {"layer": np.ones(4)}, verbose=False)
        assert result == {}

    def test_empty_scales_no_change(self):
        W = np.ones((4, 8), dtype=np.float32)
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        apply_awq_to_weights(weights, {}, verbose=False)
        np.testing.assert_array_equal(weights["model.layers.0.self_attn.q_proj.weight"], W)

    def test_mlp_weights_also_adjusted(self):
        W = np.ones((32, 16), dtype=np.float32)
        s = np.full(16, 4.0, dtype=np.float32)
        weights = {"model.layers.0.mlp.gate_proj.weight": W.copy()}
        scales  = {"model.layers.0.mlp.gate_proj": s}
        apply_awq_to_weights(weights, scales, verbose=False)
        expected = np.full((32, 16), 0.25)
        np.testing.assert_allclose(
            weights["model.layers.0.mlp.gate_proj.weight"], expected, rtol=1e-5
        )

    def test_non_linear_weight_not_modified(self):
        """Tensors without a recognized projection suffix should be unchanged."""
        E = np.ones((100, 64), dtype=np.float32)
        weights = {"model.embed_tokens.weight": E.copy()}
        scales  = {"model.embed_tokens": np.ones(64, dtype=np.float32)}
        apply_awq_to_weights(weights, scales, verbose=False)
        np.testing.assert_array_equal(weights["model.embed_tokens.weight"], E)

    def test_1d_tensors_skipped(self):
        bias = np.ones(8, dtype=np.float32)
        weights = {"model.layers.0.self_attn.q_proj.bias": bias.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": np.ones(8)}
        apply_awq_to_weights(weights, scales, verbose=False)
        np.testing.assert_array_equal(weights["model.layers.0.self_attn.q_proj.bias"], bias)

    def test_verbose_prints_info(self, capsys):
        W = np.ones((4, 8), dtype=np.float32)
        weights = {"model.layers.0.self_attn.q_proj.weight": W.copy()}
        scales  = {"model.layers.0.self_attn.q_proj": np.ones(8, dtype=np.float32)}
        apply_awq_to_weights(weights, scales, verbose=True)
        captured = capsys.readouterr()
        assert "AWQ" in (captured.out + captured.err)


# ── _preceding_norm_name ──────────────────────────────────────────────────────

class TestPrecedingNormName:
    def _weights_with_norm(self, norm_name: str, in_: int = 8):
        return {norm_name: np.ones(in_)}

    def test_q_proj_finds_input_layernorm(self):
        weights = self._weights_with_norm("model.layers.0.input_layernorm.weight")
        result = _preceding_norm_name(
            "model.layers.0.self_attn.q_proj.weight", weights
        )
        assert result == "model.layers.0.input_layernorm.weight"

    def test_mlp_gate_finds_post_attention_norm(self):
        weights = self._weights_with_norm("model.layers.0.post_attention_layernorm.weight")
        result = _preceding_norm_name(
            "model.layers.0.mlp.gate_proj.weight", weights
        )
        assert result == "model.layers.0.post_attention_layernorm.weight"

    def test_returns_none_for_non_attn_mlp(self):
        weights = {"model.embed_tokens.weight": np.ones(100)}
        result = _preceding_norm_name("model.embed_tokens.weight", weights)
        assert result is None

    def test_returns_none_when_no_matching_norm(self):
        """No norm in weights dict → return None."""
        weights = {}  # empty weights
        result = _preceding_norm_name("model.layers.5.self_attn.q_proj.weight", weights)
        assert result is None

    def test_k_proj_also_works(self):
        weights = self._weights_with_norm("model.layers.2.input_layernorm.weight")
        result = _preceding_norm_name(
            "model.layers.2.self_attn.k_proj.weight", weights
        )
        assert result == "model.layers.2.input_layernorm.weight"

    def test_ln_1_fallback(self):
        """ln_1 fallback for models using pre-GPT2-style naming."""
        weights = {"transformer.h.3.ln_1.weight": np.ones(16)}
        # self_attn style
        result = _preceding_norm_name("transformer.h.3.self_attn.q_proj.weight", weights)
        assert result == "transformer.h.3.ln_1.weight"

    def test_ln_2_fallback_for_mlp(self):
        weights = {"transformer.h.3.ln_2.weight": np.ones(16)}
        result = _preceding_norm_name("transformer.h.3.mlp.gate_proj.weight", weights)
        assert result == "transformer.h.3.ln_2.weight"
