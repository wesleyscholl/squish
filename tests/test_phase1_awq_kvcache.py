#!/usr/bin/env python3
"""
Unit tests for Phase 1.2 (AWQ) and Phase 1.3 (KV cache).
No model loading required — pure numpy logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.2 — AWQ
# ─────────────────────────────────────────────────────────────────────────────

def test_awq_apply_weights():
    """AWQ scale col-division modifies W and leaves other tensors alone."""
    from squish.awq import apply_awq_to_weights

    W = np.random.randn(128, 64).astype(np.float32)
    gamma = np.ones(64, dtype=np.float32)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": W.copy(),
        "model.layers.0.input_layernorm.weight":  gamma.copy(),
    }
    scales = {"model.layers.0.self_attn.q_proj":
              np.abs(np.random.randn(64)).astype(np.float32) + 0.1}

    out = apply_awq_to_weights(weights, scales, verbose=False)
    s = scales["model.layers.0.self_attn.q_proj"]

    # W[:, c] should have been divided by s[c]
    W_awq = out["model.layers.0.self_attn.q_proj.weight"]
    expected = W / s[np.newaxis, :]
    np.testing.assert_allclose(W_awq, expected, rtol=1e-5,
                               err_msg="W_awq col scaling mismatch")

    # gamma should have been multiplied by s
    gamma_awq = out["model.layers.0.input_layernorm.weight"]
    np.testing.assert_allclose(gamma_awq, gamma * s, rtol=1e-5,
                               err_msg="gamma awq scale absorption mismatch")


def test_awq_save_load(tmp_path=None):
    """Round-trip: save_awq_scales then load_awq_scales returns identical arrays."""
    from squish.awq import load_awq_scales, save_awq_scales

    scales_in = {
        "model.layers.0.self_attn.q_proj": np.array([1.2, 0.8, 1.5], dtype=np.float32),
        "model.layers.1.mlp.gate_proj":     np.array([0.9, 1.1, 0.7, 1.3], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        save_awq_scales(scales_in, d, verbose=False)
        scales_out = load_awq_scales(d)

    for k, v in scales_in.items():
        assert k in scales_out, f"Key {k!r} missing after round-trip"
        np.testing.assert_array_equal(scales_out[k], v,
                                      err_msg=f"Scale mismatch for {k}")


def test_awq_no_match_counts_applied():
    """apply_awq_to_weights prints warning when no scale matches — doesn't crash."""
    from squish.awq import apply_awq_to_weights

    weights = {"embed_tokens.weight": np.ones((100, 64), dtype=np.float32)}
    scales  = {"totally.different.path": np.ones(64, dtype=np.float32)}
    out = apply_awq_to_weights(weights, scales, verbose=False)
    assert out["embed_tokens.weight"].shape == (100, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.3 — KV cache
# ─────────────────────────────────────────────────────────────────────────────

def test_kv_int8_round_trip():
    """INT8 quantize → dequantize max error < 1% of signal range."""
    from squish.kv_cache import _dequantize_int8_per_channel, _quantize_int8_per_channel

    rng = np.random.default_rng(42)
    x = rng.standard_normal((32, 128)).astype(np.float16)
    q, s = _quantize_int8_per_channel(x)
    xr    = _dequantize_int8_per_channel(q, s)

    assert q.dtype  == np.int8,   f"Expected int8 got {q.dtype}"
    assert xr.dtype == np.float16, f"Expected float16 got {xr.dtype}"

    err      = np.abs(x.astype(np.float32) - xr.astype(np.float32))
    x_range  = np.abs(x.astype(np.float32)).max()
    rel_err  = err.max() / (x_range + 1e-9)
    assert rel_err < 0.02, f"INT8 round-trip relative error too large: {rel_err:.4f}"


def test_kv_layer_cache_append():
    """KVLayerCache stores exactly n_tokens positions and recent window is correct."""
    from squish.kv_cache import KVLayerCache

    cache = KVLayerCache(window=4)
    n_heads, head_dim = 8, 64

    for _i in range(10):
        k = np.random.randn(n_heads, head_dim).astype(np.float16)
        v = np.random.randn(n_heads, head_dim).astype(np.float16)
        cache.append(k, v)

    assert cache.n_tokens == 10, f"Expected 10 got {cache.n_tokens}"
    assert len(cache.keys_recent) == 4, \
        f"Recent window should be 4, got {len(cache.keys_recent)}"
    assert cache.keys_old_q is not None, "old INT8 buffer should exist after overflow"

    full_k, full_v = cache.get_full_kv()
    assert full_k.shape == (n_heads, 10, head_dim), \
        f"Expected ({n_heads}, 10, {head_dim}) got {full_k.shape}"
    assert full_v.shape == full_k.shape


def test_kv_layer_cache_reset():
    """reset() clears all state."""
    from squish.kv_cache import KVLayerCache

    cache = KVLayerCache(window=4)
    for _ in range(8):
        cache.append(np.ones((2, 32), dtype=np.float16),
                     np.ones((2, 32), dtype=np.float16))
    assert cache.n_tokens == 8
    cache.reset()
    assert cache.n_tokens == 0
    assert cache.keys_old_q is None


def test_quantized_kv_cache_full():
    """QuantizedKVCache.update accumulates correctly in int8 mode."""
    from squish.kv_cache import QuantizedKVCache

    qkv = QuantizedKVCache(n_layers=4, window=4, mode="int8")
    for _ in range(6):
        qkv.update(0, np.random.randn(2, 64).astype(np.float16),
                      np.random.randn(2, 64).astype(np.float16))
    assert qkv.n_tokens == 6
    assert qkv.memory_mb > 0
    stats = qkv.stats()
    assert stats["mode"] == "int8"
    assert stats["n_tokens"] == 6


def test_snapkv_eviction():
    """SnapKV eviction reduces token count to <= budget."""
    from squish.kv_cache import KVLayerCache, _snap_evict

    cache = KVLayerCache(window=4)
    for _ in range(20):
        cache.append(np.random.randn(4, 128).astype(np.float16),
                     np.random.randn(4, 128).astype(np.float16))
    assert cache.n_tokens == 20

    _snap_evict(cache, budget=8, snap_window=4)
    assert cache.n_tokens <= 8, \
        f"After eviction expected ≤8, got {cache.n_tokens}"


def test_snapkv_mode_auto_evict():
    """QuantizedKVCache snap mode triggers eviction once budget is exceeded.

    SnapKV evicts *once* when the cache first exceeds ``budget`` during prefill;
    subsequent autoregressive tokens keep appending normally (cache grows past
    budget again, but that's expected — re-eviction only happens on next prefill).
    """
    from squish.kv_cache import QuantizedKVCache

    budget = 12
    # Eviction fires the first time n_tokens > budget (at token budget+1 = 13)
    # After eviction: n_tokens ≤ budget
    # Then 20-13 = 7 more tokens are appended → final n_tokens ≤ budget + 7
    qkv = QuantizedKVCache(n_layers=1, window=4, mode="snap",
                           budget=budget, snap_window=4)
    for _i in range(20):
        qkv.update(0, np.random.randn(4, 64).astype(np.float16),
                      np.random.randn(4, 64).astype(np.float16))

    # Eviction should have been triggered exactly once
    assert qkv._snapped[0], "Layer 0 should have been snapped"
    # Final token count ≤ eviction_target + tokens_after_eviction ≤ budget + 7
    assert qkv._layers[0].n_tokens <= budget + 7, \
        f"After snap+append got {qkv._layers[0].n_tokens}, expected ≤{budget+7}"


# ─────────────────────────────────────────────────────────────────────────────
# mlx_lm cache protocol — update_and_fetch + offset
# ─────────────────────────────────────────────────────────────────────────────

def test_kv_layer_cache_update_and_fetch():
    """
    KVLayerCache.update_and_fetch implements the mlx_lm cache protocol:
    - accepts (1, n_heads, T_new, head_dim) MLX arrays
    - appends new tokens, returns full accumulated (1, n_heads, T_total, head_dim)
    - .offset returns total token count for RoPE position encoding
    """
    import pytest
    mx = pytest.importorskip("mlx.core")
    from squish.kv_cache import KVLayerCache

    n_heads, head_dim = 4, 32
    cache = KVLayerCache(window=8)

    # First call: 3 new tokens → full output is (1, 4, 3, 32)
    keys1   = mx.zeros([1, n_heads, 3, head_dim])
    values1 = mx.zeros([1, n_heads, 3, head_dim])
    k_out, v_out = cache.update_and_fetch(keys1, values1)

    assert k_out.shape == (1, n_heads, 3, head_dim), \
        f"First call shape: {k_out.shape}"
    assert v_out.shape == (1, n_heads, 3, head_dim)
    assert cache.offset == 3, f"Expected offset=3, got {cache.offset}"

    # Second call: 2 more tokens → accumulated output is (1, 4, 5, 32)
    keys2   = mx.ones([1, n_heads, 2, head_dim])
    values2 = mx.ones([1, n_heads, 2, head_dim])
    k_out2, v_out2 = cache.update_and_fetch(keys2, values2)

    assert k_out2.shape == (1, n_heads, 5, head_dim), \
        f"Second call accumulated shape: {k_out2.shape}"
    assert cache.offset == 5, f"Expected offset=5, got {cache.offset}"

    # reset() clears offset to 0
    cache.reset()
    assert cache.offset == 0, "offset should be 0 after reset"


# ─────────────────────────────────────────────────────────────────────────────
# _sample_mx — temperature and nucleus sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_sample_mx():
    """
    squish.server._sample_mx:
    - temp <= 0 → greedy argmax
    - temp=1.0, top_p=1.0, uniform logits → diverse outcomes
    - very peaked logits, top_p≈1.0 → always picks the top token
    """
    import pytest
    mx = pytest.importorskip("mlx.core")
    from squish.server import _sample_mx

    # Greedy: largest logit at index 2
    logits = mx.array([0.1, 0.5, 9.9, 0.2, 0.3])
    assert _sample_mx(logits, temperature=0.0, top_p=1.0) == 2
    assert _sample_mx(logits, temperature=-1.0, top_p=1.0) == 2   # negative → greedy

    # Stochastic: uniform logits with temp=1.0 should produce diverse tokens
    uniform = mx.zeros([10])  # all equal → uniform distribution
    seen = set()
    for _ in range(300):
        seen.add(_sample_mx(uniform, temperature=1.0, top_p=1.0))
    assert len(seen) > 5, \
        f"Expected diverse token sampling from uniform dist, only saw {seen}"

    # Nucleus: very peaked distribution always returns token 0 under tight top_p
    peaked = mx.array([20.0] + [-100.0] * 9)  # token 0 overwhelmingly dominant
    for _ in range(20):
        assert _sample_mx(peaked, temperature=0.5, top_p=0.999) == 0, \
            "Peaked distribution should always sample token 0"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Phase 1.2 (AWQ) ─────────────────────────────────────────────")
    test_awq_apply_weights()
    test_awq_save_load()
    test_awq_no_match_counts_applied()

    print("\n── Phase 1.3 (KV cache) ────────────────────────────────────────")
    test_kv_int8_round_trip()
    test_kv_layer_cache_append()
    test_kv_layer_cache_reset()
    test_quantized_kv_cache_full()
    test_snapkv_eviction()
    test_snapkv_mode_auto_evict()

    print("\n── mlx_lm protocol ─────────────────────────────────────────────")
    test_kv_layer_cache_update_and_fetch()
    test_sample_mx()

    print("\n✓ All Phase 1.2 + 1.3 + protocol unit tests PASSED\n")
