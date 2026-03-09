"""
squish/sub_spec.py

SubSpec — Speculative Decoding for Offloaded LLMs via Quantized Substitute
Layers.

Based on:
  "SubSpec: Speculative Decoding with Quantized Substitute Layers for
   Memory-Constrained LLM Inference"
  NeurIPS 2025 — arXiv:2509.18344

Key insight
-----------
When a model is too large for GPU memory, layers are offloaded to CPU/NVMe.
Standard speculative decoding requires a *separate* smaller draft model,
doubling memory pressure.  SubSpec avoids this by constructing the draft
entirely from GPU-resident materials:

* **Shared real layers** — transformer layers that fit in GPU memory are
  used identically for both draft and target: zero alignment gap, zero extra
  memory.

* **Quantized substitutes** — each offloaded full-precision layer gets a
  4-bit compressed resident copy that stays in GPU memory as the draft
  substitute.  These are small (8× compressed) yet closely approximate the
  offloaded computation.

* **Shared KV cache** — draft and target share the same KV state.  On each
  decode step the draft appends to the shared KV; the verifier reads from
  and extends the same cache.  This halves KV memory vs. separate
  draft/target caches.

Measured speedup
----------------
* Qwen2.5 7B  on  8 GB VRAM:  9.1×  on MT-Bench
* Qwen2.5 32B on 24 GB VRAM: 12.5×  on popular generation benchmarks

Usage
-----
    from squish.sub_spec import SubSpecConfig, SubSpecDecoder

    cfg = SubSpecConfig(n_total_layers=32, n_gpu_layers=16, gamma=4)
    dec = SubSpecDecoder(draft_fn=draft_fn, target_fn=target_fn, config=cfg)
    output_ids, stats = dec.generate(prompt_ids, max_new_tokens=128)

Provides
--------
  SubSpecConfig           — configuration dataclass.
  SubstituteLayerProxy    — INT4-quantized approximation of one offloaded layer.
  SubSpecStats            — per-generation counters.
  SubSpecDecoder          — draft → verify speculative decode loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "SubSpecConfig",
    "SubstituteLayerProxy",
    "SubSpecStats",
    "SubSpecDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D float array."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _fake_quant_int4(w: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Fake-quantize *w* to INT4 (per-group symmetric) and return dequantized
    float32.

    Simulates the INT4 weight substitution: full-precision offloaded weights
    are replaced by their nearest INT4 representation, kept in float32 format
    for portability (production: pack as nibbles in Metal buffers).
    """
    orig  = w.shape
    gs    = min(group_size, w.size)
    flat  = w.reshape(-1)
    n     = len(flat)
    out   = np.empty(n, dtype=np.float32)
    for start in range(0, n, gs):
        grp   = flat[start : start + gs].astype(np.float32)
        amax  = float(np.max(np.abs(grp)))
        if amax < 1e-12:
            out[start : start + gs] = 0.0
            continue
        scale = amax / 7.0
        q     = np.round(grp / scale).clip(-7, 7).astype(np.int8).astype(np.float32)
        out[start : start + gs] = q * scale
    return out.reshape(orig)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SubSpecConfig:
    """Configuration for SubSpec speculative decoding.

    Parameters
    ----------
    n_total_layers : int
        Total transformer layers in the model.
    n_gpu_layers : int
        Layers that remain GPU-resident (shared identically between draft and
        target).  Layers from ``n_gpu_layers`` onward are offloaded to NVMe
        for the target but are represented by INT4 substitutes for the draft.
    gamma : int
        Draft tokens generated per verification step (≥ 1).
    quant_bits : int
        Bit-width for substitute-layer weight quantization (2, 4, or 8).
        The SubSpec paper uses 4-bit as the sweet spot.
    temperature : float
        Sampling temperature (> 0).
    top_p : float
        Nucleus sampling cutoff (0, 1].
    """

    n_total_layers: int   = 32
    n_gpu_layers:   int   = 16
    gamma:          int   = 4
    quant_bits:     int   = 4
    temperature:    float = 1.0
    top_p:          float = 1.0

    def __post_init__(self) -> None:
        if self.n_total_layers < 1:
            raise ValueError("n_total_layers must be ≥ 1")
        if self.n_gpu_layers < 0:
            raise ValueError("n_gpu_layers must be ≥ 0")
        if self.n_gpu_layers > self.n_total_layers:
            raise ValueError("n_gpu_layers must be ≤ n_total_layers")
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if self.quant_bits not in (2, 4, 8):
            raise ValueError("quant_bits must be 2, 4, or 8")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def n_substitute_layers(self) -> int:
        """Number of layers represented by quantized substitutes in the draft."""
        return self.n_total_layers - self.n_gpu_layers


# ---------------------------------------------------------------------------
# Quantized Substitute Layer Proxy
# ---------------------------------------------------------------------------

class SubstituteLayerProxy:
    """
    INT4-quantized substitute for a single offloaded transformer layer.

    In SubSpec, each layer that cannot fit in GPU memory at full precision
    gets a compressed resident copy used only during the *draft* forward
    pass.  The *target* (verify) pass fetches the full-precision version from
    CPU/NVMe.

    This class quantizes the supplied weight matrix to 4-bit and provides a
    ``forward`` method that computes an approximate matrix-vector product.
    Memory footprint is approximately 8× smaller than float16.

    Parameters
    ----------
    weight : np.ndarray
        Full-precision weight matrix of shape ``(out_dim, in_dim)``.
    group_size : int
        Quantization group size along the ``in_dim`` axis (default 32).
    """

    def __init__(self, weight: np.ndarray, group_size: int = 32) -> None:
        if weight.ndim != 2:
            raise ValueError("weight must be 2-D, got shape " + str(weight.shape))
        if group_size < 1:
            raise ValueError("group_size must be ≥ 1")
        self._w_q       = _fake_quant_int4(weight.astype(np.float32), group_size)
        self.out_dim, self.in_dim = self._w_q.shape
        self._group_size = group_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the quantized substitute weight to input *x*.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(in_dim,)`` or ``(batch, in_dim)``.

        Returns
        -------
        np.ndarray of shape ``(out_dim,)`` or ``(batch, out_dim)``.
        """
        return x.astype(np.float32) @ self._w_q.T

    @property
    def compression_ratio(self) -> float:
        """Memory ratio of the substitute vs. float32: always 4/32 = 0.125."""
        return 4.0 / 32.0  # INT4 over float32


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SubSpecStats:
    """Per-generation counters returned by :class:`SubSpecDecoder`."""

    total_tokens:   int = 0
    draft_steps:    int = 0
    accepted_total: int = 0
    rejected_total: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted."""
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SubSpecDecoder:
    """
    Speculative decode loop for memory-offloaded models using quantized
    substitute layers.

    Both callables receive the current context as a list of token ids and
    return ``(vocab_size,)`` float32 logits for the *next* token position.
    The caller is responsible for wiring in the correct compute paths:

    * ``draft_fn`` — runs GPU-resident real layers
      (0..``n_gpu_layers``) + INT4 quantized substitute layers
      (``n_gpu_layers``..``n_total_layers``); everything stays on-device.

    * ``target_fn`` — runs all layers in full precision, paging offloaded
      layers from CPU/NVMe storage as needed.

    The shared KV cache is implicit: both callables always receive the same
    ``ids`` list (the complete context), so attention covers the same
    history whether drafting or verifying.

    Parameters
    ----------
    draft_fn : callable
        ``draft_fn(ids: list[int]) -> np.ndarray``
    target_fn : callable
        ``target_fn(ids: list[int]) -> np.ndarray``
    config : SubSpecConfig
    rng_seed : int, optional
    """

    def __init__(
        self,
        draft_fn:  Callable[[List[int]], np.ndarray],
        target_fn: Callable[[List[int]], np.ndarray],
        config:    SubSpecConfig,
        rng_seed:  int = 0,
    ) -> None:
        self._draft  = draft_fn
        self._target = target_fn
        self._cfg    = config
        self._rng    = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray) -> int:
        """Sample a token id from *logits* with temperature + top-p."""
        cfg   = self._cfg
        probs = _softmax(logits / max(cfg.temperature, 1e-8))
        if cfg.top_p < 1.0:
            idx    = np.argsort(-probs)
            cumsum = np.cumsum(probs[idx])
            cutoff = int((cumsum < cfg.top_p).sum()) + 1
            mask   = np.zeros_like(probs)
            mask[idx[:max(1, cutoff)]] = 1.0
            total = (probs * mask).sum()
            probs = probs * mask / (total + 1e-12)
        try:
            return int(self._rng.choice(len(probs), p=probs))
        except ValueError:
            return int(np.argmax(logits))

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], SubSpecStats]:
        """
        Generate up to *max_new_tokens* tokens using SubSpec.

        Each round:
        1. **Draft** γ tokens via the GPU-resident + substitute path.
        2. **Verify** each draft token with the full target model using the
           Leviathan accept/reject rule.
        3. Extend context with accepted tokens; on the first rejection add
           one correction token from the target distribution.
        4. On full acceptance add one bonus token from the target.

        The shared KV is maintained implicitly: both paths always see the
        full ``ids`` list, so no explicit KV synchronisation is required.

        Parameters
        ----------
        input_ids : list[int]
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats)
        """
        cfg       = self._cfg
        stats     = SubSpecStats()
        ids       = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # ── Draft phase (GPU-resident + quantized substitutes) ────────────
            draft_ids:   List[int]        = []
            draft_probs: List[np.ndarray] = []
            ctx = list(ids)

            for _ in range(cfg.gamma):
                if generated + len(draft_ids) >= max_new_tokens:
                    break
                d_logits = self._draft(ctx)
                d_probs  = _softmax(d_logits)
                d_tok    = self._sample(d_logits)
                draft_ids.append(d_tok)
                draft_probs.append(d_probs)
                ctx.append(d_tok)

            if not draft_ids:
                break

            stats.draft_steps += 1

            # ── Verify phase (full target model, may page from NVMe) ──────────
            ctx       = list(ids)
            accepted: List[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs):
                t_logits = self._target(ctx)
                t_probs  = _softmax(t_logits)
                v_tok    = self._sample(t_logits)
                p_t      = float(t_probs[d_tok])
                p_d      = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    stats.accepted_total += 1
                else:
                    # Correction token from the target distribution
                    accepted.append(v_tok)
                    ctx.append(v_tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            if not rejected and (generated + len(accepted)) < max_new_tokens:
                # Bonus token: one free target-model token on full acceptance
                bonus_logits = self._target(ctx)
                bonus_tok    = self._sample(bonus_logits)
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            ids.extend(accepted)
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
