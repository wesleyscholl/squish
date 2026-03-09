"""
squish/qspec.py

QSpec — Complementary Quantization Speculative Decoding (W4A8 draft /
W4A16 verify).

Based on:
  "QSpec: Speculative Decoding with Complementary Quantization Schemes"
  arXiv:2410.11305

Key insight
-----------
Standard speculative decoding uses the same quantization precision for both
draft and verify passes (e.g. both W4A16).  QSpec discovers a better split
by using *complementary* activation precisions on the same INT4 weight file:

* **Draft pass — W4A8** (INT4 weights, INT8 activations) — the combined
  quantization reduces compute and memory bandwidth.  On hardware with
  native INT8 matrix-multiply units (Apple M3 Neural Engine, NVIDIA tensor
  cores), INT8 activations map directly to hardware operations, yielding
  genuine throughput gains over mixed INT4×FP16.

* **Verify pass — W4A16** (INT4 weights, FP16 activations) — the standard
  MLX INT4 inference path.  Higher quality than W4A8 for a small bandwidth
  cost; acceptable because there is only one verify call per draft round.

Critically: **only one model file is required**.  Draft and verify use the
same INT4 weight tensors; the only difference is the activation precision.
On an M3 implementation this maps to two Metal shader variants rather than
two separate model checkpoints.

Performance
-----------
QSpec achieves up to **1.64×** wall-clock speedup vs. full-precision
baselines without quality degradation, as the INT4 weights are shared and
only the activation path changes.

Usage
-----
    from squish.qspec import QSpecConfig, QSpecDecoder

    cfg = QSpecConfig(gamma=4)
    dec = QSpecDecoder(
        w4a8_fn=lambda ids: model_w4a8_forward(ids),   # draft
        w4a16_fn=lambda ids: model_w4a16_forward(ids), # verify
        config=cfg,
    )
    output_ids, stats = dec.generate(prompt_ids, max_new_tokens=128)

Provides
--------
  QSpecConfig          — configuration dataclass.
  ActivationQuantizer  — per-group INT8 activation quantizer (W4A8 path).
  QSpecStats           — per-generation counters.
  QSpecDecoder         — W4A8 draft + W4A16 verify speculative loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "QSpecConfig",
    "ActivationQuantizer",
    "QSpecStats",
    "QSpecDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QSpecConfig:
    """Configuration for QSpec complementary-quantization speculative decoding.

    Parameters
    ----------
    gamma : int
        Draft tokens generated per verification step (≥ 1).
    draft_act_bits : int
        Activation quantization bits for the draft pass.  Must be 4 or 8.
        The QSpec paper uses 8 (W4A8) as the optimal draft configuration for
        M3 and NVIDIA hardware with INT8 compute units.
    verify_act_bits : int
        Activation precision for the verify pass.  Must be 8 or 16.  16 means
        FP16 (no quantization), which is the standard MLX INT4 inference path.
        Must be strictly greater than ``draft_act_bits``.
    group_size : int
        Elements-per-scale for activation quantization.  128 is a safe default.
    temperature : float
        Sampling temperature (> 0).
    top_p : float
        Nucleus sampling probability (0, 1].
    """

    gamma:           int   = 4
    draft_act_bits:  int   = 8
    verify_act_bits: int   = 16
    group_size:      int   = 128
    temperature:     float = 1.0
    top_p:           float = 1.0

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if self.draft_act_bits not in (4, 8):
            raise ValueError("draft_act_bits must be 4 or 8")
        if self.verify_act_bits not in (8, 16):
            raise ValueError("verify_act_bits must be 8 or 16")
        if self.draft_act_bits >= self.verify_act_bits:
            raise ValueError(
                f"draft_act_bits ({self.draft_act_bits}) must be < "
                f"verify_act_bits ({self.verify_act_bits})"
            )
        if self.group_size < 1:
            raise ValueError("group_size must be ≥ 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


# ---------------------------------------------------------------------------
# Activation Quantizer
# ---------------------------------------------------------------------------

class ActivationQuantizer:
    """
    Symmetric per-group activation quantizer simulating the W4A8 draft path.

    In a Metal shader implementation, activations are quantized to INT8 just
    before each weight-matrix GEMM, enabling the MAC units to perform
    INT8×INT4 operations (vs. FP16×INT4 on the standard verify path).

    This class provides:
    * ``quantize(x)`` — fake-quantizes activations to ``bits`` and returns
      a float32 array with the same shape (round-trip simulation).
    * ``bits_saved_fraction()`` — reports the memory bandwidth savings vs.
      FP16.

    Parameters
    ----------
    bits : int
        Quantization bit-width.  4 → INT4 activations (aggressive), 8 →
        INT8 (W4A8 sweet spot for M3), 16 → FP16 no-op (verify path).
    group_size : int
        Elements sharing one scale factor (≥ 1).
    """

    def __init__(self, bits: int = 8, group_size: int = 128) -> None:
        if bits not in (4, 8, 16):
            raise ValueError("bits must be 4, 8, or 16")
        if group_size < 1:
            raise ValueError("group_size must be ≥ 1")
        self.bits       = bits
        self.group_size = group_size
        self._max_int   = (1 << (bits - 1)) - 1  # 127 for INT8, 7 for INT4

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Fake-quantize activation tensor *x* to ``bits``-bit precision,
        returning a float32 array of the same shape.

        When ``bits=16`` this is a no-op (FP16 pass-through).

        Parameters
        ----------
        x : np.ndarray
            Activation array of any shape.  The last dimension is grouped
            into ``group_size`` elements sharing one scale.
        """
        if self.bits == 16:
            return x.astype(np.float32)

        flat  = x.reshape(-1).astype(np.float32)
        n     = len(flat)
        out   = np.empty(n, dtype=np.float32)
        maxi  = float(self._max_int)
        gs    = self.group_size

        for start in range(0, n, gs):
            grp  = flat[start : start + gs]
            amax = float(np.max(np.abs(grp)))
            if amax < 1e-12:
                out[start : start + gs] = 0.0
                continue
            scale = amax / maxi
            q     = np.round(grp / scale).clip(-maxi, maxi)
            out[start : start + gs] = q * scale

        return out.reshape(x.shape)

    def bits_saved_fraction(self) -> float:
        """
        Fraction of bit-budget saved vs. FP16.

        INT8 saves 50%; INT4 saves 75%; FP16 saves 0%.
        """
        return (16 - self.bits) / 16.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class QSpecStats:
    """Per-generation counters returned by :class:`QSpecDecoder`."""

    total_tokens:   int = 0
    draft_steps:    int = 0
    accepted_total: int = 0
    rejected_total: int = 0

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class QSpecDecoder:
    """
    Speculative decode loop switching between W4A8 draft and W4A16 verify.

    Both callables operate on the *same* INT4 weight tensors; the only
    difference is the activation precision used inside the forward pass:

    * ``w4a8_fn`` — quantizes activations to INT8 before each GEMM (faster
      draft, slightly more quantization noise from activations).
    * ``w4a16_fn`` — keeps activations in FP16 (standard MLX INT4 path,
      higher quality, used for verification).

    The acceptance test uses the standard Leviathan (2022) accept/reject rule
    on the token probability ratio p_target / p_draft.

    An :class:`ActivationQuantizer` is exposed as a public attribute
    (``decoder.act_quantizer``) so callers can apply the same quantization
    when building their ``w4a8_fn``.

    Parameters
    ----------
    w4a8_fn : callable
        ``w4a8_fn(ids: list[int]) -> np.ndarray``
        Forward pass with INT8 activations; returns ``(vocab_size,)`` logits.
    w4a16_fn : callable
        ``w4a16_fn(ids: list[int]) -> np.ndarray``
        Forward pass with FP16 activations; returns ``(vocab_size,)`` logits.
    config : QSpecConfig
    rng_seed : int, optional
    """

    def __init__(
        self,
        w4a8_fn:  Callable[[list[int]], np.ndarray],
        w4a16_fn: Callable[[list[int]], np.ndarray],
        config:   QSpecConfig = None,
        rng_seed: int = 0,
    ) -> None:
        if config is None:
            config = QSpecConfig()
        self._draft        = w4a8_fn
        self._verify       = w4a16_fn
        self._cfg          = config
        self._rng          = np.random.default_rng(rng_seed)
        self.act_quantizer = ActivationQuantizer(
            bits       = config.draft_act_bits,
            group_size = config.group_size,
        )

    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray) -> int:
        probs = _softmax(logits / max(self._cfg.temperature, 1e-8))
        if self._cfg.top_p < 1.0:
            idx    = np.argsort(-probs)
            cumsum = np.cumsum(probs[idx])
            cutoff = int((cumsum < self._cfg.top_p).sum()) + 1
            mask   = np.zeros_like(probs)
            mask[idx[:max(1, cutoff)]] = 1.0
            s     = (probs * mask).sum()
            probs = probs * mask / (s + 1e-12)
        try:
            return int(self._rng.choice(len(probs), p=probs))
        except ValueError:
            return int(np.argmax(logits))

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
    ) -> tuple[list[int], QSpecStats]:
        """
        Generate up to *max_new_tokens* tokens using W4A8 draft + W4A16 verify.

        Each round:
        1. **Draft** γ tokens using the W4A8 (INT8 activation) forward path.
        2. **Verify** each draft token with the W4A16 (FP16 activation) path
           using the Leviathan accept/reject rule.
        3. Append accepted tokens; on the first rejection add one correction
           token from the W4A16 distribution.
        4. On full acceptance add one bonus token from W4A16.

        Parameters
        ----------
        input_ids : list[int]
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats)
        """
        cfg       = self._cfg
        stats     = QSpecStats()
        ids       = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # ── Draft phase (W4A8 — fast INT8-activation path) ────────────────
            draft_ids:   list[int]        = []
            draft_probs: list[np.ndarray] = []
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

            # ── Verify phase (W4A16 — FP16-activation path) ───────────────────
            ctx       = list(ids)
            accepted: list[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs, strict=False):
                v_logits = self._verify(ctx)
                v_probs  = _softmax(v_logits)
                v_tok    = self._sample(v_logits)
                p_t      = float(v_probs[d_tok])
                p_d      = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    stats.accepted_total += 1
                else:
                    # Correction: one token sampled from the W4A16 distribution
                    accepted.append(v_tok)
                    ctx.append(v_tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            if not rejected and (generated + len(accepted)) < max_new_tokens:
                # Bonus token: W4A16 forward pass on the accepted context
                bonus_logits = self._verify(ctx)
                bonus_tok    = self._sample(bonus_logits)
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            ids.extend(accepted)
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
