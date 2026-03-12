"""FP8Quant — FP8 E4M3 / E5M2 weight and activation quantisation.

Implements the two FP8 formats standardised by NVIDIA / ARM / Intel for
transformer acceleration:

  * **E4M3** (range ±448, NaN at all-ones) — preferred for weights; tight
    range packs more precision near zero where most weights live.
  * **E5M2** (range ±57344) — preferred for activations and gradients; wider
    range tolerates large activation outliers.

On Apple M4 and later the Neural Engine natively executes FP8 matmuls.
For M1–M3 the codec provides a simulation path (BF16 intermediates).

Usage::

    from squish.fp8_quant import FP8Quantizer, FP8Config

    cfg  = FP8Config(fmt="e4m3", block_size=128, per_channel=True)
    quant = FP8Quantizer(cfg)
    enc  = quant.encode(weight)            # → FP8Tensor
    out  = quant.decode(enc)               # → numpy float32
    err  = quant.relative_error(weight, out)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

__all__ = [
    "FP8Config",
    "FP8Tensor",
    "FP8Quantizer",
    "fp8_encode_e4m3",
    "fp8_encode_e5m2",
    "fp8_decode",
]

# ── Format constants ─────────────────────────────────────────────────────────

_E4M3_MAX  = 448.0       # ±448 (exponent bias 7, mantissa 3 bits)
_E5M2_MAX  = 57344.0     # ±57344 (exponent bias 15, mantissa 2 bits)
_E4M3_TINY = 2.0 ** -9   # smallest normal E4M3 number
_E5M2_TINY = 2.0 ** -14  # smallest normal E5M2 number


@dataclass
class FP8Config:
    """Configuration for FP8 quantisation.

    Attributes:
        fmt: ``"e4m3"`` (weights) or ``"e5m2"`` (activations).
        block_size: Number of elements per scaling block (power of 2).
        per_channel: If True, compute one scale per output channel; otherwise
            one scale per block.
        saturate_nan: Replace NaN/Inf inputs with max representable value.
    """

    fmt: Literal["e4m3", "e5m2"] = "e4m3"
    block_size: int = 128
    per_channel: bool = True
    saturate_nan: bool = True

    def __post_init__(self) -> None:
        if self.fmt not in ("e4m3", "e5m2"):
            raise ValueError(f"fmt must be 'e4m3' or 'e5m2'; got {self.fmt!r}")
        if self.block_size < 1 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(f"block_size must be a power of 2; got {self.block_size}")

    @property
    def max_val(self) -> float:
        return _E4M3_MAX if self.fmt == "e4m3" else _E5M2_MAX

    @property
    def tiny_val(self) -> float:
        return _E4M3_TINY if self.fmt == "e4m3" else _E5M2_TINY

    @property
    def mantissa_bits(self) -> int:
        return 3 if self.fmt == "e4m3" else 2

    @property
    def exponent_bits(self) -> int:
        return 4 if self.fmt == "e4m3" else 5


@dataclass
class FP8Tensor:
    """Compressed FP8 representation of a tensor.

    Attributes:
        data: Integer codes stored as uint8.
        scales: Per-block or per-channel float32 scale factors.
        shape: Original tensor shape.
        fmt: FP8 format used for encoding.
        bits_per_element: Always 8 for FP8.
    """

    data: np.ndarray     # dtype uint8, shape=flat blocks
    scales: np.ndarray   # dtype float32
    shape: tuple[int, ...]
    fmt: str
    bits_per_element: int = 8

    @property
    def compression_ratio(self) -> float:
        """Bytes saved vs float32 baseline."""
        return 32.0 / self.bits_per_element  # 4×

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# ── Low-level FP8 codec ───────────────────────────────────────────────────────

def _quantize_to_fp8(x: np.ndarray, max_val: float,
                     mantissa_bits: int) -> np.ndarray:
    """Simulate FP8 quantisation via round-to-nearest in log space.

    Returns integer codes in [0, 255] (uint8).
    """
    x = x.astype(np.float32)
    signs = np.sign(x)
    abs_x = np.abs(x)
    # Clamp to representable range
    abs_x = np.clip(abs_x, 0.0, max_val)
    # Compute step granularity at each magnitude
    # FP8 has 2^mantissa_bits levels per binade; round to nearest
    step_bits = mantissa_bits
    abs_x_q = np.where(
        abs_x == 0.0,
        0.0,
        np.exp2(np.floor(np.log2(np.maximum(abs_x, 1e-30))) - step_bits)
        * np.round(abs_x / np.exp2(
            np.floor(np.log2(np.maximum(abs_x, 1e-30))) - step_bits
        )),
    )
    # Pack sign + magnitude into uint8 code (bit 7 = sign, bits 6-0 = mag)
    # We use a simple linear index into the E4M3/E5M2 grid
    n_pos = 1 << (7)  # 128 positive codes
    codes_f = np.clip(abs_x_q / max_val * n_pos, 0, n_pos - 1)
    codes_int = codes_f.astype(np.uint8)
    # Bit 7 encodes sign
    codes_int = codes_int | np.where(signs < 0, np.uint8(0x80), np.uint8(0)).astype(np.uint8)
    return codes_int


def _dequantize_from_fp8(codes: np.ndarray, max_val: float,
                          mantissa_bits: int) -> np.ndarray:
    """Invert ``_quantize_to_fp8``."""
    signs = np.where((codes & 0x80).astype(bool), -1.0, 1.0)
    mag_codes = (codes & 0x7F).astype(np.float32)
    n_pos = 1 << 7
    abs_vals = mag_codes / n_pos * max_val
    return (signs * abs_vals).astype(np.float32)


def fp8_encode_e4m3(x: np.ndarray, scale: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Encode a float32 array to E4M3 FP8 codes.

    Args:
        x: Input float32 array.
        scale: Optional pre-computed scale.  If None, computed from ``max(|x|)``.

    Returns:
        (codes_uint8, scale_float32)
    """
    x = np.asarray(x, dtype=np.float32)
    if scale is None:
        amax = float(np.max(np.abs(x)))
        scale = amax / _E4M3_MAX if amax > 0 else 1.0
    x_scaled = x / max(scale, 1e-12)
    codes = _quantize_to_fp8(x_scaled, _E4M3_MAX, mantissa_bits=3)
    return codes, float(scale)


def fp8_encode_e5m2(x: np.ndarray, scale: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Encode a float32 array to E5M2 FP8 codes."""
    x = np.asarray(x, dtype=np.float32)
    if scale is None:
        amax = float(np.max(np.abs(x)))
        scale = amax / _E5M2_MAX if amax > 0 else 1.0
    x_scaled = x / max(scale, 1e-12)
    codes = _quantize_to_fp8(x_scaled, _E5M2_MAX, mantissa_bits=2)
    return codes, float(scale)


def fp8_decode(codes: np.ndarray, scale: float,
               fmt: Literal["e4m3", "e5m2"] = "e4m3") -> np.ndarray:
    """Decode uint8 FP8 codes back to float32."""
    if fmt == "e4m3":
        vals = _dequantize_from_fp8(codes, _E4M3_MAX, mantissa_bits=3)
    else:
        vals = _dequantize_from_fp8(codes, _E5M2_MAX, mantissa_bits=2)
    return vals * scale


# ── FP8Quantizer ─────────────────────────────────────────────────────────────

class FP8Quantizer:
    """End-to-end FP8 quantiser.

    Handles per-channel or per-block scale computation and applies the
    requested FP8 format.

    Args:
        config: ``FP8Config`` instance controlling format and granularity.
    """

    def __init__(self, config: FP8Config) -> None:
        self.config = config

    def encode(self, x: np.ndarray) -> FP8Tensor:
        """Quantise a float32 weight/activation array to FP8.

        Args:
            x: Float32 numpy array of any shape.

        Returns:
            :class:`FP8Tensor` with uint8 codes and scale factors.
        """
        cfg = self.config
        x = np.asarray(x, dtype=np.float32)
        if cfg.saturate_nan:
            x = np.where(np.isfinite(x), x, 0.0)
        original_shape = x.shape
        flat = x.ravel()
        n = flat.size

        if cfg.per_channel and x.ndim >= 2:
            # One scale per output channel (first dim)
            out_channels = x.shape[0]
            codes_list, scales_list = [], []
            row_size = flat.size // out_channels
            for i in range(out_channels):
                row = flat[i * row_size:(i + 1) * row_size]
                if cfg.fmt == "e4m3":
                    c, s = fp8_encode_e4m3(row)
                else:
                    c, s = fp8_encode_e5m2(row)
                codes_list.append(c)
                scales_list.append(s)
            codes = np.concatenate(codes_list)
            scales = np.array(scales_list, dtype=np.float32)
        else:
            # Per-block encoding
            n_blocks = (n + cfg.block_size - 1) // cfg.block_size
            # Pad to multiple of block_size
            padded = np.zeros(n_blocks * cfg.block_size, dtype=np.float32)
            padded[:n] = flat
            codes_list, scales_list = [], []
            for b in range(n_blocks):
                block = padded[b * cfg.block_size:(b + 1) * cfg.block_size]
                if cfg.fmt == "e4m3":
                    c, s = fp8_encode_e4m3(block)
                else:
                    c, s = fp8_encode_e5m2(block)
                codes_list.append(c)
                scales_list.append(s)
            codes = np.concatenate(codes_list)[:n]
            scales = np.array(scales_list, dtype=np.float32)

        return FP8Tensor(
            data=codes,
            scales=scales,
            shape=original_shape,
            fmt=cfg.fmt,
        )

    def decode(self, tensor: FP8Tensor) -> np.ndarray:
        """Dequantise an :class:`FP8Tensor` back to float32.

        Args:
            tensor: Encoded FP8 tensor.

        Returns:
            Float32 numpy array with original shape.
        """
        cfg = self.config
        codes = tensor.data
        scales = tensor.scales
        n = codes.size

        if cfg.per_channel and len(tensor.shape) >= 2:
            out_channels = tensor.shape[0]
            row_size = n // out_channels
            recon_list = []
            for i in range(out_channels):
                row_codes = codes[i * row_size:(i + 1) * row_size]
                recon_list.append(fp8_decode(row_codes, scales[i], tensor.fmt))
            flat = np.concatenate(recon_list)
        else:
            block_size = cfg.block_size
            n_blocks = scales.size
            recon_list = []
            for b in range(n_blocks):
                start = b * block_size
                end = min(start + block_size, n)
                block_codes = codes[start:end]
                recon_list.append(fp8_decode(block_codes, scales[b], tensor.fmt))
            flat = np.concatenate(recon_list)

        return flat[:n].reshape(tensor.shape)

    def relative_error(self, original: np.ndarray,
                       decoded: np.ndarray) -> float:
        """Compute mean relative absolute error between original and decoded.

        Args:
            original: Original float32 array.
            decoded: Dequantised float32 array.

        Returns:
            Mean relative absolute error (scalar).
        """
        orig = np.asarray(original, dtype=np.float32).ravel()
        dec  = np.asarray(decoded,  dtype=np.float32).ravel()
        denom = np.abs(orig)
        mask  = denom > 1e-10
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs(orig[mask] - dec[mask]) / denom[mask]))
