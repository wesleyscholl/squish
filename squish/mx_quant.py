"""MXQuant — OCP MX Microscaling (MX4 / MX6 / MX9) quantisation.

Implements the Open Compute Project Microscaling (MX) specification (v1.0).
Microscaling groups elements into *tiles* (typically 32 values) and assigns
a shared 8-bit exponent (E8M0) to each tile, with per-element mantissas of
4, 6, or 9 bits.  This achieves better quality than uniform INT4 at the same
effective bit-width because the shared exponent captures the local dynamic
range of each tile.

Formats:
  * **MX4** (MXFP4) — 1 sign + 2 mantissa bits per element, E8M0 tile scale
  * **MX6** (MXFP6) — 1 sign + 2 exp + 3 mantissa bits per element
  * **MX9** (MXFP9) — BF8 / HF8 variant; 1 sign + 4 exp + 4 mantissa bits

Usage::

    from squish.mx_quant import MXQuantizer, MXConfig

    cfg   = MXConfig(fmt="mx4", tile_size=32)
    quant = MXQuantizer(cfg)
    enc   = quant.encode(weight_matrix)
    rec   = quant.decode(enc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

__all__ = [
    "MXConfig",
    "MXTensor",
    "MXQuantizer",
]

_FMT_BITS = {"mx4": 4, "mx6": 6, "mx9": 9}
_FMT_MANTISSA = {"mx4": 2, "mx6": 3, "mx9": 4}
_FMT_EXP      = {"mx4": 0, "mx6": 2, "mx9": 4}  # element-level exponent bits


@dataclass
class MXConfig:
    """Configuration for MX microscaling quantisation.

    Attributes:
        fmt: ``"mx4"``, ``"mx6"``, or ``"mx9"``.
        tile_size: Number of elements sharing a tile's E8M0 exponent.
        saturate: Clamp to max representable value instead of wrapping.
    """

    fmt: Literal["mx4", "mx6", "mx9"] = "mx4"
    tile_size: int = 32
    saturate: bool = True

    def __post_init__(self) -> None:
        if self.fmt not in _FMT_BITS:
            raise ValueError(f"fmt must be one of {list(_FMT_BITS)}; got {self.fmt!r}")
        if self.tile_size < 1 or (self.tile_size & (self.tile_size - 1)) != 0:
            raise ValueError(f"tile_size must be a power of 2; got {self.tile_size}")

    @property
    def bits_per_element(self) -> int:
        return _FMT_BITS[self.fmt]

    @property
    def mantissa_bits(self) -> int:
        return _FMT_MANTISSA[self.fmt]

    @property
    def element_exp_bits(self) -> int:
        return _FMT_EXP[self.fmt]

    @property
    def compression_ratio(self) -> float:
        """Effective compression vs float32."""
        return 32.0 / self.bits_per_element


@dataclass
class MXTensor:
    """Compressed MX microscaling representation.

    Attributes:
        mantissas: Integer element mantissas, uint8, shape (n_elements,).
        tile_exps: Per-tile E8M0 exponents, uint8, shape (n_tiles,).
        shape: Original tensor shape.
        fmt: MX format string.
    """

    mantissas: np.ndarray   # uint8
    tile_exps: np.ndarray   # uint8 — E8M0 packed exponents
    shape: tuple[int, ...]
    fmt: str

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bits_per_element(self) -> int:
        return _FMT_BITS[self.fmt]


class MXQuantizer:
    """MX microscaling quantiser (MX4 / MX6 / MX9).

    Each tile of ``tile_size`` contiguous elements shares a single 8-bit
    floating-point exponent (E8M0).  Within the tile, each element stores a
    mantissa of ``mantissa_bits`` bits plus sign.

    Args:
        config: ``MXConfig`` controlling format and tile size.
    """

    def __init__(self, config: MXConfig) -> None:
        self.config = config

    def _encode_tile(
        self, tile: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Encode a single tile of elements.

        Returns (mantissa codes uint8, e8m0 exponent).
        """
        cfg = self.config
        tile = tile.astype(np.float32)
        amax = float(np.max(np.abs(tile)))
        if amax == 0.0:
            return np.zeros(len(tile), dtype=np.uint8), 0

        # E8M0: encode amax as 8-bit unbiased exponent
        log2_amax = float(np.floor(np.log2(amax)))
        e8m0 = int(np.clip(log2_amax + 127, 0, 254))  # E8M0 bias=127
        tile_scale = 2.0 ** (e8m0 - 127)

        # Normalise and quantise mantissas
        m_bits = cfg.mantissa_bits
        n_levels = (1 << m_bits) - 1  # positive levels
        normalised = tile / tile_scale
        signs = np.where(normalised < 0, 1, 0).astype(np.uint8)
        abs_norm = np.abs(normalised)
        # Clamp to [0, 1]
        abs_norm = np.clip(abs_norm, 0.0, 1.0)
        man_codes = np.round(abs_norm * n_levels).astype(np.uint8)
        # Pack: sign bit at top, mantissa below
        packed = (signs << m_bits) | man_codes
        return packed.astype(np.uint8), e8m0

    def _decode_tile(
        self, packed: np.ndarray, e8m0: int
    ) -> np.ndarray:
        """Decode a tile given its packed mantissas and E8M0 exponent."""
        cfg = self.config
        m_bits = cfg.mantissa_bits
        n_levels = (1 << m_bits) - 1
        signs = ((packed >> m_bits) & 1).astype(np.float32)
        man_vals = (packed & n_levels).astype(np.float32) / n_levels
        tile_scale = 2.0 ** (e8m0 - 127)
        return (1 - 2 * signs) * man_vals * tile_scale

    def encode(self, x: np.ndarray) -> MXTensor:
        """Quantise a float32 tensor using MX microscaling.

        Args:
            x: Float32 numpy array of any shape.

        Returns:
            :class:`MXTensor` with packed mantissas and tile exponents.
        """
        cfg = self.config
        original_shape = tuple(x.shape)
        flat = np.asarray(x, dtype=np.float32).ravel()
        n = flat.size
        ts = cfg.tile_size
        n_tiles = (n + ts - 1) // ts

        padded = np.zeros(n_tiles * ts, dtype=np.float32)
        padded[:n] = flat

        man_list, exp_list = [], []
        for t in range(n_tiles):
            tile = padded[t * ts:(t + 1) * ts]
            packed, e8m0 = self._encode_tile(tile)
            man_list.append(packed)
            exp_list.append(e8m0)

        mantissas = np.concatenate(man_list)[:n]
        tile_exps = np.array(exp_list, dtype=np.uint8)
        return MXTensor(
            mantissas=mantissas,
            tile_exps=tile_exps,
            shape=original_shape,
            fmt=cfg.fmt,
        )

    def decode(self, tensor: MXTensor) -> np.ndarray:
        """Dequantise an :class:`MXTensor` back to float32.

        Args:
            tensor: Compressed MX tensor.

        Returns:
            Float32 numpy array with original shape.
        """
        cfg = self.config
        mantissas = tensor.mantissas
        tile_exps = tensor.tile_exps
        n = mantissas.size
        ts = cfg.tile_size

        # Pad mantissas to tile boundary
        n_tiles = tile_exps.size
        padded = np.zeros(n_tiles * ts, dtype=np.uint8)
        padded[:n] = mantissas

        recon_list = []
        for t in range(n_tiles):
            tile_packed = padded[t * ts:(t + 1) * ts]
            recon_list.append(self._decode_tile(tile_packed, int(tile_exps[t])))

        flat = np.concatenate(recon_list)[:n]
        return flat.reshape(tensor.shape)

    def snr_db(self, original: np.ndarray, decoded: np.ndarray) -> float:
        """Signal-to-noise ratio in dB between original and decoded tensors.

        Args:
            original: Reference float32 array.
            decoded: Dequantised float32 array.

        Returns:
            SNR in decibels (higher is better).
        """
        orig = np.asarray(original, dtype=np.float32).ravel()
        dec  = np.asarray(decoded,  dtype=np.float32).ravel()
        signal_power = float(np.mean(orig ** 2))
        noise_power  = float(np.mean((orig - dec) ** 2))
        if noise_power == 0:
            return float("inf")
        return 10.0 * np.log10(max(signal_power, 1e-30) / max(noise_power, 1e-30))
