"""
Squish — fast compressed model loader and OpenAI-compatible server for Apple Silicon.

Public API:
    load_compressed_model(model_dir, npz_path_or_dir, ...)
    load_from_npy_dir(dir_path, model_dir, ...)
    save_int4_npy_dir(npy_dir, group_size=64, verbose=True)

    compress_npy_dir(tensors_dir, level=3, ...)  # zstd entropy compression
    decompress_npy_dir(tensors_dir, ...)

    run_server(...)   # OpenAI-compatible HTTP server
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports of compressed_loader work
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compressed_loader import (  # noqa: F401
    load_compressed_model,
    load_from_npy_dir,
    save_int4_npy_dir,
)

# Entropy compression helpers (optional zstandard dep)
from squish.entropy import compress_npy_dir, decompress_npy_dir  # noqa: F401

# Speculative decoding (requires both target + draft model)
from squish.speculative import SpeculativeGenerator, load_draft_model  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "load_compressed_model",
    "load_from_npy_dir",
    "save_int4_npy_dir",
    "compress_npy_dir",
    "decompress_npy_dir",
    "SpeculativeGenerator",
    "load_draft_model",
]
