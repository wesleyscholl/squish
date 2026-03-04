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
# Phase 1.2 — AWQ calibration and scale application
from squish.awq import (  # noqa: F401
    apply_awq_to_weights,
    collect_activation_scales,
    load_awq_scales,
    save_awq_scales,
)

# Model catalog + pull (requires huggingface_hub for download)
from squish.catalog import (  # noqa: F401
    CatalogEntry,
    list_catalog,
    load_catalog,
)
from squish.catalog import (
    pull as pull_model,
)
from squish.catalog import (
    resolve as resolve_model,
)

# Entropy compression helpers (optional zstandard dep)
from squish.entropy import compress_npy_dir, decompress_npy_dir  # noqa: F401

# Phase 2.3 — Flash Attention status + benchmarking
from squish.flash_attention import (  # noqa: F401
    PatchResult,
    attention_status,
    patch_model_attention,
    predict_memory_savings,
    print_memory_table,
)

# Phase 1.3 — Quantized KV cache (KIVI + SnapKV)
from squish.kv_cache import (  # noqa: F401
    QuantizedKVCache,
    make_quantized_cache,
    patch_model_kv_cache,
)

# Phase 2.2 — Layer-wise streaming for 70B+ models
from squish.layerwise_loader import (  # noqa: F401
    LayerCache,
    LayerwiseLoader,
    LoadStats,
    recommend_cache_size,
    shard_model,
)

# Quantizer public API (self-contained, no external deps beyond numpy)
from squish.quantizer import (  # noqa: F401
    QuantizationResult,
    dequantize_int4,
    get_backend_info,
    mean_cosine_similarity,
    quantize_embeddings,
    quantize_int4,
    reconstruct_embeddings,
)

# Speculative decoding (requires both target + draft model)
from squish.speculative import SpeculativeGenerator, load_draft_model  # noqa: F401

# Phase 2.1C — CPU/GPU split loading (auto-offloads layers if model > Metal budget)
from squish.split_loader import (  # noqa: F401
    OffloadedLayer,
    SplitInfo,
    SplitLayerLoader,
    print_layer_profile,
    profile_model_layers,
)

from .compressed_loader import (  # noqa: F401
    load_compressed_model,
    load_from_npy_dir,
    save_int4_npy_dir,
)

__version__ = "1.0.0"
__all__ = [
    "load_compressed_model",
    "load_from_npy_dir",
    "save_int4_npy_dir",
    "compress_npy_dir",
    "decompress_npy_dir",
    "SpeculativeGenerator",
    "load_draft_model",
    # Phase 1.2 — AWQ
    "collect_activation_scales",
    "save_awq_scales",
    "load_awq_scales",
    "apply_awq_to_weights",
    # Phase 1.3 — KV cache
    "QuantizedKVCache",
    "make_quantized_cache",
    "patch_model_kv_cache",
    # Phase 2.1C — CPU/GPU split
    "SplitLayerLoader",
    "SplitInfo",
    "OffloadedLayer",
    "profile_model_layers",
    "print_layer_profile",
    # Phase 2.2 — Layerwise streaming
    "LayerCache",
    "LayerwiseLoader",
    "LoadStats",
    "shard_model",
    "recommend_cache_size",
    # Phase 2.3 — Flash Attention
    "patch_model_attention",
    "attention_status",
    "predict_memory_savings",
    "print_memory_table",
    "PatchResult",
    # Quantizer API
    "QuantizationResult",
    "quantize_embeddings",
    "reconstruct_embeddings",
    "quantize_int4",
    "dequantize_int4",
    "mean_cosine_similarity",
    "get_backend_info",
    # Catalog + pull
    "CatalogEntry",
    "load_catalog",
    "list_catalog",
    "resolve_model",
    "pull_model",
    # Phase 2.1 — BatchScheduler  (import: from squish.scheduler import BatchScheduler)
    # Phase 2.2 — Tool calling    (import: from squish.tool_calling import ...)
    # Phase 2.2 — Ollama compat   (import: from squish.ollama_compat import mount_ollama)
]

