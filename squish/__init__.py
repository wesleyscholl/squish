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
from __future__ import annotations

__version__ = "9.0.0"

# ── Lazy import registry ───────────────────────────────────────────────────────
# Every public name is loaded on first access via __getattr__.
# This keeps `import squish` fast regardless of how many wave modules exist.
_LAZY_IMPORTS: dict[str, str] = {
    # squish.ada_serve
    "AdaServeConfig":            "squish.ada_serve",
    "AdaServeRequest":           "squish.ada_serve",
    "AdaServeScheduler":         "squish.ada_serve",
    "AdaServeStats":             "squish.ada_serve",
    "SLOTarget":                 "squish.ada_serve",

    # squish.awq
    "apply_awq_to_weights":      "squish.awq",
    "collect_activation_scales": "squish.awq",
    "load_awq_scales":           "squish.awq",
    "save_awq_scales":           "squish.awq",

    # squish.catalog
    "CatalogEntry":              "squish.catalog",
    "list_catalog":              "squish.catalog",
    "load_catalog":              "squish.catalog",
    "pull_model":                "squish.catalog",
    "resolve_model":             "squish.catalog",

    # squish.cla
    "CLAConfig":                 "squish.cla",
    "CLALayerSpec":              "squish.cla",
    "CLASchedule":               "squish.cla",
    "CLAStats":                  "squish.cla",

    # squish.compressed_loader
    "load_compressed_model":     "squish.compressed_loader",
    "load_from_npy_dir":         "squish.compressed_loader",
    "save_int4_npy_dir":         "squish.compressed_loader",

    # squish.conf_spec
    "ConfSpecConfig":            "squish.conf_spec",
    "ConfSpecDecision":          "squish.conf_spec",
    "ConfSpecStats":             "squish.conf_spec",
    "ConfSpecVerifier":          "squish.conf_spec",

    # squish.dfloat11
    "CompressedBlock":           "squish.dfloat11",
    "CompressedModel":           "squish.dfloat11",
    "DFloat11Compressor":        "squish.dfloat11",
    "DFloat11Config":            "squish.dfloat11",
    "HuffmanCodec":              "squish.dfloat11",
    "compress_model":            "squish.dfloat11",

    # squish.diffkv
    "CompactedKVSlot":           "squish.diffkv",
    "DiffKVConfig":              "squish.diffkv",
    "DiffKVPolicy":              "squish.diffkv",
    "DiffKVPolicyManager":       "squish.diffkv",
    "DiffKVStats":               "squish.diffkv",
    "HeadSparsityProfile":       "squish.diffkv",
    "TokenImportanceTier":       "squish.diffkv",

    # squish.dovetail
    "DovetailCPUVerifier":       "squish.dovetail",
    "DovetailConfig":            "squish.dovetail",
    "DovetailDecoder":           "squish.dovetail",
    "DovetailDraftRunner":       "squish.dovetail",
    "DovetailStats":             "squish.dovetail",

    # squish.duo_decoding
    "DuoCandidate":              "squish.duo_decoding",
    "DuoCPUVerifier":            "squish.duo_decoding",
    "DuoDecodingConfig":         "squish.duo_decoding",
    "DuoDecodingDecoder":        "squish.duo_decoding",
    "DuoDecodingStats":          "squish.duo_decoding",
    "DuoScheduler":              "squish.duo_decoding",

    # squish.entropy
    "compress_npy_dir":          "squish.entropy",
    "decompress_npy_dir":        "squish.entropy",

    # squish.flash_attention
    "PatchResult":               "squish.flash_attention",
    "attention_status":          "squish.flash_attention",
    "patch_model_attention":     "squish.flash_attention",
    "predict_memory_savings":    "squish.flash_attention",
    "print_memory_table":        "squish.flash_attention",

    # squish.forelen
    "EGTPPredictor":             "squish.forelen",
    "ForelenConfig":             "squish.forelen",
    "ForelenStats":              "squish.forelen",
    "PLPPredictor":              "squish.forelen",

    # squish.fr_spec
    "FRSpecCalibrator":          "squish.fr_spec",
    "FRSpecConfig":              "squish.fr_spec",
    "FRSpecHead":                "squish.fr_spec",
    "FRSpecStats":               "squish.fr_spec",
    "FreqTokenSubset":           "squish.fr_spec",

    # squish.gemfilter
    "AttentionScoreBuffer":      "squish.gemfilter",
    "GemFilterConfig":           "squish.gemfilter",
    "GemFilterStats":            "squish.gemfilter",
    "GemSelector":               "squish.gemfilter",

    # squish.ipw
    "IPWConfig":                 "squish.ipw",
    "IPWMeasurement":            "squish.ipw",
    "IPWSummary":                "squish.ipw",
    "IPWTracker":                "squish.ipw",

    # squish.kv_cache
    "DiskKVCache":               "squish.kv_cache",
    "KVBudgetBroker":            "squish.kv_cache",
    "QuantizedKVCache":          "squish.kv_cache",
    "make_quantized_cache":      "squish.kv_cache",
    "patch_model_kv_cache":      "squish.kv_cache",

    # squish.kv_slab
    "KVPage":                    "squish.kv_slab",
    "KVSlabAllocator":           "squish.kv_slab",

    # squish.kvsharer
    "KVLayerCache":              "squish.kvsharer",
    "KVShareMap":                "squish.kvsharer",
    "KVSharerCalibrator":        "squish.kvsharer",
    "KVSharerConfig":            "squish.kvsharer",
    "KVSharerStats":             "squish.kvsharer",

    # squish.kvtuner
    "KVQuantConfig":             "squish.kvtuner",
    "KVTunerCalibrator":         "squish.kvtuner",
    "KVTunerConfig":             "squish.kvtuner",
    "KVTunerStats":              "squish.kvtuner",
    "LayerSensitivity":          "squish.kvtuner",

    # squish.layer_skip
    "ConfidenceEstimator":       "squish.layer_skip",
    "EarlyExitConfig":           "squish.layer_skip",
    "EarlyExitDecoder":          "squish.layer_skip",
    "EarlyExitStats":            "squish.layer_skip",

    # squish.layerwise_loader
    "LayerCache":                "squish.layerwise_loader",
    "LayerwiseLoader":           "squish.layerwise_loader",
    "LoadStats":                 "squish.layerwise_loader",
    "recommend_cache_size":      "squish.layerwise_loader",
    "shard_model":               "squish.layerwise_loader",

    # squish.long_spec
    "LongSpecConfig":            "squish.long_spec",
    "LongSpecDecoder":           "squish.long_spec",
    "LongSpecHead":              "squish.long_spec",
    "LongSpecStats":             "squish.long_spec",

    # squish.lookahead_reasoning
    "LookaheadBatch":            "squish.lookahead_reasoning",
    "LookaheadConfig":           "squish.lookahead_reasoning",
    "LookaheadReasoningEngine":  "squish.lookahead_reasoning",
    "LookaheadStats":            "squish.lookahead_reasoning",
    "LookaheadStep":             "squish.lookahead_reasoning",

    # squish.lora_manager
    "DareTiesConfig":            "squish.lora_manager",
    "DareTiesMerger":            "squish.lora_manager",
    "LoRAManager":               "squish.lora_manager",

    # squish.mirror_sd
    "MirrorDraftPipeline":       "squish.mirror_sd",
    "MirrorFuture":              "squish.mirror_sd",
    "MirrorSDConfig":            "squish.mirror_sd",
    "MirrorSDDecoder":           "squish.mirror_sd",
    "MirrorSDStats":             "squish.mirror_sd",
    "MirrorVerifyPipeline":      "squish.mirror_sd",

    # squish.paged_attention
    "BlockAllocator":            "squish.paged_attention",
    "PageBlockTable":            "squish.paged_attention",
    "PagedKVCache":              "squish.paged_attention",

    # squish.paris_kv
    "ParisKVCodebook":           "squish.paris_kv",
    "ParisKVConfig":             "squish.paris_kv",

    # squish.pipo
    "INT4BypassKernel":          "squish.pipo",
    "LayerWeightBuffer":         "squish.pipo",
    "PIPOConfig":                "squish.pipo",
    "PIPOScheduler":             "squish.pipo",

    # squish.prompt_lookup
    "NGramIndex":                "squish.prompt_lookup",
    "PromptLookupConfig":        "squish.prompt_lookup",
    "PromptLookupDecoder":       "squish.prompt_lookup",
    "PromptLookupStats":         "squish.prompt_lookup",

    # squish.qspec
    "ActivationQuantizer":       "squish.qspec",
    "QSpecConfig":               "squish.qspec",
    "QSpecDecoder":              "squish.qspec",
    "QSpecStats":                "squish.qspec",

    # squish.quantizer
    "QuantizationResult":        "squish.quantizer",
    "dequantize_int4":           "squish.quantizer",
    "get_backend_info":          "squish.quantizer",
    "mean_cosine_similarity":    "squish.quantizer",
    "quantize_embeddings":       "squish.quantizer",
    "quantize_int4":             "squish.quantizer",
    "reconstruct_embeddings":    "squish.quantizer",

    # squish.radix_cache
    "RadixNode":                 "squish.radix_cache",
    "RadixTree":                 "squish.radix_cache",

    # squish.robust_scheduler
    "ABalancedScheduler":        "squish.robust_scheduler",
    "AMaxScheduler":             "squish.robust_scheduler",
    "LengthInterval":            "squish.robust_scheduler",
    "Request":                   "squish.robust_scheduler",
    "RobustSchedulerConfig":     "squish.robust_scheduler",
    "RobustSchedulerStats":      "squish.robust_scheduler",

    # squish.sage_attention
    "KSmoother":                 "squish.sage_attention",
    "SageAttentionConfig":       "squish.sage_attention",
    "SageAttentionKernel":       "squish.sage_attention",
    "SageAttentionStats":        "squish.sage_attention",

    # squish.sage_attention2
    "SageAttention2Config":      "squish.sage_attention2",
    "SageAttention2Kernel":      "squish.sage_attention2",
    "SageAttention2Stats":       "squish.sage_attention2",
    "WarpQuantResult":           "squish.sage_attention2",

    # squish.seq_packing
    "PackedBatch":               "squish.seq_packing",
    "PackingConfig":             "squish.seq_packing",
    "PackingStats":              "squish.seq_packing",
    "SequencePacker":            "squish.seq_packing",

    # squish.shadow_kv
    "LandmarkSelector":          "squish.shadow_kv",
    "LowRankKeyCache":           "squish.shadow_kv",
    "ShadowKVCache":             "squish.shadow_kv",
    "ShadowKVConfig":            "squish.shadow_kv",

    # squish.smallkv
    "MarginalVCache":            "squish.smallkv",
    "SaliencyTracker":           "squish.smallkv",
    "SmallKVCache":              "squish.smallkv",
    "SmallKVConfig":             "squish.smallkv",
    "SmallKVStats":              "squish.smallkv",

    # squish.sparge_attn
    "BlockMask":                 "squish.sparge_attn",
    "SpargeAttnConfig":          "squish.sparge_attn",
    "SpargeAttnEngine":          "squish.sparge_attn",
    "SpargeAttnStats":           "squish.sparge_attn",

    # squish.sparse_spec
    "PillarAttnCache":           "squish.sparse_spec",
    "SparseSpecConfig":          "squish.sparse_spec",
    "SparseSpecDecoder":         "squish.sparse_spec",
    "SparseSpecDrafter":         "squish.sparse_spec",
    "SparseSpecStats":           "squish.sparse_spec",

    # squish.sparse_verify
    "InterDraftReuseCache":      "squish.sparse_verify",
    "SparseVerifyConfig":        "squish.sparse_verify",
    "SparseVerifyPass":          "squish.sparse_verify",
    "SparseVerifyStats":         "squish.sparse_verify",

    # squish.spec_reason
    "ReasoningStep":             "squish.spec_reason",
    "SpecReasonConfig":          "squish.spec_reason",
    "SpecReasonOrchestrator":    "squish.spec_reason",
    "SpecReasonStats":           "squish.spec_reason",
    "StepVerdict":               "squish.spec_reason",

    # squish.specontext
    "DistilledRetrievalHead":    "squish.specontext",
    "SpeContextCache":           "squish.specontext",
    "SpeContextConfig":          "squish.specontext",
    "SpeContextStats":           "squish.specontext",

    # squish.speculative
    "SpeculativeGenerator":      "squish.speculative",
    "load_draft_model":          "squish.speculative",

    # squish.split_loader
    "OffloadedLayer":            "squish.split_loader",
    "SplitInfo":                 "squish.split_loader",
    "SplitLayerLoader":          "squish.split_loader",
    "print_layer_profile":       "squish.split_loader",
    "profile_model_layers":      "squish.split_loader",

    # squish.squeeze_attention
    "BudgetAllocator":           "squish.squeeze_attention",
    "LayerKVBudget":             "squish.squeeze_attention",
    "SqueezeConfig":             "squish.squeeze_attention",
    "SqueezeKVCache":            "squish.squeeze_attention",
    "SqueezeStats":              "squish.squeeze_attention",

    # squish.squeeze_llm
    "OutlierDetector":           "squish.squeeze_llm",
    "SqueezeLLMConfig":          "squish.squeeze_llm",
    "SqueezeLLMLayer":           "squish.squeeze_llm",
    "SqueezeLLMQuantizer":       "squish.squeeze_llm",

    # squish.streaming_sink
    "SinkConfig":                "squish.streaming_sink",
    "SinkKVCache":               "squish.streaming_sink",
    "SinkStats":                 "squish.streaming_sink",

    # squish.sub_spec
    "SubSpecConfig":             "squish.sub_spec",
    "SubSpecDecoder":            "squish.sub_spec",
    "SubSpecStats":              "squish.sub_spec",
    "SubstituteLayerProxy":      "squish.sub_spec",

    # squish.svdq
    "HeadSVDProfile":            "squish.svdq",
    "SVDqCalibrator":            "squish.svdq",
    "SVDqConfig":                "squish.svdq",
    "SVDqPrecisionMap":          "squish.svdq",
    "SVDqStats":                 "squish.svdq",

    # squish.token_swift
    "MultiTokenHead":            "squish.token_swift",
    "PartialKVManager":          "squish.token_swift",
    "TokenSwiftConfig":          "squish.token_swift",
    "TokenSwiftDecoder":         "squish.token_swift",
    "TokenSwiftStats":           "squish.token_swift",

    # squish.trail
    "TrailConfig":               "squish.trail",
    "TrailLinearProbe":          "squish.trail",
    "TrailPredictor":            "squish.trail",
    "TrailStats":                "squish.trail",

    # squish.vptq
    "VPTQCodebook":              "squish.vptq",
    "VPTQConfig":                "squish.vptq",
    "VPTQLayer":                 "squish.vptq",
    "VPTQQuantizer":             "squish.vptq",

    # squish.yoco
    "YOCOConfig":                "squish.yoco",
    "YOCOKVStore":               "squish.yoco",
    "YOCOLayerSpec":             "squish.yoco",
    "YOCOSchedule":              "squish.yoco",
    "YOCOStats":                 "squish.yoco",
}

_lazy_cache: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Load any registered public name on first access (lazy import)."""
    if name in _lazy_cache:
        return _lazy_cache[name]
    if name in _LAZY_IMPORTS:
        import importlib
        mod_name = _LAZY_IMPORTS[name]
        # Special-case aliased names from squish.catalog
        if mod_name == "squish.catalog" and name in ("pull_model", "resolve_model"):
            mod = importlib.import_module(mod_name)
            alias_map = {"pull_model": "pull", "resolve_model": "resolve"}
            obj = getattr(mod, alias_map[name])
            _lazy_cache[name] = obj
            return obj
        try:
            mod = importlib.import_module(mod_name)
        except (ImportError, OSError) as exc:
            raise AttributeError(
                f"module 'squish' has no attribute {name!r} "
                f"(optional dependency {mod_name!r} could not be imported: {exc})"
            ) from None
        obj = getattr(mod, name)
        _lazy_cache[name] = obj
        return obj
    raise AttributeError(f"module 'squish' has no attribute {name!r}")


__all__ = [
    "__version__",
    *_LAZY_IMPORTS,
]
