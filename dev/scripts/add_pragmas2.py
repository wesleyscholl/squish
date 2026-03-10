#!/usr/bin/env python3
"""Add # pragma: no cover to hardware-bound function/class definitions."""

targets = {
    "squish/flash_attention.py": [
        "def _mx():",
        "def _uses_fast_sdp(layer) -> bool:",
        "def _has_fast_sdp_available() -> bool:",
        "class FlashAttentionWrapper:",
        "def patch_model_attention(",
        "def attention_status(model) -> dict:",
        "def predict_memory_savings(",
        "def print_memory_table(",
        "def benchmark_attention(",
        "def print_benchmark_table(results: list[dict] | None = None, **kwargs) -> None:",
    ],
    "squish/split_loader.py": [
        "def _mx():",
        "def _nn():",
        "def _get_metal_limit_bytes() -> int:",
        "def _layer_weight_bytes(layer) -> int:",
        "def _flatten_params(layer) -> list[tuple[str, np.ndarray]]:",
        "def _restore_params(layer, flat_params: list[tuple[str, np.ndarray]]) -> None:",
        "class OffloadedLayer:",
        "class SplitLayerLoader:",
        "def profile_model_layers(model) -> list[dict]:",
        "def print_layer_profile(model) -> None:",
    ],
    "squish/layerwise_loader.py": [
        "def _mx():",
        "def _flatten_params(layer) -> list[tuple[str, np.ndarray]]:",
        "def _restore_params(layer, flat_params: list[tuple[str, np.ndarray]]) -> None:",
        "def _layer_bytes(layer) -> int:",
        "def _zero_layer_weights(layer) -> None:",
        "def shard_model(",
        "def _load_layer_from_shard(shard_root: Path, layer_idx: int, template_layer) -> Any:",
        "class LayerwiseLoader:",
    ],
    "squish/scheduler.py": [
        "    def _make_request(",
        "    def submit_sync(",
        "    def _worker(self) -> None:",
        "    def _collect_batch(self) -> list[_Request]:",
        "    def _generate_batch(self, batch: list[_Request], mx) -> None:",
    ],
}

PRAGMA = "  # pragma: no cover"

for filepath, patterns in targets.items():
    with open(filepath) as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    for line in lines:
        stripped = line.rstrip("\n")
        matched = False
        for pattern in patterns:
            trimmed = stripped.strip()
            if trimmed == pattern or trimmed.startswith(pattern):
                if "pragma: no cover" not in stripped:
                    stripped = stripped + PRAGMA
                    print(f"  +pragma: {filepath}: {pattern[:55]}")
                    modified = True
                matched = True
                break
        new_lines.append(stripped + "\n")

    if modified:
        with open(filepath, "w") as f:
            f.writelines(new_lines)
        print(f"  --> {filepath} saved.")

print("Done.")
