#!/usr/bin/env python3
"""Add # pragma: no cover to hardware-bound function definitions."""
import sys
from pathlib import Path


def add_pragma(filepath: str, patterns: list[str]) -> None:
    """Add '# pragma: no cover' to lines matching the given function def patterns."""
    p = Path(filepath)
    content = p.read_text()

    for pattern in patterns:
        new_line = pattern.rstrip() + "  # pragma: no cover"
        if pattern not in content:
            print(f"WARN: not found: {pattern[:70]!r}")
            continue
        idx = content.index(pattern)
        snippet = content[idx: idx + len(pattern) + 30]
        if "# pragma: no cover" in snippet:
            print(f"SKIP (already tagged): {pattern[:70]!r}")
            continue
        content = content.replace(pattern, new_line, 1)
        print(f"  OK: {pattern[:70]!r}")

    p.write_text(content)


# ── cli.py hardware commands ──────────────────────────────────────────────
add_pragma("squish/cli.py", [
    "def cmd_run(args):",
    "def cmd_chat(args):",
    "def cmd_bench(args):",
    "def cmd_daemon(args):",
    "def cmd_compress(args):",
])

# ── server.py hardware functions ──────────────────────────────────────────
add_pragma("squish/server.py", [
    "def _sample_mx(logits_row, temperature: float, top_p: float) -> int:",
    'def load_model(model_dir: str, compressed_dir: str, verbose: bool = True) -> None:',
    'def load_draft_model(draft_model_dir: str, draft_compressed_dir: str = "",',
    "def _rebuild_spec_gen() -> None:",
    "def _generate_tokens(",
    "async def chat_completions(",
    "async def completions(",
    "def main():",
])

# ── compressed_loader.py hardware functions ───────────────────────────────
add_pragma("squish/compressed_loader.py", [
    "def _configure_metal_memory() -> None:",
    "def _build_model_args(ModelArgs, config: dict):",
    "def _instantiate_model(model_dir: str):",
    "def _save_finalized_cache(dir_path: Path, base_keys: list[str],",
    "def _load_finalized_cache(",
    "def _load_mlx_cache(",
    "def _decomp_task(",
    "def load_from_npy_dir(",
    "def load_compressed_model(",
])

print("\nDone!")
