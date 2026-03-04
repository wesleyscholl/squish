"""
squish/catalog.py  —  Model catalog, shorthand resolution, and pull/download logic.

The bundled catalog (BUNDLED_CATALOG) ships inside the package and covers the most
popular MLX-compatible models.  A remote catalog hosted at CATALOG_URL can update
it by being fetched into ~/.squish/catalog.json.

Squish pre-compressed weights (``squish_weights.npz`` / npy-dir) are hosted under
the ``squish-community`` HuggingFace organisation.  When available, ``pull`` skips
compression and downloads the already-squished artefacts directly.

Public API
----------
    from squish.catalog import resolve, load_catalog, list_catalog, pull

    entry = resolve("qwen3:8b")
    entry.hf_mlx_repo    # "mlx-community/Qwen3-8B-bf16"
    entry.squish_repo    # "squish-community/Qwen3-8B-squished", or None
    entry.size_gb        # raw model disk footprint (float)

    pull("qwen3:8b", models_dir=Path("~/models"), int4=False)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

CATALOG_URL = (
    "https://huggingface.co/datasets/squish-community/catalog/resolve/main/catalog.json"
)
SQUISH_CACHE_DIR = Path.home() / ".squish"
LOCAL_CATALOG_PATH = SQUISH_CACHE_DIR / "catalog.json"

# How often to refresh the remote catalog (seconds). Default: 24 h.
CATALOG_TTL = int(os.environ.get("SQUISH_CATALOG_TTL", str(24 * 3600)))


# ── CatalogEntry ──────────────────────────────────────────────────────────────

@dataclass
class CatalogEntry:
    """Metadata for a single model in the catalog."""

    # canonical identifier, e.g. "qwen3:8b"
    id: str

    # human-readable name shown in lists
    name: str

    # HuggingFace repo of the raw MLX bf16 weights
    hf_mlx_repo: str

    # approximate raw disk footprint in GB
    size_gb: float

    # parameter count string for display
    params: str

    # maximum context length in tokens
    context: int

    # approximate disk footprint after INT8 squish compression
    squished_size_gb: float

    # HuggingFace repo with pre-compressed Squish weights (or None)
    squish_repo: str | None = None

    # arbitrary tags for filtering: ["small", "fast", "moe", "reasoning", …]
    tags: list[str] = field(default_factory=list)

    # notes shown in squish models --catalog
    notes: str = ""

    @property
    def dir_name(self) -> str:
        """Filesystem directory name derived from hf_mlx_repo."""
        return self.hf_mlx_repo.split("/")[-1]

    @property
    def has_prebuilt(self) -> bool:
        """True when a pre-compressed Squish repo exists on HuggingFace."""
        return bool(self.squish_repo)

    def __str__(self) -> str:
        prebuilt = "⚡ prebuilt" if self.has_prebuilt else "  compress"
        return (
            f"  {self.id:<22} {self.params:>6}  "
            f"{self.size_gb:>6.1f} GB  {prebuilt}  {self.name}"
        )


# ── Bundled catalog ───────────────────────────────────────────────────────────
# Ground-truth shipped with the package.  Keys are canonical model IDs.

_BUNDLED: list[dict] = [
    # ── Qwen 2.5 ──────────────────────────────────────────────────────────────
    dict(id="qwen2.5:1.5b", name="Qwen2.5-1.5B-Instruct",
         hf_mlx_repo="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
         squish_repo="squish-community/Qwen2.5-1.5B-Instruct-squished",
         size_gb=3.1, squished_size_gb=0.9, params="1.5B", context=32768,
         tags=["small", "fast"]),
    dict(id="qwen2.5:7b", name="Qwen2.5-7B-Instruct",
         hf_mlx_repo="mlx-community/Qwen2.5-7B-Instruct-bf16",
         squish_repo="squish-community/Qwen2.5-7B-Instruct-squished",
         size_gb=14.4, squished_size_gb=3.9, params="7B", context=131072,
         tags=["balanced"]),
    dict(id="qwen2.5:14b", name="Qwen2.5-14B-Instruct",
         hf_mlx_repo="mlx-community/Qwen2.5-14B-Instruct-bf16",
         squish_repo="squish-community/Qwen2.5-14B-Instruct-squished",
         size_gb=28.2, squished_size_gb=7.5, params="14B", context=131072,
         tags=["balanced"]),
    dict(id="qwen2.5:32b", name="Qwen2.5-32B-Instruct",
         hf_mlx_repo="mlx-community/Qwen2.5-32B-Instruct-bf16",
         size_gb=64.4, squished_size_gb=17.1, params="32B", context=131072,
         tags=["large"]),
    dict(id="qwen2.5:72b", name="Qwen2.5-72B-Instruct",
         hf_mlx_repo="mlx-community/Qwen2.5-72B-Instruct-bf16",
         size_gb=144.0, squished_size_gb=38.2, params="72B", context=131072,
         tags=["large"]),

    # ── Qwen 3 ────────────────────────────────────────────────────────────────
    dict(id="qwen3:0.6b", name="Qwen3-0.6B",
         hf_mlx_repo="mlx-community/Qwen3-0.6B-bf16",
         size_gb=1.3, squished_size_gb=0.4, params="0.6B", context=32768,
         tags=["small", "fast"]),
    dict(id="qwen3:1.7b", name="Qwen3-1.7B",
         hf_mlx_repo="mlx-community/Qwen3-1.7B-bf16",
         size_gb=3.5, squished_size_gb=1.0, params="1.7B", context=32768,
         tags=["small", "fast"]),
    dict(id="qwen3:4b", name="Qwen3-4B",
         hf_mlx_repo="mlx-community/Qwen3-4B-bf16",
         size_gb=8.2, squished_size_gb=2.2, params="4B", context=32768,
         tags=["small", "fast"]),
    dict(id="qwen3:8b", name="Qwen3-8B",
         hf_mlx_repo="mlx-community/Qwen3-8B-bf16",
         size_gb=16.4, squished_size_gb=4.4, params="8B", context=131072,
         tags=["balanced"]),
    dict(id="qwen3:14b", name="Qwen3-14B",
         hf_mlx_repo="mlx-community/Qwen3-14B-bf16",
         size_gb=28.7, squished_size_gb=7.6, params="14B", context=131072,
         tags=["balanced"]),
    dict(id="qwen3:30b-a3b", name="Qwen3-30B-A3B (MoE)",
         hf_mlx_repo="mlx-community/Qwen3-30B-A3B-bf16",
         size_gb=18.5, squished_size_gb=5.0, params="30B", context=131072,
         tags=["moe", "balanced"],
         notes="MoE — only 3B active params per token"),
    dict(id="qwen3:32b", name="Qwen3-32B",
         hf_mlx_repo="mlx-community/Qwen3-32B-bf16",
         size_gb=64.0, squished_size_gb=17.0, params="32B", context=131072,
         tags=["large"]),

    # ── Llama 3.x ─────────────────────────────────────────────────────────────
    dict(id="llama3.2:1b", name="Llama-3.2-1B-Instruct",
         hf_mlx_repo="mlx-community/Llama-3.2-1B-Instruct-bf16",
         size_gb=2.5, squished_size_gb=0.7, params="1B", context=128000,
         tags=["small", "fast"]),
    dict(id="llama3.2:3b", name="Llama-3.2-3B-Instruct",
         hf_mlx_repo="mlx-community/Llama-3.2-3B-Instruct-bf16",
         size_gb=6.4, squished_size_gb=1.7, params="3B", context=128000,
         tags=["small"]),
    dict(id="llama3.1:8b", name="Llama-3.1-8B-Instruct",
         hf_mlx_repo="mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
         size_gb=16.1, squished_size_gb=4.3, params="8B", context=131072,
         tags=["balanced"]),
    dict(id="llama3.3:70b", name="Llama-3.3-70B-Instruct",
         hf_mlx_repo="mlx-community/Llama-3.3-70B-Instruct-bf16",
         size_gb=140.0, squished_size_gb=37.5, params="70B", context=131072,
         tags=["large"]),

    # ── Gemma 3 ───────────────────────────────────────────────────────────────
    dict(id="gemma3:1b", name="Gemma-3-1B-Instruct",
         hf_mlx_repo="mlx-community/gemma-3-1b-it-bf16",
         size_gb=2.0, squished_size_gb=0.6, params="1B", context=32768,
         tags=["small", "fast"]),
    dict(id="gemma3:4b", name="Gemma-3-4B-Instruct",
         hf_mlx_repo="mlx-community/gemma-3-4b-it-bf16",
         size_gb=8.1, squished_size_gb=2.2, params="4B", context=131072,
         tags=["small"]),
    dict(id="gemma3:12b", name="Gemma-3-12B-Instruct",
         hf_mlx_repo="mlx-community/gemma-3-12b-it-bf16",
         size_gb=24.2, squished_size_gb=6.5, params="12B", context=131072,
         tags=["balanced"]),
    dict(id="gemma3:27b", name="Gemma-3-27B-Instruct",
         hf_mlx_repo="mlx-community/gemma-3-27b-it-bf16",
         size_gb=54.0, squished_size_gb=14.5, params="27B", context=131072,
         tags=["large"]),

    # ── DeepSeek-R1 ───────────────────────────────────────────────────────────
    dict(id="deepseek-r1:7b", name="DeepSeek-R1-Distill-Qwen-7B",
         hf_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16",
         size_gb=14.4, squished_size_gb=3.9, params="7B", context=131072,
         tags=["reasoning"]),
    dict(id="deepseek-r1:14b", name="DeepSeek-R1-Distill-Qwen-14B",
         hf_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16",
         size_gb=28.2, squished_size_gb=7.5, params="14B", context=131072,
         tags=["reasoning"]),
    dict(id="deepseek-r1:32b", name="DeepSeek-R1-Distill-Qwen-32B",
         hf_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16",
         size_gb=64.0, squished_size_gb=17.1, params="32B", context=131072,
         tags=["reasoning", "large"]),

    # ── Phi-4 ─────────────────────────────────────────────────────────────────
    dict(id="phi4:14b", name="Phi-4",
         hf_mlx_repo="mlx-community/phi-4-bf16",
         size_gb=28.0, squished_size_gb=7.5, params="14B", context=16384,
         tags=["balanced"],
         notes="Microsoft Phi-4"),

    # ── Mistral ───────────────────────────────────────────────────────────────
    dict(id="mistral:7b", name="Mistral-7B-Instruct-v0.3",
         hf_mlx_repo="mlx-community/Mistral-7B-Instruct-v0.3-bf16",
         size_gb=14.5, squished_size_gb=3.9, params="7B", context=32768,
         tags=["balanced"]),
    dict(id="mistral-small:22b", name="Mistral-Small-Instruct-2409",
         hf_mlx_repo="mlx-community/Mistral-Small-Instruct-2409-bf16",
         size_gb=44.0, squished_size_gb=11.8, params="22B", context=131072,
         tags=["large"]),

    # ── SmolLM2 ───────────────────────────────────────────────────────────────
    dict(id="smollm2:135m", name="SmolLM2-135M-Instruct",
         hf_mlx_repo="mlx-community/SmolLM2-135M-Instruct-bf16",
         size_gb=0.3, squished_size_gb=0.1, params="135M", context=8192,
         tags=["small", "fast", "edge"]),
    dict(id="smollm2:360m", name="SmolLM2-360M-Instruct",
         hf_mlx_repo="mlx-community/SmolLM2-360M-Instruct-bf16",
         size_gb=0.7, squished_size_gb=0.2, params="360M", context=8192,
         tags=["small", "fast", "edge"]),
    dict(id="smollm2:1.7b", name="SmolLM2-1.7B-Instruct",
         hf_mlx_repo="mlx-community/SmolLM2-1.7B-Instruct-bf16",
         size_gb=3.5, squished_size_gb=1.0, params="1.7B", context=8192,
         tags=["small", "fast"]),
]

# Legacy shorthand aliases → canonical id (backward compat)
_ALIASES: dict[str, str] = {
    "1.5b":  "qwen2.5:1.5b",
    "7b":    "qwen2.5:7b",
    "14b":   "qwen2.5:14b",
    "32b":   "qwen2.5:32b",
    "72b":   "qwen2.5:72b",
    # convenience short forms
    "r1:7b":  "deepseek-r1:7b",
    "r1:14b": "deepseek-r1:14b",
    "r1:32b": "deepseek-r1:32b",
}


# ── Catalog loading ───────────────────────────────────────────────────────────

def _entry_from_dict(d: dict) -> CatalogEntry:
    return CatalogEntry(
        id=d["id"],
        name=d["name"],
        hf_mlx_repo=d["hf_mlx_repo"],
        size_gb=d["size_gb"],
        squished_size_gb=d.get("squished_size_gb", d["size_gb"] / 3.8),
        params=d["params"],
        context=d["context"],
        squish_repo=d.get("squish_repo"),
        tags=d.get("tags", []),
        notes=d.get("notes", ""),
    )


def _bundled_catalog() -> dict[str, CatalogEntry]:
    return {d["id"]: _entry_from_dict(d) for d in _BUNDLED}


def _try_refresh_catalog(catalog: dict[str, CatalogEntry]) -> dict[str, CatalogEntry]:
    """
    Merge the remote catalog over the bundled one.

    Behaviour:
    • If the local cache is fresh (within CATALOG_TTL), load it synchronously
      and return immediately — no network call, no blocking.
    • If stale (or missing), return the bundled/cached catalog immediately and
      launch a *daemon thread* to fetch + update in the background.  The next
      process start will pick up the freshened catalog.

    This design ensures ``import squish`` never blocks on a CDN timeout.
    """
    import threading as _threading

    def _merge(data: dict) -> None:
        for entry in data.get("models", []):
            try:
                catalog[entry["id"]] = _entry_from_dict(entry)
            except (KeyError, TypeError):
                pass

    # ── Offline mode: skip all network activity ───────────────────────────────
    if os.environ.get("SQUISH_OFFLINE"):
        if LOCAL_CATALOG_PATH.exists():
            try:
                _merge(json.loads(LOCAL_CATALOG_PATH.read_text()))
            except Exception:
                pass
        return catalog

    # ── Serve from warm local cache if fresh enough ───────────────────────────
    if LOCAL_CATALOG_PATH.exists():
        age = time.time() - LOCAL_CATALOG_PATH.stat().st_mtime
        if age < CATALOG_TTL:
            try:
                _merge(json.loads(LOCAL_CATALOG_PATH.read_text()))
                return catalog
            except Exception:
                pass

    # ── Stale (or absent) — return now, refresh asynchronously ───────────────
    def _background_fetch() -> None:
        try:
            import importlib.metadata as _imeta
            try:
                _ver = _imeta.version("squish")
            except Exception:
                _ver = "0.0.0"
            req = urllib.request.Request(
                CATALOG_URL, headers={"User-Agent": f"squish/{_ver}"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read()
            SQUISH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Atomic write: temp file + rename avoids partial reads on crash
            tmp = LOCAL_CATALOG_PATH.with_suffix(".tmp")
            tmp.write_bytes(raw)
            tmp.rename(LOCAL_CATALOG_PATH)
        except Exception:
            pass  # Offline / unreachable — bundled catalog stays in effect.

    t = _threading.Thread(target=_background_fetch, daemon=True,
                          name="squish-catalog-refresh")
    t.start()

    # If a stale local cache exists, load it while the refresh runs
    if LOCAL_CATALOG_PATH.exists():
        try:
            _merge(json.loads(LOCAL_CATALOG_PATH.read_text()))
        except Exception:
            pass

    return catalog


_CATALOG_CACHE: dict[str, CatalogEntry] | None = None


def load_catalog(refresh: bool = False) -> dict[str, CatalogEntry]:
    """
    Return the full catalog as ``{id: CatalogEntry}``.

    The first call may attempt a background refresh from HuggingFace.
    Pass ``refresh=True`` to force a re-fetch ignoring the TTL.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and not refresh:
        return _CATALOG_CACHE
    catalog = _bundled_catalog()
    if refresh and LOCAL_CATALOG_PATH.exists():
        LOCAL_CATALOG_PATH.unlink()
    _CATALOG_CACHE = _try_refresh_catalog(catalog)
    return _CATALOG_CACHE


def list_catalog(
    tag: str | None = None,
    refresh: bool = False,
) -> list[CatalogEntry]:
    """
    Return all catalog entries, optionally filtered by tag.
    Sorted by parameter count ascending.
    """
    catalog = load_catalog(refresh=refresh)
    entries = list(catalog.values())
    if tag:
        entries = [e for e in entries if tag in e.tags]
    # sort: extract numeric param count for ordering
    def _sort_key(e: CatalogEntry) -> float:
        s = e.params.upper()
        for unit, mult in (("B", 1.0), ("M", 0.001)):
            if s.endswith(unit):
                try:
                    return float(s[:-1]) * mult
                except ValueError:
                    pass
        return 9999.0

    return sorted(entries, key=_sort_key)


def search(
    query: str,
    refresh: bool = False,
) -> list[CatalogEntry]:
    """
    Search catalog entries whose ``id``, ``name``, ``tags``, or ``params``
    contain *query* (case-insensitive substring match).

    Returns entries sorted by parameter count ascending (same as
    :func:`list_catalog`).
    """
    q = query.lower()
    return [
        e for e in list_catalog(refresh=refresh)
        if q in e.id.lower()
        or q in e.name.lower()
        or any(q in t.lower() for t in e.tags)
        or q in e.params.lower()
    ]


def resolve(name: str, refresh: bool = False) -> CatalogEntry | None:
    """
    Resolve a model name/shorthand to a ``CatalogEntry``.

    Accepts:
      - canonical ids:  ``"qwen3:8b"``
      - legacy aliases: ``"7b"`` → ``"qwen2.5:7b"``
      - prefix matches: ``"qwen3"`` → first qwen3 entry by param count
    """
    # normalise
    name = name.strip().lower()

    # legacy alias
    canonical = _ALIASES.get(name, name)

    catalog = load_catalog(refresh=refresh)

    # exact match
    if canonical in catalog:
        return catalog[canonical]

    # prefix / fuzzy match (e.g. "qwen3" or "gemma3")
    matches = [e for k, e in catalog.items() if k.startswith(canonical + ":")]
    if matches:
        return sorted(matches, key=lambda e: e.size_gb)[0]

    return None


# ── Download helpers ──────────────────────────────────────────────────────────

def _hf_download(repo: str, local_dir: Path, token: str | None = None) -> None:  # pragma: no cover
    """
    Download a HuggingFace repo to ``local_dir``.

    Prefers ``huggingface_hub.snapshot_download`` when available,
    otherwise raises ImportError with an install hint.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                             "rust_model.ot", "*.ot"],
        )
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for `squish pull`.\n"
            "Install with: pip install huggingface_hub"
        ) from None


def _hf_file_download(repo: str, filename: str, local_dir: Path,  # pragma: no cover
                       token: str | None = None) -> Path:
    """Download a single file from a HuggingFace repo."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        dest = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
        return Path(dest)
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for `squish pull`.\n"
            "Install with: pip install huggingface_hub"
        ) from None


def _hf_list_files(repo: str, token: str | None = None) -> list[str]:  # pragma: no cover
    """Return all filenames in a HuggingFace repo (returns [] on error)."""
    try:
        from huggingface_hub import list_repo_files  # type: ignore[import]
        return list(list_repo_files(repo, token=token))
    except Exception:
        return []


def _has_squish_weights(repo: str, token: str | None = None) -> bool:
    """
    Return True when the squish-community repo contains pre-compressed weights.
    Checks for either ``squish_weights.npz`` or a ``squish_npy/`` directory marker.
    """
    files = _hf_list_files(repo, token=token)
    return any(
        f.startswith("squish_npy/") or f == "squish_weights.npz"
        for f in files
    )


# ── Public pull entry-point ───────────────────────────────────────────────────

def pull(  # pragma: no cover
    name: str,
    models_dir: Path | None = None,
    int4: bool = False,
    token: str | None = None,
    refresh_catalog: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Download and (if needed) compress a model.  Returns the compressed dir path.

    Steps
    -----
    1. Resolve name → CatalogEntry.
    2. If pre-compressed squish_repo exists on HuggingFace → download it directly.
    3. Otherwise download the bf16 MLX repo then run ``squish.convert`` to compress.
    4. Return the path to the local compressed directory.

    Parameters
    ----------
    name        : squish model id, e.g. ``"qwen3:8b"``
    models_dir  : base directory for models (default: ``~/.squish/models``)
    int4        : use INT4 nibble-packed compression instead of INT8
    token       : HuggingFace user access token (for gated models)
    verbose     : pass ``--verbose`` to the underlying compress step
    """

    if models_dir is None:
        models_dir = Path.home() / ".squish" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    entry = resolve(name, refresh=refresh_catalog)
    if entry is None:
        raise ValueError(
            f"Unknown model: {name!r}\n"
            f"Run `squish catalog` to see available models."
        )

    raw_dir        = models_dir / entry.dir_name
    compressed_dir = models_dir / (entry.dir_name + "-compressed")
    quant_label    = "INT4" if int4 else "INT8"

    # ── Already done? ─────────────────────────────────────────────────────────
    if compressed_dir.exists() and any(compressed_dir.iterdir()):
        print(f"  ✓  {entry.id} already compressed at {compressed_dir}")
        return compressed_dir

    # ── Try pre-compressed weights first ──────────────────────────────────────
    if entry.squish_repo:
        print(f"  Checking for pre-compressed weights in {entry.squish_repo} …")
        try:
            if _has_squish_weights(entry.squish_repo, token=token):
                print("  ⚡ Pre-compressed weights found!  Downloading …")
                squish_local = models_dir / (entry.dir_name + "-squish-src")
                _hf_download(entry.squish_repo, squish_local, token=token)

                # Move/detect: npy-dir or npz
                npy_dir = squish_local / "squish_npy"
                if npy_dir.exists():
                    import shutil
                    if compressed_dir.exists():
                        shutil.rmtree(compressed_dir)
                    npy_dir.rename(compressed_dir)
                    shutil.rmtree(squish_local, ignore_errors=True)
                else:
                    # npz variant — just keep the squish_local dir as compressed_dir
                    import shutil
                    if compressed_dir.exists():
                        shutil.rmtree(compressed_dir)
                    squish_local.rename(compressed_dir)

                print(f"  ✓  Pre-compressed {entry.id} ready at {compressed_dir}")
                return compressed_dir
        except Exception as exc:
            if verbose:
                print(f"  ⚠  Pre-compressed download failed ({exc}); falling back …")

    # ── Download raw bf16 weights ─────────────────────────────────────────────
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"  Downloading {entry.hf_mlx_repo}  ({entry.size_gb:.1f} GB) …")
        _hf_download(entry.hf_mlx_repo, raw_dir, token=token)
        print(f"  ✓  Downloaded to {raw_dir}")
    else:
        print(f"  ✓  Raw weights already in {raw_dir}")

    # ── Compress ──────────────────────────────────────────────────────────────
    est_size = entry.squished_size_gb
    print(f"\n  Compressing with Squish  ({quant_label}, ~{est_size:.1f} GB output) …")
    cmd = [
        sys.executable, "-m", "squish.convert",
        "--model-dir", str(raw_dir),
        "--output",    str(compressed_dir),
        "--format",    "npy-dir",
    ]
    if int4:
        cmd.append("--int4")
    if verbose:
        cmd.append("--verbose")

    import subprocess as _sp  # noqa
    result = _sp.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"squish.convert failed (exit {result.returncode}). "
            "Check output above."
        )

    print(f"\n  ✓  Compressed to {compressed_dir}")
    return compressed_dir
