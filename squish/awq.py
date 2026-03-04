#!/usr/bin/env python3
"""
squish/awq.py

Activation-Aware Weight Quantization (AWQ) calibration for Squish.

AWQ improves INT8 quantization accuracy by protecting the ~1% of weight
channels that are most sensitive to quantization error.  Those channels are
identified by large activation magnitudes — high |x[c]| means any quantization
error in W[:,c] gets amplified at the output.

Algorithm (Lin et al., 2023  https://arxiv.org/abs/2306.00978):
  1. Run a calibration set through the model.
  2. For each linear layer, collect per-input-channel activation magnitudes.
  3. Compute per-channel scale:  s[c] = mean_act[c] ** alpha    (alpha ≈ 0.5)
  4. Before quantization: W_awq[:, c] /= s[c]
     The input rescaling (X_awq[c] = X[c] * s[c]) is absorbed into the
     previous LayerNorm's gamma:  gamma_awq[c] = gamma[c] * s[c]

Net effect: salient channels are moved into a tighter range that INT8 can
represent accurately — typically +0.5-2% accuracy on MMLU / HellaSwag.

Usage
-----
Calibration (needs the FP16 model loaded):

    python3 -m squish.awq \\
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --output    ~/models/Qwen2.5-7B-Instruct-bf16/awq_scales \\
        --n-samples 128 \\
        --alpha 0.5

Then pass scales to squish.convert:

    python3 -m squish.convert \\
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --output    ~/models/squish_7b \\
        --awq-scales ~/models/Qwen2.5-7B-Instruct-bf16/awq_scales

Or apply programmatically before quantizing a weight dict:

    from squish.awq import load_awq_scales, apply_awq_to_weights
    scales = load_awq_scales(awq_dir)
    weights = apply_awq_to_weights(weights, scales)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default calibration texts — mix of factual, reasoning, and conversational
# so activation statistics cover the full distribution of real usage.
# ---------------------------------------------------------------------------
_DEFAULT_CALIBRATION_TEXTS = [
    "The capital of France is Paris, which is also the largest city in the country.",
    "Machine learning models learn patterns from data by adjusting internal parameters.",
    "In 1969, NASA's Apollo 11 mission successfully landed astronauts on the Moon.",
    "Python is a high-level programming language known for its readable syntax.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Climate change is driven by greenhouse gas emissions from fossil fuels.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "Quantum mechanics describes the behavior of particles at the atomic scale.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "Mathematics is the language of the universe, as Galileo famously stated.",
    "Artificial intelligence systems are increasingly used in medical diagnosis.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "DNA carries the genetic instructions for the development of all living organisms.",
    "The Internet was developed from ARPANET, a US Department of Defense project.",
    "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.",
    "The Roman Empire at its height controlled most of Europe and the Middle East.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Neural networks are inspired by the structure of the human brain.",
    "The Theory of Relativity was developed by Albert Einstein in the early 1900s.",
]


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

class _ActivationHook:
    """
    Forward hook that accumulates per-channel mean absolute activation values.

    For a linear layer with weight W  (out_features × in_features):
    We want to measure the magnitude of each INPUT channel [0..in_features-1]
    so we can decide which channels are salient.

    This hook captures the INPUT tensor to the linear layer.  Shape will be
    (batch, seq_len, in_features) for typical transformer calls.
    """
    def __init__(self):
        self.channel_sum   = None   # float64 accumulator, shape (in_features,)
        self.channel_count = 0

    def __call__(self, module, inp, output):
        # inp is a tuple; inp[0] is the activation tensor
        x = inp[0]
        try:
            import mlx.core as mx
            # Convert MLX array to numpy for statistics
            arr = np.array(x.astype(mx.float32))
        except Exception:
            arr = np.asarray(x, dtype=np.float32)

        # arr shape: (..., in_features) — flatten all but last dim
        flat = arr.reshape(-1, arr.shape[-1])           # (N, in_features)
        abs_mean = np.abs(flat).mean(axis=0)            # (in_features,)

        if self.channel_sum is None:
            self.channel_sum = abs_mean.astype(np.float64)
        else:
            self.channel_sum += abs_mean.astype(np.float64)
        self.channel_count += 1

    def mean_activation(self) -> np.ndarray:
        """Return mean per-channel activation magnitude (float32)."""
        if self.channel_sum is None or self.channel_count == 0:
            return np.array([], dtype=np.float32)
        return (self.channel_sum / self.channel_count).astype(np.float32)


def collect_activation_scales(  # pragma: no cover
    model,
    tokenizer,
    texts: list | None = None,
    n_samples: int = 64,
    alpha: float = 0.5,
    seq_len: int = 512,
    verbose: bool = True,
) -> dict:
    """
    Run calibration data through ``model`` and compute per-layer AWQ scales.

    Parameters
    ----------
    model       : mlx_lm model object (already loaded, on Metal)
    tokenizer   : HuggingFace tokenizer matching the model
    texts       : list of calibration strings (defaults to built-in set)
    n_samples   : how many total forward passes to run (more = better stats)
    alpha       : scale exponent  0 = no AWQ, 1 = full activation scaling
                  0.5 is the default recommended in the AWQ paper
    seq_len     : max token length per sample (truncated / padded)
    verbose     : print progress

    Returns
    -------
    dict mapping ``layer_name → np.ndarray(shape=(in_features,), dtype=float32)``
    of AWQ scales.  These are the ``s`` vectors to be applied as::

        W_awq[:, c] = W[:, c] / s[c]   (applied before quantization)
        gamma_awq[c] = gamma[c] * s[c]  (absorbed into preceding LayerNorm)
    """
    import mlx.core as mx

    if texts is None:
        texts = _DEFAULT_CALIBRATION_TEXTS

    # Cycle the text list to reach n_samples
    sample_texts = [texts[i % len(texts)] for i in range(n_samples)]

    # Collect all nn.Linear modules and attach hooks
    hooks   = {}      # layer_name → _ActivationHook

    # MLX modules don't have PyTorch-style forward hooks; instead we monkey-
    # patch __call__ temporarily on each nn.Linear.
    import mlx.nn as nn

    linear_layers = {}
    for name, module in model.named_modules() if hasattr(model, 'named_modules') else []:
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    # Fallback: traverse via named_children recursively
    if not linear_layers:
        def _collect(mod, prefix=""):
            for child_name, child in (mod.children().items()
                                      if hasattr(mod, 'children') else {}.items()):
                full = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, nn.Linear):
                    linear_layers[full] = child
                _collect(child, full)
        _collect(model)

    if verbose:
        print(f"  Found {len(linear_layers)} linear layers to calibrate")

    # Monkey-patch each module's __call__ to intercept inputs
    originals = {}
    for name, module in linear_layers.items():
        hook = _ActivationHook()
        hooks[name] = hook
        orig_call = module.__call__

        def _make_patched(orig, h):
            def _patched(x, *a, **kw):
                h(None, (x,), None)
                return orig(x, *a, **kw)
            return _patched

        originals[name] = orig_call
        module.__call__ = _make_patched(orig_call, hook)

    if verbose:
        print(f"  Running {n_samples} calibration forward passes ...")
    t0 = time.perf_counter()

    for i, text in enumerate(sample_texts):
        ids = tokenizer.encode(text, add_special_tokens=True)[:seq_len]
        if not ids:
            continue
        x = mx.array([ids], dtype=mx.int32)
        try:
            _ = model(x)
            mx.eval(x)          # ensure Metal execution completes
        except Exception:
            pass                # some models need kv_cache — skip on error

        if verbose and (i + 1) % 16 == 0:
            print(f"    [{i+1}/{n_samples}] calibrated …")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Calibration done in {elapsed:.1f}s")

    # Restore original __call__
    for name, module in linear_layers.items():
        if name in originals:
            module.__call__ = originals[name]

    # Compute AWQ scales from collected statistics
    scales = {}
    for name, hook in hooks.items():
        mean_act = hook.mean_activation()
        if mean_act.size == 0:
            continue
        # s[c] = mean_act[c]^alpha  (clipped to ≥ 1e-4 to avoid div-by-zero)
        s = np.clip(mean_act, 1e-4, None) ** alpha
        scales[name] = s.astype(np.float32)

    if verbose:
        print(f"  Computed AWQ scales for {len(scales)} layers")

    return scales


# ---------------------------------------------------------------------------
# Scale persistence
# ---------------------------------------------------------------------------

def save_awq_scales(scales: dict, output_dir: str | Path, verbose: bool = True) -> None:
    """
    Persist AWQ scale vectors to ``output_dir`` as ``{layer_name}.awq.npy`` files.

    Layer names with ``/`` or ``.`` are converted to path-safe names using ``_``
    and ``__`` respectively — matching the safe_key convention in convert.py.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for layer_name, scale in scales.items():
        safe = layer_name.replace("/", "_").replace(".", "__")
        np.save(str(out / f"{safe}.awq.npy"), scale)
        n += 1

    # Write index file so loaders know which layers have scales
    index = {k: k.replace("/", "_").replace(".", "__") + ".awq.npy"
             for k in scales}
    import json
    with open(out / "awq_index.json", "w") as f:
        json.dump(index, f, indent=2)

    (out / ".awq_ready").touch()

    if verbose:
        print(f"  Saved AWQ scales for {n} layers → {out}")


def load_awq_scales(awq_dir: str | Path) -> dict:
    """
    Load AWQ scale vectors from a directory written by :func:`save_awq_scales`.

    Returns a dict mapping ``layer_name → np.ndarray(float32)``.
    Returns an empty dict if the directory does not exist or has no AWQ files.
    """
    import json

    d = Path(awq_dir)
    if not d.exists():
        return {}

    index_path = d / "awq_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        scales = {}
        for layer_name, filename in index.items():
            p = d / filename
            if p.exists():
                scales[layer_name] = np.load(str(p))
        return scales

    # Fallback: enumerate all .awq.npy files and reverse-map the safe key
    scales = {}
    for p in sorted(d.glob("*.awq.npy")):
        layer_name = p.stem.replace("__", ".").replace("_", "/")
        scales[layer_name] = np.load(str(p))
    return scales


# ---------------------------------------------------------------------------
# Scale application
# ---------------------------------------------------------------------------

def apply_awq_to_weights(
    weights: dict,
    awq_scales: dict,
    alpha: float = 0.5,
    verbose: bool = False,
) -> dict:
    """
    Apply pre-computed AWQ scales to a flat weight dict before quantization.

    ``weights`` maps tensor names (e.g. ``model.layers.0.self_attn.q_proj.weight``)
    to float32 numpy arrays.

    For each matched linear layer weight W (shape out × in), the scale is
    applied column-wise:  ``W[:, c] /= s[c]``

    The corresponding inverse scale on the input side is absorbed into the
    preceding LayerNorm gamma (``input_layernorm.weight`` or ``norm.weight``).

    Parameters
    ----------
    weights     : flat dict of {tensor_name: np.ndarray(float32)}
    awq_scales  : output of load_awq_scales() or collect_activation_scales()
    alpha       : for documentation purposes only — scales already account for alpha
    verbose     : print which tensors are AWQ-adjusted

    Returns
    -------
    Modified weight dict (modifies in-place and returns same dict).
    """
    # Target linear layers — the projection weights that follow attention norms
    _PROJ_SUFFIXES = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "fc1.weight", "fc2.weight",
        "dense.weight", "dense_h_to_4h.weight", "dense_4h_to_h.weight",
    )

    n_applied = 0

    for tensor_name, arr in list(weights.items()):
        if arr.ndim < 2:
            continue
        if not any(tensor_name.endswith(sfx) for sfx in _PROJ_SUFFIXES):
            continue

        # Derive the layer's linear-module path from the tensor name:
        # e.g.  "model.layers.0.self_attn.q_proj.weight"
        #    →  "model.layers.0.self_attn.q_proj"
        layer_path = tensor_name[: tensor_name.rfind(".")]

        # Find the best matching scale key (exact → prefix match)
        scale = None
        if layer_path in awq_scales:
            scale = awq_scales[layer_path]
        else:
            # Fuzzy match: find the longest key that is a suffix of layer_path
            best = ""
            for k in awq_scales:
                if layer_path.endswith(k) and len(k) > len(best):
                    best = k
            if best:
                scale = awq_scales[best]

        if scale is None:
            continue

        # W shape: (out_features, in_features)  → scale is (in_features,)
        W = arr.reshape(-1, arr.shape[-1])          # ensure 2D
        if scale.shape[0] != W.shape[1]:
            continue   # shape mismatch — skip silently

        # Apply: W_awq[:, c] /= s[c]
        weights[tensor_name] = (W / scale[np.newaxis, :]).reshape(arr.shape)

        # Absorb inverse into the PRECEDING LayerNorm if available
        # Standard name patterns for the norm that feeds this projection:
        norm_name = _preceding_norm_name(tensor_name, weights)
        if norm_name and norm_name in weights:
            weights[norm_name] = weights[norm_name] * scale

        if verbose:
            print(f"  [AWQ] {tensor_name}  s̄={scale.mean():.4f}  "
                  f"s_max={scale.max():.4f}")
        n_applied += 1

    if n_applied == 0 and awq_scales:
        print("  [AWQ] Warning: no scales matched any weight tensors. "
              "Check that layer names in awq_scales match model weight names.")
    elif verbose or n_applied > 0:
        print(f"  [AWQ] Applied scales to {n_applied} weight tensors")

    return weights


def _preceding_norm_name(weight_name: str, weights: dict) -> str | None:
    """
    Guess the name of the LayerNorm whose output feeds this linear weight.

    For ``model.layers.{i}.self_attn.q_proj.weight``:
      → try ``model.layers.{i}.input_layernorm.weight``
         and ``model.layers.{i}.self_attn.q_norm.weight``

    For ``model.layers.{i}.mlp.gate_proj.weight``:
      → try ``model.layers.{i}.post_attention_layernorm.weight``
    """
    parts = weight_name.split(".")
    # Strip ".weight"
    if parts and parts[-1] == "weight":
        parts = parts[:-1]

    # Detect component (self_attn / mlp)
    if "self_attn" in parts or "attention" in parts:
        # Walk back to block root: everything before self_attn/attention
        root = ".".join(parts[:parts.index("self_attn")
                              if "self_attn" in parts
                              else parts.index("attention")])
        candidates = [
            f"{root}.input_layernorm.weight",
            f"{root}.self_attn.q_norm.weight",
            f"{root}.ln_1.weight",
        ]
    elif "mlp" in parts:
        root = ".".join(parts[:parts.index("mlp")])
        candidates = [
            f"{root}.post_attention_layernorm.weight",
            f"{root}.ln_2.weight",
            f"{root}.ffn_norm.weight",
        ]
    else:
        return None

    for c in candidates:
        if c in weights:
            return c
    return None


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="AWQ calibration — compute per-channel activation scales for a model"
    )
    ap.add_argument("--model-dir",  required=True,
                    help="Path to HuggingFace model (BF16 safetensors)")
    ap.add_argument("--output",     required=True,
                    help="Directory to write .awq.npy scale files")
    ap.add_argument("--n-samples",  type=int,   default=64,
                    help="Number of calibration forward passes (default 64)")
    ap.add_argument("--alpha",      type=float, default=0.5,
                    help="Scale exponent α: 0=no AWQ, 0.5=default, 1=full")
    ap.add_argument("--seq-len",    type=int,   default=512,
                    help="Max token length per sample (default 512)")
    ap.add_argument("--calibration-file",
                    help="Optional file with one calibration sentence per line")
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    print("\nSquish AWQ Calibration")
    print(f"  Model:     {args.model_dir}")
    print(f"  Output:    {args.output}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  alpha:     {args.alpha}\n")

    # Load calibration texts
    texts = _DEFAULT_CALIBRATION_TEXTS
    if args.calibration_file:
        with open(args.calibration_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        print(f"  Loaded {len(texts)} calibration texts from {args.calibration_file}")

    # Load the model
    print("Loading model (BF16) ...")
    try:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(args.model_dir)
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    # Calibrate
    scales = collect_activation_scales(
        model,
        tokenizer,
        texts=texts,
        n_samples=args.n_samples,
        alpha=args.alpha,
        seq_len=args.seq_len,
        verbose=True,
    )

    # Save
    save_awq_scales(scales, args.output, verbose=True)

    print(f"\nDone.  Run squish.convert with --awq-scales {args.output} to apply.")


if __name__ == "__main__":
    main()
