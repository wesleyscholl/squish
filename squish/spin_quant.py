"""
squish/spin_quant.py

SpinQuant — Cayley-SGD rotation calibration for improved INT4/INT8 weight
quantization quality.

Background
----------
SpinQuant (Liu et al. 2024, arXiv 2405.16406) applies a learned orthogonal
rotation R to the weight matrices of each transformer layer before quantization.
The rotation makes the weight distributions more uniform and isotropic, which
reduces rounding error without any changes to the model architecture or
inference code.

  W_rotated = W @ R.T     (stored quantized)
  output    = (x @ R) @ W_rotated.T = x @ W.T      (equivalent at inference)

Because x is rotated at inference time, the effective dot product is unchanged —
making SpinQuant a *zero-cost* rotation at deployment: the only cost is paid
once during calibration.

Optimization (Cayley-SGD on Stiefel manifold)
---------------------------------------------
We want to minimize quantization error E(R) = ‖W - quant(W @ R.T) @ R‖²_F
subject to R ∈ Stiefel(d, d): R @ R.T = I.

The Cayley update preserves orthogonality exactly:
    R_new = (I + α/2 · A)⁻¹ (I - α/2 · A) R
where A = grad_R @ R.T - R @ grad_R.T  (the skew-symmetric anti-symmetric lift
of the Riemannian gradient).

This module implements a lightweight CPU/NumPy-only version suitable for small
to mid-size calibration runs on a Mac.  For large models, pass ``--steps`` at
a low value (≤ 200) or use pre-computed rotations.

Usage
-----
    from squish.spin_quant import run_rotation

    run_rotation(
        model_dir  = "~/squish/models/Qwen3-8B-mlx-int4",
        output_dir = "~/squish/models/Qwen3-8B-spinquant",
        steps      = 100,
        lr         = 0.01,
        seed       = 42,
    )

    # Or from CLI:
    squish rotate qwen3:8b --output-dir ~/models/Qwen3-8B-spinquant --steps 100
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a uniformly-distributed random orthogonal matrix of shape (dim, dim).

    Uses the QR decomposition of a Gaussian random matrix, which gives a
    Haar-measure uniform distribution over O(d).
    """
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Ensure uniform distribution by fixing signs
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    return (Q * diag).astype(np.float32)


def _quantize_fake_int8(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fake-quantize *W* to INT8 (per-row, symmetric).

    Returns ``(W_dequant, scale)`` where W_dequant has the same shape as W
    and represents the dequantized values.
    """
    # Per-row absolute max → scale
    scale = np.max(np.abs(W), axis=-1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    W_int = np.round(W / scale).clip(-127, 127).astype(np.int8).astype(np.float32)
    return W_int * scale, scale.squeeze(-1)


def _quantize_fake_int4(W: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Fake-quantize *W* to INT4 with per-group symmetric quantization.

    The last dimension of W is split into groups of *group_size* elements.
    Returns the dequantized float32 representation.
    """
    orig_shape = W.shape
    _d_out, _d_in = W.shape[0], W.shape[1] if W.ndim > 1 else W.shape[0]
    flat = W.reshape(-1, group_size)
    scale = np.max(np.abs(flat), axis=-1, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    quantized = np.round(flat / scale).clip(-7, 7).astype(np.int8).astype(np.float32)
    return (quantized * scale).reshape(orig_shape)


def _quant_error(W: np.ndarray, R: np.ndarray, bits: int = 8) -> float:
    """
    Compute the Frobenius-norm quantization error for rotating W by R.

    E(R) = ‖W_rot - quant(W_rot)‖²_F  where W_rot = W @ R.T
    """
    W_rot = W @ R.T
    if bits == 8:
        W_dq, _ = _quantize_fake_int8(W_rot)
    else:
        W_dq = _quantize_fake_int4(W_rot)
    diff = W_rot - W_dq
    return float(np.sum(diff * diff))


def _riemannian_grad(W: np.ndarray, R: np.ndarray, bits: int = 8,
                     eps: float = 1e-4) -> np.ndarray:
    """
    Approximate Riemannian gradient of the quantization error w.r.t. R.

    Uses finite differences with perturbation *eps* on each basis element of
    the tangent space (skew-symmetric matrices).  This is slow but safe and
    only practical for small head_dim values; for production use a closed-form
    gradient or automatic differentiation via MLX/JAX.
    """
    d = R.shape[0]
    E0 = _quant_error(W, R, bits)
    G  = np.zeros_like(R)   # Euclidean gradient

    # Approximate: perturb each element of R
    for i in range(d):
        for j in range(d):
            R_pert       = R.copy()
            R_pert[i, j] += eps
            # Re-orthogonalize by QR to stay on manifold
            Q, _ = np.linalg.qr(R_pert)
            G[i, j] = (_quant_error(W, Q, bits) - E0) / eps

    return G


def _cayley_update(R: np.ndarray, G: np.ndarray, lr: float) -> np.ndarray:
    """
    Apply one Cayley retraction step to keep R on the Stiefel manifold.

    A  = (G @ R.T - R @ G.T)      # skew-symmetric anti-symmetric lift
    R' = (I + α/2 · A)⁻¹ (I - α/2 · A) R
    """
    d  = R.shape[0]
    A  = G @ R.T - R @ G.T        # skew-symmetric (d × d)
    I  = np.eye(d, dtype=np.float32)
    L  = I + (lr / 2) * A
    Ri = I - (lr / 2) * A
    # Solve L @ R_new = Ri @ R
    try:
        R_new = np.linalg.solve(L, Ri @ R)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse
        R_new = np.linalg.pinv(L) @ (Ri @ R)
    # Numeric cleanup — re-orthogonalize via QR
    Q, S = np.linalg.qr(R_new)
    diag = np.sign(np.diag(S))
    diag[diag == 0] = 1.0
    return (Q * diag).astype(np.float32)


# ---------------------------------------------------------------------------
# Weight loading / saving (MLX safetensors format)
# ---------------------------------------------------------------------------

def _load_safetensors_numpy(model_dir: Path) -> dict[str, np.ndarray]:
    """
    Load all safetensors weight files in *model_dir* into a flat dict.

    Falls back to ``npz`` files if safetensors is unavailable.
    """
    weights: dict[str, np.ndarray] = {}
    try:
        from safetensors import safe_open
        for st_file in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(st_file), framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        if weights:
            return weights
    except ImportError:  # pragma: no cover
        pass

    # Fallback: npz
    for npz_file in sorted(model_dir.glob("*.npz")):
        data = np.load(str(npz_file))
        for key in data.files:
            weights[key] = data[key]
    return weights


def _save_safetensors_or_npz(
    weights: dict[str, np.ndarray],
    output_dir: Path,
    shard_name: str = "model.npz",
) -> None:
    """
    Save *weights* dict to *output_dir*.

    Prefers safetensors when the library is available; falls back to npz.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.numpy import save_file
        save_file(weights, str(output_dir / "model.safetensors"))
        return
    except ImportError:  # pragma: no cover
        pass
    np.savez(str(output_dir / "model.npz"), **weights)


# ---------------------------------------------------------------------------
# Layer-wise rotation
# ---------------------------------------------------------------------------

_WEIGHT_SUFFIXES_TO_ROTATE = (
    # Attention projections
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    # FFN projections (SwiGLU / Llama style)
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    # Fallback conventional names
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)


def _build_rotation_matrices(
    weights:    dict[str, np.ndarray],
    steps:      int,
    lr:         float,
    bits:       int,
    rng:        np.random.Generator,
    verbose:    bool = True,
) -> dict[int, np.ndarray]:
    """
    Learn one shared rotation R per unique weight dimension seen in
    *weights*, optimized via Cayley-SGD for *steps* iterations.

    Returns a dict mapping ``head_dim → R``.

    Note: For speed, we learn a single per-dimension rotation rather than
    per-layer rotations.  Per-layer would be strictly better but would
    multiply the calibration time by ``num_layers``.
    """
    # Collect unique dimensions
    dim_set: set[int] = set()
    for key, w in weights.items():
        if any(key.endswith(suf) for suf in _WEIGHT_SUFFIXES_TO_ROTATE):
            if w.ndim == 2:
                dim_set.add(w.shape[-1])   # rotate along input dimension

    if verbose:
        print(f"[SpinQuant] Found weight dimensions to rotate: {sorted(dim_set)}")

    rotations: dict[int, np.ndarray] = {}

    for dim in sorted(dim_set):
        if verbose:
            print(f"[SpinQuant] Optimizing R for dim={dim}, steps={steps}, lr={lr} ...")

        R = _random_orthogonal(dim, rng)

        # Gather representative weight rows from this dimension
        sample_weights: list[np.ndarray] = []
        for key, w in weights.items():
            if (any(key.endswith(suf) for suf in _WEIGHT_SUFFIXES_TO_ROTATE)
                    and w.ndim == 2 and w.shape[-1] == dim):
                # Take up to 256 rows to keep calibration fast
                rows = min(w.shape[0], 256)
                sample_weights.append(w[:rows].astype(np.float32))

        if not sample_weights:
            rotations[dim] = R
            continue

        # Concatenate sample rows
        W_sample = np.concatenate(sample_weights, axis=0)   # (N, dim)

        for step in range(steps):
            # Compute approximate Euclidean gradient via finite differences
            # For performance, limit to a random subset of rows per step
            batch_size = min(64, W_sample.shape[0])
            idx  = rng.integers(0, W_sample.shape[0], size=batch_size)
            W_b  = W_sample[idx]

            E0   = _quant_error(W_b, R, bits)

            # Fast approximate gradient via one-shot perturbation of R
            # (cheaper than full finite differences: perturb in a random direction)
            d    = R.shape[0]
            # Random skew-symmetric direction
            B    = rng.standard_normal((d, d)).astype(np.float32)
            A_dir = B - B.T   # skew-symmetric
            eps_step = 1e-3
            R_pert = R + eps_step * A_dir @ R
            Qt, _  = np.linalg.qr(R_pert)
            E1     = _quant_error(W_b, Qt, bits)

            # Gradient in skew-symmetric direction
            G_skew = ((E1 - E0) / eps_step) * A_dir
            # Recover approximate Euclidean gradient of R
            G = G_skew @ R

            R = _cayley_update(R, G, lr)

            if verbose and (step + 1) % max(1, steps // 5) == 0:
                print(f"  [dim={dim}] step {step+1}/{steps}  error={E0:.4f}")

        rotations[dim] = R
        if verbose:
            print(f"  [dim={dim}] final error={_quant_error(W_sample, R, bits):.4f}")

    return rotations


def _apply_rotations(
    weights:   dict[str, np.ndarray],
    rotations: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Apply pre-computed rotations to all eligible weight matrices.

    For a weight ``W`` with input dimension ``d``:
        W_rotated = W @ R.T     (rotate input channel basis)
    """
    rotated: dict[str, np.ndarray] = {}
    for key, w in weights.items():
        if (any(key.endswith(suf) for suf in _WEIGHT_SUFFIXES_TO_ROTATE)
                and w.ndim == 2):
            dim = w.shape[-1]
            if dim in rotations:
                R = rotations[dim]
                rotated[key] = (w.astype(np.float32) @ R.T).astype(w.dtype)
                continue
        rotated[key] = w
    return rotated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rotation(
    model_dir:  str,
    output_dir: str,
    steps:      int   = 100,
    lr:         float = 0.01,
    seed:       int   = 42,
    bits:       int   = 8,
    verbose:    bool  = True,
) -> None:
    """
    Learn and apply SpinQuant rotations to the model in *model_dir*, writing
    the rotated model to *output_dir*.

    Parameters
    ----------
    model_dir  : source model directory (must contain safetensors or npz files
                 and a ``config.json``)
    output_dir : destination directory for the rotated model
    steps      : Cayley-SGD optimization steps per weight dimension
    lr         : Cayley-SGD learning rate (0.005 – 0.05 is typical)
    seed       : random seed for reproducibility
    bits       : target quantization bits (4 or 8) for error estimation
    verbose    : print progress information
    """
    src  = Path(model_dir).expanduser().resolve()
    dst  = Path(output_dir).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"model_dir not found: {src}")
    if dst.exists() and dst != src:
        shutil.rmtree(dst)

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[SpinQuant] Loading weights from {src} ...")
    weights = _load_safetensors_numpy(src)
    if not weights:
        raise RuntimeError(
            f"No weight files (.safetensors or .npz) found in {src}"
        )
    if verbose:
        print(f"[SpinQuant] Loaded {len(weights)} tensors.")

    rotations = _build_rotation_matrices(weights, steps=steps, lr=lr,
                                         bits=bits, rng=rng, verbose=verbose)
    if not rotations:
        if verbose:
            print("[SpinQuant] No rotatable weight dimensions found; nothing to do.")
        return

    if verbose:
        print("[SpinQuant] Applying rotations to weights ...")
    rotated_weights = _apply_rotations(weights, rotations)

    if verbose:
        print(f"[SpinQuant] Saving rotated model to {dst} ...")
    dst.mkdir(parents=True, exist_ok=True)

    # Copy config.json and tokenizer files unchanged
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"):
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)

    # Copy any other non-weight files (e.g. .model tokenizer, merges.txt)
    for f in src.iterdir():
        if f.is_file() and f.suffix not in (".safetensors", ".npz", ".bin"):
            dst_f = dst / f.name
            if not dst_f.exists():
                shutil.copy2(f, dst_f)

    _save_safetensors_or_npz(rotated_weights, dst)

    if verbose:
        n_rotated = sum(
            1 for k in rotated_weights
            if any(k.endswith(suf) for suf in _WEIGHT_SUFFIXES_TO_ROTATE)
        )
        print(f"[SpinQuant] Done. {n_rotated} weight tensors rotated.")
        print(f"[SpinQuant] Output: {dst}")
