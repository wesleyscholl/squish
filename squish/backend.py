"""squish/backend.py — Unified compute backend.

On macOS/Apple Silicon with MLX installed → delegates to mlx.core / mlx_lm.
On Linux / CPU-only / CUDA / ROCm         → delegates to torch + transformers.

Exported singleton
──────────────────
    from squish.backend import BE

    BE.IS_APPLE  : bool — True when running on Apple Silicon with MLX
    BE.device    : str  — "metal" | "cuda" | "cpu"

    # Tensor ops
    BE.array(data, dtype="int32")          → mlx.array or torch.Tensor
    BE.eval(*tensors)                      → None  (no-op on PyTorch)
    BE.to_numpy(tensor)                    → np.ndarray float32

    # Model forward pass (normalises mlx / HF output differences)
    BE.forward(model, input_ids, cache=None)            → logits tensor
    BE.forward_np(model, input_ids, cache=None)         → np.ndarray float32

    # Model loading
    BE.load_model(path, **kw)                           → (model, tokenizer)

    # Token streaming
    BE.stream_generate(model, tok, prompt, **kw)        → Iterator[(text, finish)]

    # Weight I/O
    BE.save_tensors(path, weight_dict)                  → None
    BE.load_tensors(path)                               → dict[str, np.ndarray]

    # Memory management
    BE.configure_memory(fraction=0.90)                  → None
"""
from __future__ import annotations

import sys
from typing import Iterator

import numpy as np

# ── Platform detection ────────────────────────────────────────────────────────
_IS_APPLE: bool = False
if sys.platform == "darwin":
    try:
        import mlx.core as _mlx_probe  # noqa: F401
        _mlx_probe.array([0], dtype=_mlx_probe.int32)  # ensure Metal is live
        _IS_APPLE = True
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Apple / MLX backend
# ═══════════════════════════════════════════════════════════════════════════════

class _AppleBackend:
    """All operations delegate to mlx.core + mlx_lm."""

    IS_APPLE: bool = True
    device: str = "metal"

    # ── Tensor ops ────────────────────────────────────────────────────────────

    def array(self, data, dtype: str = "int32"):
        import mlx.core as mx
        _dtype_map = {
            "int32":    mx.int32,
            "float32":  mx.float32,
            "float16":  mx.float16,
            "bfloat16": mx.bfloat16,
        }
        dt = _dtype_map.get(dtype, mx.int32)
        return mx.array(data, dtype=dt)

    def eval(self, *tensors) -> None:
        import mlx.core as mx
        for t in tensors:
            if t is not None:
                mx.eval(t)

    def to_numpy(self, tensor) -> np.ndarray:
        import mlx.core as mx
        return np.array(tensor.astype(mx.float32), dtype=np.float32)

    # ── Model ops ─────────────────────────────────────────────────────────────

    def forward(self, model, input_ids, cache=None):
        """Run one forward pass; return raw logits tensor (mlx.array)."""
        if cache is not None:
            return model(input_ids, cache=cache)
        return model(input_ids)

    def forward_np(self, model, input_ids, cache=None) -> np.ndarray:
        """Run forward pass; return logits as float32 numpy."""
        import mlx.core as mx
        out = self.forward(model, input_ids, cache=cache)
        mx.eval(out)
        return np.array(out.astype(mx.float32), dtype=np.float32)

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, path: str, **kwargs):
        """Load model + tokenizer; returns (model, tokenizer)."""
        import mlx_lm
        return mlx_lm.load(path)

    # ── Streaming inference ───────────────────────────────────────────────────

    def stream_generate(
        self,
        model,
        tokenizer,
        prompt: str,
        **kwargs,
    ) -> Iterator[tuple[str, str | None]]:
        """Yield (text_chunk, finish_reason) tuples."""
        import mlx_lm
        max_tokens = kwargs.get("max_tokens", 512)
        temp       = kwargs.get("temperature", 0.7)
        top_p      = kwargs.get("top_p", 0.9)
        max_kv     = kwargs.get("max_kv_size", None)

        gen_kw: dict = dict(max_tokens=max_tokens, temp=temp, top_p=top_p)
        if max_kv is not None:
            gen_kw["max_kv_size"] = max_kv

        for result in mlx_lm.stream_generate(model, tokenizer, prompt, **gen_kw):
            if hasattr(result, "text"):
                yield result.text, getattr(result, "finish_reason", None)
            else:
                yield str(result), None

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def save_tensors(self, path: str, weight_dict: dict) -> None:
        import mlx.core as mx
        mx.save_safetensors(str(path), weight_dict)

    def load_tensors(self, path: str) -> dict:
        """Returns dict of {name → mlx.array}."""
        import mlx.core as mx
        return mx.load(str(path))

    # ── Memory management ─────────────────────────────────────────────────────

    def configure_memory(self, fraction: float = 0.90) -> None:
        """Raise the MLX Metal allocator ceiling (macOS only)."""
        try:
            import ctypes
            import mlx.core as mx

            if not (0.5 <= fraction <= 0.99):
                return
            libc = ctypes.CDLL("libSystem.dylib")
            memsize  = ctypes.c_uint64(0)
            size_ptr = ctypes.c_size_t(ctypes.sizeof(memsize))
            ret = libc.sysctlbyname(
                b"hw.memsize",
                ctypes.byref(memsize),
                ctypes.byref(size_ptr),
                None, 0,
            )
            if ret == 0:
                mx.metal.set_memory_limit(int(memsize.value * fraction), relaxed=True)
        except Exception:
            pass  # non-fatal


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch backend (Linux / CUDA / ROCm / CPU)
# ═══════════════════════════════════════════════════════════════════════════════

class _TorchBackend:
    """All operations delegate to torch + HuggingFace transformers."""

    IS_APPLE: bool = False

    def __init__(self) -> None:
        import torch  # raises ImportError on install without torch
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self.device  = "cuda"
        else:
            self._device = torch.device("cpu")
            self.device  = "cpu"
        self._torch = torch

    # ── Tensor ops ────────────────────────────────────────────────────────────

    def array(self, data, dtype: str = "int32"):
        import torch
        _dtype_map = {
            "int32":    torch.int32,
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dt = _dtype_map.get(dtype, torch.int32)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(data)).to(dtype=dt, device=self._device)
        return torch.tensor(data, dtype=dt, device=self._device)

    def eval(self, *tensors) -> None:
        pass  # PyTorch is eager — no deferred execution graph

    def to_numpy(self, tensor) -> np.ndarray:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().float().cpu().numpy()
        return np.array(tensor, dtype=np.float32)

    # ── Model ops ─────────────────────────────────────────────────────────────

    def forward(self, model, input_ids, cache=None):
        """Return raw output (CausalLMOutputWithPast or plain tensor)."""
        import torch
        with torch.no_grad():
            if cache is not None:
                return model(input_ids, past_key_values=cache, use_cache=True)
            return model(input_ids, use_cache=False)

    def forward_np(self, model, input_ids, cache=None) -> np.ndarray:
        """Run forward pass; return logits float32 numpy (B, T, vocab)."""
        out = self.forward(model, input_ids, cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        return self.to_numpy(logits)

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, path: str, **kwargs):
        """Load a HuggingFace model + tokenizer from *path*.

        Keyword args
        ─────────────
        load_in_4bit : bool   — enable bitsandbytes int4 (requires CUDA)
        torch_dtype  : dtype  — default float16
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_in_4bit = kwargs.get("load_in_4bit", False)
        torch_dtype  = kwargs.get("torch_dtype", torch.float16)

        load_kw: dict = dict(device_map="auto")
        if load_in_4bit:
            try:
                load_kw["load_in_4bit"] = True
            except Exception:
                load_kw["torch_dtype"] = torch_dtype
        else:
            load_kw["torch_dtype"] = torch_dtype

        model     = AutoModelForCausalLM.from_pretrained(path, **load_kw)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()
        return model, tokenizer

    # ── Streaming inference ───────────────────────────────────────────────────

    def stream_generate(
        self,
        model,
        tokenizer,
        prompt: str,
        **kwargs,
    ) -> Iterator[tuple[str, str | None]]:
        """Yield (text_chunk, finish_reason) tuples."""
        import threading
        import torch
        from transformers import TextIteratorStreamer

        max_tokens = kwargs.get("max_tokens", 512)
        temp       = float(kwargs.get("temperature", 0.7))
        top_p      = float(kwargs.get("top_p", 0.9))

        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        gen_kw = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temp > 1e-4),
            temperature=max(temp, 1e-4),
            top_p=top_p,
            streamer=streamer,
        )
        thread = threading.Thread(
            target=model.generate, kwargs=gen_kw, daemon=True,
        )
        thread.start()
        for text in streamer:
            yield text, None
        thread.join()
        # Emit a terminal tuple with finish_reason so callers can detect end
        yield "", "stop"

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def save_tensors(self, path: str, weight_dict: dict) -> None:
        import torch
        from safetensors.torch import save_file as _sf

        torch_dict = {}
        for k, v in weight_dict.items():
            if isinstance(v, torch.Tensor):
                torch_dict[k] = v.contiguous()
            else:
                torch_dict[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
        _sf(torch_dict, str(path))

    def load_tensors(self, path: str) -> dict:
        """Returns dict of {name → numpy float32 ndarray}."""
        try:
            from safetensors.torch import load_file as _lf
            return {k: v.float().numpy() for k, v in _lf(str(path)).items()}
        except Exception:
            from safetensors.numpy import load_file as _nf
            return dict(_nf(str(path)))

    # ── Memory management ─────────────────────────────────────────────────────

    def configure_memory(self, fraction: float = 0.90) -> None:
        """Set CUDA per-process memory fraction when CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available() and 0.0 < fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Stub backend — neither MLX nor torch installed (import-only / test env)
# ═══════════════════════════════════════════════════════════════════════════════

class _StubBackend:
    IS_APPLE: bool = False
    device: str = "cpu"

    def _fail(self, *_, **__):
        raise RuntimeError(
            "squish: no compute backend available. "
            "On macOS install mlx: pip install mlx mlx-lm. "
            "On Linux install torch: pip install torch transformers."
        )

    array            = _fail
    eval             = lambda self, *a, **k: None  # noqa: E731
    to_numpy         = _fail
    forward          = _fail
    forward_np       = _fail
    load_model       = _fail
    stream_generate  = _fail
    save_tensors     = _fail
    load_tensors     = _fail
    configure_memory = lambda self, *a, **k: None  # noqa: E731


# ── Module-level singleton ────────────────────────────────────────────────────

if _IS_APPLE:
    BE = _AppleBackend()
else:
    try:
        BE = _TorchBackend()
    except ImportError:
        BE = _StubBackend()  # type: ignore[assignment]
