# squish-quant

High-throughput INT8 symmetric weight quantizer for the Squish LLM compression toolkit.

Implemented in Rust with Rayon parallel processing and PyO3 Python bindings.

## Performance

- **8–12 GB/s** sustained quantization throughput (Apple Silicon M-series)
- **5–8×** faster than vectorized numpy baseline (~1.5 GB/s)
- 14B model (29.6 GB bf16): ~3s vs ~16s with numpy

## Usage

```python
from squish_quant import quantize_int8_f32, quantize_int8_bf16, quantize_int8_grouped

# Float32 input
q, scales = quantize_int8_f32(arr)   # arr: (N,D) float32 → q: (N,D) int8, scales: (N,)

# BF16 input (native HuggingFace format — avoids Python-side cast)
q, scales = quantize_int8_bf16(arr)  # arr: (N,D) float16

# Grouped quantization (higher accuracy, larger scale array)
q, scales = quantize_int8_grouped(arr, group_size=64)  # scales: (N, D//64)
```

## Build

```bash
pip install maturin
python3 -m maturin build --release
pip install target/wheels/squish_quant-*.whl
```
