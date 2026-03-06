"""Phase 3A — Chunked prefill for long COMPRESS_PATH requests.

Splits a long token sequence into equal-sized chunks and runs one model
forward pass per chunk.  The KV cache accumulates state across chunks via
the shared ``layer_caches`` argument, so each chunk sees all previous KV
context — this is identical in expressiveness to full-length prefill.

Why chunk at all?
  • On Apple M-series, large MLX graphs (> ~512 tokens) hold the Neural Engine
    / GPU compute tile for the entire duration.  Chunking releases the tile
    between chunks, letting the OS interleave other work and preventing thermal
    buildup.
  • When ``interleave_decode=True``, callers can yield one greedy token between
    chunks, streaming the first output token as soon as the first chunk
    completes rather than waiting for the full prefill.  This cuts TTFT from
    O(total_len) to O(chunk_size) prefill latency.

CRITICAL conflict rule (plan §3):
  Speculative decode (EAGLE-3 / N-gram) MUST NOT start until the final chunk
  completes.  The caller must check ``is_final_chunk`` and only activate spec
  decode when True.  On intermediate chunks the caller should at most emit one
  greedy token for the TTFT benefit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx


@dataclass
class ChunkedPrefillConfig:
    """Configuration for chunked prefill.

    Attributes:
        chunk_size:        Maximum tokens processed per forward pass.
                           Should be a power-of-two multiple of the model's
                           attention block size (default 512).
        interleave_decode: When True, callers should yield one greedy decode
                           token between non-final chunks for minimum TTFT.
                           The token counts against ``max_tokens`` but is not
                           submitted through the speculative decode path.
    """
    chunk_size: int = 512
    interleave_decode: bool = True


def chunk_prefill(
    model,
    input_ids: list[int],
    layer_caches,
    config: ChunkedPrefillConfig | None = None,
) -> Iterator[tuple["mx.array", bool]]:
    """Yield ``(last_token_logit, is_final_chunk)`` for each prefill chunk.

    Parameters
    ----------
    model:
        MLX model callable — ``model(x, cache=layer_caches)`` interface.
    input_ids:
        Full token-id list for the prompt (already tokenised).
    layer_caches:
        Per-layer KV cache objects (e.g. from ``_kv_cache._layers``).
        The cache is updated **in-place** on every chunk; the caller must
        NOT reset the cache between chunks.
    config:
        Chunking configuration.  Defaults to ``ChunkedPrefillConfig()``.

    Yields
    ------
    logit_mx : mx.array
        The last-position logit vector from the chunk, shape ``[vocab_size]``.
        Materialised with ``mx.eval`` before yielding — safe to sample from
        immediately without a second eval call.
    is_final_chunk : bool
        True only for the last chunk.  Callers should activate speculative
        decode **only** when this is True.

    Notes
    -----
    * If ``len(input_ids) <= chunk_size`` a single pair is yielded with
      ``is_final_chunk=True`` — identical to non-chunked prefill.
    * On error inside the loop the generator raises; the caller's except block
      should fall back to standard single-shot prefill (see server.py wiring).
    """
    import mlx.core as mx

    if config is None:
        config = ChunkedPrefillConfig()

    n_tokens = len(input_ids)
    chunk_size = max(1, config.chunk_size)
    n_chunks = max(1, (n_tokens + chunk_size - 1) // chunk_size)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_tokens)
        chunk_ids = input_ids[start:end]
        is_final = chunk_idx == n_chunks - 1

        x = mx.array(chunk_ids, dtype=mx.int32)[None]    # shape [1, chunk_len]
        logits = model(x, cache=layer_caches)             # shape [1, chunk_len, vocab]
        mx.eval(logits)                                   # materialise before yield

        # Yield the last-position logit (shape [vocab_size]) as a flat vector
        # so callers can pass it directly to _sample_mx(logit, temp, top_p).
        yield logits[0, -1], is_final
