# ANCF — Access-Native Compression Format

> Compressed storage that is also the working format.  
> Any block is independently seekable and decompressable in O(1) seek + O(block) decode.  
> No full-file decompression ever required.

---

## Problem Statement

Most compression algorithms encode data as a sequential stream with inter-block dependencies — reading byte 10,000 may require decoding bytes 1–9,999 first. This forces a full decompress-then-use cycle that is prohibitively expensive for large files (LLM weights, embeddings, columnar datasets, large blobs).

The goal of ANCF is to design a format where the **compressed form is the working form**: individual records or byte ranges can be accessed by seeking directly to the correct block and decoding only that block.

---

## Theoretical Basis

### Shannon's Source Coding Theorem (1948)

Every data source has an **entropy** — the theoretical minimum number of bits required to represent its information losslessly. No algorithm can compress below this floor without losing data.

$$H(X) = -\sum_{i} p_i \log_2 p_i$$

This is a hard mathematical limit, not an engineering one.

### Practical Shannon Entropy by Data Type

| Data Type | Entropy Density | Best Lossless Ratio | ANC Viable? | Notes |
|-----------|----------------|---------------------|-------------|-------|
| Columnar / tabular data | 10–30% | 5x–20x | ✓ Excellent | Most viable target |
| Log files / text | 10–25% | 5x–15x | ✓ Excellent | Dictionary encoding is extremely effective |
| Genomic sequences | 5–15% | 10x–50x | ✓ Excellent | Nucleotide codecs can hit 100x on raw FASTQ |
| Scientific sensor data | 15–40% | 5x–15x | ✓ Good | Delta/Gorilla encoding works well on time-series |
| LLM weights (fp32) | 40–60% | 2x–4x | ~ Moderate | Quantization (4-bit) is the main lever |
| LLM weights (quantized) | 70–85% | 1.2x–1.5x | ✗ Limited | Already near entropy floor after quantization |
| Generic binary files | 60–80% | 10–30% | ✗ Limited | No exploitable structure by definition |
| Encrypted / random data | ~100% | ~0% | ✗ None | Shannon entropy ≈ file size. Mathematically impossible. |

**Key insight**: a 10x lossless compression gain is achievable on specific high-redundancy data types.  
100x losslessly is almost never achievable on real data. Knowing your data's actual entropy is the first step.

---

## Factors That Determine Feasibility

| Factor | Tier | Impact |
|--------|------|--------|
| **Shannon Entropy** | ABSOLUTE LIMIT | The mathematical floor. No engineering overcomes it. |
| **Domain Structure** | BIGGEST LEVER | Columnar formats, genomic codecs, float delta-encoding exploit structure generic compressors miss entirely. |
| **Block Granularity** | DESIGN TRADEOFF | Smaller blocks → better random access, worse ratios. Sweet spot: 32–64 KB. |
| **Access Patterns** | DATA-DEPENDENT | Sequential, columnar, random, range — each requires different codec design. |
| **Hardware: SIMD / GPU** | HARDWARE-BOUND | AVX-512 operates on 4-bit integers natively. GPU tensor cores handle INT8/INT4. |
| **Storage vs. CPU Tradeoff** | SYSTEMS TRADEOFF | On slow storage (S3, HDD), compress+decompress can be faster than reading uncompressed. Profile, don't assume. |

---

## Core Architectural Concept

Instead of:
```
Compressed Storage → DECOMPRESS (full file) → RAM Buffer → CPU / GPU
```

ANCF uses:
```
Compressed Storage (block-indexed) → Index Lookup (O(1)) → Partial Decode (1 block) → CPU / GPU
```

**To access any value**: `seek(block_offset[i])` → `decode(32–64 KB)` → return result.  
The rest of the file is never touched.

---

## Real-World Analogues (Already Exist in Pieces)

| System | What it does |
|--------|-------------|
| Apache Parquet | Columnar block format with per-column codecs |
| DuckDB | Operates directly on compressed columnar data |
| GGUF / AWQ / GPTQ | Quantized LLM weights with direct access |
| Gorilla TSDB (Meta) | Time-series with native compressed operations |
| Apache Arrow | In-memory columnar with vectorized compressed ops |

ANCF's innovation is a **unified, general-purpose format** that brings this pattern to arbitrary data types via a pluggable `Codec` trait.

---

## Binary Format Specification: ANCF1

```
┌──────────────────────────────────────────────┐
│  HEADER  (56 bytes, fixed)                   │
│  magic[14] | version:u16 | codec_id:u16      │
│  block_size:u32 | block_count:u64            │
│  flags:u64 | reserved[16]                   │
├──────────────────────────────────────────────┤
│  BLOCK 0  (compressed bytes, self-contained) │
│  [per-block metadata prefix, if flags set]   │
├──────────────────────────────────────────────┤
│  BLOCK 1                                     │
│  ...                                         │
│  BLOCK N-1                                   │
├──────────────────────────────────────────────┤
│  BLOCK INDEX  (32 bytes × N)                 │
│  For each block:                             │
│    offset:u64 | comp_len:u32 | raw_len:u32   │
│    checksum:u32 | metadata_len:u16 | pad[10] │
├──────────────────────────────────────────────┤
│  INDEX FOOTER  (8 bytes)                     │
│  u64 LE — byte offset of block index start  │
└──────────────────────────────────────────────┘
```

### Header Fields

| Field | Type | Description |
|-------|------|-------------|
| `magic` | `[u8; 14]` | `b"ANCF1\n\0\0\0\0\0\0\0\0"` |
| `version` | `u16 LE` | Format version (currently 1) |
| `codec_id` | `u16 LE` | Codec used for all blocks |
| `block_size` | `u32 LE` | Nominal raw bytes per block (default: 65536) |
| `block_count` | `u64 LE` | Total number of blocks in file |
| `flags` | `u64 LE` | Bit field (see below) |
| `reserved` | `[u8; 16]` | Future use, must be zero |

### Flags Bit Field

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `HAS_CHECKSUM` | Each block's xxhash3 checksum is valid |
| 1 | `PER_BLOCK_META` | Each block is prefixed with `metadata_len` bytes of sidecar data |
| 2 | `IS_COLUMNAR` | Data is columnar; block index includes column metadata |
| 3–63 | Reserved | Must be zero |

### Why Footer Index (Not Header)?

A footer-based index allows **streaming writes** — the writer doesn't need to know the total block count or byte offsets before writing data. After all blocks are written, the index is appended, followed by a single 8-byte footer giving the index's start offset. Reading requires one `seek(file_end - 8)` to find the index.

---

## Codec Trait

```rust
pub trait Codec: Send + Sync {
    fn id(&self) -> u16;
    fn name(&self) -> &'static str;
    fn compress_block(&self, raw: &[u8], meta: &mut BlockMeta) -> Result<Vec<u8>>;
    fn decompress_block(&self, compressed: &[u8], meta: &BlockMeta) -> Result<Vec<u8>>;
}
```

`BlockMeta` is a small struct the codec can write per-block sidecar data into (e.g., float min/max for the `FloatQuant` codec). This makes every block **self-contained** — a critical requirement for independent decompression.

---

## Bundled Codecs

| Codec | ID | Best for | Crate dependency |
|-------|----|----------|-----------------|
| `PassThrough` | 0 | Testing, already-compressed data | — |
| `Zstd` | 1 | General text, JSON, logs, mixed data | `zstd` |
| `Lz4` | 2 | Fast general, NVMe / low-latency workloads | `lz4_flex` |
| `DeltaInt` | 3 | Monotonic integers, timestamps, sorted columns | custom |
| `FloatQuant` | 4 | f32 arrays, embeddings, neural net weights | custom (extends vectro-plus per-block quant) |
| `BitPack` | 5 | Low-cardinality integers, enum columns | `bitpacking` |
| `Rle` | 6 | Sparse / repetitive runs (null columns, bool arrays) | custom |

---

## Workspace Structure

Mirrors the established pattern from `vectro-plus` and `ZipGraph`:

```
squish/
├── Cargo.toml                 — workspace root
├── PLAN.md                    — this file
├── LICENSE
│
├── ancf_core/                 — format spec, Codec trait, Reader, Writer
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── format.rs          — ANCF1Header, BlockEntry, magic constants
│       ├── codec.rs           — Codec trait, BlockMeta, CodecRegistry
│       ├── writer.rs          — streaming block writer with parallel codec dispatch
│       └── reader.rs          — block index reader, O(1) seek, read_block, read_range
│
├── ancf_codecs/               — bundled codec implementations
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── passthrough.rs
│       ├── zstd_codec.rs
│       ├── lz4_codec.rs
│       ├── delta_int.rs
│       ├── float_quant.rs     — per-block f32→u8 quantization (extends vectro-plus)
│       ├── bitpack.rs
│       └── rle.rs
│
├── ancf_cli/                  — compress / decompress / inspect / bench subcommands
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── ancf_py/                   — PyO3 + maturin Python bindings
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs
│
└── ancf_bench/                — Criterion benchmarks
    ├── Cargo.toml
    └── benches/
        ├── compress.rs        — compress 1MB / 10MB / 100MB per codec
        ├── random_access.rs   — 1,000 random block seeks, measure p50/p99
        └── sequential.rs      — full scan throughput (MB/s)
```

---

## 5-Layer Runtime Architecture

| Layer | Responsibility |
|-------|---------------|
| **1 — Storage** | Compressed blocks (4–64 KB each), independently decodable, indexed by footer |
| **2 — Block Index** | Byte offset map loaded once on `open()`, `Vec<BlockEntry>` in RAM |
| **3 — Codec Engine** | Domain-specific codec: FloatQuant, DeltaInt, Zstd, LZ4, BitPack |
| **4 — Access Runtime** | API: `read_block(idx)` → seek → decode → checksum → return bytes |
| **5 — Compute Interface** | Python bindings / CLI / future: vectorized / SIMD ops on decoded slices |

---

## Access Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Open file | O(index) | One seek + read of `32N` bytes for N-block index |
| Read any single block | O(block) | 1 seek + decode ~64 KB |
| Read a byte range | O(k·block) | k = number of blocks the range spans, usually 1–2 |
| Compress (write) | O(N) | Parallel block encoding via rayon |
| Full sequential scan | O(N·block) | Equivalent to streaming decompress |

---

## Language Decision

**Rust** — chosen for:
- Zero-copy slice operations and `memmap2` for mmap-backed random access
- SIMD-friendly byte slice ops for codec inner loops (relevant for `FloatQuant`, `BitPack`)
- No GC pauses during block decode (critical for latency SLOs)
- Established pattern in this codebase (`vectro-plus`, `ZipGraph`)
- Native PyO3 for Python bindings without a C ABI bridge

**Python bindings** via PyO3 + maturin for data science / ML ecosystem integration.

**Go client** (future) — a thin service-layer wrapper following the `generic-go-service` pattern if ANCF needs to plug into the microservice fleet.

---

## Differences from vectro-plus That Make This General-Purpose

| vectro-plus limitation | ANCF solution |
|-----------------------|---------------|
| Sequential-only (no seek index) | Block index footer → O(1) random block access |
| Global quantization tables (must load whole file to compute min/max) | Per-block `BlockMeta` — each block carries its own quant tables |
| Single fixed codec (u8 scalar quant) | `Codec` trait + `CodecRegistry` — pluggable at write time, auto-discovered at read |
| Vector-specific format (STREAM1/QSTREAM1) | Generic `&[u8]` interface — any data type |
| No checksums | xxhash3 per block — detectable corruption |
| No partial range access | `read_range(byte_start, len)` resolves to minimum set of blocks |

---

## Build Plan (POC → Production)

### Phase 1 — POC (prove the concept)
- [ ] Workspace bootstrap (`Cargo.toml`)
- [ ] `ancf_core`: `ANCF1Header`, `BlockEntry`, `Codec` trait, `BlockMeta`
- [ ] `ancf_core`: `Writer` — sequential block write with zstd, footer index append
- [ ] `ancf_core`: `Reader` — open, load index, `read_block(idx)`, `read_range(start, len)`
- [ ] `ancf_codecs`: `PassThrough` + `Zstd` codec implementations
- [ ] Integration test: compress 10 MB, seek to block 80 without reading blocks 0–79, assert bytes match
- [ ] `ancf_cli`: `compress`, `decompress`, `inspect` subcommands

### Phase 2 — Codec Coverage
- [ ] `FloatQuant` codec with per-block min/max tables (self-contained decode)
- [ ] `DeltaInt` codec (for sorted integer columns / timestamps)
- [ ] `Lz4` codec (low-latency alternative to zstd)
- [ ] `BitPack` codec (low-cardinality integer columns)
- [ ] `Rle` codec (null arrays, boolean columns)

### Phase 3 — Python Bindings
- [ ] `ancf_py` crate with PyO3 + maturin
- [ ] `compress(src, dst, codec)`, `metadata(path)`, `read_block(path, idx)` Python API
- [ ] `read_blocks_numpy(path, indices)` returning `np.ndarray` for float data

### Phase 4 — Benchmarks & Validation
- [ ] Criterion benchmarks: compress throughput per codec (1 MB / 10 MB / 100 MB)
- [ ] Random-access benchmark: 1,000 random block reads — measure p50 / p99 latency
- [ ] Compare: ANCF/zstd vs. raw zstd vs. gzip vs. uncompressed mmap
- [ ] Validate Shannon floor: run `ancf inspect` on encrypted data, show ratio ≈ 1.0

### Phase 5 — Columnar Extension (future)
- [ ] Column-aware write API for tabular data (one codec per column)
- [ ] Column strip index (find all blocks containing column K without decoding others)
- [ ] Predicate pushdown: skip blocks whose per-block min/max exclude the query range

---

## Verification Strategy

| Test | What it proves |
|------|---------------|
| Round-trip write → read_block (each codec) | Codec correctness |
| Read block N without reading 0..N-1 | True random access — the core claim |
| `read_range` across a block boundary | Multi-block range reads work correctly |
| Corrupt compressed bytes → checksum error | Data integrity detection |
| Compress `/dev/urandom` 1MB → inspect ratio | Shannon floor demonstrated |
| Criterion random_access bench | Latency numbers for O(1) seek claim |

---

*Last updated: 2026-02-23*  
*Status: Phase 1 in progress*
