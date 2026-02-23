use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_64;

use crate::codec::{BlockMeta, Codec};
use crate::format::{Ancf1Header, BlockEntry, BLOCK_ENTRY_SIZE, FLAG_HAS_CHECKSUM, HEADER_SIZE};

/// Random-access reader for ANCF1 files.
///
/// # Open sequence
/// 1. Read the 56-byte header (magic check, codec_id, block_count, block_size).
/// 2. Seek to `file_end - 8`, read the `index_offset` u64.
/// 3. Seek to `index_offset`, load the full block index into RAM (`Vec<BlockEntry>`).
///
/// The entire block index is small: 32 bytes × N blocks.
/// A 100 GB file with 64 KB blocks has ~1.6 million blocks → ~50 MB index.
/// For typical usage the index fits comfortably in RAM.
///
/// # Access pattern
/// [`read_block`] seeks directly to the block's byte offset and decodes only
/// that block. No other blocks are touched.
///
/// [`read_range`] resolves the byte range to a minimal span of blocks
/// (usually 1–2), decodes only those, and slices the result precisely.
pub struct Reader {
    file: File,
    pub header: Ancf1Header,
    entries: Vec<BlockEntry>,
    codec: Arc<dyn Codec>,
}

impl Reader {
    /// Open an ANCF1 file.
    ///
    /// `codec` must match the `codec_id` stored in the file header. Use
    /// `ancf_codecs::codec_by_id(header_codec_id)` to obtain the right codec
    /// after a first-pass header read, or pre-select when the codec is known.
    pub fn open(path: impl AsRef<Path>, codec: Arc<dyn Codec>) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;

        // ── Read and validate header ────────────────────────────────────────
        let mut header_buf = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut header_buf)?;
        let header = Ancf1Header::from_bytes(&header_buf)?;

        if header.version != 1 {
            anyhow::bail!(
                "unsupported ANCF version {} (only version 1 is supported)",
                header.version
            );
        }
        if header.codec_id != codec.id() {
            anyhow::bail!(
                "codec mismatch: file uses codec {} but provided codec has id {}",
                header.codec_id,
                codec.id()
            );
        }

        // ── Read footer → index offset ──────────────────────────────────────
        file.seek(SeekFrom::End(-8))?;
        let mut footer_buf = [0u8; 8];
        file.read_exact(&mut footer_buf)?;
        let index_offset = u64::from_le_bytes(footer_buf);

        // ── Load block index ────────────────────────────────────────────────
        file.seek(SeekFrom::Start(index_offset))?;
        let mut entries = Vec::with_capacity(header.block_count as usize);
        let mut entry_buf = [0u8; BLOCK_ENTRY_SIZE as usize];
        for _ in 0..header.block_count {
            file.read_exact(&mut entry_buf)?;
            entries.push(BlockEntry::from_bytes(&entry_buf)?);
        }

        Ok(Self {
            file,
            header,
            entries,
            codec,
        })
    }

    /// Total number of blocks in the file.
    #[inline]
    pub fn block_count(&self) -> u64 {
        self.header.block_count
    }

    /// Nominal raw bytes per block (the last block may be smaller).
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.header.block_size
    }

    /// Total uncompressed size of all blocks in bytes.
    pub fn raw_size(&self) -> u64 {
        self.entries.iter().map(|e| e.raw_len as u64).sum()
    }

    /// Total compressed size of all blocks in bytes (excluding index/header).
    pub fn compressed_size(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| e.compressed_len as u64 + e.metadata_len as u64)
            .sum()
    }

    /// Compression ratio (raw / compressed).
    pub fn ratio(&self) -> f64 {
        let raw = self.raw_size();
        let compressed = self.compressed_size();
        if compressed == 0 {
            return 1.0;
        }
        raw as f64 / compressed as f64
    }

    /// Access the raw `BlockEntry` slice (for inspection / benchmarks).
    pub fn entries(&self) -> &[BlockEntry] {
        &self.entries
    }

    /// Decompress and return the raw bytes of block `idx`.
    ///
    /// Only the single block at `entries[idx].offset` is read from disk.
    /// All other blocks are untouched — this is the core O(1) seek guarantee.
    pub fn read_block(&mut self, idx: u64) -> anyhow::Result<Vec<u8>> {
        let entry = self
            .entries
            .get(idx as usize)
            .ok_or_else(|| anyhow::anyhow!("block index {} out of range (total {})", idx, self.header.block_count))?
            .clone();

        // Seek to block start
        self.file.seek(SeekFrom::Start(entry.offset))?;

        // Read optional per-block metadata sidecar
        let meta = if entry.metadata_len > 0 {
            // The block starts with [metadata_len:u16][sidecar bytes]
            let mut len_buf = [0u8; 2];
            self.file.read_exact(&mut len_buf)?;
            let on_disk_meta_len = u16::from_le_bytes(len_buf);
            if on_disk_meta_len != entry.metadata_len {
                anyhow::bail!(
                    "block {} metadata_len mismatch: index says {} but on-disk prefix says {}",
                    idx,
                    entry.metadata_len,
                    on_disk_meta_len
                );
            }
            let mut sidecar = vec![0u8; entry.metadata_len as usize];
            self.file.read_exact(&mut sidecar)?;
            BlockMeta { sidecar }
        } else {
            BlockMeta::default()
        };

        // Read compressed payload
        let mut compressed = vec![0u8; entry.compressed_len as usize];
        self.file.read_exact(&mut compressed)?;

        // Verify checksum if the flag is set
        if self.header.has_flag(FLAG_HAS_CHECKSUM) {
            let computed = xxh3_64(&compressed);
            if computed != entry.checksum {
                anyhow::bail!(
                    "block {} checksum mismatch: expected {:016x}, got {:016x}",
                    idx,
                    entry.checksum,
                    computed
                );
            }
        }

        // Decompress
        let raw = self.codec.decompress_block(&compressed, &meta)?;

        if raw.len() != entry.raw_len as usize {
            anyhow::bail!(
                "block {} decompressed to {} bytes but index says {}",
                idx,
                raw.len(),
                entry.raw_len
            );
        }

        Ok(raw)
    }

    /// Decompress and return exactly `len` bytes starting at raw byte offset
    /// `start` within the logical (uncompressed) file.
    ///
    /// Internally this resolves to the minimal set of blocks that cover the
    /// range, decodes only those blocks, and slices the result precisely.
    pub fn read_range(&mut self, start: u64, len: u64) -> anyhow::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let raw_total = self.raw_size();
        if start >= raw_total {
            anyhow::bail!("read_range start {} is beyond raw file size {}", start, raw_total);
        }

        let end = (start + len).min(raw_total); // clamp to file boundary
        let block_size = self.header.block_size as u64;

        let first_block = start / block_size;
        let last_block = (end - 1) / block_size;

        let mut result = Vec::with_capacity(len as usize);

        for block_idx in first_block..=last_block {
            let block_raw = self.read_block(block_idx)?;
            let block_start_in_file = block_idx * block_size;

            // Slice within this block
            let slice_start = if block_idx == first_block {
                (start - block_start_in_file) as usize
            } else {
                0
            };
            let slice_end = if block_idx == last_block {
                ((end - block_start_in_file) as usize).min(block_raw.len())
            } else {
                block_raw.len()
            };

            result.extend_from_slice(&block_raw[slice_start..slice_end]);
        }

        Ok(result)
    }
}
