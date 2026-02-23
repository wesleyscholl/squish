use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use xxhash_rust::xxh3::xxh3_64;

use crate::codec::{BlockMeta, Codec};
use crate::format::{
    Ancf1Header, BlockEntry, BLOCK_ENTRY_SIZE, FLAG_HAS_CHECKSUM, HEADER_SIZE,
};

/// Streaming writer for ANCF1 files.
///
/// # Write contract
/// Call [`write`] any number of times with arbitrary-sized byte slices.
/// The writer accumulates data and flushes independent compressed blocks
/// whenever `block_size` bytes of raw data have been gathered.
/// Call [`finish`] to flush any remaining partial block, append the block
/// index and footer, and write back the final header.
///
/// # Format layout written
/// ```text
/// [HEADER: 56 bytes placeholder]
/// [BLOCK 0] [BLOCK 1] ... [BLOCK N-1]      ← independent compressed blocks
/// [BLOCK INDEX: 32 bytes × N]
/// [FOOTER: 8 bytes — u64 LE offset of block index]
/// ← seek back to 0, overwrite header with real values
/// ```
pub struct Writer {
    file: File,
    codec: Box<dyn Codec>,
    block_size: u32,
    /// Pending raw bytes not yet flushed into a block.
    pending: Vec<u8>,
    /// In-memory block index, appended to file on `finish()`.
    entries: Vec<BlockEntry>,
    /// Current write position in the file (mirrors the file cursor).
    current_offset: u64,
}

impl Writer {
    /// Create a new ANCF1 file at `path`.
    ///
    /// Overwrites any existing file. `block_size` controls the nominal raw bytes
    /// per compressed block; use [`DEFAULT_BLOCK_SIZE`] (64 KB) if unsure.
    pub fn create(
        path: impl AsRef<Path>,
        codec: Box<dyn Codec>,
        block_size: u32,
    ) -> anyhow::Result<Self> {
        let mut file = File::create(path)?;
        // Write placeholder header (will be overwritten in finish())
        file.write_all(&[0u8; HEADER_SIZE as usize])?;
        Ok(Self {
            file,
            codec,
            block_size,
            pending: Vec::with_capacity(block_size as usize * 2),
            entries: Vec::new(),
            current_offset: HEADER_SIZE,
        })
    }

    /// Buffer `data` and flush complete blocks as they fill up.
    pub fn write(&mut self, data: &[u8]) -> anyhow::Result<()> {
        self.pending.extend_from_slice(data);
        while self.pending.len() >= self.block_size as usize {
            let raw: Vec<u8> = self.pending.drain(..self.block_size as usize).collect();
            self.flush_block(&raw)?;
        }
        Ok(())
    }

    /// Compress `raw` as a single block and write it to the file.
    fn flush_block(&mut self, raw: &[u8]) -> anyhow::Result<()> {
        let mut meta = BlockMeta::default();
        let compressed = self.codec.compress_block(raw, &mut meta)?;
        let checksum = xxh3_64(&compressed);

        let block_offset = self.current_offset;
        let metadata_len = meta.sidecar.len() as u16;

        // Write optional per-block metadata sidecar
        if metadata_len > 0 {
            self.file.write_all(&metadata_len.to_le_bytes())?;
            self.file.write_all(&meta.sidecar)?;
            self.current_offset += 2 + meta.sidecar.len() as u64;
        }

        // Write compressed payload
        self.file.write_all(&compressed)?;
        let compressed_len = compressed.len() as u32;
        self.current_offset += compressed_len as u64;

        self.entries.push(BlockEntry {
            offset: block_offset,
            compressed_len,
            raw_len: raw.len() as u32,
            checksum,
            metadata_len,
        });

        Ok(())
    }

    /// Flush remaining buffered data, write the block index + footer, and seal
    /// the file by writing the final header.
    ///
    /// Returns the number of blocks written.
    pub fn finish(mut self) -> anyhow::Result<u64> {
        // Flush any partial trailing block
        if !self.pending.is_empty() {
            let remaining = std::mem::take(&mut self.pending);
            self.flush_block(&remaining)?;
        }

        // ── Block index ────────────────────────────────────────────────────
        let index_offset = self.current_offset;
        for entry in &self.entries {
            self.file.write_all(&entry.to_bytes())?;
        }
        let index_bytes = self.entries.len() as u64 * BLOCK_ENTRY_SIZE;
        self.current_offset += index_bytes;

        // ── Footer: 8-byte u64 LE offset of block index start ──────────────
        self.file.write_all(&index_offset.to_le_bytes())?;

        // ── Seek back to 0 and write the real header ────────────────────────
        let block_count = self.entries.len() as u64;
        let header = Ancf1Header {
            version: 1,
            codec_id: self.codec.id(),
            block_size: self.block_size,
            block_count,
            flags: FLAG_HAS_CHECKSUM,
        };
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header.to_bytes())?;
        self.file.flush()?;

        Ok(block_count)
    }
}
