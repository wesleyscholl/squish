use ancf_core::codec::{BlockMeta, Codec};
use ancf_core::format::CODEC_ZSTD;

/// Zstandard block codec.
///
/// Each block is compressed independently with `zstd` at the configured level
/// (default: 3). Because each block is independent, any block can be
/// decompressed without touching adjacent blocks.
///
/// Best for: general text, JSON, logs, mixed structured data.
pub struct ZstdCodec {
    /// Compression level (1 = fast / larger, 22 = slow / smallest).
    pub level: i32,
}

impl Default for ZstdCodec {
    fn default() -> Self {
        Self { level: 3 }
    }
}

impl ZstdCodec {
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl Codec for ZstdCodec {
    fn id(&self) -> u16 {
        CODEC_ZSTD
    }

    fn name(&self) -> &'static str {
        "zstd"
    }

    fn compress_block(&self, raw: &[u8], _meta: &mut BlockMeta) -> anyhow::Result<Vec<u8>> {
        let compressed = zstd::bulk::compress(raw, self.level)?;
        Ok(compressed)
    }

    fn decompress_block(&self, compressed: &[u8], _meta: &BlockMeta) -> anyhow::Result<Vec<u8>> {
        // We know the original block size from BlockEntry.raw_len but the
        // zstd frame also carries its own content size, so we let zstd decode
        // into a fresh Vec without needing to pre-size it. For production
        // we'd pass raw_len as a hint; for the POC this is sufficient.
        let raw = zstd::decode_all(compressed)?;
        Ok(raw)
    }
}
