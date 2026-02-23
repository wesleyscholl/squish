use ancf_core::codec::{BlockMeta, Codec};
use ancf_core::format::CODEC_PASSTHROUGH;

/// No-op codec: stores blocks verbatim, with no compression.
///
/// Useful for:
/// - Verifying the format round-trip independently of any codec.
/// - Data that is already compressed (e.g., JPEG, MP4) where further
///   compression would expand the file.
pub struct PassThroughCodec;

impl Codec for PassThroughCodec {
    fn id(&self) -> u16 {
        CODEC_PASSTHROUGH
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn compress_block(&self, raw: &[u8], _meta: &mut BlockMeta) -> anyhow::Result<Vec<u8>> {
        Ok(raw.to_vec())
    }

    fn decompress_block(&self, compressed: &[u8], _meta: &BlockMeta) -> anyhow::Result<Vec<u8>> {
        Ok(compressed.to_vec())
    }
}
