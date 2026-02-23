use ancf_core::codec::{BlockMeta, Codec};
use ancf_core::format::CODEC_LZ4;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

/// LZ4 block codec.
///
/// Fastest decompression of all bundled codecs — typically 3–5 GB/s on
/// modern hardware. Best for NVMe/local workloads where I/O latency is low
/// and decode speed matters more than size reduction.
///
/// Best for: hot data, low-latency random access workloads.
pub struct Lz4Codec;

impl Codec for Lz4Codec {
    fn id(&self) -> u16 {
        CODEC_LZ4
    }

    fn name(&self) -> &'static str {
        "lz4"
    }

    fn compress_block(&self, raw: &[u8], _meta: &mut BlockMeta) -> anyhow::Result<Vec<u8>> {
        Ok(compress_prepend_size(raw))
    }

    fn decompress_block(&self, compressed: &[u8], _meta: &BlockMeta) -> anyhow::Result<Vec<u8>> {
        let raw = decompress_size_prepended(compressed)
            .map_err(|e| anyhow::anyhow!("lz4 decompress error: {}", e))?;
        Ok(raw)
    }
}
