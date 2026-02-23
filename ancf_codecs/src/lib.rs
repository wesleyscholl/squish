mod lz4_codec;
mod passthrough;
mod zstd_codec;

pub use lz4_codec::Lz4Codec;
pub use passthrough::PassThroughCodec;
pub use zstd_codec::ZstdCodec;

use ancf_core::Codec;
use ancf_core::format::{CODEC_LZ4, CODEC_PASSTHROUGH, CODEC_ZSTD};
use std::sync::Arc;

/// Resolve a codec from its on-disk `codec_id`.
///
/// Called by the CLI and Python bindings when opening an existing ANCF1 file,
/// so the reader can be initialized with the right codec automatically.
pub fn codec_by_id(id: u16) -> anyhow::Result<Arc<dyn Codec>> {
    match id {
        CODEC_PASSTHROUGH => Ok(Arc::new(PassThroughCodec)),
        CODEC_ZSTD => Ok(Arc::new(ZstdCodec::default())),
        CODEC_LZ4 => Ok(Arc::new(Lz4Codec)),
        _ => anyhow::bail!("unknown codec id {}; POC supports 0 (passthrough), 1 (zstd), 2 (lz4)", id),
    }
}
