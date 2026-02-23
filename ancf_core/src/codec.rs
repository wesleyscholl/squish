/// Per-block sidecar metadata written and read by the codec.
///
/// For generic codecs (PassThrough, Zstd, Lz4) this is always empty.
/// For domain codecs like FloatQuant it carries per-block min/max tables,
/// which make every block independently decompressable without loading the
/// full file's global quantization state.
#[derive(Default, Debug, Clone)]
pub struct BlockMeta {
    pub sidecar: Vec<u8>,
}

/// Core compression abstraction.
///
/// Each `Codec` implementation:
/// - Is identified by a stable numeric `id()` stored in the ANCF1 header.
/// - Must compress/decompress individual blocks independently — no cross-block
///   state is permitted. This is the invariant that makes random access possible.
/// - May write per-block metadata into `BlockMeta.sidecar`; this sidecar is
///   stored alongside the compressed bytes and provided back on decompress.
pub trait Codec: Send + Sync {
    /// Stable codec ID stored in the ANCF1 file header.
    fn id(&self) -> u16;

    /// Human-readable codec name for CLI display.
    fn name(&self) -> &'static str;

    /// Compress a single independent block.
    ///
    /// Codecs may write domain-specific metadata (e.g., per-block float min/max)
    /// into `meta.sidecar`. This sidecar is stored in the block index entry and
    /// passed back to `decompress_block` verbatim.
    fn compress_block(&self, raw: &[u8], meta: &mut BlockMeta) -> anyhow::Result<Vec<u8>>;

    /// Decompress a single independent block.
    ///
    /// `meta` contains the sidecar written by `compress_block` for this block.
    /// For codecs with `meta.sidecar.is_empty()`, this argument can be ignored.
    fn decompress_block(&self, compressed: &[u8], meta: &BlockMeta) -> anyhow::Result<Vec<u8>>;
}
