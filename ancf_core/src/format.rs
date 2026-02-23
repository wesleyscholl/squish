/// Magic bytes for ANCF version 1 files.
/// 14 bytes: "ANCF1\n" followed by 8 null bytes.
pub const MAGIC: &[u8; 14] = b"ANCF1\n\x00\x00\x00\x00\x00\x00\x00\x00";

/// Fixed size of the ANCF1 file header in bytes.
///   magic[14] + version:u16 + codec_id:u16 + block_size:u32
///   + block_count:u64 + flags:u64 + reserved[18]
///   = 14 + 2 + 2 + 4 + 8 + 8 + 18 = 56
pub const HEADER_SIZE: u64 = 56;

/// Size of each BlockEntry in the block index, in bytes.
///   offset:u64 + compressed_len:u32 + raw_len:u32
///   + checksum:u64 + metadata_len:u16 + _pad[6]
///   = 8 + 4 + 4 + 8 + 2 + 6 = 32
pub const BLOCK_ENTRY_SIZE: u64 = 32;

/// Size of the index footer (single u64 offset) in bytes.
pub const FOOTER_SIZE: u64 = 8;

/// Default block size: 64 KB.
pub const DEFAULT_BLOCK_SIZE: u32 = 64 * 1024;

// ── Flags ──────────────────────────────────────────────────────────────────

/// Each block carries an xxhash3-64 checksum.
pub const FLAG_HAS_CHECKSUM: u64 = 1 << 0;

/// Each block is prefixed with a per-block metadata sidecar (for domain codecs
/// like FloatQuant that embed per-block min/max tables).
pub const FLAG_PER_BLOCK_META: u64 = 1 << 1;

// ── Codec IDs ──────────────────────────────────────────────────────────────

pub const CODEC_PASSTHROUGH: u16 = 0;
pub const CODEC_ZSTD: u16 = 1;
pub const CODEC_LZ4: u16 = 2;
pub const CODEC_DELTA_INT: u16 = 3;
pub const CODEC_FLOAT_QUANT: u16 = 4;
pub const CODEC_BITPACK: u16 = 5;
pub const CODEC_RLE: u16 = 6;

// ── Header ─────────────────────────────────────────────────────────────────

/// Decoded representation of the 56-byte ANCF1 file header.
#[derive(Debug, Clone)]
pub struct Ancf1Header {
    pub version: u16,
    pub codec_id: u16,
    /// Nominal raw bytes per block (the last block may be smaller).
    pub block_size: u32,
    pub block_count: u64,
    pub flags: u64,
}

impl Ancf1Header {
    /// Serialize to exactly `HEADER_SIZE` bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE as usize] {
        let mut buf = [0u8; HEADER_SIZE as usize];
        buf[..14].copy_from_slice(MAGIC);
        buf[14..16].copy_from_slice(&self.version.to_le_bytes());
        buf[16..18].copy_from_slice(&self.codec_id.to_le_bytes());
        buf[18..22].copy_from_slice(&self.block_size.to_le_bytes());
        buf[22..30].copy_from_slice(&self.block_count.to_le_bytes());
        buf[30..38].copy_from_slice(&self.flags.to_le_bytes());
        // reserved[18] stays zero
        buf
    }

    /// Deserialize from `HEADER_SIZE` bytes, checking the magic.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE as usize]) -> anyhow::Result<Self> {
        if &buf[..14] != MAGIC {
            anyhow::bail!("invalid ANCF magic bytes — not an ANCF1 file");
        }
        Ok(Self {
            version: u16::from_le_bytes(buf[14..16].try_into()?),
            codec_id: u16::from_le_bytes(buf[16..18].try_into()?),
            block_size: u32::from_le_bytes(buf[18..22].try_into()?),
            block_count: u64::from_le_bytes(buf[22..30].try_into()?),
            flags: u64::from_le_bytes(buf[30..38].try_into()?),
        })
    }

    pub fn has_flag(&self, flag: u64) -> bool {
        self.flags & flag != 0
    }
}

// ── Block index entry ───────────────────────────────────────────────────────

/// One entry in the block index — locates and describes a single compressed block.
#[derive(Debug, Clone, Default)]
pub struct BlockEntry {
    /// Byte offset of this block from the start of the file.
    pub offset: u64,
    /// Length of the compressed block payload in bytes (excluding metadata prefix).
    pub compressed_len: u32,
    /// Length of the original uncompressed data in bytes.
    pub raw_len: u32,
    /// xxhash3-64 of the compressed bytes.
    pub checksum: u64,
    /// Bytes of per-block sidecar metadata written before the compressed payload.
    /// Zero for codecs that don't use per-block metadata (PassThrough, Zstd, Lz4).
    pub metadata_len: u16,
}

impl BlockEntry {
    /// Serialize to exactly `BLOCK_ENTRY_SIZE` bytes.
    pub fn to_bytes(&self) -> [u8; BLOCK_ENTRY_SIZE as usize] {
        let mut buf = [0u8; BLOCK_ENTRY_SIZE as usize];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.compressed_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.raw_len.to_le_bytes());
        buf[16..24].copy_from_slice(&self.checksum.to_le_bytes());
        buf[24..26].copy_from_slice(&self.metadata_len.to_le_bytes());
        // buf[26..32] = 6 bytes padding, stays zero
        buf
    }

    /// Deserialize from `BLOCK_ENTRY_SIZE` bytes.
    pub fn from_bytes(buf: &[u8; BLOCK_ENTRY_SIZE as usize]) -> anyhow::Result<Self> {
        Ok(Self {
            offset: u64::from_le_bytes(buf[0..8].try_into()?),
            compressed_len: u32::from_le_bytes(buf[8..12].try_into()?),
            raw_len: u32::from_le_bytes(buf[12..16].try_into()?),
            checksum: u64::from_le_bytes(buf[16..24].try_into()?),
            metadata_len: u16::from_le_bytes(buf[24..26].try_into()?),
        })
    }
}
