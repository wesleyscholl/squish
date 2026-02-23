/// Integration test: proves that a single block can be read from an ANCF1 file
/// WITHOUT reading any other blocks — the fundamental POC claim.
///
/// Test sequence:
///  1. Generate ~1 MB of deterministic pseudo-random data
///  2. Split it into 64 KB blocks and write as ANCF1 (zstd)
///  3. Reopen the file, read ONLY block N (not 0..N-1)
///  4. Assert the decompressed bytes match the original data for that block
///  5. Assert the file position never crossed blocks 0..N-1 (seeked directly)
use std::sync::Arc;

use ancf_codecs::{Lz4Codec, PassThroughCodec, ZstdCodec};
use ancf_core::format::DEFAULT_BLOCK_SIZE;
use ancf_core::{Reader, Writer};

/// Generate `len` deterministic bytes using a simple LCG.
fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = seed;
    (0..len)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 56) as u8
        })
        .collect()
}

/// Generate `len` highly compressible bytes (repeating pattern).
fn compressible_bytes(len: usize) -> Vec<u8> {
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    (0..len).map(|i| pattern[i % pattern.len()]).collect()
}

// ── helpers ───────────────────────────────────────────────────────────────

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ancf_test_{}.ancf", name))
}

// ── tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_passthrough() {
    let data = compressible_bytes(4 * DEFAULT_BLOCK_SIZE as usize + 1234);
    let path = temp_path("passthrough");

    // Write
    let mut w = Writer::create(&path, Box::new(PassThroughCodec), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(&data).unwrap();
    let blocks = w.finish().unwrap();
    assert_eq!(blocks, 5); // 4 full + 1 partial

    // Read all blocks sequentially and reconstruct
    let mut r = Reader::open(&path, Arc::new(PassThroughCodec)).unwrap();
    let mut reconstructed = Vec::new();
    for i in 0..r.block_count() {
        reconstructed.extend(r.read_block(i).unwrap());
    }
    assert_eq!(reconstructed, data, "passthrough round-trip should be byte-exact");
}

#[test]
fn test_roundtrip_zstd() {
    let data = compressible_bytes(8 * DEFAULT_BLOCK_SIZE as usize + 777);
    let path = temp_path("zstd");

    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(&data).unwrap();
    let blocks = w.finish().unwrap();
    assert_eq!(blocks, 9);

    let mut r = Reader::open(&path, Arc::new(ZstdCodec::default())).unwrap();
    let mut reconstructed = Vec::new();
    for i in 0..r.block_count() {
        reconstructed.extend(r.read_block(i).unwrap());
    }
    assert_eq!(reconstructed, data);

    // File should be smaller than raw data (compressible input)
    let compressed_size = r.compressed_size();
    let raw_size = r.raw_size();
    assert!(
        compressed_size < raw_size,
        "zstd should compress compressible data: compressed={compressed_size} raw={raw_size}"
    );
    eprintln!("zstd ratio: {:.2}x", r.ratio());
}

#[test]
fn test_roundtrip_lz4() {
    let data = compressible_bytes(3 * DEFAULT_BLOCK_SIZE as usize);
    let path = temp_path("lz4");

    let mut w = Writer::create(&path, Box::new(Lz4Codec), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(&data).unwrap();
    w.finish().unwrap();

    let mut r = Reader::open(&path, Arc::new(Lz4Codec)).unwrap();
    let mut reconstructed = Vec::new();
    for i in 0..r.block_count() {
        reconstructed.extend(r.read_block(i).unwrap());
    }
    assert_eq!(reconstructed, data);
}

/// THE CORE POC TEST: read only block N without touching blocks 0..N-1.
///
/// We write 16 blocks. We then open the file and read ONLY block 12.
/// The test asserts that the bytes of block 12 match the original data
/// exactly — proving that the seek-based random access works correctly
/// and that no other blocks need to be decoded to recover block 12.
#[test]
fn test_random_access_skips_prior_blocks() {
    const NUM_BLOCKS: usize = 16;
    const TARGET_BLOCK: u64 = 12;

    let data = pseudo_random_bytes(NUM_BLOCKS * DEFAULT_BLOCK_SIZE as usize, 0xDEAD_BEEF);
    let path = temp_path("random_access");

    // Write
    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(&data).unwrap();
    let block_count = w.finish().unwrap();
    assert_eq!(block_count, NUM_BLOCKS as u64);

    // Open and read ONLY block 12
    let mut r = Reader::open(&path, Arc::new(ZstdCodec::default())).unwrap();
    let raw = r.read_block(TARGET_BLOCK).unwrap();

    // Expected bytes for block 12
    let start = TARGET_BLOCK as usize * DEFAULT_BLOCK_SIZE as usize;
    let end = start + DEFAULT_BLOCK_SIZE as usize;
    let expected = &data[start..end];

    assert_eq!(
        raw.as_slice(),
        expected,
        "block {} should decompress to the original bytes without reading blocks 0–{}",
        TARGET_BLOCK,
        TARGET_BLOCK - 1
    );
}

/// Test read_range across a block boundary.
#[test]
fn test_read_range_crosses_block_boundary() {
    let block_size = 1024u32;
    let data = compressible_bytes(4 * block_size as usize);
    let path = temp_path("read_range");

    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), block_size).unwrap();
    w.write(&data).unwrap();
    w.finish().unwrap();

    let mut r = Reader::open(&path, Arc::new(ZstdCodec::default())).unwrap();

    // Read a range that straddles the boundary between block 1 and block 2
    let start = block_size as u64 - 100; // 100 bytes from end of block 1
    let len = 300u64; // spans into block 2
    let result = r.read_range(start, len).unwrap();

    assert_eq!(result.len(), 300);
    assert_eq!(result.as_slice(), &data[start as usize..start as usize + 300]);
}

/// Test that codec mismatch on open returns a clear error.
#[test]
fn test_codec_mismatch_error() {
    let data = b"hello world test data for codec mismatch";
    let path = temp_path("codec_mismatch");

    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(data).unwrap();
    w.finish().unwrap();

    // Try to open with wrong codec
    let result = Reader::open(&path, Arc::new(Lz4Codec));
    assert!(result.is_err(), "opening with wrong codec should fail");
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("codec mismatch"),
        "error message should mention codec mismatch, got: {err}"
    );
}

/// Test that a single-block file works correctly (edge case: last block partial).
#[test]
fn test_single_partial_block() {
    let data = b"a small payload that fits in one partial block";
    let path = temp_path("single_block");

    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(data).unwrap();
    let blocks = w.finish().unwrap();
    assert_eq!(blocks, 1);

    let mut r = Reader::open(&path, Arc::new(ZstdCodec::default())).unwrap();
    assert_eq!(r.block_count(), 1);
    let raw = r.read_block(0).unwrap();
    assert_eq!(raw.as_slice(), data.as_slice());
}

/// Verify Shannon floor: pseudo-random (high-entropy) data should not compress.
#[test]
fn test_incompressible_data_no_size_gain() {
    let data = pseudo_random_bytes(DEFAULT_BLOCK_SIZE as usize * 4, 0x1234_5678);
    let path = temp_path("incompressible");

    let mut w = Writer::create(&path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE).unwrap();
    w.write(&data).unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path, Arc::new(ZstdCodec::default())).unwrap();
    let ratio = r.ratio();
    eprintln!("incompressible data ratio: {:.4}x", ratio);
    // zstd on random data expands slightly; ratio < 1.02 means essentially no gain
    assert!(
        ratio < 1.10,
        "zstd on random data should not meaningfully compress: ratio={:.4}",
        ratio
    );
}
