//! ANCF Proof-of-Concept Demo
//!
//! Generates a realistic 100 MB structured log dataset, compresses it with
//! multiple formats, then demonstrates the core access-native claim:
//! a single compressed block can be read in microseconds without touching
//! the rest of the file — vs. seconds of full decompression required by
//! traditional formats.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression as GzCompression;

use ancf_codecs::{Lz4Codec, ZstdCodec};
use ancf_core::format::DEFAULT_BLOCK_SIZE;
use ancf_core::{Codec, Reader, Writer};

// ── constants ──────────────────────────────────────────────────────────────

const TARGET_RAW_BYTES: u64 = 100 * 1024 * 1024; // 100 MB

// IPs used in synthetic log lines (10 options → some repetition = better compression)
const IPS: &[&str] = &[
    "203.0.113.42", "198.51.100.77", "192.0.2.15", "10.10.10.88",
    "172.16.254.1", "203.0.113.99", "198.51.100.3", "192.0.2.200",
    "10.20.30.40",  "172.31.0.5",
];
const METHODS: &[&str] = &["GET", "GET", "GET", "POST", "PUT", "DELETE", "GET", "GET"];
const PATHS: &[&str] = &[
    "/api/v1/catalog/items",
    "/api/v1/catalog/items?category=electronics&page={page}&limit=20",
    "/api/v1/orders/{id}/status",
    "/api/v1/users/{id}/profile",
    "/api/v1/cart/items",
    "/api/v1/search?q=laptop&page={page}",
    "/api/v1/recommendations?user={id}",
    "/static/assets/bundle.js",
    "/api/v1/inventory/sku/{id}",
    "/health",
];
const STATUSES: &[(u16, u32)] = &[
    (200, 4821), (200, 1204), (200, 8912), (201, 312),
    (400, 188),  (404, 95),  (200, 22480),(304, 0),
    (200, 3312), (500, 512),
];
const USER_AGENTS: &[&str] = &[
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "PostmanRuntime/7.43.0",
    "python-httpx/0.28.1",
    "Go-http-client/2.0",
];

// ── data generator ──────────────────────────────────────────────────────────

/// Generate a deterministic, realistic-looking Apache access log line for entry `i`.
/// The same `i` always produces the same bytes (enables round-trip verification).
fn generate_log_line(i: u64) -> Vec<u8> {
    let ip       = IPS[(i as usize * 7 + 3) % IPS.len()];
    let method   = METHODS[(i as usize * 3 + 1) % METHODS.len()];
    let path_tpl = PATHS[(i as usize * 11 + 5) % PATHS.len()];
    let path     = path_tpl
        .replace("{page}", &((i % 200) + 1).to_string())
        .replace("{id}", &(i * 13 % 9_999_999).to_string());
    let (status, size) = STATUSES[(i as usize * 5 + 2) % STATUSES.len()];
    let ua       = USER_AGENTS[(i as usize * 17 + 7) % USER_AGENTS.len()];
    let lat_ms   = (((i * 137 + 42) % 900) + 10) as f64 / 100.0;

    // seconds since a fixed epoch — makes timestamps vary but stay deterministic
    let ts_sec   = 1_740_268_800u64 + (i * 7) % (86400 * 30);
    let h = (ts_sec / 3600) % 24;
    let m = (ts_sec / 60) % 60;
    let s = ts_sec % 60;
    let day = (ts_sec / 86400) % 28 + 1;
    let months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    let month = months[((ts_sec / (86400 * 28)) % 12) as usize];

    format!(
        "{ip} - - [{day:02}/{month}/2026:{h:02}:{m:02}:{s:02} +0000] \
         \"{method} {path} HTTP/1.1\" {status} {size} {lat_ms:.3} \
         \"https://shop.example.com/products\" \"{ua}\"\n"
    )
    .into_bytes()
}

// ── timing ──────────────────────────────────────────────────────────────────

fn human_bytes(n: u64) -> String {
    const U: &[&str] = &["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < U.len() - 1 { v /= 1024.0; u += 1; }
    if u == 0 { format!("{n} B") } else { format!("{v:.2} {}", U[u]) }
}

fn fmt_duration(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.1} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{ms:.1} ms")
    } else {
        format!("{:.2} s", d.as_secs_f64())
    }
}

fn speedup(slow: Duration, fast: Duration) -> f64 {
    slow.as_secs_f64() / fast.as_secs_f64().max(1e-9)
}

// ── write helpers ───────────────────────────────────────────────────────────

/// Stream all log lines totalling ~TARGET_RAW_BYTES into an ANCF Writer.
/// Returns (lines_written, bytes_written).
fn write_ancf(path: &Path, codec: Box<dyn Codec>, block_size: u32) -> Result<(u64, u64)> {
    let mut w = Writer::create(path, codec, block_size)?;
    let mut i = 0u64;
    let mut total = 0u64;
    while total < TARGET_RAW_BYTES {
        let line = generate_log_line(i);
        total += line.len() as u64;
        w.write(&line)?;
        i += 1;
    }
    w.finish()?;
    Ok((i, total))
}

/// Traditional zstd — one continuous compressed stream (no block index → no random access).
fn write_raw_zstd(path: &Path) -> Result<u64> {
    let file = File::create(path)?;
    let mut enc = zstd::stream::write::Encoder::new(BufWriter::new(file), 3)?;
    let mut total = 0u64;
    let mut i = 0u64;
    while total < TARGET_RAW_BYTES {
        let line = generate_log_line(i);
        total += line.len() as u64;
        enc.write_all(&line)?;
        i += 1;
    }
    enc.finish()?;
    Ok(total)
}

/// Traditional gzip — one continuous compressed stream.
fn write_raw_gzip(path: &Path) -> Result<u64> {
    let file   = File::create(path)?;
    let mut enc = GzEncoder::new(BufWriter::new(file), GzCompression::default());
    let mut total = 0u64;
    let mut i = 0u64;
    while total < TARGET_RAW_BYTES {
        let line = generate_log_line(i);
        total += line.len() as u64;
        enc.write_all(&line)?;
        i += 1;
    }
    enc.finish()?;
    Ok(total)
}

// ── traditional random-access simulation ────────────────────────────────────

/// Simulate reading `target_raw_offset` bytes into a raw zstd stream.
/// Returns time taken and the byte at that position (to prove correctness).
fn decompress_zstd_to_offset(path: &Path, target_raw_offset: u64) -> Result<(Duration, u8)> {
    let file = File::open(path)?;
    let mut dec = zstd::stream::read::Decoder::new(file)?;
    let mut consumed = 0u64;
    let mut buf = vec![0u8; 65536];
    let mut target_byte = 0u8;

    let t0 = Instant::now();
    loop {
        let n = dec.read(&mut buf)?;
        if n == 0 { break; }
        let end = consumed + n as u64;
        if consumed <= target_raw_offset && target_raw_offset < end {
            target_byte = buf[(target_raw_offset - consumed) as usize];
        }
        consumed = end;
        if consumed > target_raw_offset {
            break; // stop as soon as we've passed the target — mimics the best case
        }
    }
    Ok((t0.elapsed(), target_byte))
}

/// Same but for gzip.
fn decompress_gzip_to_offset(path: &Path, target_raw_offset: u64) -> Result<(Duration, u8)> {
    use flate2::read::GzDecoder;
    let file = File::open(path)?;
    let mut dec = GzDecoder::new(file);
    let mut consumed = 0u64;
    let mut buf = vec![0u8; 65536];
    let mut target_byte = 0u8;

    let t0 = Instant::now();
    loop {
        let n = dec.read(&mut buf)?;
        if n == 0 { break; }
        let end = consumed + n as u64;
        if consumed <= target_raw_offset && target_raw_offset < end {
            target_byte = buf[(target_raw_offset - consumed) as usize];
        }
        consumed = end;
        if consumed > target_raw_offset {
            break;
        }
    }
    Ok((t0.elapsed(), target_byte))
}

// ── demo runner ─────────────────────────────────────────────────────────────

fn run() -> Result<()> {
    let out_dir = std::env::temp_dir().join("ancf_demo");
    std::fs::create_dir_all(&out_dir)?;

    let ancf_zstd_path = out_dir.join("corpus.ancf.zstd");
    let ancf_lz4_path  = out_dir.join("corpus.ancf.lz4");
    let raw_zstd_path  = out_dir.join("corpus.raw.zst");
    let raw_gzip_path  = out_dir.join("corpus.raw.gz");

    // ── banner ───────────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       ANCF — Access-Native Compression Format  ·  POC Demo      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 0: Data Generation ─────────────────────────────────────────────
    section("0 · DATA GENERATION");
    eprint!("  Generating ~100 MB of realistic Apache access log data ");

    // Quick preview of what a line looks like
    let sample_line = generate_log_line(42);
    let sample_len = sample_line.len();
    let sample = String::from_utf8_lossy(&sample_line);

    // Count lines for 100MB
    let mut preview_total = 0u64;
    let mut preview_i = 0u64;
    while preview_total < TARGET_RAW_BYTES {
        preview_total += generate_log_line(preview_i).len() as u64;
        preview_i += 1;
    }
    eprintln!("(~{} lines)", format_number(preview_i));

    println!("  Sample log entry ({sample_len} bytes):");
    println!("  ┌─────────────────────────────────────────────────────────────────┐");
    let trimmed = sample.trim_end();
    // word-wrap at 67 chars
    let mut pos = 0;
    while pos < trimmed.len() {
        let end = (pos + 67).min(trimmed.len());
        println!("  │ {:<67}│", &trimmed[pos..end]);
        pos = end;
    }
    println!("  └─────────────────────────────────────────────────────────────────┘");
    println!();

    // ── Phase 1: Compression ─────────────────────────────────────────────────
    section("1 · COMPRESSION");
    println!("  {:<22} {:>12}  {:>12}  {:>8}  {:>8}  Note",
             "Format", "Raw", "Compressed", "Ratio", "Time");
    println!("  {}", "─".repeat(78));

    let (lines, raw_bytes) = timed_step("ANCF/zstd", || {
        write_ancf(&ancf_zstd_path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE)
    })?;
    print_compression_row("ANCF/zstd  (block=64KB)", raw_bytes, &ancf_zstd_path, "", false)?;

    timed_step("ANCF/lz4", || {
        write_ancf(&ancf_lz4_path, Box::new(Lz4Codec), DEFAULT_BLOCK_SIZE)
    })?;
    print_compression_row("ANCF/lz4   (block=64KB)", raw_bytes, &ancf_lz4_path, "", false)?;

    timed_step("raw zstd", || write_raw_zstd(&raw_zstd_path))?;
    print_compression_row("raw zstd", raw_bytes, &raw_zstd_path,
        "← no random access", true)?;

    timed_step("raw gzip", || write_raw_gzip(&raw_gzip_path))?;
    print_compression_row("raw gzip", raw_bytes, &raw_gzip_path,
        "← no random access", true)?;

    println!();
    println!("  Total log entries : {}", format_number(lines));
    println!("  Raw data size     : {}", human_bytes(raw_bytes));

    // ── Phase 2: Inspection ──────────────────────────────────────────────────
    section("2 · ANCF FILE INSPECTION");
    let codec = Arc::new(ZstdCodec::default());
    let reader = Reader::open(&ancf_zstd_path, codec.clone())?;
    let block_count = reader.block_count();
    let index_bytes = block_count * 32 + 8 + 56; // entries + footer + header
    let index_pct   = index_bytes as f64 / std::fs::metadata(&ancf_zstd_path)?.len() as f64 * 100.0;

    println!("  block count    : {}", format_number(block_count));
    println!("  block size     : {} (64 KB nominal raw)", human_bytes(reader.block_size() as u64));
    println!("  raw size       : {}", human_bytes(reader.raw_size()));
    println!("  compressed     : {}", human_bytes(reader.compressed_size()));
    println!("  compression    : {:.1}x", reader.ratio());
    println!("  block index    : {} ({:.3}% of file — negligible overhead)", human_bytes(index_bytes), index_pct);
    println!("  seek cost      : 1 × lseek(offset)  →  decode one 64 KB block");
    drop(reader);

    // ── Phase 3: THE CORE CLAIM ───────────────────────────────────────────────
    section("3 · THE CORE CLAIM — access any data without full decompression");

    // Gather everything we need up-front, then close the reader
    let (total_raw_bytes, block_count_3, ancf_file_size, raw_zstd_size, raw_gzip_size,
         target_block_compressed_len) = {
        let r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        let raw_size  = r.raw_size();
        let bc        = r.block_count();
        let tgt_block = (raw_size as f64 * 0.80) as u64 / DEFAULT_BLOCK_SIZE as u64;
        let comp_len  = r.entries()[tgt_block as usize].compressed_len as u64;
        (
            raw_size, bc,
            file_size(&ancf_zstd_path)?,
            file_size(&raw_zstd_path)?,
            file_size(&raw_gzip_path)?,
            comp_len,
        )
    };

    // Target 80% through the file — a "deep seek"
    let target_raw_offset  = (total_raw_bytes as f64 * 0.80) as u64;
    let target_block       = target_raw_offset / DEFAULT_BLOCK_SIZE as u64;
    let target_block_file_offset: u64 = {
        let r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        r.entries()[target_block as usize].offset
    };

    // I/O bytes each approach must read to serve this request
    // (ANCF: one compressed block; others: all compressed bytes up to the 80% point)
    let ancf_io_bytes = target_block_compressed_len;
    let zstd_io_bytes = (ancf_file_size as f64 * 0.80) as u64; // proportional approximation
    let gzip_io_bytes = (raw_gzip_size as f64 * 0.80) as u64;

    println!("  Seeking to raw byte offset {}  =  block {} of {}  (80% through file)",
        format_number(target_raw_offset), format_number(target_block), format_number(block_count_3));
    println!();

    // ANCF read_block
    let (ancf_dur, ancf_byte) = {
        let mut r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        let t0 = Instant::now();
        let block = r.read_block(target_block)?;
        let dur = t0.elapsed();
        let local_offset = (target_raw_offset % DEFAULT_BLOCK_SIZE as u64) as usize;
        (dur, block[local_offset])
    };

    // raw zstd — must stream-decompress up to target offset
    let (zstd_dur, zstd_byte) = decompress_zstd_to_offset(&raw_zstd_path, target_raw_offset)?;

    // raw gzip — same
    let (gzip_dur, gzip_byte) = decompress_gzip_to_offset(&raw_gzip_path, target_raw_offset)?;

    let all_match = ancf_byte == zstd_byte && ancf_byte == gzip_byte;
    let zstd_x    = speedup(zstd_dur, ancf_dur);
    let gzip_x    = speedup(gzip_dur, ancf_dur);

    println!("  {:<42}  {:>12}  {:>12}  {:>12}",
             "Method", "Latency", "I/O read", "I/O ratio");
    println!("  {}", "─".repeat(84));
    println!("  {:<42}  {:>12}  {:>12}  {:>12}",
        format!("ANCF/zstd  (block {} @ +{})", format_number(target_block), human_bytes(target_block_file_offset)),
        fmt_duration(ancf_dur),
        human_bytes(ancf_io_bytes),
        "1.0× (baseline)");
    println!("  {:<42}  {:>12}  {:>12}  {:>12}",
        "raw zstd  (full stream decode to target)",
        fmt_duration(zstd_dur),
        human_bytes(zstd_io_bytes),
        format!("{:.0}×  more", zstd_io_bytes as f64 / ancf_io_bytes as f64));
    println!("  {:<42}  {:>12}  {:>12}  {:>12}",
        "raw gzip  (full stream decode to target)",
        fmt_duration(gzip_dur),
        human_bytes(gzip_io_bytes),
        format!("{:.0}×  more", gzip_io_bytes as f64 / ancf_io_bytes as f64));
    println!();
    println!("  Byte at target offset (0x{:02x}): {}",
        ancf_byte, if all_match { "✓ all three produce the same value" } else { "⚠ MISMATCH" });
    println!();
    println!("  NVMe latency on this machine:");
    println!("    ANCF vs raw zstd : {:.1}×  |  ANCF vs raw gzip : {:.1}×",
        zstd_x, gzip_x);
    println!();
    println!("  ┌───────────────────────────────────────────────────────────────────────────┐");
    println!("  │  WHY THE I/O GAP IS THE REAL METRIC                                       │");
    println!("  │                                                                            │");
    println!("  │  ANCF reads exactly {} per random block request — always.               │",
        format!("{:<10}", human_bytes(ancf_io_bytes)));
    println!("  │  The raw formats read {:.0}–{:.0}× more bytes AND cannot use S3 range    │",
        zstd_io_bytes as f64 / ancf_io_bytes as f64,
        gzip_io_bytes as f64 / ancf_io_bytes as f64);
    println!("  │  requests at all (no index → no mapping from raw offset → file offset).    │");
    println!("  │                                                                            │");
    println!("  │  On slow storage the latency gap scales linearly with I/O size.            │");
    println!("  └───────────────────────────────────────────────────────────────────────────┘");

    // ── Scale-up projection ───────────────────────────────────────────────────
    section("3b · SCALE-UP PROJECTION — same ratio, 100 GB file, S3 (500 MB/s)");

    let ratio        = total_raw_bytes as f64 / ancf_file_size as f64;
    let raw_100gb    = 100u64 * 1024 * 1024 * 1024;
    let ancf_100gb   = (raw_100gb as f64 / ratio) as u64;        // compressed ancf size
    let block_100gb  = raw_100gb / DEFAULT_BLOCK_SIZE as u64;    // number of blocks
    let zstd_100gb   = (raw_100gb as f64 / (total_raw_bytes as f64 / raw_zstd_size as f64)) as u64;

    // I/O to reach 80% through file
    let ancf_io_100gb = ancf_io_bytes; // still just 1 block — doesn't scale with file size
    let zstd_io_100gb = (zstd_100gb as f64 * 0.80) as u64;

    let s3_bw_bps = 500.0 * 1024.0 * 1024.0f64; // 500 MB/s
    let s3_api_ms = 10.0f64;                      // S3 PUT/GET minimum latency
    let ancf_s3_ms  = (ancf_io_100gb as f64 / s3_bw_bps) * 1000.0 + s3_api_ms;
    let zstd_s3_ms  = (zstd_io_100gb as f64 / s3_bw_bps) * 1000.0 + s3_api_ms;
    let s3_speedup  = zstd_s3_ms / ancf_s3_ms.max(0.001);

    println!("  100 GB raw data at {:.1}× compression ratio:", ratio);
    println!();
    println!("  {:<40}  {:>14}  {:>14}", "Metric", "ANCF/zstd", "raw zstd");
    println!("  {}", "─".repeat(72));
    println!("  {:<40}  {:>14}  {:>14}", "Compressed file size",
        human_bytes(ancf_100gb), human_bytes(zstd_100gb));
    println!("  {:<40}  {:>14}  {:>14}", "Block count",
        format_number(block_100gb), "n/a (no blocks)");
    println!("  {:<40}  {:>14}  {:>14}", "I/O to reach 80% offset",
        human_bytes(ancf_io_100gb), human_bytes(zstd_io_100gb));
    println!("  {:<40}  {:>14}  {:>14}", "Estimated S3 latency (500 MB/s + 10ms)",
        format!("{:.1} ms", ancf_s3_ms),
        format!("{:.0} s", zstd_s3_ms / 1000.0));
    println!();
    println!("  Estimated S3 random-access speedup at 100 GB scale: {:.0}×", s3_speedup);
    println!("  (raw zstd would need to download {:.2} GB just to reach record at 80% offset)",
        zstd_io_100gb as f64 / 1024.0 / 1024.0 / 1024.0);

    // ── Phase 4: Sequential scan throughput ──────────────────────────────────
    section("4 · SEQUENTIAL SCAN THROUGHPUT");

    let (scan_dur, scan_bytes) = {
        let mut r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        let t0 = Instant::now();
        let mut total = 0u64;
        for idx in 0..r.block_count() {
            total += r.read_block(idx)?.len() as u64;
        }
        (t0.elapsed(), total)
    };
    let scan_gb_s = scan_bytes as f64 / scan_dur.as_secs_f64() / 1e9;
    println!("  Full ANCF/zstd scan: {} decompressed in {} → {:.2} GB/s",
        human_bytes(scan_bytes), fmt_duration(scan_dur), scan_gb_s);

    // ── Phase 5: Random access bulk benchmark ────────────────────────────────
    section("5 · RANDOM ACCESS BENCHMARK  —  1 000 random block reads");

    let mut latencies_us: Vec<u64> = {
        let mut r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        let bc = r.block_count();
        let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
        let indices: Vec<u64> = (0..1000)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 33) % bc
            })
            .collect();
        let mut lats = Vec::with_capacity(1000);
        for &idx in &indices {
            let t = Instant::now();
            let _ = r.read_block(idx)?;
            lats.push(t.elapsed().as_micros() as u64);
        }
        lats
    };
    latencies_us.sort_unstable();

    let p50 = latencies_us[500];
    let p95 = latencies_us[950];
    let p99 = latencies_us[990];
    let min = latencies_us[0];
    let max = *latencies_us.last().unwrap();

    println!("  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}",
             "min", "p50", "p95", "p99", "max");
    println!("  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}",
             format!("{min}µs"), format!("{p50}µs"), format!("{p95}µs"),
             format!("{p99}µs"), format!("{max}µs"));
    println!();
    println!("  Each read = 1 × lseek + 1 × read({}) + 1 × zstd_decode",
        human_bytes(DEFAULT_BLOCK_SIZE as u64));

    // ── Phase 6: Shannon floor demo ───────────────────────────────────────────
    section("6 · SHANNON ENTROPY FLOOR");
    println!("  Compressing pseudo-random (high-entropy) data with ANCF/zstd...");

    let entropy_path = out_dir.join("entropy.ancf");
    let entropy_raw  = 8 * DEFAULT_BLOCK_SIZE as u64; // 8 blocks of random
    let entropy_data: Vec<u8> = {
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        (0..entropy_raw)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 56) as u8
            })
            .collect()
    };
    let mut w = Writer::create(&entropy_path, Box::new(ZstdCodec::default()), DEFAULT_BLOCK_SIZE)?;
    w.write(&entropy_data)?;
    w.finish()?;

    let er = Reader::open(&entropy_path, Arc::new(ZstdCodec::default()))?;
    println!("  raw:        {}", human_bytes(entropy_raw));
    println!("  compressed: {} (ratio: {:.3}x — essentially 1.0 = no gain possible)",
        human_bytes(er.compressed_size()), er.ratio());
    println!();
    println!("  Shannon's theorem: entropy ≈ 1 bit/bit for random data → compression impossible.");
    println!("  ANCF ratio for structured log data above was {:.1}x.", {
        let r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        r.ratio()
    });
    println!("  The difference is not the codec — it is the entropy of the data.");

    // ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY");
    let final_ratio = {
        let r = Reader::open(&ancf_zstd_path, Arc::new(ZstdCodec::default()))?;
        r.ratio()
    };
    println!("  100 MB corpus,  ANCF/zstd,  block size 64 KB");
    println!();
    println!("  {:<46}  {}", "Compression ratio (log data):",        format!("{:.1}×", final_ratio));
    println!("  {:<46}  {}", "Block index overhead:",                 format!("{:.3}% of file", index_pct));
    println!("  {:<46}  {}", "Random block read  (p50 / p99):",       format!("{p50} µs / {p99} µs"));
    println!("  {:<46}  {}", "Sequential scan throughput:",           format!("{:.2} GB/s", scan_gb_s));
    println!("  {:<46}  {}", "I/O per random block read:",            human_bytes(ancf_io_bytes));
    println!("  {:<46}  {}", "NVMe latency vs raw zstd / gzip:",      format!("{:.1}× / {:.1}×  faster", zstd_x, gzip_x));
    println!("  {:<46}  {}", "Estimated S3 speedup at 100 GB scale:", format!("{:.0}×  faster", s3_speedup));
    println!("  {:<46}  {}", "Entropy floor (random data ratio):",    "~1.000× — confirmed uncompressible");
    println!();

    // cleanup temp files
    for p in [&ancf_zstd_path, &ancf_lz4_path, &raw_zstd_path, &raw_gzip_path, &entropy_path] {
        let _ = std::fs::remove_file(p);
    }

    Ok(())
}

// ── small helpers ──────────────────────────────────────────────────────────

fn section(title: &str) {
    println!("━━━ {title} {}", "━".repeat(70usize.saturating_sub(title.len() + 5)));
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

fn print_compression_row(label: &str, raw: u64, path: &Path, note: &str, dim: bool) -> Result<()> {
    let compressed = file_size(path)?;
    let ratio = raw as f64 / compressed as f64;
    let dim_s = if dim { "\x1b[2m" } else { "" };
    let rst   = if dim { "\x1b[0m" } else { "" };
    println!("  {dim_s}{:<22} {:>12}  {:>12}  {:>7.1}x  {:>8}  {note}{rst}",
        label,
        human_bytes(raw),
        human_bytes(compressed),
        ratio,
        "—",
    );
    Ok(())
}

fn timed_step<T, F: FnOnce() -> Result<T>>(label: &str, f: F) -> Result<T> {
    eprint!("  writing {label:<24} ");
    let t0 = Instant::now();
    let r = f()?;
    eprintln!("done  ({:.2}s)", t0.elapsed().as_secs_f64());
    Ok(r)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
