use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, Subcommand};

use ancf_codecs::{codec_by_id, Lz4Codec, PassThroughCodec, ZstdCodec};
use ancf_core::format::DEFAULT_BLOCK_SIZE;
use ancf_core::{Codec, Reader, Writer};

// ── CLI definition ─────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "ancf",
    about = "Access-Native Compression Format — compress, inspect, and randomly access ANCF1 files",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a file into ANCF1 format
    Compress {
        /// Source file to compress ("-" reads stdin)
        input: PathBuf,
        /// Destination ANCF1 file
        output: PathBuf,
        /// Codec to use: passthrough | zstd | lz4
        #[arg(short, long, default_value = "zstd")]
        codec: String,
        /// Zstd compression level (1–22, only used with --codec zstd)
        #[arg(long, default_value_t = 3)]
        zstd_level: i32,
        /// Raw bytes per block (default: 65536 = 64 KB)
        #[arg(short, long, default_value_t = DEFAULT_BLOCK_SIZE)]
        block_size: u32,
    },
    /// Fully decompress an ANCF1 file back to raw bytes
    Decompress {
        /// Source ANCF1 file
        input: PathBuf,
        /// Destination file ("-" writes to stdout)
        output: PathBuf,
    },
    /// Print header metadata and block index statistics
    Inspect {
        /// ANCF1 file to inspect
        file: PathBuf,
        /// Print per-block details
        #[arg(long)]
        blocks: bool,
    },
    /// Decompress a single block by index
    ///
    /// This is the core POC demonstration: only the requested block is
    /// read from disk — no other blocks are touched.
    ReadBlock {
        /// ANCF1 file
        file: PathBuf,
        /// Zero-based block index to read
        #[arg(short, long)]
        index: u64,
        /// Write raw bytes to a file instead of printing a hex dump
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Benchmark random-access reads across N randomly chosen blocks
    Bench {
        /// ANCF1 file
        file: PathBuf,
        /// Number of random blocks to read
        #[arg(short, long, default_value_t = 1000)]
        count: u64,
        /// Fixed random seed for reproducibility
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn codec_from_name(name: &str, zstd_level: i32) -> anyhow::Result<Box<dyn Codec>> {
    match name {
        "passthrough" | "pass" | "none" => Ok(Box::new(PassThroughCodec)),
        "zstd" | "z" => Ok(Box::new(ZstdCodec::new(zstd_level))),
        "lz4" | "l" => Ok(Box::new(Lz4Codec)),
        other => anyhow::bail!(
            "unknown codec '{}'. Valid options: passthrough, zstd, lz4",
            other
        ),
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} B", n)
    } else {
        format!("{:.2} {}", v, UNITS[unit])
    }
}

// ── Subcommand implementations ─────────────────────────────────────────────

fn run_compress(
    input: PathBuf,
    output: PathBuf,
    codec_name: &str,
    zstd_level: i32,
    block_size: u32,
) -> anyhow::Result<()> {
    let codec = codec_from_name(codec_name, zstd_level)?;
    let codec_display = codec.name().to_string();

    let mut writer = Writer::create(&output, codec, block_size)
        .with_context(|| format!("creating output file {:?}", output))?;

    let bytes_read: u64;
    let t0 = Instant::now();

    if input.to_str() == Some("-") {
        let stdin = io::stdin();
        let mut src = stdin.lock();
        let mut buf = vec![0u8; block_size as usize];
        let mut total = 0u64;
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            writer.write(&buf[..n])?;
            total += n as u64;
        }
        bytes_read = total;
    } else {
        let file = File::open(&input)
            .with_context(|| format!("opening input file {:?}", input))?;
        let meta = file.metadata()?;
        bytes_read = meta.len();
        let mut src = BufReader::new(file);
        let mut buf = vec![0u8; block_size as usize];
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            writer.write(&buf[..n])?;
        }
    }

    let block_count = writer.finish()?;
    let elapsed = t0.elapsed();

    // Re-open to get compressed size
    let out_meta = std::fs::metadata(&output)?;
    let compressed_size = out_meta.len();
    let ratio = bytes_read as f64 / compressed_size as f64;

    eprintln!(
        "  codec       : {}",
        codec_display
    );
    eprintln!("  block size  : {}", human_bytes(block_size as u64));
    eprintln!("  blocks      : {}", block_count);
    eprintln!("  raw size    : {}", human_bytes(bytes_read));
    eprintln!("  compressed  : {}", human_bytes(compressed_size));
    eprintln!("  ratio       : {:.2}x", ratio);
    eprintln!(
        "  throughput  : {}/s",
        human_bytes((bytes_read as f64 / elapsed.as_secs_f64()) as u64)
    );
    eprintln!("  elapsed     : {:.3}s", elapsed.as_secs_f64());
    Ok(())
}

fn run_decompress(input: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    // Read just the header codec_id first to pick the right codec
    let codec_id = {
        use std::io::Read;
        use ancf_core::format::{Ancf1Header, HEADER_SIZE};
        let mut f = File::open(&input)?;
        let mut buf = [0u8; HEADER_SIZE as usize];
        f.read_exact(&mut buf)?;
        Ancf1Header::from_bytes(&buf)?.codec_id
    };

    let codec = codec_by_id(codec_id)?;
    let mut reader = Reader::open(&input, codec)?;

    let is_stdout = output.to_str() == Some("-");
    let mut dst: Box<dyn Write> = if is_stdout {
        Box::new(io::stdout())
    } else {
        Box::new(
            File::create(&output).with_context(|| format!("creating output file {:?}", output))?,
        )
    };

    let t0 = Instant::now();
    let block_count = reader.block_count();
    let mut total_raw = 0u64;

    for idx in 0..block_count {
        let block = reader.read_block(idx)?;
        total_raw += block.len() as u64;
        dst.write_all(&block)?;
    }

    let elapsed = t0.elapsed();
    eprintln!("  blocks      : {}", block_count);
    eprintln!("  raw size    : {}", human_bytes(total_raw));
    eprintln!(
        "  throughput  : {}/s",
        human_bytes((total_raw as f64 / elapsed.as_secs_f64()) as u64)
    );
    eprintln!("  elapsed     : {:.3}s", elapsed.as_secs_f64());
    Ok(())
}

fn run_inspect(file: PathBuf, show_blocks: bool) -> anyhow::Result<()> {
    // Read header to get codec_id
    let codec_id = {
        use std::io::Read;
        use ancf_core::format::{Ancf1Header, HEADER_SIZE};
        let mut f = File::open(&file)?;
        let mut buf = [0u8; HEADER_SIZE as usize];
        f.read_exact(&mut buf)?;
        Ancf1Header::from_bytes(&buf)?.codec_id
    };
    let codec = codec_by_id(codec_id)?;
    let reader = Reader::open(&file, codec.clone())?;

    let file_meta = std::fs::metadata(&file)?;
    let file_size = file_meta.len();

    println!("=== ANCF1 File: {:?} ===", file);
    println!();
    println!("  format version : {}", reader.header.version);
    println!("  codec          : {} (id={})", codec.name(), reader.header.codec_id);
    println!("  block size     : {}", human_bytes(reader.header.block_size as u64));
    println!("  block count    : {}", reader.block_count());
    println!("  raw size       : {}", human_bytes(reader.raw_size()));
    println!("  compressed     : {}", human_bytes(reader.compressed_size()));
    println!("  file on disk   : {}", human_bytes(file_size));
    println!("  ratio          : {:.2}x", reader.ratio());
    println!("  flags          : 0x{:016x}", reader.header.flags);

    if show_blocks {
        println!();
        println!(
            "  {:>8}  {:>14}  {:>12}  {:>12}  {:>16}",
            "block", "file offset", "compressed", "raw", "checksum"
        );
        println!("  {}", "-".repeat(66));
        for (i, e) in reader.entries().iter().enumerate() {
            println!(
                "  {:>8}  {:>14}  {:>12}  {:>12}  {:016x}",
                i,
                e.offset,
                human_bytes(e.compressed_len as u64),
                human_bytes(e.raw_len as u64),
                e.checksum
            );
        }
    }

    Ok(())
}

fn run_read_block(file: PathBuf, index: u64, output: Option<PathBuf>) -> anyhow::Result<()> {
    let codec_id = {
        use std::io::Read;
        use ancf_core::format::{Ancf1Header, HEADER_SIZE};
        let mut f = File::open(&file)?;
        let mut buf = [0u8; HEADER_SIZE as usize];
        f.read_exact(&mut buf)?;
        Ancf1Header::from_bytes(&buf)?.codec_id
    };
    let codec = codec_by_id(codec_id)?;
    let mut reader = Reader::open(&file, codec)?;

    eprintln!(
        "seeking to block {} (offset {} bytes from file start)...",
        index,
        reader.entries()[index as usize].offset
    );

    let t0 = Instant::now();
    let raw = reader.read_block(index)?;
    let elapsed = t0.elapsed();

    eprintln!(
        "  decoded {} in {:.3}ms",
        human_bytes(raw.len() as u64),
        elapsed.as_secs_f64() * 1000.0
    );

    match output {
        Some(path) => {
            std::fs::write(&path, &raw)?;
            eprintln!("  written to {:?}", path);
        }
        None => {
            // Print a hex dump of the first 256 bytes
            let preview = &raw[..raw.len().min(256)];
            println!("--- block {} ({} bytes, first {} shown) ---", index, raw.len(), preview.len());
            for (i, chunk) in preview.chunks(16).enumerate() {
                print!("  {:04x}  ", i * 16);
                for b in chunk {
                    print!("{:02x} ", b);
                }
                // padding
                for _ in chunk.len()..16 {
                    print!("   ");
                }
                print!("  |");
                for b in chunk {
                    if b.is_ascii_graphic() || *b == b' ' {
                        print!("{}", *b as char);
                    } else {
                        print!(".");
                    }
                }
                println!("|");
            }
            if raw.len() > 256 {
                println!("  ... ({} bytes remaining not shown)", raw.len() - 256);
            }
        }
    }

    Ok(())
}

fn run_bench(file: PathBuf, count: u64, seed: u64) -> anyhow::Result<()> {
    let codec_id = {
        use std::io::Read;
        use ancf_core::format::{Ancf1Header, HEADER_SIZE};
        let mut f = File::open(&file)?;
        let mut buf = [0u8; HEADER_SIZE as usize];
        f.read_exact(&mut buf)?;
        Ancf1Header::from_bytes(&buf)?.codec_id
    };
    let codec = codec_by_id(codec_id)?;
    let mut reader = Reader::open(&file, codec)?;
    let block_count = reader.block_count();

    if block_count == 0 {
        anyhow::bail!("file has no blocks");
    }

    // Simple LCG for reproducible random block indices (no external dep)
    let indices: Vec<u64> = {
        let mut rng = seed;
        (0..count)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 33) % block_count
            })
            .collect()
    };

    eprintln!(
        "benchmarking {} random block reads across {} blocks...",
        count, block_count
    );

    let t0 = Instant::now();
    let mut total_raw = 0u64;
    let mut latencies_us: Vec<u64> = Vec::with_capacity(count as usize);

    for &idx in &indices {
        let t = Instant::now();
        let block = reader.read_block(idx)?;
        latencies_us.push(t.elapsed().as_micros() as u64);
        total_raw += block.len() as u64;
    }

    let elapsed = t0.elapsed();
    latencies_us.sort_unstable();

    let p50 = latencies_us[latencies_us.len() / 2];
    let p95 = latencies_us[(latencies_us.len() as f64 * 0.95) as usize];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];
    let min = latencies_us[0];
    let max = *latencies_us.last().unwrap();

    println!();
    println!("=== Random Block Access Benchmark ===");
    println!("  blocks read : {}", count);
    println!("  total raw   : {}", human_bytes(total_raw));
    println!("  elapsed     : {:.3}s", elapsed.as_secs_f64());
    println!(
        "  throughput  : {}/s",
        human_bytes((total_raw as f64 / elapsed.as_secs_f64()) as u64)
    );
    println!("  latency:");
    println!("    min  : {} µs", min);
    println!("    p50  : {} µs", p50);
    println!("    p95  : {} µs", p95);
    println!("    p99  : {} µs", p99);
    println!("    max  : {} µs", max);

    Ok(())
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compress {
            input,
            output,
            codec,
            zstd_level,
            block_size,
        } => run_compress(input, output, &codec, zstd_level, block_size),
        Commands::Decompress { input, output } => run_decompress(input, output),
        Commands::Inspect { file, blocks } => run_inspect(file, blocks),
        Commands::ReadBlock {
            file,
            index,
            output,
        } => run_read_block(file, index, output),
        Commands::Bench { file, count, seed } => run_bench(file, count, seed),
    }
}
