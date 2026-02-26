#!/usr/bin/env python3
"""
scripts/launch_eval.py

Launch a Squish eval run in a detached background process.
Avoids shell quoting issues while keeping full flexibility.

Usage:
    # ARC + HellaSwag, 200 samples, skip reference  (quick 20-30 min)
    python3 scripts/launch_eval.py

    # Full eval, all tasks, no limit
    python3 scripts/launch_eval.py --no-limit --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa

    # Specific model
    python3 scripts/launch_eval.py \\
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --compressed-dir ~/models/squish_7b

    # With speculative decoding
    python3 scripts/launch_eval.py \\
        --draft-model ~/models/Qwen2.5-0.5B-Instruct-bf16 \\
        --draft-compressed ~/models/squish_0.5b
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Launch Squish eval in background")
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16"))
    ap.add_argument("--compressed-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-7B-Instruct-bf16-compressed"))
    ap.add_argument("--tasks",    default="arc_easy,hellaswag")
    ap.add_argument("--limit",    type=int, default=200)
    ap.add_argument("--no-limit", action="store_true",
                    help="Remove --limit (full eval — hours)")
    ap.add_argument("--skip-reference", action="store_true", default=True)
    ap.add_argument("--batch-size",     type=int, default=4,
                    help="Eval batch size (default 4 — uses padded batch inference)")
    ap.add_argument("--output-dir",     default="")
    ap.add_argument("--log",            default="/tmp/squish_eval.log")
    ap.add_argument("--foreground",     action="store_true",
                    help="Run in foreground (not detached)")
    args = ap.parse_args()

    cmd = [
        sys.executable, str(ROOT / "evals" / "run_eval.py"),
        "--model-dir",       args.model_dir,
        "--compressed-dir",  args.compressed_dir,
        "--tasks",           args.tasks,
        "--batch-size",      str(args.batch_size),
    ]
    if args.no_limit:
        cmd.append("--no-limit")
    elif args.limit:
        cmd += ["--limit", str(args.limit)]

    if args.skip_reference:
        cmd.append("--skip-reference")

    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]

    print("Command:", " ".join(cmd))
    print(f"Log:    {args.log}")
    print()

    if args.foreground:
        os.execv(cmd[0], cmd)   # replace this process — clean exit on completion
        return

    log_fh = open(args.log, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()

    time.sleep(0.5)
    if proc.poll() is not None:
        print(f"ERROR: process died immediately (exit={proc.returncode})")
        sys.exit(1)

    print(f"PID={proc.pid}  log={args.log}")
    print()
    print("Monitor with:")
    print(f"  tail -f {args.log}")
    print()
    print("Check progress:")
    print(f"  ps aux | grep run_eval | grep -v grep")


if __name__ == "__main__":
    main()
