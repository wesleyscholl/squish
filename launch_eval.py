#!/usr/bin/env python3
"""
launch_eval.py — Start a squish eval in a detached background process.

Usage:
    python3 launch_eval.py [--limit N] [--tasks TASKS] [--no-limit] [--model 7b|1.5b]

Defaults: arc_easy,hellaswag,winogrande,piqa  --limit 200  --skip-reference
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
MROOT  = Path("/Users/wscholl/models")   # machine-specific model storage

# Known model locations
MODEL_PRESETS = {
    "7b":  {
        "model_dir":      MROOT / "Qwen2.5-7B-Instruct-bf16",
        "compressed_dir": MROOT / "Qwen2.5-7B-Instruct-bf16-compressed",
        "name":           "Qwen2.5-7B-Instruct",
        "output_dir":     ROOT / "results" / "eval_output_7b",
    },
    "14b": {
        "model_dir":      MROOT / "Qwen2.5-14B-Instruct-bf16",
        "compressed_dir": MROOT / "Qwen2.5-14B-Instruct-bf16-compressed",
        "name":           "Qwen2.5-14B-Instruct",
        "output_dir":     ROOT / "results" / "eval_output_14b",
    },
    "1.5b": {
        "model_dir":      MROOT / "Qwen2.5-1.5B-Instruct-bf16",
        "compressed_dir": MROOT / "Qwen2.5-1.5B-Instruct-bf16-compressed",
        "name":           "Qwen2.5-1.5B-Instruct",
        "output_dir":     ROOT / "results" / "eval_output_1.5b",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="7b", choices=list(MODEL_PRESETS))
    ap.add_argument("--limit",      type=int, default=200)
    ap.add_argument("--no-limit",   action="store_true")
    ap.add_argument("--tasks",      default="arc_easy,hellaswag,winogrande,piqa")
    ap.add_argument("--skip-reference", action="store_true", default=True)
    ap.add_argument("--log",        default="/tmp/squish_eval.log")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]

    cmd = [
        sys.executable,
        str(ROOT / "evals" / "run_eval.py"),
        "--tasks",         args.tasks,
        "--model-name",    preset["name"],
        "--model-dir",     str(preset["model_dir"]),
        "--compressed-dir",str(preset["compressed_dir"]),
        "--output-dir",    str(preset["output_dir"]),
    ]
    if not args.no_limit:
        cmd += ["--limit", str(args.limit)]
    if args.skip_reference:
        cmd += ["--skip-reference"]

    log_path = args.log
    print(f"Launching eval — model={args.model}  limit={'none' if args.no_limit else args.limit}")
    print(f"Log: {log_path}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
            cwd=str(ROOT),
        )

    print(f"PID={proc.pid}")
    time.sleep(3)
    try:
        with open(log_path) as f:
            head = f.read(1000)
        if head:
            print("--- first log lines ---")
            print(head)
    except Exception:
        pass


if __name__ == "__main__":
    main()
