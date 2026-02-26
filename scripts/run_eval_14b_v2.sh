#!/bin/bash
# Run 14B eval with fresh output dir to avoid file collision
set -e
mkdir -p /Users/wscholl/poc/eval_output_14b
python3 /Users/wscholl/poc/run_eval.py \
  --model-dir /Users/wscholl/models/Qwen2.5-14B-Instruct-bf16 \
  --compressed-dir /Users/wscholl/models/Qwen2.5-14B-Instruct-bf16-compressed \
  --tasks winogrande,piqa,arc_easy,hellaswag \
  --limit 200 \
  --skip-reference \
  --model-name Qwen2.5-14B-squish4bit \
  --output-dir /Users/wscholl/poc/eval_output_14b \
  > /tmp/eval_14b_run2.log 2>&1
echo "EXIT:$?" >> /tmp/eval_14b_run2.log
