#!/bin/bash
cd /Users/wscholl/poc
python3 run_eval.py \
  --model-dir /Users/wscholl/models/Qwen2.5-14B-Instruct-bf16 \
  --compressed-dir /Users/wscholl/models/Qwen2.5-14B-Instruct-bf16-compressed \
  --tasks winogrande,piqa,arc_easy,hellaswag \
  --limit 200 \
  --skip-reference \
  --model-name Qwen2.5-14B-squish4bit \
  > /tmp/eval_14b.log 2>&1
echo "EXIT:$?" >> /tmp/eval_14b.log
