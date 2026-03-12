# Phase 3+4 Completion Guide — Squish v9.0.0

## Overview

Phase 3+4 requires hardware validation and community publication. This guide covers all remaining tasks.

---

## Phase 3: Hardware Validation (Requires M-series Mac)

### 3.1 End-to-End Performance Benchmark (bench_eoe.py)

**Goal:** Measure real TTFT (time to first token) and throughput on live Squish server

**Files:**
- Script: `dev/benchmarks/bench_eoe.py` (ready to run)
- Output: `results/eoe_${DATE}.json`
- Destination: README (cold load metrics) + paper.md Section 4.1

**Steps:**

1. **Start the Squish server:**
   ```bash
   # Terminal 1: Start server with qwen2.5:1.5b
   squish serve qwen2.5:1.5b --port 11435
   # Wait for: "Listening on http://localhost:11435"
   ```

2. **Run the benchmark:**
   ```bash
   # Terminal 2: Run 5 iterations (will take ~5 min)
   python3 dev/benchmarks/bench_eoe.py \
       --squish-port 11435 \
       --squish-model squish \
       --runs 5 \
       --max-tokens 256 \
       --output results/eoe_2026_03_12_m1_pro.json
   ```

3. **Expected output:**
   ```
   ─────────────────────────────────────────────────────────
     Squish  (squish)
   ─────────────────────────────────────────────────────────
     run 01                             234.5 tok/s  TTFT 47ms  256 toks
     run 02                             245.2 tok/s  TTFT 42ms  256 toks
     ...
   
     mean TTFT                      43 ms  ± 4 ms
     mean throughput               240.1 tok/s  ± 5.8
     best TTFT                      42 ms
     peak throughput               245.2 tok/s
   ```

4. **Optional: Compare vs Ollama**
   ```bash
   # Terminal 1 (alt): Start Ollama
   ollama serve
   
   # Terminal 2: Run comparison
   python3 dev/benchmarks/bench_eoe.py \
       --squish-port 11435 \
       --squish-model squish \
       --ollama-port 11434 \
       --ollama-model qwen2.5:1.5b \
       --runs 5 \
       --output results/eoe_squish_vs_ollama.json
   ```

5. **Update documentation:**
   - Extract `mean TTFT`, `mean throughput`, `peak throughput` from JSON
   - Update [README.md](../README.md) Section 2.1 (cold start metrics)
   - Update [docs/paper.md](../docs/paper.md) Section 4.1 "Inference Performance"
   - Commit: `git add results/eoe_*.json README.md docs/paper.md && git commit -m "benchmark: add real hardware TTFT/tok-s from bench_eoe.py"`

---

### 3.2 MMLU Evaluation (lm-evaluation-harness)

**Goal:** Validate accuracy on MMLU (14,042 questions) using INT8-quantized weights

**Files:**
- Script: `lm_eval` CLI (install: `pip install lm-eval`)
- Output: accuracy metrics per subject and overall
- Destination: [docs/RESULTS.md](../docs/RESULTS.md) + paper.md Section 4.2

**Steps:**

1. **Install lm-evaluation-harness:**
   ```bash
   pip install lm-eval
   ```

2. **Run MMLU evaluation:**
   ```bash
   lm_eval \
       --model squish \
       --model_args "base_url=http://localhost:11435,model_name=squish" \
       --tasks mmlu \
       --batch_size 1 \
       --limit 14042 \
       --output_path results/mmlu_squish_v9_int8.json
   ```

   (This may take 30–60 minutes on M-series hardware)

3. **Expected output format:**
   ```json
   {
     "results": {
       "mmlu": {
         "acc": 0.627,
         "acc_norm": 0.632
       },
       "mmlu_humanities": {"acc": 0.64},
       "mmlu_social_sciences": {"acc": 0.65},
       ...
     },
     "config": {...}
   }
   ```

4. **Update documentation:**
   - Extract accuracy and per-subject scores
   - Format as table in [docs/RESULTS.md](../docs/RESULTS.md)
   - Add row to paper.md Section 4.2 "Accuracy Validation"
   - Commit: `git add results/mmlu_*.json docs/RESULTS.md docs/paper.md && git commit -m "benchmark: add MMLU eval results (14042 questions, INT8)"`

---

## Phase 4: Community & Publication (Can Start Now)

### 4.1 Publish Pre-Squished Weights to Hugging Face

**Goal:** Publish ZIP with squish_weights.safetensors so users skip the one-time conversion

**Prerequisites:**
1. HuggingFace account (free, huggingface.co)
2. Write-access API token from https://huggingface.co/settings/tokens
3. Squish model already pre-compressed: `~/.cache/squish/Qwen2.5-1.5B-Instruct/squish_weights.safetensors` (from `squish pull`)

**Steps:**

1. **Create HF token:**
   - Visit https://huggingface.co/settings/tokens
   - Click "New token" → enter name "squish-publish" → select "Write" access → Create
   - Copy the token

2. **Set environment variable:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

3. **Run publish script (example for Qwen2.5-1.5B):**
   ```bash
   python3 dev/publish_hf.py \
       --model-dir ~/.cache/squish/Qwen2.5-1.5B-Instruct-Int8 \
       --repo squish-community/Qwen2.5-1.5B-Instruct-Int8 \
       --base-model Qwen/Qwen2.5-1.5B-Instruct
   ```

4. **Expected output:**
   ```
     Repository : squish-community/Qwen2.5-1.5B-Instruct-Int8
     Model dir  : ~/.cache/squish/Qwen2.5-1.5B-Instruct-Int8
     Files      : 4
     Total size : 1.2 GB
   
       squish_weights.safetensors                      1.1 GB
       tokenizer.json                                  2.3 MB
       ...
   
     Repo ready: https://huggingface.co/squish-community/Qwen2.5-1.5B-Instruct-Int8
     Committed  : https://huggingface.co/squish-community/.../commit/...
   
     Model page : https://huggingface.co/squish-community/Qwen2.5-1.5B-Instruct-Int8
     Pull with  : squish pull squish-community/Qwen2.5-1.5B-Instruct-Int8
   ```

5. **Repeat for each pre-squished model** (if multiple):
   - Qwen2.5-1.5B-Instruct
   - Llama-3.2-3B
   - Other key models

6. **Document in README – add HF hub models section:**
   Update [README.md](../README.md) with links to published models

---

### 4.2 Create GitHub Release

✅ **Already completed!** (v9.0.0 released on 2026-03-12)

Release notes: https://github.com/wesleyscholl/squish/releases/tag/v9.0.0

---

### 4.3 Community Posts

**Files:** [dev/community_posts.md](community_posts.md) (templates ready)

**Platforms & Timing:**

| Platform | When | Template | Est. Traffic |
|----------|------|----------|--------------|
| Hacker News | Day 1, 9–10 AM PT (Tue–Thu) | `community_posts.md` → HN section | 500–2K upvotes = 10K–50K views |
| Reddit r/LocalLLaMA | Day 1, 9–10 AM PT (same time as HN) | `community_posts.md` → Reddit section | 100–500 upvotes = 2K–10K views |
| Twitter/X | Day 1, staggered 2–3h after HN | `community_posts.md` → Tweet threads | 5K–50K impressions |
| LinkedIn | Day 1–2, midday (professional hours) | `community_posts.md` → LinkedIn section | 500–5K impressions |

**HN Submission:**
1. Go to https://news.ycombinator.com/submit
2. Paste: `https://github.com/wesleyscholl/squish/releases/tag/v9.0.0`
3. Title: `Squish – 54× faster local LLM cold-start on Apple Silicon (222 modules)`
4. Submit
5. Monitor comment threads; respond to technical questions

**Reddit Submission:**
1. Go to https://reddit.com/r/LocalLLaMA
2. Click "Create Post"
3. Use template from `community_posts.md` → Reddit section
4. Engage in comments

**Twitter/X Threads:**
1. Post as thread (not individual tweets)
2. Space replies 30 min apart
3. Use templates from `community_posts.md` → Twitter section
4. Tag relevant accounts: @openai, @AnthropicAI, @ollama_ai, key ML researchers

---

### 4.4 arXiv Submission

**Goal:** Submit formal paper to arXiv.org

**Files:**
- Source (Markdown): [docs/paper.md](../docs/paper.md) (ready, just needs real numbers)
- Output format: PDF (via LaTeX → pdflatex)

**Steps:**

1. **Fill in real numbers** (from Phase 3):
   - Section 4.1: TTFT from bench_eoe.py
   - Section 4.2: Accuracy from MMLU eval
   - Any revised benchmark tables

2. **Convert Markdown → LaTeX** (use Pandoc):
   ```bash
   # Install pandoc if needed
   brew install pandoc
   
   pandoc docs/paper.md -o docs/squish_paper.tex --standalone \
       --include-in-header=docs/preamble.tex
   ```

3. **Create LaTeX preamble** (`docs/preamble.tex`):
   (See template below)

4. **Compile to PDF:**
   ```bash
   cd docs && pdflatex squish_paper.tex
   ```

5. **Upload to arXiv:**
   - Visit https://arxiv.org/submit
   - Create account (free)
   - Upload `squish_paper.pdf`
   - Fill metadata:
     - **Title:** *Squish: Sub-Second Model Loading and Modular Inference Optimisation for Apple Silicon*
     - **Authors:** Wesley Scholl (Independent Research)
     - **Abstract:** (from docs/paper.md intro)
     - **Categories:** cs.AI, cs.LG, cs.DC
   - Submit
   - arXiv will assign ID (e.g., `2403.12345`) within 24h

6. **Update docs and README:**
   ```bash
   # Add arXiv link to README
   echo "📄 Paper: https://arxiv.org/abs/2403.XXXXX" >> README.md
   
   # Commit
   git add docs/paper.md docs/squish_paper.pdf README.md
   git commit -m "papers: submit to arXiv [2403.XXXXX]"
   ```

---

## LaTeX Preamble Template (`docs/preamble.tex`)

```latex
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[margin=1in]{geometry}

\lstset{
  language=python,
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue},
  commentstyle=\color{gray},
  stringstyle=\color{red},
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false,
}

\title{Squish: Sub-Second Model Loading and Modular Inference Optimisation for Apple Silicon}
\author{Wesley Scholl \\ Independent Research}
\date{\today}

\begin{document}
\maketitle
```

---

## Checkoff Checklist

### Phase 3: Hardware Validation
- [ ] Run bench_eoe.py (5+ runs), collect TTFT/tok-s metrics
- [ ] Update README.md with real cold-load times
- [ ] Run MMLU eval (14042 questions), collect accuracy
- [ ] Update docs/RESULTS.md with accuracy table
- [ ] Update paper.md Sections 4.1–4.2 with real numbers
- [ ] Commit: `git commit -m "benchmark: Phase 3 hardware validation complete"`

### Phase 4: Community & Publication
- [ ] Publish 1–3 pre-squished models to HuggingFace Hub
- [ ] GitHub release v9.0.0 ✅ (done)
- [ ] Post to Hacker News (monitor for 48h)
- [ ] Post to r/LocalLLaMA (engage in comments)
- [ ] Post tweet threads on Twitter/X (stagger over 3h)
- [ ] Post on LinkedIn (optional, for professional reach)
- [ ] Convert docs/paper.md → LaTeX via Pandoc
- [ ] Submit to arXiv.org
- [ ] Update README with arXiv link
- [ ] Commit: `git commit -m "papers: submit to arXiv [XXXX.XXXXX], announce v9.0.0"`

---

## Expected Timeline

| Task | Duration | Start | End |
|------|----------|-------|-----|
| bench_eoe.py | 15 min (5 runs) | Day 1 | Day 1 |
| MMLU eval | 45 min – 1h | Day 1 | Day 1–2 |
| Publish HF weights | 10 min per model | Day 1–2 | Day 2 |
| HN + Reddit + Twitter | 30 min posting + 24h+ monitoring | Day 1 | Day 2+ |
| arXiv conversion + submission | 30 min | Day 2 | Day 2 |
| **Total** | **2–3 hours active work** | **Day 1** | **Day 2** |

---

## Support & Troubleshooting

**bench_eoe.py not connecting?**
- Verify server is running: `curl http://localhost:11435/health`
- Check port matches `--squish-port` flag
- Increase timeout: `--timeout 120`

**MMLU eval slow?**
- Reduce `--limit 100` for quick test
- Monitor GPU with `powermetrics` (macOS):
  ```bash
  sudo powermetrics -n 1 | grep "ANE\|GPU"
  ```

**HF token issues?**
- Verify token has "write" permission (not "read" only)
- Test: `huggingface-cli login` (interactive)

**arXiv submission rejected?**
- PDF must be <100 MB (compress images if needed)
- Title + abstract must be <10KB each
- See arXiv submission guidelines: https://arxiv.org/help/submit

---

## Key Contacts & Links

- **GitHub Issues:** https://github.com/wesleyscholl/squish/issues
- **HuggingFace Model Hub:** https://huggingface.co/squish-community
- **arXiv Submissions:** https://arxiv.org/submit
- **Hacker News:** https://news.ycombinator.com/
- **Reddit r/LocalLLaMA:** https://reddit.com/r/LocalLLaMA
- **Twitter/X:** https://twitter.com/

---

## Sample Results to Expect

### bench_eoe.py (M2 Pro, Qwen2.5-1.5B)
```
Mean TTFT:       ~40–60 ms
Mean throughput: ~200–250 tok/s
Peak throughput: ~270 tok/s
```

### MMLU (INT8 Squished)
```
Overall accuracy: ~62–63%
(Within ±2pp of full-precision baseline, per paper abstract)
```

### HF Hub Upload
```
Total size: ~1.1–1.5 GB per model
Upload time: ~5–15 min (depends on internet)
```

---

Enjoy the launch! 🚀
