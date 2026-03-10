# Squish Benchmarks

Custom benchmarks for evaluating squish-served models.

## Agent benchmark — `bench_agent_8b.py`

Tests Qwen3-8B (or any model) across three capability pillars needed for
agentic frameworks like OpenClaw:

| Suite | Tests |
|-------|-------|
| **Tool Calling** | Single dispatch, arg types, multi-tool selection, no-tool guard, result synthesis, code dispatch |
| **Reasoning**    | Syllogistic logic, multi-step math, code trace, JSON extraction, self-correction, constraint adherence |
| **Agentic**      | Task decomposition, sequential tool ordering, ambiguity clarification, multi-turn context, persona adherence, error recovery |

```bash
# make sure squish is serving qwen3:8b on port 11435
squish serve qwen3:8b

# run all suites
python3 dev/benchmarks/bench_agent_8b.py

# options
python3 dev/benchmarks/bench_agent_8b.py --port 11435 --model squish --verbose
python3 dev/benchmarks/bench_agent_8b.py --suite tools       # tools only
python3 dev/benchmarks/bench_agent_8b.py --suite reasoning   # reasoning only
python3 dev/benchmarks/bench_agent_8b.py --suite agentic     # agentic only
```

**Readiness thresholds:**

| Score | Verdict |
|-------|---------|
| ≥90%  | ★ AGENT READY — suitable for production agentic use |
| ≥75%  | ◑ MOSTLY READY — supervised workflows ok |
| ≥50%  | ⚠ PARTIAL — prompt engineering needed |
| <50%  | ✗ NOT READY — check model/server |

---

## Commit message benchmark — `bench_commit.py`

Compares 1.5B vs 7B (vs any model) on generating git commit messages
from realistic diffs. Mirrors the exact payload used by
`git-commit-push-script.sh` (`max_tokens=20`, `temperature=0.2`, stop on `\n`).

**10 diff types:** bug fix, new feature, refactor, config, docs, deps, tests,
security patch, env var cleanup, multi-file feature.

**7 quality criteria per message:**

1. Not empty
2. Under 72 characters
3. No leading quote
4. No conventional-commit prefix (`feat:`, `fix:` etc.)
5. Starts with imperative verb
6. Contains diff-relevant keyword
7. No generic/bad-pattern words

```bash
# default: test 1.5b and 7b
python3 dev/benchmarks/bench_commit.py

# options
python3 dev/benchmarks/bench_commit.py --models squish:1.5b squish:7b squish:8b
python3 dev/benchmarks/bench_commit.py --rounds 3         # best of 3 per diff
python3 dev/benchmarks/bench_commit.py --csv results.csv  # save to CSV
python3 dev/benchmarks/bench_commit.py --diff bug_fix     # single diff
python3 dev/benchmarks/bench_commit.py --verbose          # show each generation
```

**Exit code:** `0` if any model averages ≥70% quality, `1` otherwise.

---

## Running both

```bash
# terminal 1
squish serve qwen3:8b

# terminal 2 — agent bench
python3 dev/benchmarks/bench_agent_8b.py --verbose

# terminal 2 — commit bench (switch model between runs via squish serve)
python3 dev/benchmarks/bench_commit.py --models squish:1.5b squish:7b --rounds 3 --csv commit_results.csv
```
