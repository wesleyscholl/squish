#!/usr/bin/env python3
"""
bench_commit.py — Squish commit message generation benchmark

Measures quality and speed of commit message generation across models.
Designed for the 1.5B and 7B models used by git-commit-push-script.sh.

Usage:
    python3 benchmarks/bench_commit.py                     # 1.5b + 7b
    python3 benchmarks/bench_commit.py --models 1.5b 7b 8b
    python3 benchmarks/bench_commit.py --rounds 5          # repeat for stable timings
    python3 benchmarks/bench_commit.py --csv out.csv       # save results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import socket
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_PORT    = 11435
API_KEY         = "squish"
TIMEOUT         = 60
NO_THINK        = False  # overridden by --no-think
MAX_TOKENS      = 20
TEMPERATURE     = 0.2
STOP_SEQS       = ["\n", "\r"]

# ── Color ─────────────────────────────────────────────────────────────────────
G  = "\033[32m"; R = "\033[31m"; Y = "\033[33m"
C  = "\033[36m"; W = "\033[1;37m"; D = "\033[2m"; NC = "\033[0m"

# ── System prompt (mirrors git-commit-push-script.sh) ─────────────────────────
SYSTEM_PROMPT = (
    "Generate a concise git commit message for these changes. "
    "Start with an imperative verb. No quotes, no prefix, just the message. "
    "Under 72 chars."
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST DIFFS — realistic cases
# ══════════════════════════════════════════════════════════════════════════════

DIFFS: list[dict] = [

    # ── 1. Simple bug fix ─────────────────────────────────────────────────────
    {
        "id": "bug_fix",
        "label": "Bug fix (off-by-one)",
        "char_count": "small",
        "diff": """\
diff --git a/pkg/parser/parser.go b/pkg/parser/parser.go
index a1b2c3d..e4f5a6b 100644
--- a/pkg/parser/parser.go
+++ b/pkg/parser/parser.go
@@ -42,7 +42,7 @@ func ParseTokens(tokens []string) []Node {
     nodes := make([]Node, 0, len(tokens))
-    for i := 0; i <= len(tokens); i++ {
+    for i := 0; i < len(tokens); i++ {
         nodes = append(nodes, parseToken(tokens[i]))
     }
     return nodes
""",
        "keywords": ["fix", "off", "bound", "index", "loop", "range"],
        "bad_patterns": ["TODO", "WIP", "misc", "stuff", "thing"],
    },

    # ── 2. New feature / API addition ─────────────────────────────────────────
    {
        "id": "new_feature",
        "label": "New feature (pagination)",
        "char_count": "medium",
        "diff": """\
diff --git a/api/handlers/products.go b/api/handlers/products.go
index 1111111..2222222 100644
--- a/api/handlers/products.go
+++ b/api/handlers/products.go
@@ -15,6 +15,8 @@ import (
+    "strconv"
+
 func ListProducts(w http.ResponseWriter, r *http.Request) {
-    products, err := db.GetAllProducts()
+    page, _ := strconv.Atoi(r.URL.Query().Get("page"))
+    limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
+    if limit == 0 { limit = 20 }
+    products, err := db.GetProducts(page, limit)
     if err != nil {
         http.Error(w, err.Error(), http.StatusInternalServerError)
         return
@@ -24,6 +28,12 @@ func ListProducts(w http.ResponseWriter, r *http.Request) {
+
+func GetProductPage(w http.ResponseWriter, r *http.Request) {
+    page := r.URL.Query().Get("page")
+    products, err := db.GetProductsByPage(page)
+    if err != nil {
+        http.Error(w, err.Error(), http.StatusInternalServerError)
+    }
+    json.NewEncoder(w).Encode(products)
+}
""",
        "keywords": ["add", "feat", "pagina", "limit", "page"],
        "bad_patterns": ["update", "change", "modify"],
    },

    # ── 3. Refactor / rename ───────────────────────────────────────────────────
    {
        "id": "refactor",
        "label": "Refactor (rename + extract function)",
        "char_count": "medium",
        "diff": """\
diff --git a/internal/auth/auth.go b/internal/auth/auth.go
index aaaaaaa..bbbbbbb 100644
--- a/internal/auth/auth.go
+++ b/internal/auth/auth.go
@@ -8,14 +8,18 @@ package auth
-func checkToken(tok string) bool {
-    if tok == "" { return false }
-    decoded, err := base64.StdEncoding.DecodeString(tok)
-    if err != nil { return false }
-    parts := strings.Split(string(decoded), ":")
-    if len(parts) != 2 { return false }
-    return validateCredentials(parts[0], parts[1])
+func ValidateToken(tok string) bool {
+    if tok == "" { return false }
+    creds, err := decodeToken(tok)
+    if err != nil { return false }
+    return validateCredentials(creds.User, creds.Pass)
+}
+
+func decodeToken(tok string) (Credentials, error) {
+    decoded, err := base64.StdEncoding.DecodeString(tok)
+    if err != nil { return Credentials{}, err }
+    parts := strings.Split(string(decoded), ":")
+    if len(parts) != 2 { return Credentials{}, errors.New("invalid format") }
+    return Credentials{User: parts[0], Pass: parts[1]}, nil
 }
""",
        "keywords": ["refactor", "extract", "rename", "split", "clean", "restructure"],
        "bad_patterns": ["fix", "bug", "error"],
    },

    # ── 4. Config / infra change ───────────────────────────────────────────────
    {
        "id": "config",
        "label": "Config change (resource limits)",
        "char_count": "small",
        "diff": """\
diff --git a/deploy/k8s/api-deployment.yaml b/deploy/k8s/api-deployment.yaml
index ccccccc..ddddddd 100644
--- a/deploy/k8s/api-deployment.yaml
+++ b/deploy/k8s/api-deployment.yaml
@@ -28,9 +28,9 @@ spec:
         resources:
           requests:
-            cpu: "100m"
-            memory: "128Mi"
+            cpu: "250m"
+            memory: "256Mi"
           limits:
-            cpu: "500m"
-            memory: "512Mi"
+            cpu: "1000m"
+            memory: "1Gi"
""",
        "keywords": ["update", "increase", "adjust", "resource", "limit", "cpu", "memory", "k8s"],
        "bad_patterns": ["fix", "bug"],
    },

    # ── 5. Docs / README ───────────────────────────────────────────────────────
    {
        "id": "docs",
        "label": "Docs update (README)",
        "char_count": "small",
        "diff": """\
diff --git a/README.md b/README.md
index eeeeeee..fffffff 100644
--- a/README.md
+++ b/README.md
@@ -1,5 +1,12 @@
 # Harbor API

-A Go REST API for product catalog management.
+A high-performance Go REST API for product catalog and inventory management.
+
+## Requirements
+- Go 1.21+
+- PostgreSQL 15+
+- Redis 7+
+
+## Quick Start
+```bash
+go run ./cmd/server
+```
""",
        "keywords": ["doc", "readme", "update", "add"],
        "bad_patterns": ["fix", "bug", "refactor"],
    },

    # ── 6. Dependency / go.mod ─────────────────────────────────────────────────
    {
        "id": "deps",
        "label": "Dependency update (go.mod)",
        "char_count": "small",
        "diff": """\
diff --git a/go.mod b/go.mod
index 1234567..abcdefg 100644
--- a/go.mod
+++ b/go.mod
@@ -10,7 +10,7 @@ require (
-    github.com/gin-gonic/gin v1.9.0
+    github.com/gin-gonic/gin v1.9.1
-    golang.org/x/crypto v0.14.0
+    golang.org/x/crypto v0.17.0
     github.com/stretchr/testify v1.8.4
 )
""",
        "keywords": ["bump", "update", "upgrade", "dep", "version"],
        "bad_patterns": ["fix", "add feature"],
    },

    # ── 7. Test addition ───────────────────────────────────────────────────────
    {
        "id": "tests",
        "label": "Add tests",
        "char_count": "medium",
        "diff": """\
diff --git a/pkg/parser/parser_test.go b/pkg/parser/parser_test.go
new file mode 100644
index 0000000..9999999
--- /dev/null
+++ b/pkg/parser/parser_test.go
@@ -0,0 +1,35 @@
+package parser_test
+
+import (
+    "testing"
+    "github.com/example/harbor/pkg/parser"
+)
+
+func TestParseTokens_Empty(t *testing.T) {
+    nodes := parser.ParseTokens([]string{})
+    if len(nodes) != 0 {
+        t.Errorf("expected 0 nodes, got %d", len(nodes))
+    }
+}
+
+func TestParseTokens_Single(t *testing.T) {
+    nodes := parser.ParseTokens([]string{"hello"})
+    if len(nodes) != 1 {
+        t.Fatalf("expected 1 node, got %d", len(nodes))
+    }
+    if nodes[0].Value != "hello" {
+        t.Errorf("expected value 'hello', got '%s'", nodes[0].Value)
+    }
+}
""",
        "keywords": ["add", "test", "cover"],
        "bad_patterns": ["fix", "update"],
    },

    # ── 8. Security patch ─────────────────────────────────────────────────────
    {
        "id": "security",
        "label": "Security patch (SQL injection)",
        "char_count": "small",
        "diff": """\
diff --git a/internal/db/queries.go b/internal/db/queries.go
index 1a2b3c4..5d6e7f8 100644
--- a/internal/db/queries.go
+++ b/internal/db/queries.go
@@ -19,6 +19,6 @@ func GetUserByEmail(email string) (*User, error) {
-    row := db.QueryRow(fmt.Sprintf(
-        "SELECT id, name, email FROM users WHERE email = '%s'", email))
+    row := db.QueryRow(
+        "SELECT id, name, email FROM users WHERE email = $1", email)
     var u User
     err := row.Scan(&u.ID, &u.Name, &u.Email)
""",
        "keywords": ["fix", "secure", "sanitize", "injection", "sql", "parameterize"],
        "bad_patterns": ["update", "refactor"],
    },

    # ── 9. Env / secrets cleanup ───────────────────────────────────────────────
    {
        "id": "env",
        "label": "Config env var refactor",
        "char_count": "small",
        "diff": """\
diff --git a/cmd/server/main.go b/cmd/server/main.go
index aaaaaaa..bbbbbbb 100644
--- a/cmd/server/main.go
+++ b/cmd/server/main.go
@@ -12,7 +12,8 @@ func main() {
-    port := "8080"
-    dsn  := "postgres://localhost/harbor"
+    port := getEnvOrDefault("PORT", "8080")
+    dsn  := mustGetEnv("DATABASE_URL")
     srv := &http.Server{Addr: ":" + port, Handler: router()}
""",
        "keywords": ["config", "env", "env var", "environment", "move", "externalize"],
        "bad_patterns": ["fix bug", "update"],
    },

    # ── 10. Large multi-file change ────────────────────────────────────────────
    {
        "id": "multi_file",
        "label": "Multi-file feature (auth middleware)",
        "char_count": "large",
        "diff": """\
diff --git a/internal/middleware/auth.go b/internal/middleware/auth.go
new file mode 100644
index 0000000..1234abc
--- /dev/null
+++ b/internal/middleware/auth.go
@@ -0,0 +1,28 @@
+package middleware
+
+import (
+    "net/http"
+    "strings"
+    "github.com/example/harbor/internal/auth"
+)
+
+func RequireAuth(next http.Handler) http.Handler {
+    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
+        header := r.Header.Get("Authorization")
+        if !strings.HasPrefix(header, "Bearer ") {
+            http.Error(w, "unauthorized", http.StatusUnauthorized)
+            return
+        }
+        token := strings.TrimPrefix(header, "Bearer ")
+        if !auth.ValidateToken(token) {
+            http.Error(w, "invalid token", http.StatusUnauthorized)
+            return
+        }
+        next.ServeHTTP(w, r)
+    })
+}
diff --git a/api/router.go b/api/router.go
index ccccccc..ddddddd 100644
--- a/api/router.go
+++ b/api/router.go
@@ -14,6 +14,8 @@ func NewRouter() *http.ServeMux {
     mux := http.NewServeMux()
-    mux.HandleFunc("/api/products", handlers.ListProducts)
-    mux.HandleFunc("/api/orders", handlers.ListOrders)
+    protected := middleware.RequireAuth
+    mux.Handle("/api/products", protected(http.HandlerFunc(handlers.ListProducts)))
+    mux.Handle("/api/orders",   protected(http.HandlerFunc(handlers.ListOrders)))
     return mux
 }
""",
        "keywords": ["add", "auth", "middleware", "protect", "jwt", "bearer"],
        "bad_patterns": ["update", "fix"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP helper
# ══════════════════════════════════════════════════════════════════════════════

def complete(diff_text: str, port: int, model: str) -> tuple[str, float, int]:
    """Return (message, elapsed_s, completion_tokens)."""
    user_msg = f"Git diff:\n{diff_text[:1500]}"  # mirror MAX_DIFF_CHARS cap

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "stop":        STOP_SEQS,
        "stream":      False,
    }
    if NO_THINK:
        payload["enable_thinking"] = False
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    message = (data["choices"][0]["message"].get("content") or "").strip()
    tokens  = data.get("usage", {}).get("completion_tokens", 0)
    return message, elapsed, tokens


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════════

IMPERATIVE_VERBS = {
    "add", "adds", "added",
    "fix", "fixes", "fixed",
    "remove", "removes", "removed",
    "update", "updates", "updated",
    "refactor", "refactors", "refactored",
    "bump", "bumps", "bumped",
    "rename", "renames", "renamed",
    "extract", "extracts", "extracted",
    "improve", "improves", "improved",
    "secure", "secures", "secured",
    "increase", "increases", "increased",
    "protect", "protects", "protected",
    "implement", "implements", "implemented",
    "migrate", "migrates", "migrated",
    "enable", "enables", "enabled",
    "disable", "disables", "disabled",
    "clean", "cleans", "cleaned",
    "document", "documents", "documented",
    "test", "tests", "tested",
    "adjust", "adjusts", "adjusted",
    "sanitize", "sanitizes", "sanitized",
    "externalize", "externalizes",
    "parameterize", "parameterizes",
    "enforce", "enforces",
    "configure", "configures",
    "set", "sets",
    "use", "uses", "used",
    "replace", "replaces", "replaced",
    "introduce", "introduces",
    "expand", "expands",
    "cover", "covers",
    "split", "splits",
    "streamline", "streamlines",
}

@dataclass
class Score:
    message:      str
    model:        str
    diff_id:      str
    elapsed:      float
    tokens:       int
    tok_per_sec:  float

    # quality booleans
    not_empty:    bool = False
    under_72:     bool = False
    no_quotes:    bool = False
    no_prefix:    bool = False   # no "feat:", "fix:" prefix (we want plain imperative)
    imperative:   bool = False
    relevant:     bool = False
    no_bad:       bool = False

    @property
    def quality_score(self) -> int:
        return sum([
            self.not_empty,
            self.under_72,
            self.no_quotes,
            self.no_prefix,
            self.imperative,
            self.relevant,
            self.no_bad,
        ])

    @property
    def quality_pct(self) -> int:
        return round(self.quality_score / 7 * 100)


def score_message(message: str, diff: dict, model: str,
                  elapsed: float, tokens: int) -> Score:
    s = Score(
        message=message, model=model, diff_id=diff["id"],
        elapsed=elapsed, tokens=tokens,
        tok_per_sec=round(tokens / elapsed, 1) if elapsed > 0 else 0.0,
    )
    if not message:
        return s

    s.not_empty  = bool(message.strip())
    s.under_72   = len(message) <= 72
    s.no_quotes  = message[0] not in ('"', "'", "`")
    s.no_prefix  = not re.match(r"^(feat|fix|chore|docs|refactor|test|style|ci)\s*:", message, re.I)

    first_word = message.split()[0].rstrip(".,!:").lower() if message.split() else ""
    s.imperative = first_word in IMPERATIVE_VERBS

    msg_lower = message.lower()
    s.relevant = any(k.lower() in msg_lower for k in diff["keywords"])
    s.no_bad   = not any(b.lower() in msg_lower for b in diff["bad_patterns"])

    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_model(model: str, rounds: int, port: int, verbose: bool) -> list[Score]:
    all_scores: list[Score] = []

    for diff in DIFFS:
        round_scores: list[Score] = []
        for r in range(rounds):
            try:
                msg, elapsed, tokens = complete(diff["diff"], port, model)
                s = score_message(msg, diff, model, elapsed, tokens)
                round_scores.append(s)
                if verbose:
                    q_color = G if s.quality_pct >= 85 else (Y if s.quality_pct >= 60 else R)
                    print(f"    r{r+1}  {q_color}{s.quality_pct}%{NC}  {elapsed:.2f}s  "
                          f"{D}{msg[:60]!r}{NC}")
            except Exception as e:
                # on total failure, insert a zeroed score
                s = Score(message="", model=model, diff_id=diff["id"],
                          elapsed=0, tokens=0, tok_per_sec=0)
                round_scores.append(s)
                if verbose:
                    print(f"    r{r+1}  {R}ERROR{NC}  {e}")

        # best round (highest quality, then fastest)
        best = max(round_scores, key=lambda x: (x.quality_score, -x.elapsed))
        all_scores.append(best)

    return all_scores


def print_model_table(model: str, scores: list[Score]) -> None:
    print(f"\n{C}{'─'*70}{NC}")
    print(f"{W}  Model: {model}{NC}")
    print(f"{C}{'─'*70}{NC}")
    print(f"  {'Diff':<26} {'Msg (truncated)':<30} {'Q%':>4} {'s':>5} {'t/s':>5}")
    print(f"  {'─'*26} {'─'*30} {'─'*4} {'─'*5} {'─'*5}")

    for s in scores:
        diff = next(d for d in DIFFS if d["id"] == s.diff_id)
        q_color = G if s.quality_pct >= 85 else (Y if s.quality_pct >= 60 else R)
        label = diff["label"][:25]
        msg   = (s.message[:28] + "..") if len(s.message) > 30 else s.message.ljust(30)
        if not s.message:
            msg = f"{R}(empty){NC}"
        print(f"  {label:<26} {D}{msg:<30}{NC} {q_color}{s.quality_pct:>3}%{NC} "
              f"{s.elapsed:>5.2f} {s.tok_per_sec:>5.1f}")


def print_comparison(model_scores: dict[str, list[Score]]) -> None:
    print(f"\n{C}{'═'*70}{NC}")
    print(f"{W}  HEAD-TO-HEAD COMPARISON{NC}")
    print(f"{C}{'═'*70}{NC}")

    models = list(model_scores.keys())
    header = f"  {'Diff':<26}"
    for m in models:
        header += f"  {m:>12}"
    print(header)
    print(f"  {'─'*26}" + "  " + "─"*12 * len(models))

    for diff in DIFFS:
        row = f"  {diff['label'][:25]:<26}"
        for m in models:
            scores = model_scores[m]
            s = next((x for x in scores if x.diff_id == diff["id"]), None)
            if s:
                q_color = G if s.quality_pct >= 85 else (Y if s.quality_pct >= 60 else R)
                row += f"  {q_color}{s.quality_pct:>3}% {s.elapsed:>4.1f}s{NC}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print(f"\n  {'Metric':<26}", end="")
    for m in models:
        print(f"  {m:>12}", end="")
    print()
    print(f"  {'─'*26}" + "  " + "─"*12 * len(models))

    for metric_name, fn in [
        ("Avg quality %",   lambda ss: round(sum(s.quality_pct for s in ss) / len(ss))),
        ("Avg latency (s)", lambda ss: round(sum(s.elapsed for s in ss) / len(ss), 2)),
        ("Avg tok/s",       lambda ss: round(sum(s.tok_per_sec for s in ss) / len(ss), 1)),
        ("Perfect (100%)",  lambda ss: sum(1 for s in ss if s.quality_pct == 100)),
        ("Failed (<40%)",   lambda ss: sum(1 for s in ss if s.quality_pct < 40)),
    ]:
        row = f"  {metric_name:<26}"
        best_val = None
        vals = []
        for m in models:
            vals.append(fn(model_scores[m]))
        # for quality/tps: higher is better; for latency/failures: lower is better
        higher_better = metric_name not in ("Avg latency (s)", "Failed (<40%)")
        best_val = max(vals) if higher_better else min(vals)

        for i, m in enumerate(models):
            v = vals[i]
            color = G if v == best_val else NC
            row += f"  {color}{str(v):>12}{NC}"
        print(row)

    print(f"{C}{'═'*70}{NC}")


def save_csv(model_scores: dict[str, list[Score]], path: str) -> None:
    rows = []
    for model, scores in model_scores.items():
        for s in scores:
            rows.append({
                "model":       model,
                "diff_id":     s.diff_id,
                "message":     s.message,
                "elapsed":     s.elapsed,
                "tokens":      s.tokens,
                "tok_per_sec": s.tok_per_sec,
                "quality_pct": s.quality_pct,
                "not_empty":   int(s.not_empty),
                "under_72":    int(s.under_72),
                "no_quotes":   int(s.no_quotes),
                "no_prefix":   int(s.no_prefix),
                "imperative":  int(s.imperative),
                "relevant":    int(s.relevant),
                "no_bad":      int(s.no_bad),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{G}✓ Results saved to {path}{NC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global TIMEOUT, NO_THINK
    parser = argparse.ArgumentParser(description="Squish commit message benchmark")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    parser.add_argument("--models",   nargs="+", default=["squish:1.5b", "squish:7b"],
                        metavar="MODEL",
                        help="squish model identifiers to test (default: 1.5b 7b)")
    parser.add_argument("--rounds",   type=int, default=1,
                        help="test rounds per diff per model (best is kept)")
    parser.add_argument("--verbose",  action="store_true",
                        help="print each generation attempt")
    parser.add_argument("--csv",      metavar="PATH",
                        help="write CSV results to this path")
    parser.add_argument("--diff",     metavar="ID",
                        help="run only a specific diff by id (e.g. bug_fix)")
    parser.add_argument("--timeout",  type=int, default=TIMEOUT,
                        help="per-request timeout in seconds (default: 60)")
    parser.add_argument("--no-think", action="store_true",
                        help="disable thinking mode for Qwen3")
    args = parser.parse_args()
    TIMEOUT  = args.timeout
    NO_THINK = args.no_think

    # check server
    try:
        s = socket.create_connection(("127.0.0.1", args.port), timeout=2)
        s.close()
    except OSError:
        print(f"{R}✗ Server not reachable on :{args.port}  —  run: squish serve <model>{NC}")
        sys.exit(1)

    # filter diffs if requested
    diffs_to_run = DIFFS
    if args.diff:
        diffs_to_run = [d for d in DIFFS if d["id"] == args.diff]
        if not diffs_to_run:
            print(f"{R}Unknown diff id: {args.diff!r}{NC}")
            print(f"Available: {', '.join(d['id'] for d in DIFFS)}")
            sys.exit(1)

    global DIFFS
    DIFFS = diffs_to_run

    print(f"\n{W}Squish Commit Message Benchmark{NC}")
    print(f"  {D}models={args.models}  rounds={args.rounds}  "
          f"diffs={len(diffs_to_run)}  port={args.port}{NC}")

    model_scores: dict[str, list[Score]] = {}

    for model in args.models:
        print(f"\n{W}── {model} ──{NC}")
        scores = run_model(model, args.rounds, args.port, args.verbose)
        model_scores[model] = scores
        print_model_table(model, scores)

    if len(args.models) > 1:
        print_comparison(model_scores)

    if args.csv:
        save_csv(model_scores, args.csv)

    # exit code: 0 if any model hits avg ≥70%, else 1
    for scores in model_scores.values():
        avg = sum(s.quality_pct for s in scores) / max(len(scores), 1)
        if avg >= 70:
            sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
