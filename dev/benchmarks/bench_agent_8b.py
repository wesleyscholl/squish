#!/usr/bin/env python3
"""
bench_agent_8b.py — Squish Qwen3-8B agentic capability benchmark

Tests three capability pillars required for agent frameworks (OpenClaw etc):
  1. Tool Calling     — structured JSON function dispatch
  2. Reasoning        — multi-hop, chain-of-thought, self-correction
  3. Agentic Planning — task decomposition, sequential decisions

Usage:
    python3 benchmarks/bench_agent_8b.py
    python3 benchmarks/bench_agent_8b.py --port 11435 --model qwen3:8b
    python3 benchmarks/bench_agent_8b.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_PORT  = 11435
DEFAULT_MODEL = "squish"
API_KEY       = "squish"
TIMEOUT       = 60   # overridden by --timeout
NO_THINK      = False  # overridden by --no-think

# ── Color ─────────────────────────────────────────────────────────────────────
G  = "\033[32m"; R = "\033[31m"; Y = "\033[33m"
C  = "\033[36m"; W = "\033[1;37m"; D = "\033[2m"; NC = "\033[0m"
PASS = f"{G}✓ PASS{NC}"; FAIL = f"{R}✗ FAIL{NC}"; SKIP = f"{Y}~ SKIP{NC}"

# ── HTTP helper ───────────────────────────────────────────────────────────────

def chat(messages: list[dict], tools: list[dict] | None = None,
         port: int = DEFAULT_PORT, model: str = DEFAULT_MODEL,
         temperature: float = 0.0, max_tokens: int = 512) -> tuple[dict, float]:
    """POST /v1/chat/completions, return (response_json, elapsed_seconds)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if NO_THINK:
        payload["enable_thinking"] = False
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

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
    return data, elapsed


def extract_text(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def extract_tool_calls(resp: dict) -> list[dict]:
    return resp["choices"][0]["message"].get("tool_calls") or []


def tok_per_sec(resp: dict, elapsed: float) -> float:
    usage = resp.get("usage", {})
    total = usage.get("completion_tokens", 0)
    return round(total / elapsed, 1) if elapsed > 0 else 0.0


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Result:
    name:    str
    passed:  bool
    elapsed: float
    tps:     float
    notes:   list[str] = field(default_factory=list)
    skipped: bool = False


results: list[Result] = []

def record(name: str, passed: bool, elapsed: float, tps: float,
           notes: list[str] = (), skipped: bool = False) -> Result:
    r = Result(name, passed, elapsed, tps, list(notes), skipped)
    results.append(r)
    status = SKIP if skipped else (PASS if passed else FAIL)
    print(f"  {status}  {W}{name}{NC}  {D}({elapsed:.2f}s, {tps} tok/s){NC}")
    for n in notes:
        print(f"         {D}{n}{NC}")
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 1 — TOOL CALLING
# ══════════════════════════════════════════════════════════════════════════════

TOOLS_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":    {"type": "string", "description": "City name"},
                    "unit":    {"type": "string", "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"},
                },
                "required": ["city"],
            },
        },
    }
]

TOOLS_MULTI = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

TOOLS_CODE = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute a Python code snippet and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Python code to run"},
                    "timeout": {"type": "integer", "description": "Max seconds", "default": 10},
                },
                "required": ["code"],
            },
        },
    }
]


def suite_tool_calling(port: int, model: str, verbose: bool) -> None:
    print(f"\n{C}{'═'*60}{NC}")
    print(f"{W}  SUITE 1 — Tool Calling{NC}")
    print(f"{C}{'═'*60}{NC}")

    # ── Test 1.1: single tool dispatch ────────────────────────────────────────
    try:
        resp, elapsed = chat(
            [{"role": "user", "content": "What's the weather in Tokyo right now?"}],
            tools=TOOLS_WEATHER, port=port, model=model,
        )
        calls  = extract_tool_calls(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = (len(calls) == 1
                  and calls[0]["function"]["name"] == "get_weather")
        args   = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
            except Exception:
                pass
        notes = [f"tool={calls[0]['function']['name'] if calls else 'none'}",
                 f"args={args}"]
        if verbose:
            notes.append(f"raw={json.dumps(calls, indent=2)[:300]}")
        record("1.1 Single tool dispatch", passed, elapsed, tps, notes)
    except Exception as e:
        record("1.1 Single tool dispatch", False, 0, 0, [str(e)])

    # ── Test 1.2: correct argument types ──────────────────────────────────────
    try:
        resp, elapsed = chat(
            [{"role": "user",
              "content": "Get the weather in Paris in fahrenheit."}],
            tools=TOOLS_WEATHER, port=port, model=model,
        )
        calls = extract_tool_calls(resp)
        tps   = tok_per_sec(resp, elapsed)
        args  = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
            except Exception:
                pass
        passed = (len(calls) == 1
                  and args.get("city", "").lower() in ("paris",)
                  and args.get("unit", "") == "fahrenheit")
        record("1.2 Argument type correctness", passed, elapsed, tps,
               [f"city={args.get('city')}, unit={args.get('unit')}"])
    except Exception as e:
        record("1.2 Argument type correctness", False, 0, 0, [str(e)])

    # ── Test 1.3: tool selection from multiple tools ───────────────────────────
    try:
        resp, elapsed = chat(
            [{"role": "user",
              "content": "Search the web for the latest MLX release notes."}],
            tools=TOOLS_MULTI, port=port, model=model,
        )
        calls  = extract_tool_calls(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = (len(calls) >= 1
                  and calls[0]["function"]["name"] == "search_web")
        args = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
            except Exception:
                pass
        record("1.3 Tool selection (3 available)", passed, elapsed, tps,
               [f"chose={calls[0]['function']['name'] if calls else 'none'}",
                f"query={args.get('query', '')[:60]}"])
    except Exception as e:
        record("1.3 Tool selection (3 available)", False, 0, 0, [str(e)])

    # ── Test 1.4: no tool when not needed ─────────────────────────────────────
    try:
        resp, elapsed = chat(
            [{"role": "user", "content": "What is 2 + 2?"}],
            tools=TOOLS_MULTI, port=port, model=model,
        )
        calls  = extract_tool_calls(resp)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = len(calls) == 0 and ("4" in text)
        record("1.4 No tool when not needed", passed, elapsed, tps,
               [f"tool_calls={len(calls)}", f"answer={text[:60].strip()}"])
    except Exception as e:
        record("1.4 No tool when not needed", False, 0, 0, [str(e)])

    # ── Test 1.5: tool result synthesis ───────────────────────────────────────
    try:
        messages = [
            {"role": "user",
             "content": "What's the weather in London? Use the tool then summarize."},
            {"role": "assistant", "content": None,
             "tool_calls": [{
                 "id": "call_abc123",
                 "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": '{"city":"London","unit":"celsius"}'},
             }]},
            {"role": "tool",
             "tool_call_id": "call_abc123",
             "content": '{"city":"London","temp":14,"condition":"cloudy","humidity":72}'},
        ]
        resp, elapsed = chat(messages, port=port, model=model, max_tokens=150)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = ("14" in text or "cloudy" in text.lower()) and len(text) > 20
        record("1.5 Tool result synthesis", passed, elapsed, tps,
               [f"response={text[:100].strip()}"])
    except Exception as e:
        record("1.5 Tool result synthesis", False, 0, 0, [str(e)])

    # ── Test 1.6: code execution tool dispatch ────────────────────────────────
    try:
        resp, elapsed = chat(
            [{"role": "user",
              "content": "Run Python to compute the first 10 fibonacci numbers."}],
            tools=TOOLS_CODE, port=port, model=model,
        )
        calls  = extract_tool_calls(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = (len(calls) == 1
                  and calls[0]["function"]["name"] == "execute_python")
        args = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
            except Exception:
                pass
        code_snippet = args.get("code", "")[:80]
        record("1.6 Code execution dispatch", passed, elapsed, tps,
               [f"code={code_snippet!r}"])
    except Exception as e:
        record("1.6 Code execution dispatch", False, 0, 0, [str(e)])


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 2 — REASONING
# ══════════════════════════════════════════════════════════════════════════════

def suite_reasoning(port: int, model: str, verbose: bool) -> None:
    print(f"\n{C}{'═'*60}{NC}")
    print(f"{W}  SUITE 2 — Reasoning{NC}")
    print(f"{C}{'═'*60}{NC}")

    # ── Test 2.1: basic logic ──────────────────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system", "content": "Answer concisely and precisely."},
            {"role": "user",
             "content": ("If all Bloops are Razzles, and all Razzles are Lazzles, "
                         "are all Bloops definitely Lazzles? Answer yes or no, then explain.")},
        ], port=port, model=model, max_tokens=150)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = text.lower().startswith("yes") or ("yes" in text[:20].lower())
        record("2.1 Syllogistic logic", passed, elapsed, tps,
               [f"answer={text[:100].strip()}"])
    except Exception as e:
        record("2.1 Syllogistic logic", False, 0, 0, [str(e)])

    # ── Test 2.2: multi-hop math ───────────────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "user",
             "content": ("A train travels at 120 km/h. It departs at 09:15 and "
                         "arrives at 11:45. How many km did it travel? "
                         "Show your working, then give the final number.")},
        ], port=port, model=model, max_tokens=200)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = "300" in text
        record("2.2 Multi-step arithmetic (300 km)", passed, elapsed, tps,
               [f"contains_300={'300' in text}", f"snippet={text[:120].strip()}"])
    except Exception as e:
        record("2.2 Multi-step arithmetic", False, 0, 0, [str(e)])

    # ── Test 2.3: code reasoning ───────────────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "user",
             "content": textwrap.dedent("""\
                What does this Python function return for f(3)?

                def f(n):
                    if n <= 1:
                        return n
                    return f(n-1) + f(n-2)

                Reply with just the number.""")},
        ], port=port, model=model, max_tokens=20)
        text   = extract_text(resp).strip()
        tps    = tok_per_sec(resp, elapsed)
        passed = "2" in text[:10]
        record("2.3 Code trace (fibonacci f(3)=2)", passed, elapsed, tps,
               [f"answer={text[:30]}"])
    except Exception as e:
        record("2.3 Code trace", False, 0, 0, [str(e)])

    # ── Test 2.4: JSON structured output ──────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system",
             "content": "Always respond with valid JSON only. No markdown, no prose."},
            {"role": "user",
             "content": ('Extract entities from: "John Smith, 34, is a senior engineer '
                         'at Acme Corp in San Francisco." '
                         'Return: {"name":str,"age":int,"title":str,"company":str,"city":str}')},
        ], port=port, model=model, max_tokens=100)
        text = extract_text(resp).strip()
        tps  = tok_per_sec(resp, elapsed)
        parsed = None
        try:
            # strip markdown fences if present
            clean = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except Exception:
            pass
        passed = (parsed is not None
                  and parsed.get("name", "").lower().startswith("john")
                  and parsed.get("age") == 34)
        record("2.4 JSON structured extraction", passed, elapsed, tps,
               [f"parsed={passed}", f"raw={text[:100]}"])
    except Exception as e:
        record("2.4 JSON structured extraction", False, 0, 0, [str(e)])

    # ── Test 2.5: self-correction ──────────────────────────────────────────────
    try:
        messages = [
            {"role": "user",
             "content": "What is the capital of Australia? (Hint: it's not Sydney)"},
            {"role": "assistant", "content": "The capital of Australia is Sydney."},
            {"role": "user",
             "content": "That's incorrect. Please reconsider."},
        ]
        resp, elapsed = chat(messages, port=port, model=model, max_tokens=80)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = "canberra" in text.lower()
        record("2.5 Self-correction (Canberra)", passed, elapsed, tps,
               [f"corrected={'canberra' in text.lower()}", f"response={text[:80].strip()}"])
    except Exception as e:
        record("2.5 Self-correction", False, 0, 0, [str(e)])

    # ── Test 2.6: instruction following / constraint adherence ────────────────
    try:
        resp, elapsed = chat([
            {"role": "user",
             "content": ("List exactly 3 programming languages invented before 1980. "
                         "Format: a numbered list, nothing else.")},
        ], port=port, model=model, max_tokens=80)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        # count numbered items
        lines  = [l.strip() for l in text.splitlines() if l.strip() and l.strip()[0].isdigit()]
        passed = len(lines) == 3
        record("2.6 Constraint adherence (exactly 3 items)", passed, elapsed, tps,
               [f"items_found={len(lines)}", f"response={text[:100].strip()}"])
    except Exception as e:
        record("2.6 Constraint adherence", False, 0, 0, [str(e)])


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 3 — AGENTIC PLANNING
# ══════════════════════════════════════════════════════════════════════════════

def suite_agentic(port: int, model: str, verbose: bool) -> None:
    print(f"\n{C}{'═'*60}{NC}")
    print(f"{W}  SUITE 3 — Agentic Planning{NC}")
    print(f"{C}{'═'*60}{NC}")

    # ── Test 3.1: task decomposition ──────────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system",
             "content": ("You are an AI agent. When given a task, respond ONLY with "
                         "a JSON array of steps: [{\"step\":1,\"action\":str,\"tool\":str|null},...]. "
                         "No prose.")},
            {"role": "user",
             "content": ("Task: Research the latest version of Python, then write a "
                         "summary to a file called python_summary.txt")},
        ], port=port, model=model, max_tokens=300)
        text = extract_text(resp).strip()
        tps  = tok_per_sec(resp, elapsed)
        steps = None
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            steps = json.loads(clean)
        except Exception:
            pass
        passed = (isinstance(steps, list) and len(steps) >= 2
                  and any("search" in str(s).lower() or "web" in str(s).lower()
                          for s in steps)
                  and any("write" in str(s).lower() or "file" in str(s).lower()
                          for s in steps))
        record("3.1 Task decomposition into steps", passed, elapsed, tps,
               [f"steps={len(steps) if steps else 'parse_failed'}",
                f"raw={text[:120]}"])
    except Exception as e:
        record("3.1 Task decomposition", False, 0, 0, [str(e)])

    # ── Test 3.2: sequential tool use planning ────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system",
             "content": "You are an agent with tools: search_web, read_file, write_file, execute_python."},
            {"role": "user",
             "content": ("I need to: 1) find the fibonacci formula online, 2) implement it in Python, "
                         "3) save the implementation to fib.py. "
                         "What is the CORRECT order of tool calls? List tool names only, one per line.")},
        ], port=port, model=model, max_tokens=100)
        text  = extract_text(resp).lower()
        tps   = tok_per_sec(resp, elapsed)
        # search should come before execute_python which should come before write_file
        idx_search = text.find("search")
        idx_exec   = text.find("execute")
        idx_write  = text.find("write")
        passed = (idx_search != -1 and idx_write != -1
                  and idx_search < idx_write)
        record("3.2 Sequential tool ordering", passed, elapsed, tps,
               [f"order_correct={passed}",
                f"search_pos={idx_search}, write_pos={idx_write}",
                f"response={text[:100].strip()}"])
    except Exception as e:
        record("3.2 Sequential tool ordering", False, 0, 0, [str(e)])

    # ── Test 3.3: ambiguity clarification ─────────────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system",
             "content": "You are a helpful agent. Ask for clarification when a task is ambiguous."},
            {"role": "user", "content": "Delete the file."},
        ], port=port, model=model, max_tokens=100)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        # should ask which file
        passed = ("?" in text and
                  any(w in text.lower() for w in
                      ["which", "what", "specify", "name", "file", "clarify", "path"]))
        record("3.3 Ambiguity clarification", passed, elapsed, tps,
               [f"asked_question={passed}", f"response={text[:100].strip()}"])
    except Exception as e:
        record("3.3 Ambiguity clarification", False, 0, 0, [str(e)])

    # ── Test 3.4: multi-turn context retention ────────────────────────────────
    try:
        messages = [
            {"role": "user",
             "content": "I'm building a REST API in Go. The project is called 'Harbor'."},
            {"role": "assistant",
             "content": "Got it — a Go REST API project named Harbor. How can I help?"},
            {"role": "user",
             "content": "What language and project name did I just tell you?"},
        ]
        resp, elapsed = chat(messages, port=port, model=model, max_tokens=60)
        text   = extract_text(resp).lower()
        tps    = tok_per_sec(resp, elapsed)
        passed = "go" in text and "harbor" in text
        record("3.4 Multi-turn context retention", passed, elapsed, tps,
               [f"remembered_go={'go' in text}, remembered_harbor={'harbor' in text}",
                f"response={text[:80].strip()}"])
    except Exception as e:
        record("3.4 Multi-turn context retention", False, 0, 0, [str(e)])

    # ── Test 3.5: agent persona / role adherence ──────────────────────────────
    try:
        resp, elapsed = chat([
            {"role": "system",
             "content": ("You are a DevOps agent. You ONLY answer questions related to "
                         "infrastructure, CI/CD, Kubernetes, or cloud platforms. "
                         "For anything else say: 'Outside my DevOps scope.'")},
            {"role": "user",
             "content": "Write me a sonnet about love."},
        ], port=port, model=model, max_tokens=80)
        text   = extract_text(resp)
        tps    = tok_per_sec(resp, elapsed)
        passed = ("scope" in text.lower() or "devops" in text.lower()
                  or "outside" in text.lower() or "can't" in text.lower()
                  or "cannot" in text.lower())
        record("3.5 Persona / role boundary adherence", passed, elapsed, tps,
               [f"stayed_in_role={passed}", f"response={text[:100].strip()}"])
    except Exception as e:
        record("3.5 Persona adherence", False, 0, 0, [str(e)])

    # ── Test 3.6: error recovery suggestion ───────────────────────────────────
    try:
        messages = [
            {"role": "user",
             "content": "Run this command: kubectl apply -f deployment.yaml"},
            {"role": "assistant",
             "content": "Running kubectl apply -f deployment.yaml..."},
            {"role": "user",
             "content": ('Error: error validating "deployment.yaml": '
                         'unknown field "spec.template.spec.containers[0].ressources". '
                         "What should I do?")},
        ]
        resp, elapsed = chat(messages, port=port, model=model, max_tokens=150)
        text   = extract_text(resp).lower()
        tps    = tok_per_sec(resp, elapsed)
        passed = ("resource" in text and
                  any(w in text for w in ["typo", "spell", "correct", "fix", "change"]))
        record("3.6 Error recovery / typo diagnosis", passed, elapsed, tps,
               [f"identified_typo={passed}",
                f"response={text[:120].strip()}"])
    except Exception as e:
        record("3.6 Error recovery", False, 0, 0, [str(e)])


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    passed  = sum(1 for r in results if r.passed and not r.skipped)
    failed  = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    total   = len(results)
    scored  = total - skipped
    pct     = round(passed / scored * 100) if scored else 0
    avg_tps = round(sum(r.tps for r in results if not r.skipped) / max(scored, 1), 1)
    avg_t   = round(sum(r.elapsed for r in results if not r.skipped) / max(scored, 1), 2)

    print(f"\n{C}{'═'*60}{NC}")
    print(f"{W}  BENCHMARK SUMMARY{NC}")
    print(f"{C}{'═'*60}{NC}")
    print(f"  Total      : {total}")
    print(f"  {G}Passed{NC}     : {passed}")
    print(f"  {R}Failed{NC}     : {failed}")
    print(f"  {Y}Skipped{NC}    : {skipped}")
    print(f"  Score      : {W}{pct}%{NC}  ({passed}/{scored})")
    print(f"  Avg latency: {C}{avg_t}s{NC}")
    print(f"  Avg tok/s  : {C}{avg_tps}{NC}")
    print(f"{C}{'═'*60}{NC}")

    if failed:
        print(f"\n{R}  Failed tests:{NC}")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"    {R}✗{NC} {r.name}")

    # Agent readiness verdict
    print()
    if pct >= 90:
        print(f"  {G}★ AGENT READY{NC}  — Excellent tool calling + reasoning. Suitable for OpenClaw/production agentic use.")
    elif pct >= 75:
        print(f"  {Y}◑ MOSTLY READY{NC} — Good baseline, minor gaps. Suitable for supervised agentic workflows.")
    elif pct >= 50:
        print(f"  {Y}⚠ PARTIAL{NC}      — Tool calling works but reasoning gaps. Needs prompt engineering for reliability.")
    else:
        print(f"  {R}✗ NOT READY{NC}    — Significant failures. Check model, server, or consider a larger model.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global TIMEOUT, NO_THINK
    parser = argparse.ArgumentParser(description="Squish Qwen3-8B agent benchmark")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--suite",    choices=["tools", "reasoning", "agentic", "all"],
                        default="all")
    parser.add_argument("--timeout",  type=int, default=TIMEOUT,
                        help="Per-request timeout in seconds (default: 60)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode for Qwen3 (faster, less thorough)")
    args = parser.parse_args()
    TIMEOUT  = args.timeout
    NO_THINK = args.no_think

    # check server
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", args.port), timeout=2)
        s.close()
    except OSError:
        print(f"{R}✗ Server not reachable on :{args.port}  —  run: squish serve qwen3:8b{NC}")
        sys.exit(1)

    print(f"\n{W}Squish Agent Benchmark{NC}  {D}→ model={args.model}  port={args.port}{NC}")

    if args.suite in ("tools", "all"):
        suite_tool_calling(args.port, args.model, args.verbose)
    if args.suite in ("reasoning", "all"):
        suite_reasoning(args.port, args.model, args.verbose)
    if args.suite in ("agentic", "all"):
        suite_agentic(args.port, args.model, args.verbose)

    print_summary()


if __name__ == "__main__":
    main()
