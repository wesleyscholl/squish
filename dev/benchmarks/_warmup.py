#!/usr/bin/env python3
"""JIT warmup script — sends a single request and waits for the response."""
import urllib.request, json, time, sys

print("JIT warmup request (Qwen3-8B: may take up to 120s)...", flush=True)
req = urllib.request.Request(
    'http://127.0.0.1:11435/v1/chat/completions',
    data=json.dumps({
        'model': 'squish',
        'messages': [{'role': 'user', 'content': '/no_think Say hi.'}],
        'max_tokens': 4,
        'temperature': 0,
        'enable_thinking': False,
    }).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer squish'},
)
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=250) as r:
        d = json.loads(r.read())
    elapsed = time.time() - t0
    print(f"Warmup done in {elapsed:.1f}s")
    print("Response:", d['choices'][0]['message']['content'])
    sys.exit(0)
except Exception as exc:
    print(f"Error after {time.time()-t0:.1f}s: {exc}")
    sys.exit(1)
