import urllib.request, urllib.error, time, sys

port = 11435
deadline = time.time() + 90
attempt = 0
while time.time() < deadline:
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=4)
        if r.status == 200:
            elapsed = 90 - (deadline - time.time())
            print(f"UP after ~{elapsed:.0f}s")
            sys.exit(0)
    except Exception as e:
        attempt += 1
        if attempt % 5 == 0:
            print(f"  still waiting… ({attempt} attempts, {deadline-time.time():.0f}s left)")
        time.sleep(3)
print("TIMEOUT — server not ready in 90s")
sys.exit(1)
