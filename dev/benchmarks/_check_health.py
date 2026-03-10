import urllib.request, urllib.error, sys
try:
    r = urllib.request.urlopen("http://localhost:11435/health", timeout=4)
    print("UP", r.status)
except urllib.error.URLError as e:
    print("DOWN", e.reason)
    sys.exit(1)
