#!/usr/bin/env python3
"""Poll until 14B eval (PID 51502) completes, then show results."""
import subprocess, time, sys

pid = 51502
start = time.time()

for i in range(300):
    r = subprocess.run(['ps', '-p', str(pid)], capture_output=True)
    if r.returncode != 0:
        print(f'\nEval finished after {(time.time()-start)/60:.1f} min', flush=True)
        break
    elapsed = time.time() - start
    if i % 8 == 0:
        tail = subprocess.run(
            ['tail', '-2', '/tmp/eval_14b_run2.log'],
            capture_output=True, text=True
        )
        print(f't={elapsed/60:.1f}min: {tail.stdout.strip()[-120:]}', flush=True)
    time.sleep(15)
else:
    print('Timeout waiting for eval', flush=True)
    sys.exit(1)

import subprocess
result = subprocess.run(['tail', '-80', '/tmp/eval_14b_run2.log'], capture_output=True, text=True)
print(result.stdout)
