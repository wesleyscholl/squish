import subprocess, sys, json, statistics

POC_DIR = "/Users/wscholl/poc"
COMP_DIR = "/Users/wscholl/models/Qwen2.5-14B-Instruct-bf16-compressed"
MODEL_DIR = "/Users/wscholl/models/Qwen2.5-14B-Instruct-bf16"
N = 5

SCRIPT = (
    "import time, json, sys;"
    f"sys.path.insert(0, '{POC_DIR}');"
    "from compressed_loader import load_from_npy_dir;"
    "import mlx_lm;"
    f"t0=time.perf_counter();model,tok,stats=load_from_npy_dir('{COMP_DIR}',model_dir='{MODEL_DIR}',verbose=False,return_stats=True);load_s=time.perf_counter()-t0;"
    "t1=time.perf_counter();resp=mlx_lm.generate(model,tok,prompt='The capital of France is',max_tokens=32,verbose=False);gen_s=time.perf_counter()-t1;"
    "n_tok=len(tok.encode(resp,add_special_tokens=False));tps=n_tok/max(gen_s,0.001);"
    "print(json.dumps({'load_s':load_s,'tps':tps,'loader':stats.get('loader'),'n_tok':n_tok}))"
)

times, tps_list = [], []
for i in range(N):
    print(f"  run {i+1}/{N} starting...", flush=True)
    r = subprocess.run([sys.executable, "-c", SCRIPT],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"  run {i+1}: ERROR\n{r.stderr[-500:]}")
        continue
    try:
        d = json.loads(r.stdout.strip())
    except Exception as e:
        print(f"  run {i+1}: parse error: {e}\n{r.stdout[:200]}")
        continue
    times.append(d["load_s"])
    tps_list.append(d["tps"])
    print(f"  run {i+1:2d}: load={d['load_s']:.3f}s  tps={d['tps']:.1f}  loader={d.get('loader')}  n_tok={d.get('n_tok')}")

if times:
    print(f"\nLoad time  — mean={statistics.mean(times):.3f}s  "
          f"stddev={statistics.stdev(times) if len(times)>1 else 0.0:.3f}s  "
          f"min={min(times):.3f}s  max={max(times):.3f}s")
    print(f"Throughput — mean={statistics.mean(tps_list):.1f}  "
          f"stddev={statistics.stdev(tps_list) if len(tps_list)>1 else 0.0:.1f}  "
          f"min={min(tps_list):.1f}  max={max(tps_list):.1f}")
