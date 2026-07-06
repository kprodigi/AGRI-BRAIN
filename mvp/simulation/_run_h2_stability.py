"""Cross-hash-seed stability check launcher.

Runs a small set of base seeds at several PYTHONHASHSEED values to bound how
much the channel-attribution flip rates depend on the hash-seed pin (the
simulator has hash-ordering nondeterminism in MCP dispatch / retrieval).

hash seed 0 is reused from the main run (decision_ledger_h2); this launcher
runs the NON-zero hash seeds into decision_ledger_h2_hs<N>/seed_<base>/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(Path(sys.executable))

BASE_SEEDS = [7, 99, 707, 2024]
HASH_SEEDS = [1, 2]            # hs0 reused from decision_ledger_h2
POOL = 8
STAGGER_S = 4.0

runner = str(HERE / "_run_h2_seed.py")
logdir = HERE / "results" / "_stability_logs"
logdir.mkdir(parents=True, exist_ok=True)

jobs = [(b, h) for h in HASH_SEEDS for b in BASE_SEEDS]
pending = list(jobs)
running = {}
done, failed = [], []
t0 = time.time()
_last = 0.0
print(f"Stability: {len(jobs)} runs ({len(BASE_SEEDS)} base x {len(HASH_SEEDS)} hash), pool={POOL}", flush=True)
while pending or running:
    now = time.time()
    if pending and len(running) < POOL and (now - _last) >= STAGGER_S:
        b, h = pending.pop(0)
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(h)
        env["DETERMINISTIC_MODE"] = "false"
        env["H2_LEDGER_ROOT"] = f"decision_ledger_h2_hs{h}"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = "1"
        lf = open(logdir / f"hs{h}_seed{b}.log", "w")
        p = subprocess.Popen([PY, runner, str(b)], stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[(b, h)] = (p, lf)
        _last = time.time()
        print(f"  [{time.time()-t0:6.0f}s] started base={b} hash={h} ({len(running)} running)", flush=True)
    time.sleep(2)
    for key in list(running):
        p, lf = running[key]
        rc = p.poll()
        if rc is None:
            continue
        lf.close(); del running[key]
        (done if rc == 0 else failed).append(key)
        print(f"  [{time.time()-t0:6.0f}s] {'DONE' if rc==0 else 'FAIL'} base={key[0]} hash={key[1]} ({len(done)}/{len(jobs)})", flush=True)
print(f"\nFinished in {time.time()-t0:.0f}s. done={len(done)} failed={failed}", flush=True)
