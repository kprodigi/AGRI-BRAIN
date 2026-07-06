"""Parallel launcher for the instrumented 20-seed H2 channel-attribution run.

Runs ``_run_h2_seed.py <seed>`` for each canonical benchmark seed with a
bounded process pool. Each seed writes its own ledger directory, so there is
no cross-process file contention (mirrors the HPC per-seed isolation).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(Path(sys.executable))

# Reproducible per-seed environment: PYTHONHASHSEED pins dict/set iteration
# order (MCP tool dispatch / retrieval), DETERMINISTIC_MODE=false keeps the
# 8-source stochastic layer active (the canonical benchmark setting).
CHILD_ENV = dict(os.environ)
CHILD_ENV["PYTHONHASHSEED"] = "0"
CHILD_ENV["DETERMINISTIC_MODE"] = "false"
# Pin BLAS/OpenMP to one thread per process: the sim's matrix ops are tiny
# (3x10, 3x5) so multi-threaded BLAS only oversubscribes cores when many
# seeds run in parallel. One thread per process keeps the pool CPU-efficient.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    CHILD_ENV[_k] = "1"

# Canonical 20-seed benchmark seeds (from benchmark_summary.json _meta).
SEEDS = [7, 42, 99, 101, 202, 303, 404, 505, 606, 707,
         808, 909, 1010, 1111, 1212, 1313, 1337, 1414, 1515, 2024]

POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
logdir = HERE / "results" / "decision_ledger_h2" / "_logs"
logdir.mkdir(parents=True, exist_ok=True)

runner = str(HERE / "_run_h2_seed.py")
pending = list(SEEDS)
running = {}   # seed -> (Popen, logfile handle)
done, failed = [], []
t0 = time.time()
# Stagger launches so shared first-import caches (matplotlib font cache, KB /
# TF-IDF index build) are warmed by the first process before the pool ramps;
# concurrent cold-cache builds were the cause of the pool=20 init crash.
STAGGER_S = 4.0
_last_launch = 0.0

print(f"Launching {len(SEEDS)} seeds, pool={POOL}", flush=True)
while pending or running:
    now = time.time()
    while pending and len(running) < POOL and (now - _last_launch) >= STAGGER_S:
        seed = pending.pop(0)
        lf = open(logdir / f"seed_{seed}.log", "w")
        p = subprocess.Popen([PY, runner, str(seed)], stdout=lf,
                             stderr=subprocess.STDOUT, env=CHILD_ENV)
        running[seed] = (p, lf)
        _last_launch = time.time()
        print(f"  [{time.time()-t0:6.0f}s] started seed {seed} ({len(running)} running)", flush=True)
        break  # one launch per loop tick so the stagger gate applies
    time.sleep(2)
    for seed in list(running):
        p, lf = running[seed]
        rc = p.poll()
        if rc is None:
            continue
        lf.close()
        del running[seed]
        if rc == 0:
            done.append(seed)
            print(f"  [{time.time()-t0:6.0f}s] DONE seed {seed} ({len(done)}/{len(SEEDS)})", flush=True)
        else:
            failed.append(seed)
            print(f"  [{time.time()-t0:6.0f}s] FAILED seed {seed} rc={rc}", flush=True)

print(f"\nFinished in {time.time()-t0:.0f}s. done={len(done)} failed={failed}", flush=True)
