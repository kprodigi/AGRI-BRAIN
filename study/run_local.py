"""Run study tasks on one machine with several worker processes.

usage: run_local.py --output DIR --indices 0-99,200,305 [--workers N]
                    [--development-runtime DIR]

Each task is executed by ``study.py task`` in its own process, exactly as a
cluster array task would be. A task that is already complete is verified and
skipped. The script exits with status 1 if any task failed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_indices(text):
    indices = []
    for part in text.split(','):
        if '-' in part:
            first, last = part.split('-')
            indices.extend(range(int(first), int(last) + 1))
        elif part:
            indices.append(int(part))
    return sorted(set(indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--development-runtime', type=Path)
    args = parser.parse_args()
    base = [sys.executable, str(HERE / 'study.py')]
    if args.development_runtime:
        base += ['--development-runtime', str(args.development_runtime)]
    logs = args.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    def one(index):
        started = time.monotonic()
        proc = subprocess.run(base + ['task', '--output', str(args.output), '--index', str(index)],
                              capture_output=True, text=True)
        (logs / f'task_{index:04d}.out').write_text(proc.stdout, encoding='utf-8')
        (logs / f'task_{index:04d}.err').write_text(proc.stderr, encoding='utf-8')
        last = (proc.stdout.strip().splitlines() or [''])[-1]
        if proc.returncode != 0:
            last = 'FAILED: ' + (proc.stderr.strip().splitlines() or ['no error text'])[-1]
        return index, proc.returncode, time.monotonic() - started, last

    indices = parse_indices(args.indices)
    failed = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, code, seconds, last in pool.map(one, indices):
            print(f'[{index:4d}] {seconds:6.0f}s {last[:150]}', flush=True)
            if code != 0:
                failed.append(index)
    print(f'{len(indices) - len(failed)} of {len(indices)} tasks complete in '
          f'{time.monotonic() - started:.0f}s; failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
