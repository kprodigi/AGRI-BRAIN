#!/usr/bin/env python3
"""Run one HPC worker command and atomically retain process/resource evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - publication HPC is Linux
    resource = None


_SLURM_FIELDS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_ARRAY_TASK_STEP",
    "SLURM_RESTART_COUNT",
    "SLURM_JOB_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_QOS",
    "SLURM_JOB_RESERVATION",
    "SLURM_JOB_CONSTRAINTS",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_GPUS_ON_NODE",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "SLURM_JOB_NUM_NODES",
    "SLURM_DISTRIBUTION",
    "SLURM_SUBMIT_HOST",
    "SLURM_SUBMIT_DIR",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _snapshot() -> dict[str, int | float] | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        field: getattr(usage, field)
        for field in (
            "ru_utime", "ru_stime", "ru_maxrss", "ru_minflt", "ru_majflt",
            "ru_inblock", "ru_oublock", "ru_nvcsw", "ru_nivcsw",
        )
    }


def _delta(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    result = {}
    for field, end in after.items():
        # ru_maxrss is a peak, not an additive cumulative counter.
        result[field] = end if field == "ru_maxrss" else end - before[field]
    return result


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime receipt: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.output.exists():
        raise FileExistsError(
            f"refusing to execute without a fresh runtime-receipt target: {args.output}"
        )

    start_utc = _utc_now()
    start_wall = time.perf_counter()
    before = _snapshot()
    completed = subprocess.run(command, check=False)
    after = _snapshot()
    wall_seconds = time.perf_counter() - start_wall
    payload = {
        "schema_version": 1,
        "label": args.label,
        "measurement_scope": (
            "child worker process from subprocess start through process exit; "
            "excludes scheduler queue time and post-exit publisher work"
        ),
        "utc_start": start_utc,
        "utc_end": _utc_now(),
        "wall_seconds": float(wall_seconds),
        "returncode": int(completed.returncode),
        "command": command,
        "cwd": str(Path.cwd()),
        "python_wrapper": sys.version,
        "resource_units": {
            "ru_utime": "seconds",
            "ru_stime": "seconds",
            "ru_maxrss": (
                "KiB on Linux; platform-defined elsewhere"
                if sys.platform.startswith("linux") else "platform-defined"
            ),
            "ru_minflt": "count",
            "ru_majflt": "count",
            "ru_inblock": "count",
            "ru_oublock": "count",
            "ru_nvcsw": "count",
            "ru_nivcsw": "count",
        },
        "resource_available": before is not None and after is not None,
        "resource_children_before": before,
        "resource_children_after": after,
        "resource_child_delta_or_peak": _delta(before, after),
        "slurm": {name: os.environ.get(name) for name in _SLURM_FIELDS},
        "run_identity": {
            "run_tag": os.environ.get("RUN_TAG", ""),
            "source_commit": os.environ.get("AGRIBRAIN_GIT_COMMIT", ""),
            "source_tree_sha256": os.environ.get("AGRIBRAIN_SOURCE_TREE_SHA256", ""),
            "source_snapshot": os.environ.get("AGRIBRAIN_SOURCE_SNAPSHOT", ""),
        },
        "interpretation": (
            "Observed runtime/resource evidence only. It is not an electricity, "
            "carbon, water, or monetary estimate without separately measured "
            "node power and site-specific conversion factors."
        ),
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _atomic_write(args.output, payload)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
