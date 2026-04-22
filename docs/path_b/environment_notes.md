# AgriBrain Environment Notes

Recorded: 2026-04-22

## Dependency management

The backend uses `agri-brain-mvp-1.0.0/backend/pyproject.toml` with pinned
version ranges (upper and lower bounds on every direct dependency). There
is no full lockfile (`poetry.lock` / `pip-compile`'d `requirements.lock`).
The WARN in `pre_hpc_check_report.md` (Section 2.2) captures this.

Why no lockfile: the tool that would generate one from this repo is pip's
`pip freeze`, but running it on Windows produces a Windows-specific pin
list (platform-tagged wheels, `pywin32`, etc.) that fails or drifts on a
Linux HPC node. Cross-generating a Linux lock from Windows is not
something pip supports cleanly. Per the pre-HPC verification prompt,
generating a lockfile from a heterogeneous environment is worse than
having none, so this gap is documented rather than papered over.

## HPC reproducibility path

1. The SLURM orchestrator (`hpc_run.sh`) installs the backend with
   `pip install -e agri-brain-mvp-1.0.0/backend` on the login node. pip
   resolves each range against the Linux wheel index.
2. The Path B load assertion in `hpc_run.sh`, `hpc_seed.sh`, and
   `hpc_aggregate.sh` confirms that `yield_query` is present and
   `THETA_CONTEXT` has shape `(3, 6)` before any compute is spent. If the
   resolver pulls a broken combination, this fails fast.
3. If the HPC run needs to be frozen for archival, regenerate a Linux
   lockfile from the HPC login node after the first successful install:

   ```bash
   source .venv/bin/activate
   pip freeze > agri-brain-mvp-1.0.0/backend/requirements.linux.lock
   ```

   Commit that file. Future runs can then `pip install -r
   requirements.linux.lock` instead of `pip install -e`.

## Local development (Windows)

For a Windows developer, Python 3.13 with the versions currently resolved
against `pyproject.toml` works. As of the 2026-04-22 pre-HPC audit, those
are:

- fastapi 0.135.1
- uvicorn 0.42.0
- pydantic 2.10.3
- numpy 2.1.3
- pandas 2.2.3
- matplotlib 3.10.0
- reportlab 4.4.10
- orjson 3.11.7
- requests 2.32.3
- web3 7.14.1
- python-multipart 0.0.22
- pyyaml 6.0.2
- pytest 8.3.4

These are indicative, not authoritative. The HPC node resolves against
its own wheel index.
