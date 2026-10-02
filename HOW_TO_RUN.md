# AGRI-BRAIN run guide

How to install the application, run the API and dashboard, and run the tests. For recomputing the reported statistics and for rerunning the experiments on Slurm, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## 1. Requirements

- Python 3.11 (the experiment scripts reject other Python minors)
- Node.js 22.12 or later for the dashboard only (required by the locked Vite toolchain)
- Git
- A Slurm cluster only when rerunning the experiments

The experiment scripts fix BLAS-related thread counts to one and record the interpreter, installed-package versions, environment contract, platform and source hashes. This is a version-resolved runtime inventory, not a claim of byte-identical wheels, BLAS binaries or a container image.

## 2. Install the backend

For ordinary development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "agribrain/backend[dev]"
```

On Windows PowerShell, use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e "agribrain/backend[dev]"
```

For the closest local match to the publication environment:

```bash
python3.11 -m venv .venv-publication
source .venv-publication/bin/activate
python -m pip install --upgrade pip
python -m pip install -r agribrain/backend/requirements-lock.txt
python -m pip install --no-deps -e agribrain/backend
```

## 3. Run the API and dashboard

Start the API from the repository root:

```bash
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100
```

Start the dashboard in another terminal:

```bash
cd agribrain/frontend
npm ci
npm run dev
```

Load the bundled synthetic telemetry trace and verify health:

```bash
curl -X POST http://127.0.0.1:8100/case/load
curl http://127.0.0.1:8100/health
```

## 4. Run tests

```bash
python -m pytest agribrain/backend/tests agribrain/backend/pirag/tests -q
```

The default selection excludes tests marked `slow`, and `test_install_imports.py` needs the backend installed as in section 2. The study's own checks are `scripts/verify_package.py`, `analysis/primary_statistics.py` and `reproduction/test_study.py`.

Internal paths and identifiers retained for compatibility may use historical names. User-facing output refers to institutional retrieval.

## 5. Rerun the experiments

The primary comparison and the weight-sensitivity study are rerun with the Slurm scripts described in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).
