# Reproduction

## Verify the published results

From the repository root:

```bash
python scripts/verify_package.py
```

It needs only the Python standard library. It checks the hash of every file, each published table against [`data/study/MANIFEST.json`](../data/study/MANIFEST.json), and the coverage of modes, seeds, scenarios and settings in the per-seed endpoints. From those endpoints it recomputes the principal comparison, the adjustment at the guard and the gain in every sensitivity setting, and it checks the figures, the figure inputs and the relative links of the documentation. No simulation is run.

## Regenerate the figures and the dashboard data

```bash
python -m pip install -r analysis/requirements.txt
python analysis/make_figures.py --output figures_regenerated --compare
python analysis/export_dashboard_data.py --check
```

`make_figures.py` runs the scripts in `analysis/figures` on `data/study` and `data/figure_inputs` and, with `--compare`, reports for each figure whether it is byte-identical to the file in `figures/`. With the versions in `analysis/requirements.txt` and the Arial font it is; elsewhere only the rendering differs. `export_dashboard_data.py --check` confirms that the dashboard's results file matches `data/study`.

## Rerun the study

The design, the comparisons, the analysis and the commands that run the study on a Slurm cluster are in [study/PROTOCOL.md](../study/PROTOCOL.md), Section 11. The study has 6,300 tasks over 63 settings; `COMPLETE.json` in the results folder marks a finished analysis, whose tables are those of which `data/study` is the published subset. A rerun from this repository records its own commit and does not replace the published results.

The repository is the source the study ran from, so it also holds the files of the submitted configuration:

- `reproduction/` holds the engine and harness of the submitted configuration. The study reads `reproduction/primary_reference.json`, the stored results of that configuration, and checks its submitted setting against them in every seed–scenario cell.
- `hpc/` holds the publication environment and the source and evidence validators that the study scripts call.
- `validation/` holds the interface and frozen-control scripts of the submitted configuration; they need that configuration's raw results, which are not part of this repository.
- `analysis/primary_statistics.py` holds the interval and test procedure that `study/inference.py` uses. Its command-line check reads the submitted configuration's tables, which are not part of this repository either.

## Environment

The simulations use Python 3.11 and the dependency lock in `agribrain/backend/requirements-lock.txt`; `study/setup.sh` builds that environment. A full study needs a writable project or scratch directory with ample quota. Cluster account and partition options are site-specific.
