# Methods Repro Appendix (Publication Gate)

**Project:** AGRI-BRAIN  
**Repository:** `C:/AgriBrain`  
**Run mode:** Deterministic gate (`DETERMINISTIC_MODE=true`)  
**Date:** 2026-04-06

## 1) Reproducibility Objective

This appendix documents the exact deterministic reproduction workflow used to regenerate core simulation outputs, verify metric constraints, run regression checks, execute robustness stress tests, and produce publication figures and evidence artifacts.

## 2) Exact Command Chain

From repository root:

```bash
python mvp/simulation/generate_results.py
python mvp/simulation/validate_results.py
python mvp/simulation/run_regression_guard.py
python mvp/simulation/run_stress_suite.py
python mvp/simulation/generate_figures.py
python mvp/simulation/aggregate_seeds.py
python mvp/simulation/export_paper_evidence.py
python mvp/simulation/build_artifact_manifest.py
```

Additional deterministic guard initialization used once for baseline synchronization:

```bash
REGRESSION_GUARD_INIT=true python mvp/simulation/run_regression_guard.py
python mvp/simulation/run_regression_guard.py
```

## 3) Acceptance Criteria and Gate Status

- **Deterministic validation rules (`validate_results.py`)**: **PASS**
- **Regression drift guard (`run_regression_guard.py`)**: **PASS**
- **Stress robustness suite (`run_stress_suite.py`)**: **PASS**
- **Figure generation (Fig2–Fig10)**: **PASS**
- **Evidence export (`export_paper_evidence.py`)**: **PASS**
- **Artifact hash manifest (`build_artifact_manifest.py`)**: **PASS**

Gate-discovered failures were corrected during run:
- Fault-injection null handling in `backend/pirag/context_builder.py`
- Fig9 CI null-safe plotting in `mvp/simulation/generate_figures.py`

## 4) Deterministic Core Results (Table 1, AGRI-BRAIN rows)

From `mvp/simulation/results/table1_summary.csv`:

| Scenario | ARI | RLE | Waste | SLCA | Carbon | Equity |
|---|---:|---:|---:|---:|---:|---:|
| heatwave | 0.614 | 0.994 | 0.021 | 0.756 | 2108.0 | 0.900 |
| overproduction | 0.632 | 0.985 | 0.035 | 0.726 | 2033.0 | 0.899 |
| cyber_outage | 0.649 | 0.808 | 0.033 | 0.733 | 2407.0 | 0.861 |
| adaptive_pricing | 0.742 | 0.909 | 0.019 | 0.805 | 2006.0 | 0.887 |
| baseline | 0.760 | 0.960 | 0.018 | 0.817 | 1971.0 | 0.884 |

## 5) Benchmark Summary (Multi-seed Evidence)

From `mvp/simulation/results/benchmark_summary.json` (AGRI-BRAIN ARI means, n=5):

- `heatwave`: mean `0.6175`, 95% CI `[0.6139, 0.6218]`
- `overproduction`: mean `0.6383`, 95% CI `[0.6352, 0.6417]`
- `cyber_outage`: mean `0.6646`, 95% CI `[0.6586, 0.6707]`
- `adaptive_pricing`: mean `0.7387`, 95% CI `[0.7308, 0.7469]`
- `baseline`: mean `0.7541`, 95% CI `[0.7513, 0.7569]`

## 6) Artifact Integrity (SHA-256)

From `mvp/simulation/results/artifact_manifest.json`:

- `artifact_manifest.json`: `fff742c8825ad9666e07708e5c876c4ebe685f5c72a2f5a13c0faa0f86859608`
- `table1_summary.csv`: `fa9302da1bc047f758cba4f09965b80a59130cc4fad12e674a970dc4a1d228bb`
- `table2_ablation.csv`: `9888eaf05b260562d38d0b7940d0298175b313996763d27ba799623b5e527932`
- `benchmark_summary.json`: `f2d55eb0e7e2a1d99d5f3fd5e6658804955749fdfdd5ee0e5d2db12f23d88e3a`
- `benchmark_significance.json`: `258c3ce23aa90ebac96e5c3a7cff3c567b2a0bc2145cd5a67618a4c4d6fe7349`
- `stress_summary.json`: `bbb92097c5e3e020415b9eaeec622c2ce57337be900404ca7459e8ce51e84f21`
- `stress_degradation.csv`: `6d4e6cf36f768e725d1623a5660849177443eba44e6b4cc400efc185b64c2076`
- `fig9_mcp_pirag_robustness.png`: `7e32d5e20e695223616e827a9da940347460be339605aa1ad4d9f32cbb2cfb7e`
- `fig10_latency_quality_frontier.png`: `8c97cba79c7d175afc59040c1687ce864f30d6904beddfb3fd1e3811be184425`
- `paper_benchmark_table.json`: `9e13a3ef7f8101976c43d813eed08eff0b174e91b6fb9e24cb5d355cacd8a981`

## 7) Sign-off

**Publication reproducibility gate:** ✅ **PASSED**  
All required deterministic checks, stress robustness runs, figures, evidence exports, and hash-manifested artifacts were regenerated and verified in the current repository state.

