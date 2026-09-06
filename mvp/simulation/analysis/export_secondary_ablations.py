#!/usr/bin/env python3
"""Export the prespecified, seed-paired secondary AGRI-BRAIN ablations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from ..benchmarks.aggregate_seeds import (
        _BCA_STATS,
        _WILCOXON_FALLBACK_CELLS,
        SCENARIOS,
        SEEDS,
        _ci_method_since,
        _resampling_identity,
        _reset_bca_fallback_stats,
        benjamini_yekutieli,
        bootstrap_mean_diff_ci,
        cohens_dz,
        hedges_g,
        wilcoxon_signed_rank_pvalue,
    )
except ImportError:
    import sys

    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from mvp.simulation.benchmarks.aggregate_seeds import (  # noqa: E402
        _BCA_STATS,
        _WILCOXON_FALLBACK_CELLS,
        SCENARIOS,
        SEEDS,
        _ci_method_since,
        _resampling_identity,
        _reset_bca_fallback_stats,
        benjamini_yekutieli,
        bootstrap_mean_diff_ci,
        cohens_dz,
        hedges_g,
        wilcoxon_signed_rank_pvalue,
    )


REFERENCE_MODE = "agribrain"
COMPARISONS = (
    ("standard_rag", "agribrain_standard_rag"),
    ("no_peer", "agribrain_no_peer"),
    ("sign_unconstrained", "agribrain_sign_unconstrained"),
)
JSON_NAME = "secondary_ablation_analysis.json"
CSV_NAME = "secondary_ablation_analysis.csv"
N_BOOT = 10_000
ALPHA = 0.05


def _load_panel(seed_root: Path, seeds: tuple[int, ...]) -> tuple[dict[int, dict], dict]:
    expected = {f"seed_{seed}.json" for seed in seeds}
    actual = {path.name for path in seed_root.glob("seed_*.json") if path.is_file()}
    if actual != expected:
        raise RuntimeError(
            f"secondary-ablation seed panel mismatch: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    panel: dict[int, dict] = {}
    provenance: dict[str, set[Any]] = {"source_commit": set(), "run_tag": set()}
    for seed in seeds:
        path = seed_root / f"seed_{seed}.json"
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value, path=path: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r} in {path}")
            ),
        )
        if payload.get("seed") != seed:
            raise RuntimeError(f"seed identity mismatch in {path}: {payload.get('seed')!r}")
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, dict):
            raise RuntimeError(f"missing scenarios object in {path}")
        if set(scenarios) != set(SCENARIOS):
            raise RuntimeError(f"scenario panel mismatch in {path}")
        meta = payload.get("_meta", {})
        if isinstance(meta, dict):
            for key in provenance:
                value = meta.get(key) or meta.get("git_commit" if key == "source_commit" else key)
                if value is not None:
                    provenance[key].add(value)
        for scenario in SCENARIOS:
            for mode in (REFERENCE_MODE, *(mode for _, mode in COMPARISONS)):
                record = scenarios[scenario].get(mode)
                value = record.get("ari") if isinstance(record, dict) else None
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite ARI in {path}: {scenario}/{mode}")
        panel[seed] = scenarios
    for key, values in provenance.items():
        if len(values) > 1:
            raise RuntimeError(f"mixed {key} values in seed panel: {sorted(values)}")
    return panel, {key: next(iter(values), None) for key, values in provenance.items()}


def _wilcoxon_statistic(a: list[float], b: list[float]) -> float:
    differences = [x - y for x, y in zip(a, b, strict=True) if x != y]
    if len(differences) < 2:
        return 0.0
    from scipy.stats import wilcoxon

    return float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto").statistic)


def analyse(seed_root: Path, seeds: tuple[int, ...] = tuple(SEEDS)) -> dict[str, Any]:
    panel, provenance = _load_panel(seed_root.resolve(), seeds)
    _reset_bca_fallback_stats()
    _WILCOXON_FALLBACK_CELLS.clear()
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        records: list[dict[str, Any]] = []
        raw_p: dict[str, float] = {}
        for label, comparator_mode in COMPARISONS:
            a = [float(panel[seed][scenario][REFERENCE_MODE]["ari"]) for seed in seeds]
            b = [float(panel[seed][scenario][comparator_mode]["ari"]) for seed in seeds]
            cell_key = ("secondary_ablation", scenario, label, "ari")
            before = dict(_BCA_STATS)
            ci_low, ci_high = bootstrap_mean_diff_ci(
                a, b, n_boot=N_BOOT, alpha=ALPHA, paired=True, cell_key=cell_key
            )
            ci_method = _ci_method_since(before, ci_low, ci_high)
            p_value = wilcoxon_signed_rank_pvalue(
                a, b, cell_key=cell_key, alternative="two-sided"
            )
            raw_p[label] = p_value
            records.append({
                "comparison": label,
                "reference_mode": REFERENCE_MODE,
                "comparator_mode": comparator_mode,
                "metric": "ari",
                "n_pairs": len(seeds),
                "reference_mean": float(sum(a) / len(a)),
                "comparator_mean": float(sum(b) / len(b)),
                "mean_paired_difference": float(
                    sum(x - y for x, y in zip(a, b, strict=True)) / len(a)
                ),
                "mean_paired_difference_ci_low": ci_low,
                "mean_paired_difference_ci_high": ci_high,
                "mean_paired_difference_ci_method": ci_method,
                "wilcoxon_statistic": _wilcoxon_statistic(a, b),
                "p_value_raw_two_sided": p_value,
                "cohens_dz": cohens_dz(a, b),
                "hedges_g_paired": hedges_g(a, b, paired=True),
            })
        adjusted = benjamini_yekutieli(raw_p)
        for record in records:
            record["p_value_adj_by_within_scenario"] = adjusted[record["comparison"]]
        by_scenario[scenario] = records
    return {
        "_meta": {
            "schema_version": 1,
            "status": "prespecified_diagnostic_not_confirmatory",
            "source_commit": provenance["source_commit"],
            "run_tag": provenance["run_tag"] or os.environ.get("RUN_TAG") or None,
            "metric": "ari",
            "unit": "seed",
            "paired": True,
            "seed_order": list(seeds),
            "n_seeds": len(seeds),
            "scenarios": list(SCENARIOS),
            "reference_mode": REFERENCE_MODE,
            "comparators": [mode for _, mode in COMPARISONS],
            "test": "two_sided_paired_wilcoxon",
            "multiplicity": "benjamini_yekutieli_within_scenario_across_exactly_three_contrasts",
            "confidence_interval": "95_percent_bca_paired_bootstrap_interval",
            "bootstrap_resamples": N_BOOT,
            "resampling_rng": _resampling_identity(list(seeds)),
            "claim_rule": "direction_and_magnitude_are_observed; no_required_superiority",
            "wilcoxon_fallback_cells": [list(cell) for cell in sorted(_WILCOXON_FALLBACK_CELLS)],
        },
        "by_scenario": by_scenario,
    }


CSV_FIELDS = (
    "scenario", "comparison", "reference_mode", "comparator_mode", "metric",
    "n_pairs", "reference_mean", "comparator_mean", "mean_paired_difference",
    "mean_paired_difference_ci_low", "mean_paired_difference_ci_high",
    "mean_paired_difference_ci_method", "wilcoxon_statistic",
    "p_value_raw_two_sided", "p_value_adj_by_within_scenario", "cohens_dz",
    "hedges_g_paired",
)


def write_outputs(payload: dict[str, Any], json_output: Path, csv_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for scenario in SCENARIOS:
            for record in payload["by_scenario"][scenario]:
                writer.writerow({"scenario": scenario, **record})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = analyse(args.seed_root)
    write_outputs(payload, args.output_dir / JSON_NAME, args.output_dir / CSV_NAME)
    print(f"Saved {args.output_dir / JSON_NAME}")
    print(f"Saved {args.output_dir / CSV_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
