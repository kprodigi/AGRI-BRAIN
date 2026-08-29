#!/usr/bin/env python3
"""Validate required publication artifacts and schema fields.

Fails fast when key reproducibility/statistics fields are missing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPO_ROOT = RESULTS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from mvp.simulation.analysis.recovery_provenance import (  # noqa: E402
    validate_recovery_context,
)

VALIDATION_RECEIPT_NAME = "publication_validation_receipt.json"
RECOVERY_RECEIPT_PATH: Path | None = None
_RUN_TAG_RE = re.compile(r"^([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$")
EXPECTED_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)
EXPECTED_SCENARIOS = (
    "heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline",
)
EXPECTED_STRESSORS = (
    "sensor_noise", "missing_data", "telemetry_delay",
    "mcp_fault_injection", "compounded",
)
EXPECTED_MODES = (
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "mcp_only",
    "pirag_only", "agribrain", "agribrain_standard_rag",
    "agribrain_no_peer", "agribrain_sign_unconstrained",
)
EXPECTED_STRESS_TASK_FILES = (
    "stress_summary.json", "stress_degradation.csv",
    "stress_passfail.csv", "stress_h3_test.json",
)
EXPECTED_TOP_LEVEL_ARTIFACTS = {
    "benchmark_summary.json", "benchmark_significance.json",
    "h2_directional_evidence.csv",
    "table1_summary.csv", "table2_ablation.csv",
    "secondary_ablation_analysis.json", "secondary_ablation_analysis.csv",
    "channel_attribution_aggregate.json",
    "channel_complementarity_test.json",
    "channel_saturation_analysis.json", "explainability_metrics.json",
    "stress_summary.json", "stress_degradation.csv",
    "stress_passfail.csv", "stress_h3_test.json",
    "paper_benchmark_table.json", "publication_environment.json",
    "forecast_validation_summary.json", "forecast_validation_predictions.csv",
    "figure_provenance.json",
    "heatwave.png", "heatwave.pdf",
    "overproduction.png", "overproduction.pdf",
    "cyber_outage.png", "cyber_outage.pdf",
    "adaptive_pricing.png", "adaptive_pricing.pdf",
    "cross_scenario.png", "cross_scenario.pdf",
    "ablation.png", "ablation.pdf",
    "transport_emissions.png", "transport_emissions.pdf",
    "performance_efficiency.png", "performance_efficiency.pdf",
    "context_value.png", "context_value.pdf",
    "stress_robustness.png", "stress_robustness.pdf",
}
EXPECTED_STRESS_THRESHOLDS = {
    "ari_abs_delta_max": 0.01,
    "waste_delta_max": 0.04,
    "slca_delta_min": -0.10,
    "rle_delta_min": -0.12,
    "carbon_delta_max": 250.0,
    "equity_delta_min": -0.06,
    "constraint_violation_delta_max": 0.15,
    "latency_ms_delta_max": 100.0,
}
EXPECTED_FIGURE_AGGREGATE_INPUTS = (
    "benchmark_summary.json",
    "benchmark_significance.json",
    "channel_attribution_aggregate.json",
    "stress_passfail.csv",
)
EXPECTED_DERIVED_REPLAY_ARTIFACTS = (
    "forecast_validation_summary.json",
    "forecast_validation_predictions.csv",
    "channel_attribution_aggregate.json",
    "channel_complementarity_test.json",
    "channel_saturation_analysis.json",
    "explainability_metrics.json",
)
H2_PUBLICATION_COLUMNS = (
    "source_commit", "run_tag", "scenario", "comparison",
    "numerator_mode", "denominator_mode", "direction", "endpoint",
    "n_seeds", "paired_design", "test", "alternative",
    "mean_difference", "mean_difference_ci_low",
    "mean_difference_ci_high", "mean_difference_ci_method",
    "cohens_dz", "cohens_dz_ci_low", "cohens_dz_ci_high",
    "cohens_dz_ci_method", "raw_directional_p_value",
    "holm_adjusted_p_value", "holm_family_size", "alpha",
    "positive_mean", "cell_supported",
)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _load_json(path: Path) -> Any:
    if not path.exists():
        _fail(f"Missing required file: {path}")
    try:
        def _reject_constant(value: str):
            raise ValueError(f"non-finite JSON constant {value!r}")

        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
        )
    except Exception as exc:
        _fail(f"Invalid JSON in {path}: {exc}")


def _safe_manifest_payload(raw_name: object) -> Path:
    """Resolve one manifested result as an in-tree, non-symlink regular file."""

    if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
        _fail(f"artifact manifest contains an unsafe payload path: {raw_name!r}")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        _fail(f"artifact manifest contains an unsafe payload path: {raw_name!r}")
    base = RESULTS_DIR.resolve()
    source = RESULTS_DIR.joinpath(*relative.parts)
    cursor = source
    while cursor != RESULTS_DIR:
        if cursor.is_symlink():
            _fail(f"artifact manifest payload traverses a symlink: {raw_name}")
        cursor = cursor.parent
    try:
        resolved = source.resolve(strict=True)
    except OSError:
        _fail(f"artifact_manifest.json lists a missing payload: {raw_name}")
    if not resolved.is_relative_to(base) or not resolved.is_file():
        _fail(f"artifact manifest payload escapes results or is irregular: {raw_name}")
    return resolved


def _as_bool(value: Any, *, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    _fail(f"{where} is not a Boolean value")


def _recompute_tost(values: list[float], margin: float = 0.01) -> dict[str, Any]:
    """Independent one-sample t-TOST recomputation for the artifact gate."""
    import math

    from scipy.stats import t as student_t

    if len(values) != len(EXPECTED_SEEDS) or not all(math.isfinite(v) for v in values):
        _fail("TOST recomputation requires the exact finite 20-seed panel")
    n = len(values)
    mean = sum(values) / n
    sd = (sum((value - mean) ** 2 for value in values) / (n - 1)) ** 0.5
    se = sd / (n ** 0.5)
    df = n - 1
    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci90_low = ci90_high = mean
        ci95_low = ci95_high = mean
        p_two_sided = 0.0 if mean != 0.0 else 1.0
    else:
        p_lower = float(1.0 - student_t.cdf((mean + margin) / se, df))
        p_upper = float(student_t.cdf((mean - margin) / se, df))
        critical = float(student_t.ppf(0.95, df))
        ci90_low = mean - critical * se
        ci90_high = mean + critical * se
        critical95 = float(student_t.ppf(0.975, df))
        ci95_low = mean - critical95 * se
        ci95_high = mean + critical95 * se
        p_two_sided = float(2.0 * student_t.sf(abs(mean / se), df))
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05
    positive = p_two_sided < 0.05 and mean > 0.0
    negative = p_two_sided < 0.05 and mean < 0.0
    if equivalent and positive:
        verdict = "positive_but_equivalent"
    elif equivalent and negative:
        verdict = "negative_but_equivalent"
    elif equivalent:
        verdict = "equivalent_within_margin"
    elif positive:
        verdict = "positive_difference"
    elif negative:
        verdict = "negative_difference"
    else:
        verdict = "inconclusive"
    return {
        "mean": mean,
        "p_tost": p_tost,
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p_two_sided": p_two_sided,
        "equivalent": equivalent,
        "verdict": verdict,
    }


def _load_raw_benchmark_panel() -> dict[int, dict[str, Any]]:
    """Load the exact flat seed panel used by the publication aggregators."""
    seed_root = RESULTS_DIR / "benchmark_seeds"
    expected_names = {f"seed_{seed}.json" for seed in EXPECTED_SEEDS}
    found_names = {path.name for path in seed_root.glob("seed_*.json")}
    if found_names != expected_names:
        _fail(
            "raw benchmark seed inventory differs from the declared panel: "
            f"missing={sorted(expected_names - found_names)}, "
            f"unexpected={sorted(found_names - expected_names)}"
        )
    panel: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        path = seed_root / f"seed_{seed}.json"
        payload = _load_json(path)
        if not isinstance(payload, dict) or payload.get("seed") != seed:
            _fail(f"{path.name} does not contain its filename seed")
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, dict) or set(scenarios) != set(
            EXPECTED_SCENARIOS
        ):
            _fail(f"{path.name} does not contain the exact scenario panel")
        panel[seed] = scenarios
    return panel


def _raw_metric(
    panel: dict[int, dict[str, Any]], scenario: str, mode: str, metric: str,
) -> list[float]:
    import math

    values: list[float] = []
    for seed in EXPECTED_SEEDS:
        try:
            value = float(panel[seed][scenario][mode][metric])
        except (KeyError, TypeError, ValueError):
            _fail(f"raw seed {seed}/{scenario}/{mode}/{metric} is missing or invalid")
        if not math.isfinite(value):
            _fail(f"raw seed {seed}/{scenario}/{mode}/{metric} is non-finite")
        values.append(value)
    return values


def _raw_wilcoxon(
    a: list[float], b: list[float], *, alternative: str = "two-sided",
) -> float:
    """Independently reproduce a declared paired Wilcoxon rule."""
    import numpy as np
    from scipy.stats import wilcoxon

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unsupported Wilcoxon alternative: {alternative!r}")
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = differences[differences != 0.0]
    if len(nonzero) < 2:
        return 1.0
    return float(wilcoxon(
        nonzero, zero_method="wilcox", alternative=alternative, method="auto",
    ).pvalue)


def _validate_h1_h2_against_raw() -> None:
    """Bind every stored H1/H2 contrast to the raw seed envelopes."""
    import math

    panel = _load_raw_benchmark_panel()
    payload = _load_json(RESULTS_DIR / "benchmark_significance.json")
    data = payload.get("significance") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        _fail("benchmark_significance.json lacks its significance panel")
    comparison_modes = {
        "agribrain_vs_mcp_only": ("agribrain", "mcp_only"),
        "agribrain_vs_pirag_only": ("agribrain", "pirag_only"),
        "agribrain_vs_no_context": ("agribrain", "no_context"),
        "agribrain_vs_no_slca": ("agribrain", "no_slca"),
        "agribrain_vs_hybrid_rl": ("agribrain", "hybrid_rl"),
        "agribrain_vs_static": ("agribrain", "static"),
        "mcp_only_vs_no_context": ("mcp_only", "no_context"),
        "pirag_only_vs_no_context": ("pirag_only", "no_context"),
    }
    h2_directional_comparisons = {
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
    }
    metrics = ("ari", "waste", "rle", "slca", "carbon", "equity")
    for scenario in EXPECTED_SCENARIOS:
        for comparison, (left_mode, right_mode) in comparison_modes.items():
            for metric in metrics:
                left = _raw_metric(panel, scenario, left_mode, metric)
                right = _raw_metric(panel, scenario, right_mode, metric)
                expected_mean = sum(
                    a - b for a, b in zip(left, right, strict=True)
                ) / len(left)
                expected_p = _raw_wilcoxon(left, right)
                try:
                    record = data[scenario][comparison][metric]
                    stored_mean = float(record["mean_diff"])
                    stored_p = float(record["p_value"])
                except (KeyError, TypeError, ValueError):
                    _fail(f"{scenario}/{comparison}/{metric} is missing")
                if not math.isclose(
                    stored_mean, expected_mean, rel_tol=1e-12, abs_tol=1e-14,
                ):
                    _fail(
                        f"{scenario}/{comparison}/{metric} mean_diff "
                        "disagrees with raw seeds"
                    )
                if not math.isclose(
                    stored_p, expected_p, rel_tol=1e-12, abs_tol=1e-15,
                ):
                    _fail(
                        f"{scenario}/{comparison}/{metric} Wilcoxon p-value "
                        "disagrees with raw seeds"
                    )
                if metric == "ari" and (
                    comparison == "agribrain_vs_no_context"
                    or comparison in h2_directional_comparisons
                ):
                    expected_directional = _raw_wilcoxon(
                        left, right, alternative="greater",
                    )
                    try:
                        stored_directional = float(
                            record["p_value_directional_greater"]
                        )
                    except (KeyError, TypeError, ValueError):
                        _fail(
                            f"{scenario}/{comparison}/ari lacks its "
                            "directional Wilcoxon p-value"
                        )
                    if not math.isclose(
                        stored_directional, expected_directional,
                        rel_tol=1e-12, abs_tol=1e-15,
                    ):
                        _fail(
                            f"{scenario}/{comparison}/ari directional "
                            "Wilcoxon p-value disagrees with raw seeds"
                        )
                    if comparison == "agribrain_vs_no_context":
                        try:
                            h1_raw = float(
                                record["h1_raw_p_value_directional_greater"]
                            )
                        except (KeyError, TypeError, ValueError):
                            _fail(
                                f"{scenario}/{comparison}/ari lacks its H1 "
                                "directional audit field"
                            )
                        if not math.isclose(
                            h1_raw, expected_directional,
                            rel_tol=1e-12, abs_tol=1e-15,
                        ):
                            _fail(
                                f"{scenario}/{comparison}/ari H1 directional "
                                "audit field disagrees with raw seeds"
                            )

        full = _raw_metric(panel, scenario, "agribrain", "ari")
        mcp = _raw_metric(panel, scenario, "mcp_only", "ari")
        pirag = _raw_metric(panel, scenario, "pirag_only", "ari")
        no_context = _raw_metric(panel, scenario, "no_context", "ari")
        interactions = [
            a - b - c + d
            for a, b, c, d in zip(full, mcp, pirag, no_context, strict=True)
        ]
        expected_interaction_mean = sum(interactions) / len(interactions)
        expected_interaction_p = _raw_wilcoxon(
            interactions, [0.0] * len(interactions), alternative="greater",
        )
        try:
            interaction_record = data[scenario]["h2_synergy_interaction"]["ari"]
            stored_interaction_mean = float(
                interaction_record["mean_interaction"]
            )
            stored_interaction_p = float(
                interaction_record["p_value_directional_greater"]
            )
        except (KeyError, TypeError, ValueError):
            _fail(f"{scenario}/h2_synergy_interaction/ari is missing")
        if not math.isclose(
            stored_interaction_mean, expected_interaction_mean,
            rel_tol=1e-12, abs_tol=1e-14,
        ) or not math.isclose(
            stored_interaction_p, expected_interaction_p,
            rel_tol=1e-12, abs_tol=1e-15,
        ):
            _fail(
                f"{scenario}/h2_synergy_interaction/ari disagrees with raw seeds"
            )
    print(
        "[PASS] H1/H2 means, directional tests, and Wilcoxon tests "
        "recomputed from raw seeds"
    )


def _compare_reaggregated_core_artifacts(regenerated_dir: Path) -> None:
    """Require deterministic reaggregation to reproduce all core statistics."""

    json_names = (
        "benchmark_summary.json",
        "benchmark_significance.json",
        "secondary_ablation_analysis.json",
    )
    csv_names = (
        "table1_summary.csv", "table2_ablation.csv",
        "secondary_ablation_analysis.csv",
        "h2_directional_evidence.csv",
    )
    observed = {
        path.name for path in regenerated_dir.iterdir() if path.is_file()
    }
    expected = {*json_names, *csv_names}
    if observed != expected:
        _fail(
            "isolated statistical reaggregation produced an unexpected inventory: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for name in json_names:
        regenerated = _load_json(regenerated_dir / name)
        canonical = _load_json(RESULTS_DIR / name)
        # The isolated replay deliberately does not consume a recovery
        # authorization: it writes outside the canonical publication tree and
        # must never masquerade as another shippable recovery.  When validating
        # an authorized recovery, normalize only the provenance labels before
        # the exact statistical-payload comparison; all means, intervals,
        # tests, resampling identities, and tables remain byte-for-value exact.
        canonical_meta = (
            canonical.get("_meta") if isinstance(canonical, dict) else None
        )
        regenerated_meta = (
            regenerated.get("_meta") if isinstance(regenerated, dict) else None
        )
        if (
            isinstance(canonical_meta, dict)
            and canonical_meta.get("dual_provenance") is True
            and isinstance(regenerated_meta, dict)
        ):
            for key in (
                "git_commit",
                "source_commit",
                "simulation_source_commit",
                "analysis_code_commit",
                "dual_provenance",
                "recovery_authorization",
            ):
                regenerated_meta[key] = canonical_meta.get(key)
        if regenerated != canonical:
            _fail(
                f"{name} is not the deterministic reaggregation of the raw "
                "20-seed envelopes"
            )
    for name in csv_names:
        regenerated_bytes = (regenerated_dir / name).read_bytes()
        canonical_bytes = (RESULTS_DIR / name).read_bytes()
        if regenerated_bytes != canonical_bytes:
            _fail(
                f"{name} literal bytes are not the deterministic projection "
                "of the raw 20-seed envelopes"
            )


def _validate_reaggregated_core_statistics() -> None:
    """Rerun the complete prespecified aggregation in an isolated directory.

    This binds every mean, sample standard deviation, BCa interval, paired
    interval, effect size, raw/adjusted p-value, and both publication CSVs to
    the exact raw seed panel. It complements the independent H1/H2 spot
    recomputation above and rejects coherent edits to aggregate artifacts.
    """

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    source_commit = str(manifest.get("simulation_source_commit", ""))
    run_tag = str(manifest.get("artifact_run_tag", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        _fail("cannot reaggregate statistics without a full source commit")
    if _RUN_TAG_RE.fullmatch(run_tag) is None:
        _fail("cannot reaggregate statistics without a valid run tag")
    script = (
        REPO_ROOT / "mvp" / "simulation" / "benchmarks" / "aggregate_seeds.py"
    )
    if not script.is_file() or script.is_symlink():
        _fail("canonical statistical aggregator source is unavailable")
    env = os.environ.copy()
    env.update({
        "AGRIBRAIN_GIT_COMMIT": source_commit,
        "RUN_TAG": run_tag,
        "BENCHMARK_SEEDS": ",".join(str(seed) for seed in EXPECTED_SEEDS),
        "STRICT_VALIDATION": "1",
    })
    # Repair/dual-provenance inputs are forbidden for a fresh single-commit
    # release and must not influence the independent reaggregation process.
    for name in (
        "AGRIBRAIN_RECOVERY_RECEIPT",
        "AGRIBRAIN_PUBLICATION_CODE_COMMIT",
        "AGRIBRAIN_SIMULATION_COMMIT",
        "AGRIBRAIN_PUBLICATION_AGGREGATION",
    ):
        env.pop(name, None)
    with tempfile.TemporaryDirectory(
        prefix="agribrain_reaggregate_",
    ) as temporary_name:
        regenerated_dir = Path(temporary_name)
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--seed-root",
                str(RESULTS_DIR / "benchmark_seeds"),
                "--output-dir",
                str(regenerated_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=1_800,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
            _fail(
                "isolated 20-seed statistical reaggregation failed: "
                + diagnostic[-8000:]
            )
        _compare_reaggregated_core_artifacts(regenerated_dir)
    print(
        "[PASS] all core means, CIs, effect sizes, p-values, and CSV tables "
        "reaggregated from raw seeds"
    )


def _compare_exact_replay_artifacts(
    regenerated_dir: Path,
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require an isolated producer replay to match canonical literal bytes."""

    observed = {
        path.relative_to(regenerated_dir).as_posix()
        for path in regenerated_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected = set(names)
    if observed != expected:
        _fail(
            f"{label} replay produced an unexpected inventory: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for name in names:
        replayed = regenerated_dir / name
        canonical = RESULTS_DIR / name
        if replayed.is_symlink() or not replayed.is_file():
            _fail(f"{label} replay output is irregular: {name}")
        if replayed.read_bytes() != canonical.read_bytes():
            _fail(
                f"{name} literal bytes are not the deterministic {label} replay"
            )


def _run_isolated_producer(
    script: Path,
    arguments: list[str],
    *,
    env: dict[str, str],
    label: str,
) -> None:
    if script.is_symlink() or not script.is_file():
        _fail(f"{label} producer source is unavailable: {script}")
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=1_800,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
        _fail(f"isolated {label} replay failed: {diagnostic[-8000:]}")


def _validate_derived_evidence_replay() -> None:
    """Replay every non-core derived-evidence producer from raw inputs.

    Schema and internal-arithmetic checks cannot detect a coherent rewrite of
    an entire derived artifact.  These isolated reruns make the producer code
    and the manifested seed/ledger/data inputs authoritative, then compare the
    complete regenerated files byte for byte.
    """

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    source_commit = str(manifest.get("simulation_source_commit", "")).strip()
    run_tag = str(manifest.get("artifact_run_tag", "")).strip()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        _fail("derived-evidence replay requires a full simulation source commit")
    if _RUN_TAG_RE.fullmatch(run_tag) is None:
        _fail("derived-evidence replay requires a valid run tag")

    ledger_rel = Path(
        "mvp/simulation/results/decision_ledger_per_seed"
    ) / run_tag
    seed_rel = Path("mvp/simulation/results/benchmark_seeds")
    if (REPO_ROOT / ledger_rel).resolve() != (
        RESULTS_DIR / "decision_ledger_per_seed" / run_tag
    ).resolve():
        _fail("derived-evidence replay ledger root is not canonical")
    if (REPO_ROOT / seed_rel).resolve() != (
        RESULTS_DIR / "benchmark_seeds"
    ).resolve():
        _fail("derived-evidence replay seed root is not canonical")

    env = os.environ.copy()
    env.update({
        "AGRIBRAIN_GIT_COMMIT": source_commit,
        "RUN_TAG": run_tag,
        "ARTIFACT_RUN_TAG": run_tag,
        "STRICT_VALIDATION": "1",
        "AGRIBRAIN_PUBLICATION_REPLAY": "1",
    })
    env.pop("AGRIBRAIN_PUBLICATION_AGGREGATION", None)
    if manifest.get("dual_provenance") is True:
        if RECOVERY_RECEIPT_PATH is None:
            _fail("derived recovery replay lacks --recovery-receipt")
        env.update({
            "AGRIBRAIN_RECOVERY_RECEIPT": str(
                RECOVERY_RECEIPT_PATH.resolve(strict=True)
            ),
            "AGRIBRAIN_SIMULATION_COMMIT": source_commit,
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT": str(
                manifest.get("publication_code_commit", "")
            ),
        })
    else:
        for name in (
            "AGRIBRAIN_RECOVERY_RECEIPT",
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT",
            "AGRIBRAIN_SIMULATION_COMMIT",
        ):
            env.pop(name, None)

    with tempfile.TemporaryDirectory(
        prefix="agribrain_derived_replay_",
    ) as temporary_name:
        output_dir = Path(temporary_name)
        producers = (
            (
                REPO_ROOT / "mvp/simulation/benchmarks/aggregate_channel_attribution.py",
                [
                    "--ledger-root", ledger_rel.as_posix(),
                    "--output", str(output_dir / "channel_attribution_aggregate.json"),
                    "--modes", "agribrain",
                ],
                "channel-attribution",
            ),
            (
                REPO_ROOT / "mvp/simulation/_h2_permutation_test.py",
                [
                    "--ledger-root", ledger_rel.as_posix(),
                    "--output", str(output_dir / "channel_complementarity_test.json"),
                ],
                "channel-complementarity",
            ),
            (
                REPO_ROOT / "mvp/simulation/analysis/channel_saturation_analysis.py",
                [
                    "--seed-root", seed_rel.as_posix(),
                    "--output", str(output_dir / "channel_saturation_analysis.json"),
                    "--source-commit", source_commit,
                    "--run-tag", run_tag,
                ],
                "channel-saturation",
            ),
            (
                REPO_ROOT / "mvp/simulation/analysis/explainability_metrics.py",
                [
                    "--ledger", ledger_rel.as_posix(),
                    "--output", str(output_dir / "explainability_metrics.json"),
                    "--threshold", "0.10",
                ],
                "explainability",
            ),
            (
                REPO_ROOT / "mvp/simulation/validation/validate_forecasts.py",
                [
                    "--output-dir", str(output_dir),
                    "--source-commit", source_commit,
                    "--run-tag", run_tag,
                    "--publication-replay",
                ],
                "forecast-validation",
            ),
        )
        for script, arguments, label in producers:
            _run_isolated_producer(
                script, arguments, env=env, label=label,
            )
        _compare_exact_replay_artifacts(
            output_dir,
            EXPECTED_DERIVED_REPLAY_ARTIFACTS,
            label="derived-evidence producer",
        )
    print(
        "[PASS] forecast, attribution, complementarity, saturation, and "
        "explainability artifacts replay exactly from raw inputs"
    )


def _stage_manifested_h3_task_inputs(
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    """Copy the exact 5 x 4 manifested scenario-task byte inventory."""

    run_tag = str(manifest.get("artifact_run_tag", ""))
    expected = {
        f"stress_runs/{run_tag}/{scenario}/{name}"
        for scenario in EXPECTED_SCENARIOS
        for name in EXPECTED_STRESS_TASK_FILES
    }
    records = {
        str(record.get("file")): record
        for record in manifest.get("artifacts", [])
        if isinstance(record, dict) and str(record.get("file")) in expected
    }
    if set(records) != expected:
        _fail("manifest lacks the exact raw H3 scenario-task inventory")
    for manifest_name in sorted(expected):
        record = records[manifest_name]
        source = _safe_manifest_payload(manifest_name)
        payload = source.read_bytes()
        if (
            len(payload) != record.get("bytes")
            or hashlib.sha256(payload).hexdigest() != record.get("sha256")
        ):
            _fail(f"raw H3 task bytes changed during replay staging: {manifest_name}")
        relative = PurePosixPath(manifest_name)
        target = destination.joinpath(*relative.parts[2:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    observed = {
        f"stress_runs/{run_tag}/" + path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != expected:
        _fail(
            "isolated raw H3 staging has an unexpected inventory: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _validate_h3_aggregation_replay() -> None:
    """Rebuild all four top-level H3 artifacts from raw scenario tasks."""

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    source_commit = str(manifest.get("simulation_source_commit", "")).strip()
    run_tag = str(manifest.get("artifact_run_tag", "")).strip()
    env = os.environ.copy()
    env.update({
        "AGRIBRAIN_GIT_COMMIT": source_commit,
        "RUN_TAG": run_tag,
        "ARTIFACT_RUN_TAG": run_tag,
        "STRICT_VALIDATION": "1",
    })
    with tempfile.TemporaryDirectory(
        prefix="agribrain_h3_replay_",
    ) as temporary_name:
        temporary = Path(temporary_name)
        input_root = temporary / "raw_tasks"
        output_root = temporary / "regenerated"
        _stage_manifested_h3_task_inputs(input_root, manifest)
        _run_isolated_producer(
            REPO_ROOT / "mvp/simulation/benchmarks/aggregate_stress_outputs.py",
            [
                "--input-root", str(input_root),
                "--output-dir", str(output_root),
            ],
            env=env,
            label="H3 scenario-task aggregation",
        )
        _compare_exact_replay_artifacts(
            output_root,
            EXPECTED_STRESS_TASK_FILES,
            label="raw H3 scenario-task aggregation",
        )
    print(
        "[PASS] all four top-level H3 artifacts replay exactly from the "
        "manifested 5 x 4 raw task bytes"
    )


def _validate_channel_saturation_against_raw() -> None:
    """Recompute every inferential saturation TOST from raw seed ARI."""
    import math

    panel = _load_raw_benchmark_panel()
    artifact = _load_json(RESULTS_DIR / "channel_saturation_analysis.json")

    def compare(result: Any, differences: list[float], *, where: str) -> None:
        expected = _recompute_tost(differences, 0.01)
        fields = {
            "mean_diff": "mean",
            "ci90_low": "ci90_low",
            "ci90_high": "ci90_high",
            "ci95_low": "ci95_low",
            "ci95_high": "ci95_high",
            "p_two_sided": "p_two_sided",
            "p_tost": "p_tost",
        }
        if not isinstance(result, dict):
            _fail(f"{where} is missing")
        for stored_key, expected_key in fields.items():
            try:
                stored = float(result[stored_key])
            except (KeyError, TypeError, ValueError):
                _fail(f"{where}/{stored_key} is missing or invalid")
            if not math.isclose(
                stored, float(expected[expected_key]),
                rel_tol=1e-12, abs_tol=1e-14,
            ):
                _fail(f"{where}/{stored_key} disagrees with raw seeds")
        if result.get("verdict") != expected["verdict"]:
            _fail(f"{where}/verdict disagrees with raw seeds")

    by_scenario = artifact.get("by_scenario")
    if not isinstance(by_scenario, dict):
        _fail("channel saturation artifact lacks by_scenario")
    scenario_differences: dict[str, dict[str, list[float]]] = {}
    for scenario in EXPECTED_SCENARIOS:
        full = _raw_metric(panel, scenario, "agribrain", "ari")
        mcp = _raw_metric(panel, scenario, "mcp_only", "ari")
        pirag = _raw_metric(panel, scenario, "pirag_only", "ari")
        differences = {
            "add_pirag_on_mcp": [
                a - b for a, b in zip(full, mcp, strict=True)
            ],
            "add_mcp_on_pirag": [
                a - b for a, b in zip(full, pirag, strict=True)
            ],
        }
        scenario_differences[scenario] = differences
        for name, values in differences.items():
            compare(
                by_scenario[scenario].get(name), values,
                where=f"channel saturation {scenario}/{name}",
            )

    perturbed = EXPECTED_SCENARIOS[:-1]
    pooled = artifact.get("pooled_perturbed")
    if not isinstance(pooled, dict):
        _fail("channel saturation artifact lacks pooled_perturbed")
    for name in ("add_pirag_on_mcp", "add_mcp_on_pirag"):
        pooled_differences = [
            sum(scenario_differences[scenario][name][index]
                for scenario in perturbed) / len(perturbed)
            for index in range(len(EXPECTED_SEEDS))
        ]
        compare(
            pooled.get(name), pooled_differences,
            where=f"channel saturation pooled/{name}",
        )
    print("[PASS] H2 saturation TOST recomputed from raw seeds")


def _stress_cell(panel: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    seed_panel = panel.get(str(seed), panel.get(seed))
    if not isinstance(seed_panel, dict) or not isinstance(seed_panel.get(mode), dict):
        _fail(f"stress summary missing seed={seed}/mode={mode}")
    return seed_panel[mode]


def _holm_adjusted(p_values: dict[str, float]) -> dict[str, float]:
    """Independently recompute Holm step-down adjusted p-values."""
    ordered = sorted(p_values.items(), key=lambda item: float(item[1]))
    m = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (key, raw) in enumerate(ordered):
        value = float(raw)
        if not (0.0 <= value <= 1.0):
            _fail(f"Holm family contains invalid p-value {key}={value!r}")
        running = max(running, min(1.0, value * (m - index)))
        adjusted[key] = running
    return adjusted


def _validate_significance() -> None:
    path = RESULTS_DIR / "benchmark_significance.json"
    payload = _load_json(path)
    meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
    # 2026-04 schema: per-(scenario, comparison, metric) records are
    # nested under top-level "significance" alongside "_meta" and
    # "primary_h1_holm_adjusted". Unwrap so the traversal works on
    # both wrapped and legacy-flat formats.
    if isinstance(payload, dict) and isinstance(payload.get("significance"), dict):
        data = payload["significance"]
    else:
        data = payload
    required = {
        "p_value",
        "p_value_adj",
        "cohens_d",
        "cohens_dz",
        "mean_diff",
        "mean_diff_ci_low",
        "mean_diff_ci_high",
        "mean_diff_ci_method",
        "effect_size_ci_method",
    }
    missing = []
    # Comparison-level metadata fields that are NOT per-metric records
    # (so the inner schema check should skip them). ``_family`` is the
    # string family label (e.g. "channel_decomposition") that
    # aggregate_seeds.py attaches to the channel-decomposition contrasts
    # (mcp_only_vs_no_context / pirag_only_vs_no_context) to drive the
    # within-family Holm correction (``p_value_adj_holm_channel``); it sits
    # at the comparison level next to is_paired_design / test_type, so it
    # must be skipped here rather than validated as a per-metric object.
    _COMP_META_KEYS = {
        "is_paired_design", "test_type", "effect_size_primary",
        "_family", "_meta",
    }
    for scenario, comps in data.items():
        if not isinstance(comps, dict):
            missing.append(f"{scenario} (not an object)")
            continue
        for comp, metrics in comps.items():
            if not isinstance(metrics, dict):
                missing.append(f"{scenario}.{comp} (not an object)")
                continue
            for metric, rec in metrics.items():
                if metric in _COMP_META_KEYS:
                    continue
                if comp == "h2_synergy_interaction":
                    # Exploratory superadditivity has its own schema below;
                    # it is not one of the ordinary pairwise metric records.
                    continue
                if not isinstance(rec, dict):
                    missing.append(f"{scenario}.{comp}.{metric} (not an object)")
                    continue
                absent = sorted(required.difference(rec.keys()))
                if absent:
                    missing.append(f"{scenario}.{comp}.{metric}: missing {', '.join(absent)}")
    if missing:
        _fail("benchmark_significance schema violations:\n  - " + "\n  - ".join(missing[:20]))

    if not isinstance(meta, dict):
        _fail("benchmark_significance.json has no metadata object")
    if int(meta.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
        _fail("benchmark_significance.json metadata does not report 20 seeds")
    if meta.get("paired") is not True:
        _fail("benchmark_significance.json does not declare the matched-seed design")
    if int(meta.get("wilcoxon_fallback_count", -1)) != 0:
        _fail("canonical significance used a non-declared Wilcoxon fallback")
    if meta.get("confirmatory_test") != "directional_wilcoxon_signed_rank":
        _fail("benchmark_significance.json has the wrong confirmatory test label")
    n_perm_scope = str(meta.get("n_perm_scope", ""))
    if (
        int(meta.get("legacy_sign_flip_resamples", -1)) != 10_000
        or "legacy" not in n_perm_scope
        or "not the canonical H1/H2 test" not in n_perm_scope
    ):
        _fail("legacy sign-flip resampling is not explicitly scoped away from H1/H2")
    if meta.get("primary_h1_correction") != "holm_bonferroni":
        _fail("benchmark_significance.json has the wrong H1 correction")
    if meta.get("pinn_ablation_correction") != "holm_bonferroni":
        _fail("benchmark_significance.json has the wrong PINN-ablation correction")
    if meta.get("pinn_ablation_scope") != (
        "separate prespecified paired mechanistic-residual ablation; "
        "not part of H1 or H2"
    ):
        _fail("benchmark_significance.json has the wrong PINN-ablation scope")
    if meta.get("h2_directional_correction") != "holm_bonferroni":
        _fail("benchmark_significance.json has the wrong H2 correction")
    if meta.get("h2_directional_canonical_field") != (
        "p_value_adj_holm_h2_directional"
    ):
        _fail("benchmark_significance.json has the wrong canonical H2 field")

    if not isinstance(data, dict) or set(data) != set(EXPECTED_SCENARIOS):
        _fail("benchmark_significance.json does not contain the exact scenario panel")
    baseline_comparisons = {
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
        "agribrain_vs_no_context", "agribrain_vs_no_pinn",
        "agribrain_vs_no_slca", "agribrain_vs_hybrid_rl",
        "agribrain_vs_static",
    }
    channel_comparisons = {
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
    }
    h2_comparisons = {
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
    }
    expected_comparisons = (
        baseline_comparisons | channel_comparisons
        | {"h2_synergy_interaction"}
    )
    expected_metrics = {"ari", "waste", "rle", "slca", "carbon", "equity"}
    allowed_ci_methods = {"BCa", "deterministic", "percentile_fallback"}
    for scenario in EXPECTED_SCENARIOS:
        comparisons = data[scenario]
        if set(comparisons) != expected_comparisons:
            _fail(f"{scenario} significance comparison panel is incomplete or unexpected")
        for comparison in expected_comparisons:
            comp = comparisons[comparison]
            if comparison == "h2_synergy_interaction":
                interaction = comp.get("ari") if isinstance(comp, dict) else None
                if comp.get("exploratory") is not True or not isinstance(
                    interaction, dict
                ):
                    _fail(f"{scenario}/h2_synergy_interaction is not exploratory")
                if int(interaction.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
                    _fail(f"{scenario}/h2_synergy_interaction is not 20-seed")
                for field in (
                    "p_value_directional_greater", "mean_interaction",
                    "mean_interaction_ci_low", "mean_interaction_ci_high",
                ):
                    try:
                        value = float(interaction[field])
                    except (KeyError, TypeError, ValueError):
                        _fail(
                            f"{scenario}/h2_synergy_interaction/{field} is invalid"
                        )
                    if not __import__("math").isfinite(value):
                        _fail(
                            f"{scenario}/h2_synergy_interaction/{field} is non-finite"
                        )
                continue
            if comp.get("is_paired_design") is not True:
                _fail(f"{scenario}/{comparison} is not labelled as paired")
            if comp.get("test_type") != "wilcoxon_signed_rank":
                _fail(f"{scenario}/{comparison} has the wrong declared test")
            if comp.get("effect_size_primary") != "cohens_dz":
                _fail(f"{scenario}/{comparison} has the wrong primary effect size")
            metric_keys = set(comp) - _COMP_META_KEYS
            if metric_keys != expected_metrics:
                _fail(f"{scenario}/{comparison} has an incomplete metric panel")
            for metric in expected_metrics:
                record = comp[metric]
                if int(record.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
                    _fail(f"{scenario}/{comparison}/{metric} is not a 20-seed contrast")
                if record.get("test_type_actual") != "wilcoxon_signed_rank":
                    _fail(f"{scenario}/{comparison}/{metric} changed inferential test")
                if comparison == "agribrain_vs_no_pinn" and metric == "ari":
                    if record.get("directional_test_type_actual") != (
                        "wilcoxon_signed_rank"
                    ):
                        _fail(
                            f"{scenario}/{comparison}/{metric} changed the "
                            "directional PINN-ablation test"
                        )
                    if int(record.get("pinn_ablation_family_size", -1)) != len(
                        EXPECTED_SCENARIOS
                    ):
                        _fail(
                            f"{scenario}/{comparison}/{metric} has the wrong "
                            "PINN-ablation family size"
                        )
                if record.get("mean_diff_ci_method") not in allowed_ci_methods:
                    _fail(f"{scenario}/{comparison}/{metric} has an invalid mean CI label")
                effect_method = record.get("effect_size_ci_method")
                if effect_method not in allowed_ci_methods | {"undefined"}:
                    _fail(f"{scenario}/{comparison}/{metric} has an invalid effect CI label")
                for field in ("p_value", "p_value_adj", "mean_diff"):
                    try:
                        value = float(record[field])
                    except (KeyError, TypeError, ValueError):
                        _fail(f"{scenario}/{comparison}/{metric}/{field} is invalid")
                    if not __import__("math").isfinite(value):
                        _fail(f"{scenario}/{comparison}/{metric}/{field} is non-finite")
                if not (0.0 <= float(record["p_value"]) <= 1.0):
                    _fail(f"{scenario}/{comparison}/{metric} p-value is outside [0,1]")
                if not (0.0 <= float(record["p_value_adj"]) <= 1.0):
                    _fail(f"{scenario}/{comparison}/{metric} adjusted p-value is outside [0,1]")

    h1_map = payload.get("primary_h1_holm_adjusted")
    expected_h1_keys = set(EXPECTED_SCENARIOS)
    if not isinstance(h1_map, dict) or set(h1_map) != expected_h1_keys:
        _fail("primary H1 Holm map is not the exact five-scenario family")
    h1_support_map = payload.get("primary_h1_supported_by_cell")
    if not isinstance(h1_support_map, dict) or set(h1_support_map) != expected_h1_keys:
        _fail("primary H1 support map is not the exact five-scenario family")
    if any(type(value) is not bool for value in h1_support_map.values()):
        _fail("primary H1 support map contains a non-Boolean claim flag")
    pinn_map = payload.get("pinn_ablation_holm_adjusted")
    expected_pinn_keys = set(EXPECTED_SCENARIOS)
    if not isinstance(pinn_map, dict) or set(pinn_map) != expected_pinn_keys:
        _fail("PINN-ablation Holm map is not the exact five-scenario family")
    pinn_support_map = payload.get("pinn_ablation_supported_by_cell")
    if (
        not isinstance(pinn_support_map, dict)
        or set(pinn_support_map) != expected_pinn_keys
    ):
        _fail("PINN-ablation support map is not the exact five-scenario family")
    if any(type(value) is not bool for value in pinn_support_map.values()):
        _fail("PINN-ablation support map contains a non-Boolean claim flag")
    h2_map = payload.get("h2_directional_holm_adjusted")
    expected_h2_keys = {
        f"{scenario}:{comparison}"
        for scenario in EXPECTED_SCENARIOS
        for comparison in h2_comparisons
    }
    if not isinstance(h2_map, dict) or set(h2_map) != expected_h2_keys:
        _fail("directional Holm map is not the exact twenty-test H2 family")
    h2_support_map = payload.get("h2_directional_supported_by_cell")
    if not isinstance(h2_support_map, dict) or set(h2_support_map) != expected_h2_keys:
        _fail("directional H2 support map is not the exact twenty-test family")
    if any(type(value) is not bool for value in h2_support_map.values()):
        _fail("directional H2 support map contains a non-Boolean claim flag")
    expected_h1_adjusted = _holm_adjusted({
        scenario: float(
            data[scenario]["agribrain_vs_no_context"]["ari"]
            ["h1_raw_p_value_directional_greater"]
        )
        for scenario in EXPECTED_SCENARIOS
    })
    expected_h2_adjusted = _holm_adjusted({
        f"{scenario}:{comparison}": float(
            data[scenario][comparison]["ari"]["p_value_directional_greater"]
        )
        for scenario in EXPECTED_SCENARIOS
        for comparison in h2_comparisons
    })
    expected_pinn_adjusted = _holm_adjusted({
        scenario: float(
            data[scenario]["agribrain_vs_no_pinn"]["ari"]
            ["pinn_ablation_raw_p_value_directional_greater"]
        )
        for scenario in EXPECTED_SCENARIOS
    })
    for key, expected in expected_h1_adjusted.items():
        if not __import__("math").isclose(
            float(h1_map[key]), expected, rel_tol=1e-12, abs_tol=1e-15,
        ):
            _fail(f"primary H1 Holm map is miscomputed at {key}")
    for key, expected in expected_h2_adjusted.items():
        if not __import__("math").isclose(
            float(h2_map[key]), expected, rel_tol=1e-12, abs_tol=1e-15,
        ):
            _fail(f"directional H2 Holm map is miscomputed at {key}")
    for key, expected in expected_pinn_adjusted.items():
        if not __import__("math").isclose(
            float(pinn_map[key]), expected, rel_tol=1e-12, abs_tol=1e-15,
        ):
            _fail(f"PINN-ablation Holm map is miscomputed at {key}")

    for scenario in EXPECTED_SCENARIOS:
        h1_record = data[scenario]["agribrain_vs_no_context"]["ari"]
        if not __import__("math").isclose(
            float(h1_record["p_value_adj"]), float(h1_map[scenario]),
            rel_tol=1e-12, abs_tol=1e-15,
        ) or h1_record.get("correction_method") != "holm_bonferroni_across_scenarios":
            _fail(f"{scenario} H1 record disagrees with its Holm family")
        if int(h1_record.get("h1_family_size", -1)) != 5:
            _fail(f"{scenario} H1 record has the wrong family size")
        try:
            h1_audit_adjusted = float(h1_record["p_value_adj_holm"])
            h1_mean_diff = float(h1_record["mean_diff"])
        except (KeyError, TypeError, ValueError):
            _fail(f"{scenario} H1 record lacks its canonical claim evidence")
        if not __import__("math").isclose(
            h1_audit_adjusted, float(h1_map[scenario]),
            rel_tol=1e-12, abs_tol=1e-15,
        ):
            _fail(f"{scenario} H1 audit field disagrees with its Holm family")
        expected_h1_support = bool(
            h1_mean_diff > 0.0 and float(h1_map[scenario]) < 0.05
        )
        if h1_record.get("h1_positive_effect_supported") is not expected_h1_support:
            _fail(
                f"{scenario} H1 support flag contradicts its direction and "
                "Holm-adjusted p-value"
            )
        if h1_support_map[scenario] is not expected_h1_support:
            _fail(f"primary H1 support map contradicts {scenario}")
        try:
            h1_margin = float(h1_record["h1_practical_margin"])
            h1_ci_low = float(h1_record["mean_diff_ci_low"])
        except (KeyError, TypeError, ValueError):
            _fail(f"{scenario} H1 record lacks its practical-margin evidence")
        if not __import__("math").isclose(
            h1_margin, 0.005, rel_tol=0.0, abs_tol=1e-15,
        ):
            _fail(f"{scenario} H1 record changed the practical ARI margin")
        expected_practical_support = bool(
            h1_record.get("mean_diff_ci_method") == "BCa"
            and h1_ci_low > h1_margin
        )
        if h1_record.get("h1_practical_margin_supported") is not (
            expected_practical_support
        ):
            _fail(
                f"{scenario} H1 practical-margin flag is inconsistent with "
                "the prespecified 95% BCa rule"
            )
        for comparison in h2_comparisons:
            record = data[scenario][comparison]["ari"]
            key = f"{scenario}:{comparison}"
            if not __import__("math").isclose(
                float(record["p_value_adj"]), float(h2_map[key]),
                rel_tol=1e-12, abs_tol=1e-15,
            ) or record.get("correction_method") != (
                "holm_bonferroni_h2_directional_20"
            ):
                _fail(f"{scenario}/{comparison} disagrees with its H2 Holm family")
            try:
                h2_cell_adjusted = float(
                    record["p_value_adj_holm_h2_directional"]
                )
            except (KeyError, TypeError, ValueError):
                _fail(f"{scenario}/{comparison} lacks its H2 Holm audit field")
            if not __import__("math").isclose(
                h2_cell_adjusted, float(h2_map[key]),
                rel_tol=1e-12, abs_tol=1e-15,
            ):
                _fail(f"{scenario}/{comparison} H2 Holm audit field is inconsistent")
            if int(record.get("h2_family_size", -1)) != 20:
                _fail(f"{scenario}/{comparison} has the wrong H2 family size")
            try:
                h2_raw_directional = float(record["p_value_directional_greater"])
                h2_mean_diff = float(record["mean_diff"])
            except (KeyError, TypeError, ValueError):
                _fail(f"{scenario}/{comparison} lacks its H2 claim evidence")
            expected_h2_support = bool(
                h2_mean_diff > 0.0
                and h2_raw_directional < 0.05
                and float(h2_map[key]) < 0.05
            )
            if record.get("h2_cell_supported") is not expected_h2_support:
                _fail(
                    f"{scenario}/{comparison} H2 cell flag contradicts its "
                    "directional evidence"
                )
            if h2_support_map[key] is not expected_h2_support:
                _fail(
                    f"directional H2 support map contradicts {scenario}/{comparison}"
                )
    expected_h1_global = bool(h1_support_map) and all(h1_support_map.values())
    if payload.get("primary_h1_supported_all_cells") is not expected_h1_global:
        _fail(
            "global H1 support flag is not the intersection of all five "
            "recomputed directional cells"
        )
    expected_h2_global = bool(h2_support_map) and all(h2_support_map.values())
    if payload.get("h2_directional_supported_all_cells") is not expected_h2_global:
        _fail(
            "global H2 support flag is not the intersection of all 20 "
            "recomputed directional cells"
        )
    _validate_h2_publication_table(payload, data)
    print("[PASS] exact paired H1/H2 significance panels and correction families")


def _validate_h2_publication_table(payload: dict, data: dict) -> None:
    """Cross-check the persisted 20-row H2 table against canonical records."""
    rows = payload.get("h2_directional_evidence")
    if not isinstance(rows, list) or len(rows) != 20:
        _fail("benchmark_significance.json lacks the exact 20-row H2 table")
    expected_pairs = (
        ("mcp_only", "no_context"),
        ("pirag_only", "no_context"),
        ("agribrain", "mcp_only"),
        ("agribrain", "pirag_only"),
    )
    expected_order = [
        (scenario, left, right)
        for scenario in EXPECTED_SCENARIOS
        for left, right in expected_pairs
    ]
    observed_order: list[tuple[str, str, str]] = []
    meta = payload["_meta"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(H2_PUBLICATION_COLUMNS):
            _fail(f"H2 evidence row {index} has the wrong schema")
        scenario = row["scenario"]
        left = row["numerator_mode"]
        right = row["denominator_mode"]
        observed_order.append((scenario, left, right))
        comparison = f"{left}_vs_{right}"
        if (
            row["comparison"] != comparison
            or row["direction"] != f"{left} > {right}"
        ):
            _fail(f"H2 evidence row {index} has inconsistent labels")
        if row["endpoint"] != "ari" or row["alternative"] != "greater":
            _fail(f"H2 evidence row {index} changes the endpoint or alternative")
        if row["test"] != "wilcoxon_signed_rank":
            _fail(f"H2 evidence row {index} did not use directional Wilcoxon")
        if (
            row["source_commit"] != meta.get("source_commit")
            or row["run_tag"] != meta.get("run_tag")
        ):
            _fail(f"H2 evidence row {index} has inconsistent run provenance")
        record = data[scenario][comparison]["ari"]
        numeric_links = {
            "mean_difference": "mean_diff",
            "mean_difference_ci_low": "mean_diff_ci_low",
            "mean_difference_ci_high": "mean_diff_ci_high",
            "cohens_dz": "cohens_dz",
            "cohens_dz_ci_low": "cohens_dz_ci_low",
            "cohens_dz_ci_high": "cohens_dz_ci_high",
            "raw_directional_p_value": "p_value_directional_greater",
            "holm_adjusted_p_value": "p_value_adj_holm_h2_directional",
        }
        for exported, canonical in numeric_links.items():
            if row[exported] is None or record[canonical] is None:
                if row[exported] is record[canonical]:
                    continue
                _fail(
                    f"H2 evidence row {index}/{exported} has inconsistent null"
                )
            if not __import__("math").isclose(
                float(row[exported]), float(record[canonical]),
                rel_tol=1e-12, abs_tol=1e-15,
            ):
                _fail(
                    f"H2 evidence row {index}/{exported} disagrees with JSON"
                )
        if (
            int(row["n_seeds"]) != len(EXPECTED_SEEDS)
            or row["paired_design"] is not True
            or int(row["holm_family_size"]) != 20
            or float(row["alpha"]) != 0.05
            or row["mean_difference_ci_method"]
            != record["mean_diff_ci_method"]
            or row["cohens_dz_ci_method"] != record["cohens_dz_ci_method"]
            or row["positive_mean"] is not (float(record["mean_diff"]) > 0.0)
            or row["cell_supported"] is not record["h2_cell_supported"]
        ):
            _fail(f"H2 evidence row {index} disagrees with claim metadata")
    if observed_order != expected_order:
        _fail("H2 evidence table does not use the locked scenario/contrast order")

    csv_path = RESULTS_DIR / "h2_directional_evidence.csv"
    if not csv_path.is_file():
        _fail("Missing required H2 publication CSV")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != H2_PUBLICATION_COLUMNS:
            _fail("H2 publication CSV has the wrong column schema")
        csv_rows = list(reader)
    if len(csv_rows) != 20:
        _fail("H2 publication CSV is not the exact 20-cell family")
    for index, (csv_row, json_row) in enumerate(
        zip(csv_rows, rows, strict=True),
    ):
        for column in H2_PUBLICATION_COLUMNS:
            expected = "" if json_row[column] is None else str(json_row[column])
            if csv_row[column] != expected:
                _fail(
                    f"H2 publication CSV row {index}/{column} disagrees with JSON"
                )


def _validate_benchmark_summary() -> None:
    """Validate the seed-level summaries used by H1/H2 and their figures."""
    payload = _load_json(RESULTS_DIR / "benchmark_summary.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        _fail("benchmark_summary.json lacks the canonical summary envelope")
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        _fail("benchmark_summary.json lacks metadata")
    if int(meta.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
        _fail("benchmark_summary.json metadata does not report 20 seeds")
    if meta.get("seeds_loaded") != sorted(EXPECTED_SEEDS):
        _fail("benchmark_summary.json metadata has the wrong seed panel")
    if int(meta.get("std_ddof", -1)) != 1:
        _fail("benchmark_summary.json standard deviations are not sample SDs")
    bca = meta.get("bca_fallback_stats")
    if not isinstance(bca, dict):
        _fail("benchmark_summary.json lacks BCa diagnostics")
    if int(bca.get("fallback_scipy_unavailable", -1)) != 0:
        _fail("benchmark_summary.json used a CI fallback because SciPy was unavailable")
    contracts = meta.get("derived_metric_contracts")
    if not isinstance(contracts, dict):
        _fail("benchmark_summary.json lacks CE/Green-AI metric contracts")
    ce_contract = contracts.get("carbon_efficiency_ari_per_kgco2e_proxy")
    green_contract = contracts.get("green_ai_decision_path")
    if (
        not isinstance(ce_contract, dict)
        or ce_contract.get("scale_factor") != 1.0
        or ce_contract.get("uncertainty")
        != "BCa bootstrap of within-seed ratios"
        or not isinstance(green_contract, dict)
        or green_contract.get("assumed_active_power_W") != 10.0
        or green_contract.get("water_rate_L_per_server_second") != 1.8e-6
        or "not whole-job" not in str(green_contract.get("measurement_scope"))
        or "not hardware telemetry" not in str(green_contract.get("status"))
    ):
        _fail("benchmark_summary.json changes the CE/Green-AI definitions")

    summary = payload["summary"]
    if set(summary) != set(EXPECTED_SCENARIOS):
        _fail("benchmark_summary.json does not contain the exact scenario panel")
    required_modes = {"agribrain", "no_context", "mcp_only", "pirag_only"}
    required_metrics = {"ari", "waste", "rle", "slca", "carbon", "equity"}
    derived_metrics = {
        "carbon_efficiency_ari_per_kgco2e_proxy",
        "decision_path_compute_energy_estimate_j",
        "decision_path_compute_water_estimate_l",
        "decision_path_elapsed_seconds",
        "decision_step_count_energy_proxy_j",
        "decision_step_count_water_proxy_l",
    }
    allowed_methods = {"BCa", "deterministic", "percentile_fallback"}
    import math
    for scenario in EXPECTED_SCENARIOS:
        if not required_modes.issubset(summary[scenario]):
            _fail(f"benchmark_summary.json {scenario} lacks an H1/H2 arm")
        for mode in required_modes:
            cell = summary[scenario][mode]
            if not required_metrics.issubset(cell):
                _fail(f"benchmark_summary.json {scenario}/{mode} lacks a core endpoint")
            if not derived_metrics.issubset(cell):
                _fail(
                    f"benchmark_summary.json {scenario}/{mode} lacks CE/Green-AI evidence"
                )
            for metric in required_metrics:
                record = cell[metric]
                if int(record.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
                    _fail(f"benchmark_summary.json {scenario}/{mode}/{metric} is not 20-seed")
                if record.get("ci_method") not in allowed_methods:
                    _fail(f"benchmark_summary.json {scenario}/{mode}/{metric} has no honest CI label")
                try:
                    mean = float(record["mean"])
                    std = float(record["std"])
                    low = float(record["ci_low"])
                    high = float(record["ci_high"])
                except (KeyError, TypeError, ValueError):
                    _fail(f"benchmark_summary.json {scenario}/{mode}/{metric} is nonnumeric")
                if not all(math.isfinite(value) for value in (mean, std, low, high)):
                    _fail(f"benchmark_summary.json {scenario}/{mode}/{metric} is non-finite")
                if std < 0.0 or low > high:
                    _fail(f"benchmark_summary.json {scenario}/{mode}/{metric} has invalid spread")
                if metric == "ari" and not (0.0 <= mean <= 1.0):
                    _fail(f"benchmark_summary.json {scenario}/{mode}/ari is outside [0,1]")
            for metric in derived_metrics:
                record = cell[metric]
                if int(record.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
                    _fail(
                        f"benchmark_summary.json {scenario}/{mode}/{metric} "
                        "is not 20-seed"
                    )
                try:
                    value = float(record["mean"])
                except (KeyError, TypeError, ValueError):
                    _fail(
                        f"benchmark_summary.json {scenario}/{mode}/{metric} "
                        "is nonnumeric"
                    )
                if not math.isfinite(value) or value < 0.0:
                    _fail(
                        f"benchmark_summary.json {scenario}/{mode}/{metric} "
                        "is invalid"
                    )
            elapsed = float(cell["decision_path_elapsed_seconds"]["mean"])
            energy = float(
                cell["decision_path_compute_energy_estimate_j"]["mean"]
            )
            water = float(
                cell["decision_path_compute_water_estimate_l"]["mean"]
            )
            if not math.isclose(energy, 10.0 * elapsed, rel_tol=1e-12, abs_tol=1e-7):
                _fail(f"benchmark_summary.json {scenario}/{mode} energy equation fails")
            if not math.isclose(
                water, 1.8e-6 * elapsed, rel_tol=1e-12, abs_tol=1e-12,
            ):
                _fail(f"benchmark_summary.json {scenario}/{mode} water equation fails")
    print("[PASS] exact H1/H2 seed summaries and CI method labels")


def _validate_paper_benchmark_table() -> None:
    """Prove the paper-facing combined export is an exact source projection."""

    paper = _load_json(RESULTS_DIR / "paper_benchmark_table.json")
    summary = _load_json(RESULTS_DIR / "benchmark_summary.json")
    significance = _load_json(RESULTS_DIR / "benchmark_significance.json")
    if not all(isinstance(item, dict) for item in (paper, summary, significance)):
        _fail("paper benchmark inputs must be JSON objects")
    if paper.get("benchmark") != summary.get("summary"):
        _fail("paper_benchmark_table.json benchmark differs from benchmark_summary.json")
    if paper.get("significance") != significance.get("significance"):
        _fail(
            "paper_benchmark_table.json significance differs from "
            "benchmark_significance.json"
        )
    if paper.get("h2_directional_evidence") != significance.get(
        "h2_directional_evidence"
    ):
        _fail("paper benchmark H2 table differs from benchmark significance")

    meta = paper.get("_meta")
    summary_meta = summary.get("_meta")
    significance_meta = significance.get("_meta")
    if not all(
        isinstance(item, dict)
        for item in (meta, summary_meta, significance_meta)
    ):
        _fail("paper benchmark export lacks source metadata")
    for key in (
        "git_commit", "source_commit", "simulation_source_commit",
        "analysis_code_commit", "dual_provenance", "run_tag", "n_seeds",
        "seeds_loaded", "bootstrap_alpha", "n_boot", "n_perm", "std_ddof",
        "bca_fallback_stats",
    ):
        if meta.get(key) != summary_meta.get(key):
            _fail(f"paper benchmark metadata {key} differs from benchmark summary")
    if meta.get("source_artifacts") != [
        "mvp/simulation/results/benchmark_summary.json",
        "mvp/simulation/results/benchmark_significance.json",
        "mvp/simulation/results/h2_directional_evidence.csv",
    ]:
        _fail("paper benchmark export has an invalid source-artifact declaration")
    generated_at = meta.get("generated_at")
    if not isinstance(generated_at, str):
        _fail("paper benchmark export lacks generated_at")
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        _fail("paper benchmark export has invalid generated_at")
    if parsed.tzinfo is None:
        _fail("paper benchmark generated_at must be timezone-aware")

    expected_correction_meta = {
        "primary_h1_family": significance_meta.get("primary_h1_family"),
        "primary_h1_correction": significance_meta.get("primary_h1_correction"),
        "h2_directional_family": significance_meta.get("h2_directional_family"),
        "h2_directional_correction": significance_meta.get(
            "h2_directional_correction"
        ),
        "h2_directional_canonical_field": significance_meta.get(
            "h2_directional_canonical_field"
        ),
        "h2_global_support_rule": significance_meta.get("h2_global_support_rule"),
        "h2_synergy_status": significance_meta.get("h2_synergy_status"),
        "confirmatory_test": significance_meta.get("confirmatory_test"),
        "n_perm_scope": significance_meta.get("n_perm_scope"),
        "channel_decomposition_family": significance_meta.get(
            "channel_decomposition_family"
        ),
        "channel_decomposition_correction": significance_meta.get(
            "channel_decomposition_correction"
        ),
        "channel_decomposition_status": significance_meta.get(
            "channel_decomposition_status"
        ),
        "secondary_correction": significance_meta.get("secondary_correction"),
        "secondary_family_scope": significance_meta.get("secondary_family_scope"),
        "primary_h1_holm_adjusted": significance.get(
            "primary_h1_holm_adjusted"
        ),
        "primary_h1_supported_by_cell": significance.get(
            "primary_h1_supported_by_cell"
        ),
        "primary_h1_supported_all_cells": significance.get(
            "primary_h1_supported_all_cells"
        ),
        "pinn_ablation_family": significance_meta.get("pinn_ablation_family"),
        "pinn_ablation_correction": significance_meta.get(
            "pinn_ablation_correction"
        ),
        "pinn_ablation_scope": significance_meta.get("pinn_ablation_scope"),
        "pinn_ablation_holm_adjusted": significance.get(
            "pinn_ablation_holm_adjusted"
        ),
        "pinn_ablation_supported_by_cell": significance.get(
            "pinn_ablation_supported_by_cell"
        ),
        "pinn_ablation_supported_all_cells": significance.get(
            "pinn_ablation_supported_all_cells"
        ),
        "h2_directional_holm_adjusted": significance.get(
            "h2_directional_holm_adjusted"
        ),
        "h2_directional_supported_by_cell": significance.get(
            "h2_directional_supported_by_cell"
        ),
        "h2_directional_supported_all_cells": significance.get(
            "h2_directional_supported_all_cells"
        ),
        "channel_decomposition_holm_adjusted": significance.get(
            "channel_decomposition_holm_adjusted"
        ),
    }
    if meta.get("significance_correction_meta") != expected_correction_meta:
        _fail("paper benchmark correction metadata differs from significance source")
    print("[PASS] paper benchmark exact semantic projection")


def _rounded_csv_match(text: str, expected: float) -> bool:
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    tolerance = 0.5000001 * (10.0 ** (-decimals))
    return abs(float(text) - float(expected)) <= tolerance


def _validate_tables_against_summary() -> None:
    """Require both publication tables to be rounded views of one JSON source."""
    summary_doc = _load_json(RESULTS_DIR / "benchmark_summary.json")
    summary = summary_doc.get("summary")
    meta = summary_doc.get("_meta")
    if not isinstance(summary, dict) or not isinstance(meta, dict):
        _fail("benchmark_summary.json lacks canonical table source data")
    table_sources = meta.get("canonical_table_sources")
    if not isinstance(table_sources, dict) or table_sources.get(
        "ConstraintViolationRate"
    ) != "constraint_violation_rate":
        _fail("benchmark summary does not declare the canonical table metric mapping")

    metric_map = {
        "ARI": "ari",
        "RLE": "rle",
        "Waste": "waste",
        "SLCA": "slca",
        "Carbon": "carbon",
        "Equity": "equity",
        "ConstraintViolationRate": "constraint_violation_rate",
        "OperatingEnvelopeViolationRate": "operating_envelope_violation_rate",
        "DownstreamViolationRate": "downstream_violation_rate",
        "ContainedViolationRate": "contained_violation_rate",
    }
    for filename, label_column in (
        ("table1_summary.csv", "Method"),
        ("table2_ablation.csv", "Variant"),
    ):
        path = RESULTS_DIR / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row_number, row in enumerate(rows, start=2):
            scenario = row.get("Scenario", "")
            mode = row.get(label_column, "")
            bucket = summary.get(scenario, {}).get(mode)
            if not isinstance(bucket, dict):
                _fail(f"{filename}:{row_number} has an unknown scenario/mode cell")
            if row.get("n_seeds") != "20":
                _fail(f"{filename}:{row_number} does not report n_seeds=20")
            for display, metric in metric_map.items():
                if display not in row:
                    continue
                source = bucket.get(metric)
                if not isinstance(source, dict):
                    _fail(f"{filename}:{row_number} lacks source metric {metric}")
                for suffix, key in (
                    ("", "mean"),
                    ("_ci_low", "ci_low"),
                    ("_ci_high", "ci_high"),
                ):
                    column = f"{display}{suffix}"
                    text = row.get(column)
                    expected = source.get(key)
                    if text in (None, "") or not isinstance(expected, (int, float)):
                        _fail(f"{filename}:{row_number} missing {column} or {metric}.{key}")
                    if not _rounded_csv_match(text, float(expected)):
                        _fail(
                            f"{filename}:{row_number} {column} disagrees with "
                            f"benchmark_summary.json {scenario}/{mode}/{metric}/{key}"
                        )
    print("[PASS] publication CSV tables are canonical rounded summary views")


def _validate_channel_saturation() -> None:
    """Prevent scenario pseudoreplication in the secondary H2 diagnostic."""
    data = _load_json(RESULTS_DIR / "channel_saturation_analysis.json")
    meta = data.get("_meta") if isinstance(data, dict) else None
    if not isinstance(meta, dict) or int(meta.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
        _fail("channel saturation analysis is not based on the exact 20-seed panel")
    if meta.get("seed_order") != list(EXPECTED_SEEDS):
        _fail("channel saturation analysis has the wrong seed identities")
    by_scenario = data.get("by_scenario")
    if not isinstance(by_scenario, dict) or set(by_scenario) != set(EXPECTED_SCENARIOS):
        _fail("channel saturation analysis lacks the exact scenario panel")
    tests = ("add_pirag_on_mcp", "add_mcp_on_pirag")

    def _check_tost(result: Any, *, where: str) -> None:
        if not isinstance(result, dict) or int(result.get("n", -1)) != 20:
            _fail(f"{where} is not paired by seed")
        numeric: dict[str, float] = {}
        for field in (
            "mean_diff", "ci90_low", "ci90_high", "ci95_low", "ci95_high",
            "p_two_sided", "p_tost", "sesoi",
        ):
            try:
                numeric[field] = float(result[field])
            except (KeyError, TypeError, ValueError):
                _fail(f"{where}/{field} is missing or nonnumeric")
            if not __import__("math").isfinite(numeric[field]):
                _fail(f"{where}/{field} is non-finite")
        if numeric["sesoi"] != 0.01:
            _fail(f"{where} has the wrong SESOI")
        if not (0.0 <= numeric["p_two_sided"] <= 1.0):
            _fail(f"{where} has an invalid two-sided p-value")
        if not (0.0 <= numeric["p_tost"] <= 1.0):
            _fail(f"{where} has an invalid TOST p-value")
        if (
            numeric["ci90_low"] > numeric["ci90_high"]
            or numeric["ci95_low"] > numeric["ci95_high"]
            or numeric["ci95_low"] > numeric["ci90_low"]
            or numeric["ci95_high"] < numeric["ci90_high"]
        ):
            _fail(f"{where} has inconsistent confidence intervals")
        equivalent_p = numeric["p_tost"] < 0.05
        equivalent_ci = (
            numeric["ci90_low"] > -numeric["sesoi"]
            and numeric["ci90_high"] < numeric["sesoi"]
        )
        if equivalent_p != equivalent_ci:
            _fail(f"{where} TOST p-value and 90% CI disagree")
        positive = numeric["p_two_sided"] < 0.05 and numeric["mean_diff"] > 0.0
        negative = numeric["p_two_sided"] < 0.05 and numeric["mean_diff"] < 0.0
        if equivalent_p and positive:
            expected_verdict = "positive_but_equivalent"
        elif equivalent_p and negative:
            expected_verdict = "negative_but_equivalent"
        elif equivalent_p:
            expected_verdict = "equivalent_within_margin"
        elif positive:
            expected_verdict = "positive_difference"
        elif negative:
            expected_verdict = "negative_difference"
        else:
            expected_verdict = "inconclusive"
        if result.get("verdict") != expected_verdict:
            _fail(f"{where} verdict contradicts its tests")

    for scenario in EXPECTED_SCENARIOS:
        cell = by_scenario[scenario]
        if int(cell.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
            _fail(f"channel saturation {scenario} is not a 20-seed cell")
        for name in tests:
            result = cell.get(name)
            _check_tost(result, where=f"channel saturation {scenario}/{name}")

    pooled = data.get("pooled_perturbed")
    if not isinstance(pooled, dict):
        _fail("channel saturation lacks the pooled perturbed-scenario diagnostic")
    if pooled.get("inferential_unit") != "seed":
        _fail("channel saturation pooled TOST does not use seed as its unit")
    if pooled.get("scenario_aggregation") != (
        "mean paired difference across four scenarios within seed"
    ):
        _fail("channel saturation pooled TOST does not declare within-seed aggregation")
    if pooled.get("scenarios") != list(EXPECTED_SCENARIOS[:-1]):
        _fail("channel saturation pooled TOST has the wrong scenario set")
    for name in tests:
        result = pooled.get(name)
        _check_tost(result, where=f"channel saturation pooled/{name}")

    moderation = data.get("moderation")
    expected_moderation = {
        "pirag_marginal_vs_mcp_strength", "mcp_marginal_vs_pirag_strength",
    }
    if not isinstance(moderation, dict) or set(moderation) != expected_moderation:
        _fail("channel saturation moderation panel is incomplete")
    for name in expected_moderation:
        for fit_name in ("crossfit", "naive_coupled_bound"):
            fit = moderation[name].get(fit_name)
            if (
                not isinstance(fit, dict)
                or int(fit.get("n", -1)) != 4
                or fit.get("p_value") is not None
                or fit.get("inferential") is not False
                or fit.get("unit") != "scenario"
            ):
                _fail(f"channel saturation {name}/{fit_name} is not descriptive")
            estimable = fit.get("estimable")
            if estimable is True:
                try:
                    slope = float(fit["slope"])
                    r2 = float(fit["r2"])
                except (KeyError, TypeError, ValueError):
                    _fail(f"channel saturation {name}/{fit_name} has invalid fit values")
                if (
                    not __import__("math").isfinite(slope)
                    or not __import__("math").isfinite(r2)
                    or not (0.0 <= r2 <= 1.0)
                ):
                    _fail(f"channel saturation {name}/{fit_name} has non-finite fit values")
            elif estimable is False:
                if fit.get("slope") is not None or fit.get("r2") is not None:
                    _fail(f"channel saturation {name}/{fit_name} masks an unestimable fit")
            else:
                _fail(f"channel saturation {name}/{fit_name} lacks estimability status")
    print("[PASS] H2 saturation uses seed-level pooling and descriptive moderation")


def _validate_stress_passfail() -> None:
    path = RESULTS_DIR / "stress_passfail.csv"
    if not path.exists():
        _fail(f"Missing required file: {path}")
    required_cols = {
        "Scenario",
        "Stressor",
        "Method",
        "Pass",
        "Pass_Equivalence",
        "n_seeds",
        "ari_tost_p_tost",
        "ari_tost_ci90_low",
        "ari_tost_ci90_high",
        "ari_delta",
        "waste_delta",
        "slca_delta",
        "rle_delta",
        "carbon_delta",
        "equity_delta",
        "constraint_violation_delta",
        "latency_ms_delta",
        "ARI_Base",
        "ARI_Stressed",
        "Waste_Base",
        "Waste_Stressed",
        "SLCA_Base",
        "SLCA_Stressed",
        "fault_injection_scheduled_opportunity_steps_mean",
        "fault_injection_scheduled_opportunity_steps_min",
        "fault_injection_scheduled_opportunity_steps_max",
        "fault_injection_trigger_steps_mean",
        "fault_injection_trigger_steps_min",
        "fault_injection_trigger_steps_max",
        "fault_injected_tool_result_count_mean",
        "fault_injected_tool_result_count_min",
        "fault_injected_tool_result_count_max",
        "Threshold_ARI",
        "Threshold_Waste",
        "Threshold_SLCA",
        "Threshold_RLE",
        "Threshold_Carbon",
        "Threshold_Equity",
        "Threshold_CVR",
        "Threshold_LatencyMs",
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing_cols = sorted(required_cols.difference(cols))
        if missing_cols:
            _fail(f"stress_passfail.csv missing columns: {', '.join(missing_cols)}")
        rows = list(reader)
        row_count = len(rows)
        if row_count == 0:
            _fail("stress_passfail.csv has no rows")
    h3_rows = [
        row for row in rows
        if row.get("Method") == "agribrain"
        and row.get("comparison_type") != "cross_mode_under_stress"
    ]
    if len(h3_rows) != 25:
        _fail(f"stress_passfail.csv must contain 25 AGRI-BRAIN H3 cells; found {len(h3_rows)}")
    observed_keys = [(row.get("Scenario"), row.get("Stressor")) for row in h3_rows]
    expected_keys = {
        (scenario, stressor)
        for scenario in EXPECTED_SCENARIOS for stressor in EXPECTED_STRESSORS
    }
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        _fail("stress_passfail.csv is not the exact unique 5 x 5 H3 panel")
    for row in h3_rows:
        cell = f"{row.get('Scenario')}/{row.get('Stressor')}"
        try:
            if int(float(row["n_seeds"])) != 20:
                _fail(f"{cell} H3 cell does not contain 20 paired seeds")
            p_value = float(row["ari_tost_p_tost"])
            ci_low = float(row["ari_tost_ci90_low"])
            ci_high = float(row["ari_tost_ci90_high"])
        except (KeyError, TypeError, ValueError):
            _fail(f"{cell} H3 cell has non-numeric TOST fields")
        if not (0.0 <= p_value <= 1.0):
            _fail(f"{cell} ari_tost_p_tost outside [0,1]")
        if ci_low > ci_high:
            _fail(f"{cell} TOST 90% CI is inverted")
        stored_equivalent = _as_bool(
            row.get("Pass_Equivalence"), where=f"{cell}/Pass_Equivalence"
        )
        stored_pass = _as_bool(row.get("Pass"), where=f"{cell}/Pass")
        implied_equivalent = (
            p_value < 0.05 and ci_low > -0.01 and ci_high < 0.01
        )
        if stored_equivalent != implied_equivalent or stored_pass != implied_equivalent:
            _fail(f"{cell} pass flag contradicts its TOST p-value/90% CI")
        csv_thresholds = {
            "Threshold_ARI": 0.01,
            "Threshold_Waste": 0.04,
            "Threshold_SLCA": -0.10,
            "Threshold_RLE": -0.12,
            "Threshold_Carbon": 250.0,
            "Threshold_Equity": -0.06,
            "Threshold_CVR": 0.15,
            "Threshold_LatencyMs": 100.0,
        }
        for field, expected in csv_thresholds.items():
            try:
                actual = float(row[field])
            except (KeyError, TypeError, ValueError):
                _fail(f"{cell}/{field} is not numeric")
            if actual != expected:
                _fail(f"{cell}/{field} differs from the declared threshold")
    print("[PASS] stress_passfail.csv schema")


def _validate_h3_design_meta(summary_meta: Any) -> None:
    """Validate the locked H3 execution posture recorded by the aggregator."""

    if not isinstance(summary_meta, dict):
        _fail("stress_summary.json lacks design metadata")
    if summary_meta.get("max_rows") is not None:
        _fail("stress_summary.json is not a complete 288-step run")
    if int(summary_meta.get(
        "adaptation_episodes_per_stressed_condition", -1,
    )) != 3 or int(summary_meta.get(
        "frozen_evaluation_episodes_per_stressed_condition", -1,
    )) != 1:
        _fail("stress_summary.json does not use 3 adaptation + 1 frozen episode")
    if summary_meta.get("nominal_reference") != (
        "reused_primary_benchmark_episode_3"
    ):
        _fail("stress_summary.json does not reuse the primary nominal endpoint")
    if summary_meta.get("mcp_reliability_posture") != "false":
        _fail("stress_summary.json changes the canonical MCP reliability posture")
    adaptation_posture = str(summary_meta.get("adaptation_posture", ""))
    if (
        "primary nominal endpoint is reused" not in adaptation_posture
        or "retains a no-update frozen episode 3" not in adaptation_posture
    ):
        _fail("stress_summary.json lacks nominal-reuse/frozen-evaluation metadata")
    if "fresh in-memory decision history at every episode" not in str(
        summary_meta.get("decision_history_posture", "")
    ):
        _fail("stress_summary.json lacks fresh-history metadata")


def _validate_h3_test() -> None:
    data = _load_json(RESULTS_DIR / "stress_h3_test.json")
    if data.get("test") != "paired one-sample TOST on seed-level ARI differences":
        _fail("stress_h3_test.json has an unexpected inferential test")
    if int(data.get("n_cells", 0)) != 25:
        _fail("stress_h3_test.json must report 25 scenario-stressor cells")
    cells = data.get("cells")
    if not isinstance(cells, list) or len(cells) != 25:
        _fail("stress_h3_test.json cells must be a 25-record list")
    margin = float(data.get("equivalence_margin", -1.0))
    if margin != 0.01:
        _fail(f"stress_h3_test.json equivalence margin is {margin}, expected declared 0.01")
    keys = [(cell.get("Scenario"), cell.get("Stressor")) for cell in cells]
    expected_keys = {
        (scenario, stressor)
        for scenario in EXPECTED_SCENARIOS for stressor in EXPECTED_STRESSORS
    }
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        _fail("stress_h3_test.json is not the exact unique 5 x 5 H3 panel")
    by_key = {key: cell for key, cell in zip(keys, cells, strict=True)}

    passfail_path = RESULTS_DIR / "stress_passfail.csv"
    with passfail_path.open("r", encoding="utf-8", newline="") as handle:
        passfail_rows = [
            row for row in csv.DictReader(handle)
            if row.get("Method") == "agribrain"
            and row.get("comparison_type") != "cross_mode_under_stress"
        ]
    passfail_by_key = {
        (row.get("Scenario"), row.get("Stressor")): row
        for row in passfail_rows
    }
    if len(passfail_by_key) != 25 or set(passfail_by_key) != expected_keys:
        _fail("stress_passfail.csv is not the exact unique H3 panel")

    summary = _load_json(RESULTS_DIR / "stress_summary.json")
    summary_meta = summary.get("meta") if isinstance(summary, dict) else None
    _validate_h3_design_meta(summary_meta)
    if summary_meta.get("thresholds") != EXPECTED_STRESS_THRESHOLDS:
        _fail("stress_summary.json has noncanonical stress thresholds")
    dose_meta = summary_meta.get("mcp_fault_dose")
    if not isinstance(dose_meta, dict) or (
        dose_meta.get("full_trace_scheduled_opportunity_steps") != 28
        or dose_meta.get("full_trace_total_steps") != 288
    ):
        _fail("stress_summary.json has incorrect fault-dose metadata")
    result_block = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(result_block, dict) or set(result_block) != set(EXPECTED_SCENARIOS):
        _fail("stress_summary.json does not contain the exact scenario panel")
    for scenario in EXPECTED_SCENARIOS:
        scenario_result = result_block.get(scenario)
        if not isinstance(scenario_result, dict):
            _fail(f"stress_summary.json missing {scenario}")
        if scenario_result.get("baseline_seed_list") != list(EXPECTED_SEEDS):
            _fail(f"stress_summary.json {scenario} has a noncanonical seed list")
        baseline = scenario_result.get("baseline_by_seed")
        if not isinstance(baseline, dict):
            _fail(f"stress_summary.json {scenario} lacks baseline_by_seed")
        for stressor in EXPECTED_STRESSORS:
            stressed = scenario_result.get(stressor)
            if not isinstance(stressed, dict):
                _fail(f"stress_summary.json missing {scenario}/{stressor}")
            differences = [
                float(_stress_cell(stressed, seed, "agribrain")["ari"])
                - float(_stress_cell(baseline, seed, "agribrain")["ari"])
                for seed in EXPECTED_SEEDS
            ]
            recomputed = _recompute_tost(differences, margin)
            cell = by_key[(scenario, stressor)]
            numeric_pairs = {
                "ari_delta": recomputed["mean"],
                "ari_tost_p_tost": recomputed["p_tost"],
                "ari_tost_ci90_low": recomputed["ci90_low"],
                "ari_tost_ci90_high": recomputed["ci90_high"],
            }
            for field, expected in numeric_pairs.items():
                try:
                    actual = float(cell[field])
                except (KeyError, TypeError, ValueError):
                    _fail(f"{scenario}/{stressor}/{field} is missing or nonnumeric")
                if not __import__("math").isclose(
                    actual, expected, rel_tol=1e-10, abs_tol=1e-12,
                ):
                    _fail(f"{scenario}/{stressor}/{field} disagrees with raw seeds")
                try:
                    csv_value = float(passfail_by_key[(scenario, stressor)][field])
                except (KeyError, TypeError, ValueError):
                    _fail(f"stress_passfail.csv {scenario}/{stressor}/{field} is invalid")
                if not __import__("math").isclose(
                    csv_value, expected, rel_tol=1e-10, abs_tol=1e-12,
                ):
                    _fail(
                        f"stress_passfail.csv {scenario}/{stressor}/{field} "
                        "disagrees with raw seeds"
                    )
            if _as_bool(
                cell.get("Pass_Equivalence"),
                where=f"{scenario}/{stressor}/Pass_Equivalence",
            ) != recomputed["equivalent"]:
                _fail(f"{scenario}/{stressor} equivalence flag disagrees with raw seeds")
            csv_cell = passfail_by_key[(scenario, stressor)]
            for field in ("Pass", "Pass_Equivalence"):
                if _as_bool(
                    csv_cell.get(field),
                    where=f"stress_passfail.csv {scenario}/{stressor}/{field}",
                ) != recomputed["equivalent"]:
                    _fail(
                        f"stress_passfail.csv {scenario}/{stressor}/{field} "
                        "disagrees with raw seeds"
                    )

            fault_condition = stressor in {"mcp_fault_injection", "compounded"}
            expected_schedule = 28 if fault_condition else 0
            dose_by_field = {
                "fault_injection_scheduled_opportunity_steps": [],
                "fault_injection_trigger_steps": [],
                "fault_injected_tool_result_count": [],
            }
            for seed in EXPECTED_SEEDS:
                dose = _stress_cell(stressed, seed, "agribrain")
                try:
                    scheduled = int(dose["fault_injection_scheduled_opportunity_steps"])
                    triggered = int(dose["fault_injection_trigger_steps"])
                    replaced = int(dose["fault_injected_tool_result_count"])
                except (KeyError, TypeError, ValueError):
                    _fail(f"{scenario}/{stressor}/seed={seed} lacks fault-dose fields")
                dose_by_field[
                    "fault_injection_scheduled_opportunity_steps"
                ].append(scheduled)
                dose_by_field["fault_injection_trigger_steps"].append(triggered)
                dose_by_field["fault_injected_tool_result_count"].append(replaced)
                if scheduled != expected_schedule or not (0 <= triggered <= scheduled):
                    _fail(f"{scenario}/{stressor}/seed={seed} has an invalid fault dose")
                if fault_condition:
                    if triggered <= 0 or replaced < triggered:
                        _fail(
                            f"{scenario}/{stressor}/seed={seed} did not receive "
                            "the declared MCP fault treatment"
                        )
                elif triggered or replaced:
                    _fail(
                        f"{scenario}/{stressor}/seed={seed} reports fault exposure "
                        "in a non-fault condition"
                    )

            # Recompute manuscript-facing dose summaries from the seed panel.
            # This prevents a 28-step schedule from being silently reported as
            # 28 observed drops when the MCP channel was unavailable.
            import statistics
            for dose_field, values in dose_by_field.items():
                expected_aggregates = {
                    f"{dose_field}_mean": statistics.fmean(values),
                    f"{dose_field}_min": min(values),
                    f"{dose_field}_max": max(values),
                }
                for aggregate_field, expected_value in expected_aggregates.items():
                    for source_name, source_cell in (
                        ("stress_h3_test.json", cell),
                        ("stress_passfail.csv", passfail_by_key[(scenario, stressor)]),
                    ):
                        try:
                            actual_value = float(source_cell[aggregate_field])
                        except (KeyError, TypeError, ValueError):
                            _fail(
                                f"{source_name} {scenario}/{stressor}/"
                                f"{aggregate_field} is missing or nonnumeric"
                            )
                        if not __import__("math").isclose(
                            actual_value, float(expected_value),
                            rel_tol=1e-12, abs_tol=1e-12,
                        ):
                            _fail(
                                f"{source_name} {scenario}/{stressor}/"
                                f"{aggregate_field} disagrees with raw seeds"
                            )

    expected_supported = all(
        _as_bool(cell.get("Pass_Equivalence"), where="H3 cell") for cell in cells
    )
    if bool(data.get("supported_all_cells")) != expected_supported:
        _fail("stress_h3_test.json supported_all_cells is inconsistent")
    if int(data.get("n_cells_equivalent", -1)) != sum(
        _as_bool(cell.get("Pass_Equivalence"), where="H3 cell") for cell in cells
    ):
        _fail("stress_h3_test.json n_cells_equivalent is inconsistent")
    print("[PASS] exact H3 Cartesian panel + TOST recomputation")


def _expected_manifest_paths(
    run_tag: str,
    *,
    include_receipt: bool,
    include_recovery: bool = False,
) -> set[str]:
    expected = set(EXPECTED_TOP_LEVEL_ARTIFACTS)
    if include_receipt:
        expected.add(VALIDATION_RECEIPT_NAME)
    expected.add(f"core_submission_receipts/{run_tag}.json")
    if include_recovery:
        expected.update({
            f"publication_recovery_receipts/{run_tag}.json",
            f"preserved_raw_manifests/{run_tag}.json",
        })
    expected.update(
        f"benchmark_seeds/seed_{seed}.json" for seed in EXPECTED_SEEDS
    )
    expected.update(
        f"decision_ledger_per_seed/{run_tag}/seed_{seed}/{mode}__{scenario}.jsonl"
        for seed in EXPECTED_SEEDS
        for mode in EXPECTED_MODES
        for scenario in EXPECTED_SCENARIOS
    )
    expected.update(
        f"decision_ledger_h3/{run_tag}/{scenario}/{stressor}/seed_{seed}/"
        f"agribrain__{scenario}.jsonl"
        for scenario in EXPECTED_SCENARIOS
        for stressor in EXPECTED_STRESSORS
        for seed in EXPECTED_SEEDS
    )
    expected.update(
        f"stress_runs/{run_tag}/{scenario}/{name}"
        for scenario in EXPECTED_SCENARIOS
        for name in EXPECTED_STRESS_TASK_FILES
    )
    return expected


def _validate_manifest_inventory(
    manifest: dict[str, Any],
    *,
    receipt_expected: bool,
    recovery_authorization: dict[str, object] | None = None,
) -> dict[str, int]:
    """Require the exact raw evidence inventory declared by the protocol."""

    if manifest.get("schema_version") != 2:
        raise ValueError("artifact manifest schema_version must be exactly 2")
    commit = manifest.get("git_commit")
    simulation_commit = manifest.get("simulation_source_commit")
    publication_commit = manifest.get("publication_code_commit")
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
        for value in (commit, simulation_commit, publication_commit)
    ):
        raise ValueError("artifact manifest commit identities must be full Git SHA-1s")
    dual = manifest.get("dual_provenance") is True
    if dual:
        if not (
            commit == simulation_commit
            and simulation_commit != publication_commit
            and manifest.get("git_dirty") is False
            and recovery_authorization is not None
            and manifest.get("recovery_authorization") == recovery_authorization
            and recovery_authorization.get("simulation_rerun") is False
            and recovery_authorization.get("validated") is True
        ):
            raise ValueError(
                "dual-provenance publication evidence requires the exact "
                "validated recovery authorization"
            )
    elif not (
        commit == simulation_commit == publication_commit
        and manifest.get("dual_provenance") is False
        and manifest.get("git_dirty") is False
        and manifest.get("recovery_authorization") is None
        and recovery_authorization is None
    ):
        raise ValueError("publication evidence must be a clean fresh single-commit run")
    if manifest.get("includes_raw_run_artifacts") is not True:
        raise ValueError("artifact manifest must include all raw run artifacts")
    run_tag = manifest.get("artifact_run_tag")
    match = _RUN_TAG_RE.fullmatch(str(run_tag))
    if match is None or match.group(1) != str(commit)[:7]:
        raise ValueError("artifact run tag is missing or not commit-bound")

    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("artifact manifest artifacts must be a list")
    names: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"artifact manifest record {index} is not an object")
        raw_name = record.get("file")
        if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
            raise ValueError(f"artifact manifest record {index} has an unsafe path")
        path = PurePosixPath(raw_name)
        if path.is_absolute() or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError(f"artifact manifest record {index} has an unsafe path")
        name = path.as_posix()
        if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
            raise ValueError(f"artifact manifest record {name} has invalid bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise ValueError(f"artifact manifest record {name} has invalid SHA-256")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("artifact manifest contains duplicate paths")
    if manifest.get("artifact_count") != len(names):
        raise ValueError("artifact manifest artifact_count is inconsistent")
    expected = _expected_manifest_paths(
        str(run_tag),
        include_receipt=receipt_expected,
        include_recovery=dual,
    )
    observed = set(names)
    if observed != expected:
        raise ValueError(
            "artifact manifest does not contain the exact protocol inventory: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return {
        "top_level_artifacts_excluding_receipt": len(EXPECTED_TOP_LEVEL_ARTIFACTS),
        "benchmark_seed_envelopes": len(EXPECTED_SEEDS),
        "primary_retained_decision_ledgers": (
            len(EXPECTED_SEEDS) * len(EXPECTED_MODES) * len(EXPECTED_SCENARIOS)
        ),
        "h3_retained_stressed_decision_ledgers": (
            len(EXPECTED_SEEDS)
            * len(EXPECTED_SCENARIOS)
            * len(EXPECTED_STRESSORS)
        ),
        "raw_stress_task_files": (
            len(EXPECTED_SCENARIOS) * len(EXPECTED_STRESS_TASK_FILES)
        ),
        "core_slurm_submission_receipts": 1,
        "publication_recovery_receipts": int(dual),
        "preserved_raw_manifests": int(dual),
    }


def _recovery_authorization_for_manifest(
    manifest: dict[str, Any],
    *,
    results_dir: Path,
    recovery_receipt: Path | None,
) -> dict[str, object] | None:
    """Independently resolve the only permitted dual-provenance mode."""

    if manifest.get("dual_provenance") is not True:
        if recovery_receipt is not None:
            raise ValueError(
                "a recovery receipt is invalid for fresh single-provenance evidence"
            )
        return None
    if recovery_receipt is None:
        raise ValueError(
            "dual-provenance semantic validation requires --recovery-receipt"
        )
    return validate_recovery_context(
        recovery_receipt,
        results_dir=results_dir,
        run_tag=manifest.get("artifact_run_tag"),
        simulation_commit=manifest.get("simulation_source_commit"),
        publication_commit=manifest.get("publication_code_commit"),
        expected_kind="core",
    )


def _validate_manifest(*, receipt_expected: bool) -> None:
    path = RESULTS_DIR / "artifact_manifest.json"
    if not path.is_file() or path.is_symlink():
        _fail("artifact_manifest.json must be a real, non-symlink file")
    data = _load_json(path)
    if not isinstance(data, dict):
        _fail("artifact_manifest.json is not an object")
    try:
        recovery_authorization = _recovery_authorization_for_manifest(
            data,
            results_dir=RESULTS_DIR,
            recovery_receipt=RECOVERY_RECEIPT_PATH,
        )
        _validate_manifest_inventory(
            data,
            receipt_expected=receipt_expected,
            recovery_authorization=recovery_authorization,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("artifact_manifest.json has empty artifacts list")
    for i, rec in enumerate(artifacts):
        if not isinstance(rec, dict):
            _fail(f"artifact_manifest.json artifacts[{i}] is not an object")
        for key in ("file", "sha256", "bytes"):
            if key not in rec:
                _fail(f"artifact_manifest.json artifacts[{i}] missing {key}")
        source = _safe_manifest_payload(rec["file"])
        if source.stat().st_size != int(rec["bytes"]):
            _fail(f"artifact manifest byte count mismatch for {rec['file']}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != rec["sha256"]:
            _fail(f"artifact manifest literal-byte SHA-256 mismatch for {rec['file']}")
    if "publication_environment.json" not in {
        str(rec.get("file", "")) for rec in artifacts if isinstance(rec, dict)
    }:
        _fail("artifact_manifest.json does not inventory publication_environment.json")
    semantics = data.get("hash_semantics")
    if not isinstance(semantics, dict) or "literal" not in str(
        semantics.get("sha256", "")
    ).lower():
        _fail("artifact_manifest.json does not declare literal-byte hashing")
    print("[PASS] artifact_manifest.json commit + hashes")


def _artifact_set_root(records: list[dict[str, Any]]) -> str:
    """Merkle-bind manifest records without creating a receipt/hash cycle."""

    leaves = [
        hashlib.sha256(json.dumps(
            {
                "file": str(record["file"]),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).digest()
        for record in sorted(records, key=lambda item: str(item["file"]))
        if record.get("file") != VALIDATION_RECEIPT_NAME
    ]
    if not leaves:
        return "0" * 64
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _receipt_contract(
    manifest: dict[str, Any],
    *,
    repo_root: Path | None = None,
    results_dir: Path | None = None,
    recovery_receipt: Path | None = None,
) -> dict[str, Any]:
    contract_repo_root = REPO_ROOT if repo_root is None else repo_root
    contract_results_dir = RESULTS_DIR if results_dir is None else results_dir
    protocol_path = (
        contract_repo_root / "mvp" / "simulation" / "experiment_protocol.json"
    )
    if not protocol_path.is_file():
        raise ValueError("locked experiment protocol is missing")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("artifact manifest lacks records for validation receipt")
    receipt_manifested = any(
        isinstance(record, dict)
        and record.get("file") == VALIDATION_RECEIPT_NAME
        for record in records
    )
    recovery_authorization = _recovery_authorization_for_manifest(
        manifest,
        results_dir=contract_results_dir,
        recovery_receipt=recovery_receipt,
    )
    inventory = _validate_manifest_inventory(
        manifest,
        receipt_expected=receipt_manifested,
        recovery_authorization=recovery_authorization,
    )
    semantic_records = [
        record for record in records
        if isinstance(record, dict)
        and record.get("file") != VALIDATION_RECEIPT_NAME
    ]
    return {
        "schema_version": 1,
        "validation_status": "PASS",
        "validation_scope": "core_publication_evidence",
        "git_commit": manifest.get("git_commit"),
        "simulation_source_commit": manifest.get("simulation_source_commit"),
        "publication_code_commit": manifest.get("publication_code_commit"),
        "run_tag": manifest.get("artifact_run_tag"),
        "fresh_single_commit_run": bool(
            manifest.get("git_dirty") is False
            and manifest.get("dual_provenance") is False
            and manifest.get("git_commit")
            == manifest.get("simulation_source_commit")
            == manifest.get("publication_code_commit")
        ),
        "authorized_deterministic_recovery": bool(
            recovery_authorization is not None
        ),
        "simulation_rerun": (
            False if recovery_authorization is not None else True
        ),
        "recovery_authorization": recovery_authorization,
        "protocol": {
            "file": "mvp/simulation/experiment_protocol.json",
            "bytes": protocol_path.stat().st_size,
            "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        },
        "semantic_artifact_set": {
            "artifact_count_excluding_receipt": len(semantic_records),
            "merkle_root": _artifact_set_root(records),
            "excluded_from_root": [VALIDATION_RECEIPT_NAME],
            "hash_semantics": "manifested literal bytes",
        },
        "exact_manifest_inventory": inventory,
        "locked_accounting": {
            "core_unique_retained_cells": 1_600,
            "core_executed_episodes": 6_100,
            "core_simulated_steps": 1_756_800,
            "h1_directional_tests": 5,
            "h2_directional_tests": 20,
            "h3_equivalence_cells": 25,
        },
        "structural_sensitivity": {
            "included_in_core_receipt": False,
            "required_for_full_submission_evidence": True,
            "required_separate_receipt": (
                "structural_sensitivity_archive_receipt.json"
            ),
        },
    }


def _write_publication_validation_receipt() -> None:
    path = RESULTS_DIR / VALIDATION_RECEIPT_NAME
    if path.exists():
        _fail(f"refusing to overwrite existing validation receipt: {path}")
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    if not isinstance(manifest, dict):
        _fail("artifact manifest is not an object")
    try:
        contract = _receipt_contract(
            manifest,
            results_dir=RESULTS_DIR,
            recovery_receipt=RECOVERY_RECEIPT_PATH,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    payload = {
        **contract,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_checks": [
            "core_slurm_submission_dag",
            "exact_H1_H2_seed_panels_and_inference",
            "deterministic_core_statistical_reaggregation",
            "exact_H3_panel_TOST_and_treatment_exposure",
            "raw_endpoint_and_decision_ledger_recomputation",
            "table_and_paper_export_semantic_projection",
            "forecast_selection_and_predictions",
            "deterministic_derived_artifact_and_H3_replay",
            "figure_provenance_and_exact_inventory",
            "environment_source_and_run_identity",
            "literal_byte_manifest_integrity",
        ],
    }
    if not (
        payload["fresh_single_commit_run"] is True
        or (
            payload["authorized_deterministic_recovery"] is True
            and payload["simulation_rerun"] is False
        )
    ):
        _fail(
            "semantic receipt requires either a fresh clean run or an "
            "authorized deterministic recovery with no simulation rerun"
        )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote immutable semantic validation receipt: {path.name}")


def validate_publication_validation_receipt(
    results_dir: Path,
    *,
    repo_root: Path,
    recovery_receipt: Path | None = None,
) -> None:
    """Verify the semantic receipt against literal files and its manifest.

    This public, exception-based form is shared by the final artifact gate and
    archive builder, preventing packaging from substituting a presence-only
    receipt check.
    """

    try:
        manifest = json.loads(
            (results_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        receipt = json.loads(
            (results_dir / VALIDATION_RECEIPT_NAME).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"cannot load semantic validation receipt inputs: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise ValueError("publication validation receipt inputs are not objects")
    records = manifest.get("artifacts")
    receipt_record = next((
        record for record in records
        if isinstance(record, dict)
        and record.get("file") == VALIDATION_RECEIPT_NAME
    ), None) if isinstance(records, list) else None
    if receipt_record is None:
        raise ValueError(
            "artifact manifest does not hash-bind the semantic validation receipt"
        )
    receipt_bytes = (results_dir / VALIDATION_RECEIPT_NAME).read_bytes()
    if (
        receipt_record.get("bytes") != len(receipt_bytes)
        or receipt_record.get("sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
    ):
        raise ValueError("semantic validation receipt bytes disagree with the manifest")
    expected = _receipt_contract(
        manifest,
        repo_root=repo_root,
        results_dir=results_dir,
        recovery_receipt=recovery_receipt,
    )
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"publication validation receipt disagrees on {key}")
    generated_at = receipt.get("generated_at_utc")
    if not isinstance(generated_at, str):
        raise ValueError("publication validation receipt lacks generated_at_utc")
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            "publication validation receipt has invalid generated_at_utc"
        ) from None
    if parsed.tzinfo is None:
        raise ValueError(
            "publication validation receipt timestamp is not timezone-aware"
        )
    checks = receipt.get("validated_checks")
    if not isinstance(checks, list) or checks != [
        "core_slurm_submission_dag",
        "exact_H1_H2_seed_panels_and_inference",
        "deterministic_core_statistical_reaggregation",
        "exact_H3_panel_TOST_and_treatment_exposure",
        "raw_endpoint_and_decision_ledger_recomputation",
        "table_and_paper_export_semantic_projection",
        "forecast_selection_and_predictions",
        "deterministic_derived_artifact_and_H3_replay",
        "figure_provenance_and_exact_inventory",
        "environment_source_and_run_identity",
        "literal_byte_manifest_integrity",
    ] or any(
        not isinstance(item, str) or not item for item in checks
    ):
        raise ValueError(
            "publication validation receipt lacks the exact check inventory"
        )


def _validate_publication_validation_receipt() -> None:
    try:
        validate_publication_validation_receipt(
            RESULTS_DIR,
            repo_root=REPO_ROOT,
            recovery_receipt=RECOVERY_RECEIPT_PATH,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    print("[PASS] hash-bound semantic publication validation receipt")


def _publication_execution_commit(manifest: dict[str, Any]) -> object:
    """Return the code identity that executed deterministic publication."""

    return (
        manifest.get("publication_code_commit")
        if manifest.get("dual_provenance") is True
        else manifest.get("git_commit")
    )


def _validate_publication_environment() -> None:
    path = RESULTS_DIR / "publication_environment.json"
    data = _load_json(path)
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    if int(data.get("schema_version", -1)) != 2:
        _fail("publication_environment.json has an unsupported schema version")
    if data.get("environment_scope") != "version_resolved_runtime_inventory":
        _fail("publication environment overstates or omits its evidence scope")
    binary_scope = data.get("binary_reproducibility")
    if not isinstance(binary_scope, dict) or (
        binary_scope.get("byte_identical_environment_claimed") is not False
        or binary_scope.get("distribution_artifact_hashes_recorded") is not False
        or binary_scope.get("container_image_digest_recorded") is not False
        or not isinstance(binary_scope.get("interpretation"), str)
    ):
        _fail("publication environment has an inaccurate binary-reproducibility claim")
    expected_environment_commit = _publication_execution_commit(manifest)
    if data.get("git_commit") != expected_environment_commit:
        _fail(
            "publication_environment.json commit differs from the code that "
            "executed publication"
        )
    if data.get("run_tag") != manifest.get("artifact_run_tag"):
        _fail("publication_environment.json run tag differs from artifact manifest")
    packages = data.get("installed_distributions")
    if not isinstance(packages, list) or not packages:
        _fail("publication_environment.json has no installed-distribution inventory")
    if int(data.get("installed_package_count", -1)) != len(packages):
        _fail("publication_environment.json package count is inconsistent")
    normalized_names = []
    for item in packages:
        if not isinstance(item, str) or "==" not in item:
            _fail("publication_environment.json has an invalid distribution record")
        name, version = item.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
        if not normalized or not version.strip() or name != normalized:
            _fail("publication_environment.json distribution inventory is not normalized")
        normalized_names.append(normalized)
    if len(normalized_names) != len(set(normalized_names)):
        _fail("publication_environment.json contains duplicate normalized distributions")

    distribution_validation = data.get("distribution_validation")
    if not isinstance(distribution_validation, dict):
        _fail("publication_environment.json lacks distribution validation")
    for key in ("unique_normalized_names", "lock_versions_match", "core_version_match"):
        if distribution_validation.get(key) is not True:
            _fail(f"publication environment distribution check failed: {key}")
    if distribution_validation.get("unexpected_distributions") != []:
        _fail("publication environment contains distributions outside lock/core/bootstrap")
    locked_count = int(distribution_validation.get("locked_distribution_count", 0))
    if locked_count <= 0:
        _fail("publication environment lacks a positive locked-distribution count")
    applicable_lock = distribution_validation.get("applicable_lock_distributions")
    if not isinstance(applicable_lock, list) or len(applicable_lock) != locked_count:
        _fail("publication environment lacks the exact applicable lock inventory")
    lock_names = []
    for item in applicable_lock:
        if not isinstance(item, str) or "==" not in item:
            _fail("publication environment has an invalid applicable lock record")
        name, version = item.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
        if not normalized or name != normalized or not version.strip():
            _fail("publication environment applicable lock inventory is not normalized")
        lock_names.append(normalized)
        if item not in packages:
            _fail(f"publication environment is missing locked distribution {item}")
    if len(lock_names) != len(set(lock_names)):
        _fail("publication environment applicable lock inventory contains duplicates")
    core_distribution = distribution_validation.get("core_distribution")
    if core_distribution not in packages:
        _fail("publication environment core distribution is absent from inventory")
    core_name = str(core_distribution).split("==", 1)[0]
    bootstrap_names = distribution_validation.get("allowed_bootstrap_distributions")
    if not isinstance(bootstrap_names, list) or any(
        name not in {"pip", "setuptools", "wheel"} for name in bootstrap_names
    ) or len(bootstrap_names) != len(set(bootstrap_names)):
        _fail("publication environment has an invalid bootstrap distribution list")
    expected_names = set(lock_names) | {core_name} | set(bootstrap_names)
    if set(normalized_names) != expected_names:
        _fail("publication environment inventory differs from lock/core/bootstrap set")

    venv = data.get("virtual_environment")
    if not isinstance(venv, dict):
        _fail("publication_environment.json lacks virtual-environment identity")
    if venv.get("run_scoped") is not True or venv.get("isolated_from_base_prefix") is not True:
        _fail("publication environment is not an isolated run-scoped venv")
    expected_venv_id = f".publication_venvs/{data.get('run_tag')}"
    if venv.get("path_id") != expected_venv_id:
        _fail("publication environment venv identity differs from run tag")
    python = data.get("python")
    if not isinstance(python, dict) or not python.get("version"):
        _fail("publication_environment.json lacks Python version details")
    if not re.fullmatch(r"3\.11(?:\.\d+)?", str(python.get("version"))):
        _fail("publication environment does not use the locked Python 3.11 minor")

    repo_root = REPO_ROOT
    for key in ("requirements_lock", "backend_project", "publication_environment_script"):
        rec = data.get(key)
        if not isinstance(rec, dict):
            _fail(f"publication_environment.json missing {key}")
        source = repo_root / str(rec.get("path", ""))
        if not source.is_file():
            _fail(f"publication environment source missing: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != rec.get("sha256"):
            _fail(f"publication environment hash mismatch for {source}")
        if int(rec.get("bytes", -1)) != source.stat().st_size:
            _fail(f"publication environment byte count mismatch for {source}")
    project_rec = data["backend_project"]
    project_path = repo_root / str(project_rec["path"])
    project = tomllib.loads(project_path.read_text(encoding="utf-8")).get("project", {})
    project_name = re.sub(r"[-_.]+", "-", str(project.get("name", "")).strip()).lower()
    project_version = str(project.get("version", "")).strip()
    if core_distribution != f"{project_name}=={project_version}":
        _fail("publication environment core version differs from backend pyproject")
    print("[PASS] version-resolved publication environment inventory")


def _validate_forecast_receipt() -> None:
    """Bind the internal rolling-origin evidence to data, code, and protocol."""
    import math

    summary_path = RESULTS_DIR / "forecast_validation_summary.json"
    predictions_path = RESULTS_DIR / "forecast_validation_predictions.csv"
    summary = _load_json(summary_path)
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    protocol_path = REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json"
    protocol = _load_json(protocol_path)
    forecast_protocol = protocol.get("forecast_protocol")
    if not isinstance(forecast_protocol, dict):
        _fail("experiment_protocol.json lacks forecast_protocol")

    manifest_files = {
        str(rec.get("file", ""))
        for rec in manifest.get("artifacts", [])
        if isinstance(rec, dict)
    }
    required_receipts = {
        summary_path.name, predictions_path.name,
    }
    if not required_receipts.issubset(manifest_files):
        _fail("artifact manifest does not inventory both forecast receipts")

    if int(summary.get("schema_version", -1)) != 1:
        _fail("forecast validation summary has an unsupported schema version")
    if summary.get("validation_scope") != "internal synthetic benchmark only":
        _fail("forecast receipt must be limited to internal synthetic validation")
    if summary.get("external_validation") is not False:
        _fail("forecast receipt incorrectly claims external validation")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("scope") != "publication":
        _fail("forecast receipt is not publication-scoped")
    expected_source = manifest.get(
        "simulation_source_commit", manifest.get("git_commit"),
    )
    if provenance.get("source_commit") != expected_source:
        _fail("forecast receipt source commit differs from the simulation manifest")
    if provenance.get("run_tag") != manifest.get("artifact_run_tag"):
        _fail("forecast receipt run tag differs from the publication manifest")

    repo_root = REPO_ROOT
    data_path = repo_root / "agribrain" / "backend" / "src" / "data_spinach.csv"
    dataset = summary.get("dataset")
    if not isinstance(dataset, dict):
        _fail("forecast receipt lacks dataset identity")
    expected_data_rel = data_path.relative_to(repo_root).as_posix()
    expected_data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if dataset.get("path") != expected_data_rel:
        _fail("forecast receipt points to the wrong dataset")
    if dataset.get("sha256") != expected_data_hash:
        _fail("forecast receipt dataset hash differs from the repository bytes")
    if forecast_protocol.get("dataset_path") != expected_data_rel:
        _fail("locked forecast protocol points to the wrong dataset")
    if forecast_protocol.get("dataset_sha256") != expected_data_hash:
        _fail("locked forecast-protocol dataset hash is stale")

    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        data_rows = list(csv.DictReader(handle))
    n_data = len(data_rows)
    if int(dataset.get("n_rows", -1)) != n_data:
        _fail("forecast receipt dataset row count is incorrect")
    if int(forecast_protocol.get("n_rows", -1)) != n_data:
        _fail("forecast protocol dataset row count is incorrect")
    if n_data != 288:
        _fail("publication forecast receipt requires the locked 288-row series")

    rolling = summary.get("rolling_origin")
    expected_rolling = {
        "horizon_steps": 1,
        "lookback_steps": 48,
        "origin_stride": 1,
        "retrained_at_each_origin": True,
        "targets_seen_during_fit": False,
    }
    if not isinstance(rolling, dict) or any(
        rolling.get(key) != value for key, value in expected_rolling.items()
    ):
        _fail("forecast receipt does not use the locked rolling-origin design")
    temporal_split = dataset.get("temporal_split")
    if temporal_split != {
        "development": "first 60 percent",
        "validation": "next 20 percent",
        "test": "final 20 percent",
    }:
        _fail("forecast receipt does not declare the locked 60/20/20 split")

    expected_columns = [
        "model", "series", "split", "origin_index", "target_index",
        "history_start_index", "history_end_index", "history_count",
        "target", "prediction", "persistence_prediction", "interval_lower",
        "interval_upper", "interval_nominal_coverage", "no_lookahead",
    ]
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            _fail("forecast predictions CSV has the wrong columns or order")
        prediction_rows = list(reader)

    receipt = summary.get("predictions_artifact")
    if not isinstance(receipt, dict) or receipt.get("file") != predictions_path.name:
        _fail("forecast summary lacks its predictions-artifact identity")
    if int(receipt.get("row_count", -1)) != len(prediction_rows):
        _fail("forecast predictions row count disagrees with its receipt")
    if int(receipt.get("bytes", -1)) != predictions_path.stat().st_size:
        _fail("forecast predictions byte count disagrees with its receipt")
    if receipt.get("sha256") != hashlib.sha256(predictions_path.read_bytes()).hexdigest():
        _fail("forecast predictions hash disagrees with its receipt")

    model_series = {
        "lstm_demand": "demand_units",
        "holt_linear_demand_candidate": "demand_units",
        "persistence_demand": "demand_units",
        "holt_linear_supply_proxy": "inventory_units",
        "persistence_supply_proxy": "inventory_units",
    }
    validation_start = math.floor(0.60 * n_data)
    test_start = math.floor(0.80 * n_data)
    expected_targets = set(range(validation_start, n_data))
    seen: set[tuple[str, int]] = set()
    parsed_by_model_split: dict[tuple[str, str], list[dict[str, float]]] = {}
    for line_number, row in enumerate(prediction_rows, start=2):
        model = row.get("model", "")
        series = row.get("series", "")
        if model not in model_series or series != model_series[model]:
            _fail(f"forecast CSV line {line_number} has an invalid model/series pair")
        try:
            origin = int(row["origin_index"])
            target_index = int(row["target_index"])
            history_start = int(row["history_start_index"])
            history_end = int(row["history_end_index"])
            history_count = int(row["history_count"])
            numeric = {
                key: float(row[key]) for key in (
                    "target", "prediction", "persistence_prediction",
                    "interval_lower", "interval_upper",
                    "interval_nominal_coverage",
                )
            }
        except (KeyError, TypeError, ValueError):
            _fail(f"forecast CSV line {line_number} contains invalid numeric data")
        if not all(math.isfinite(value) for value in numeric.values()):
            _fail(f"forecast CSV line {line_number} contains non-finite data")
        if target_index != origin + 1 or history_end != origin:
            _fail(f"forecast CSV line {line_number} violates the no-lookahead boundary")
        if history_start != max(0, origin + 1 - 48):
            _fail(f"forecast CSV line {line_number} has the wrong history start")
        if history_count != origin + 1 - history_start:
            _fail(f"forecast CSV line {line_number} has the wrong history count")
        if row.get("no_lookahead", "").strip().lower() != "true":
            _fail(f"forecast CSV line {line_number} does not attest no-lookahead")
        expected_split = "validation" if target_index < test_start else "test"
        if row.get("split") != expected_split:
            _fail(f"forecast CSV line {line_number} has the wrong temporal split")
        if not math.isclose(
            numeric["interval_nominal_coverage"], 0.95,
            rel_tol=0.0, abs_tol=1e-15,
        ):
            _fail(f"forecast CSV line {line_number} has the wrong nominal coverage")
        expected_target = float(data_rows[target_index][series])
        if not math.isclose(
            numeric["target"], expected_target, rel_tol=0.0, abs_tol=1e-12,
        ):
            _fail(f"forecast CSV line {line_number} target differs from source data")
        key = (model, target_index)
        if key in seen:
            _fail(f"forecast CSV duplicates {model} target {target_index}")
        seen.add(key)
        parsed_by_model_split.setdefault((model, expected_split), []).append(numeric)

    expected_seen = {
        (model, target_index)
        for model in model_series for target_index in expected_targets
    }
    if seen != expected_seen:
        _fail("forecast CSV does not contain the exact model-by-target panel")

    def recompute_metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
        n = len(rows)
        errors = [row["prediction"] - row["target"] for row in rows]
        persistence_errors = [
            row["persistence_prediction"] - row["target"] for row in rows
        ]
        mae = sum(abs(value) for value in errors) / n
        persistence_mae = sum(abs(value) for value in persistence_errors) / n
        return {
            "n": n,
            "mae": mae,
            "rmse": math.sqrt(sum(value * value for value in errors) / n),
            "mean_error": sum(errors) / n,
            "persistence_mae": persistence_mae,
            "persistence_rmse": math.sqrt(
                sum(value * value for value in persistence_errors) / n
            ),
            "mae_improvement_vs_persistence_fraction": (
                (persistence_mae - mae) / persistence_mae
            ),
            "interval_coverage": sum(
                row["interval_lower"] <= row["target"] <= row["interval_upper"]
                for row in rows
            ) / n,
            "mean_interval_width": sum(
                row["interval_upper"] - row["interval_lower"] for row in rows
            ) / n,
        }

    stored_metrics = summary.get("metrics")
    if not isinstance(stored_metrics, dict) or set(stored_metrics) != set(model_series):
        _fail("forecast summary has the wrong model metric panel")
    for model in model_series:
        if set(stored_metrics.get(model, {})) != {"validation", "test"}:
            _fail(f"forecast summary has incomplete splits for {model}")
        for split in ("validation", "test"):
            expected = recompute_metrics(parsed_by_model_split[(model, split)])
            stored = stored_metrics[model][split]
            if not isinstance(stored, dict) or set(stored) != set(expected):
                _fail(f"forecast summary has malformed metrics for {model}/{split}")
            for name, expected_value in expected.items():
                try:
                    actual_value = float(stored[name])
                except (TypeError, ValueError):
                    _fail(f"forecast summary {model}/{split}/{name} is nonnumeric")
                if not math.isclose(
                    actual_value, float(expected_value), rel_tol=1e-12, abs_tol=1e-12,
                ):
                    _fail(f"forecast summary {model}/{split}/{name} disagrees with CSV")

    selection = summary.get("selection_rule")
    if not isinstance(selection, dict):
        _fail("forecast summary lacks its selection rule")
    if selection.get("criterion") != "minimum validation-segment RMSE":
        _fail("forecast receipt uses the wrong selection criterion")
    if selection.get("test_segment_used_for_selection") is not False:
        _fail("forecast receipt used the locked test segment for selection")
    demand_models = (
        "lstm_demand", "holt_linear_demand_candidate", "persistence_demand",
    )
    supply_models = (
        "holt_linear_supply_proxy", "persistence_supply_proxy",
    )
    selected_demand_model = min(
        demand_models, key=lambda model: stored_metrics[model]["validation"]["rmse"],
    )
    selected_supply_model = min(
        supply_models, key=lambda model: stored_metrics[model]["validation"]["rmse"],
    )
    if selection.get("selected_demand_model_id") != selected_demand_model:
        _fail("forecast receipt demand selection is not the validation RMSE winner")
    if selection.get("selected_supply_proxy_model_id") != selected_supply_model:
        _fail("forecast receipt supply selection is not the validation RMSE winner")
    if selection.get("selected_demand_method") != "holt_linear":
        _fail("forecast receipt does not select the locked Holt-linear demand method")
    if selection.get("selected_supply_proxy_method") != "persistence":
        _fail("forecast receipt does not select the locked persistence supply method")
    if selection.get("selected_demand_method") != forecast_protocol.get(
        "selected_demand_method"
    ):
        _fail("forecast receipt demand method differs from the locked protocol")
    if selection.get("selected_supply_proxy_method") != forecast_protocol.get(
        "selected_supply_proxy_method"
    ):
        _fail("forecast receipt supply method differs from the locked protocol")

    metric_bindings = {
        "demand_holt_linear": "holt_linear_demand_candidate",
        "demand_persistence": "persistence_demand",
        "demand_lstm": "lstm_demand",
        "supply_holt_linear": "holt_linear_supply_proxy",
        "supply_persistence": "persistence_supply_proxy",
    }
    for protocol_key, split in (
        ("validation_rmse", "validation"),
        ("test_rmse_report_only", "test"),
    ):
        locked = forecast_protocol.get(protocol_key)
        if not isinstance(locked, dict) or set(locked) != set(metric_bindings):
            _fail(f"forecast protocol has an incomplete {protocol_key} panel")
        for name, model in metric_bindings.items():
            actual = float(stored_metrics[model][split]["rmse"])
            if not math.isclose(
                actual, float(locked[name]), rel_tol=1e-12, abs_tol=1e-12,
            ):
                _fail(f"forecast receipt {name}/{split} differs from locked protocol")
    print("[PASS] forecast validation data, no-lookahead panel, metrics, and selection")


def _validate_evidence_scope_metadata() -> None:
    """Pin the scope/provenance of every ledger-derived publication result."""
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    expected_commit = str(manifest.get("git_commit", "")).strip()
    expected_run_tag = str(manifest.get("artifact_run_tag", "")).strip()
    expected_ledger_root = (
        "mvp/simulation/results/decision_ledger_per_seed/"
        f"{expected_run_tag}"
    )
    expected_episode_scope = "final episode per scenario-mode-seed arm"
    expected_history_scope = "earlier decisions in the same episode only"
    evidence_files = (
        "channel_attribution_aggregate.json",
        "channel_complementarity_test.json",
        "explainability_metrics.json",
    )

    for name in evidence_files:
        data = _load_json(RESULTS_DIR / name)
        if name == "explainability_metrics.json" and data.get("threshold") != 0.10:
            _fail(
                "explainability_metrics.json does not use the locked 0.10 "
                "headline threshold"
            )
        meta = data.get("_meta") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            _fail(f"{name} missing _meta evidence-scope object")
        required = {
            "source_commit", "ledger_root", "seed_count", "run_tag",
            "episode_scope", "decision_history_scope",
        }
        missing = sorted(required.difference(meta))
        if missing:
            _fail(f"{name} evidence scope missing: {', '.join(missing)}")

        source_commit = str(meta.get("source_commit", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            _fail(f"{name} source_commit is not a full lowercase Git SHA-1")
        if source_commit != expected_commit:
            _fail(f"{name} source_commit differs from artifact manifest")
        if str(meta.get("run_tag", "")).strip() != expected_run_tag:
            _fail(f"{name} run_tag differs from artifact manifest")
        try:
            seed_count = int(meta.get("seed_count", -1))
        except (TypeError, ValueError):
            _fail(f"{name} seed_count is not an integer")
        if seed_count != 20:
            _fail(f"{name} must report the complete 20-seed evidence panel")
        if name == "channel_complementarity_test.json":
            try:
                source_seed_count = int(meta.get("source_seed_count", -1))
            except (TypeError, ValueError):
                _fail(f"{name} source_seed_count is not an integer")
            if source_seed_count != 20:
                _fail(f"{name} does not contain all 20 source seed ledgers")
        elif name == "channel_attribution_aggregate.json":
            seeds = meta.get("seeds")
            if not isinstance(seeds, list) or len(set(seeds)) != seed_count:
                _fail(f"{name} seed list conflicts with seed_count")
            cells = data.get("by_scenario_mode")
            expected_scenarios = {
                "heatwave", "overproduction", "cyber_outage",
                "adaptive_pricing", "baseline",
            }
            if not isinstance(cells, dict) or set(cells) != expected_scenarios:
                _fail(f"{name} does not contain the exact five-scenario panel")
            for scenario in sorted(expected_scenarios):
                agribrain = cells.get(scenario, {}).get("agribrain")
                if not isinstance(agribrain, dict):
                    _fail(f"{name} missing {scenario}/agribrain attribution cell")
                if int(agribrain.get("n_seeds", -1)) != seed_count:
                    _fail(f"{name} {scenario}/agribrain cell is not a 20-seed estimate")

        ledger_root = str(meta.get("ledger_root", "")).replace("\\", "/")
        if ledger_root != expected_ledger_root:
            _fail(
                f"{name} ledger_root is {ledger_root!r}; expected the "
                "consolidated publication ledger"
            )
        if meta.get("episode_scope") != expected_episode_scope:
            _fail(f"{name} has an incorrect episode_scope")
        if meta.get("decision_history_scope") != expected_history_scope:
            _fail(f"{name} has an incorrect decision_history_scope")

        legacy_seed_count = data.get("n_seeds")
        if legacy_seed_count is None and name == "channel_attribution_aggregate.json":
            legacy_seed_count = meta.get("n_seeds")
        if legacy_seed_count is not None:
            try:
                legacy_seed_count = int(legacy_seed_count)
            except (TypeError, ValueError):
                _fail(f"{name} existing n_seeds field is not an integer")
            if legacy_seed_count != seed_count:
                _fail(f"{name} seed_count conflicts with its existing n_seeds field")

    print("[PASS] ledger-derived evidence scope + provenance metadata")


def _validate_figure_source_identity(
    figure_provenance: dict[str, Any], manifest: dict[str, Any],
) -> None:
    """Require explicit raw-simulation and renderer-code identities."""

    expected_commit = manifest.get("git_commit")
    expected_simulation_commit = manifest.get(
        "simulation_source_commit", expected_commit,
    )
    expected_renderer_commit = manifest.get(
        "publication_code_commit", expected_commit,
    )
    if figure_provenance.get("schema_version") != 3:
        _fail("figure_provenance.json does not use dual-identity schema 3")
    if figure_provenance.get("source_commit") != expected_commit:
        _fail("figure_provenance.json source_commit differs from the manifest")
    if (
        figure_provenance.get("source_commit_semantics")
        != "raw_input_simulation_commit"
    ):
        _fail("figure_provenance.json source_commit semantics are ambiguous")
    if (
        figure_provenance.get("simulation_source_commit")
        != expected_simulation_commit
    ):
        _fail("figure_provenance.json simulation commit differs from the manifest")
    if figure_provenance.get("renderer_code_commit") != expected_renderer_commit:
        _fail("figure_provenance.json renderer commit differs from the manifest")
    if figure_provenance.get("dual_provenance") is not (
        expected_renderer_commit != expected_simulation_commit
    ):
        _fail("figure_provenance.json dual-provenance flag is inconsistent")


def _validate_run_provenance() -> None:
    """Ensure staged seed/stress artifacts retain the manifest identity."""
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    expected_commit = manifest.get("git_commit")
    expected_run_tag = manifest.get("artifact_run_tag")
    for name in ("benchmark_summary.json", "benchmark_significance.json"):
        aggregate = _load_json(RESULTS_DIR / name)
        aggregate_meta = aggregate.get("_meta") if isinstance(aggregate, dict) else None
        if not isinstance(aggregate_meta, dict):
            _fail(f"{name} has no run-provenance metadata")
        if aggregate_meta.get("source_commit") != expected_commit:
            _fail(f"{name} source_commit differs from the manifest")
        if aggregate_meta.get("run_tag") != expected_run_tag:
            _fail(f"{name} run_tag differs from the manifest")
    for seed in EXPECTED_SEEDS:
        path = RESULTS_DIR / "benchmark_seeds" / f"seed_{seed}.json"
        data = _load_json(path)
        meta = data.get("_meta") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            _fail(f"{path.name} has no run-provenance metadata")
        if meta.get("source_commit") != expected_commit:
            _fail(f"{path.name} source_commit differs from the manifest")
        if meta.get("run_tag") != expected_run_tag:
            _fail(f"{path.name} run_tag differs from the manifest")
        if data.get("seed") != seed:
            _fail(f"{path.name} seed field differs from its filename")

    stress_summary = _load_json(RESULTS_DIR / "stress_summary.json")
    stress_meta = stress_summary.get("meta") if isinstance(stress_summary, dict) else None
    if not isinstance(stress_meta, dict):
        _fail("stress_summary.json has no provenance metadata")
    if stress_meta.get("source_commit") != expected_commit:
        _fail("stress_summary.json source_commit differs from the manifest")
    if stress_meta.get("run_tag") != expected_run_tag:
        _fail("stress_summary.json run_tag differs from the manifest")

    h3 = _load_json(RESULTS_DIR / "stress_h3_test.json")
    if h3.get("source_commit") != expected_commit:
        _fail("stress_h3_test.json source_commit differs from the manifest")
    if h3.get("run_tag") != expected_run_tag:
        _fail("stress_h3_test.json run_tag differs from the manifest")

    saturation = _load_json(RESULTS_DIR / "channel_saturation_analysis.json")
    saturation_meta = (
        saturation.get("_meta") if isinstance(saturation, dict) else None
    )
    if not isinstance(saturation_meta, dict):
        _fail("channel_saturation_analysis.json has no provenance metadata")
    if saturation_meta.get("git_commit") != expected_commit:
        _fail("channel_saturation_analysis.json commit differs from the manifest")
    if saturation_meta.get("benchmark_run") != expected_run_tag:
        _fail("channel_saturation_analysis.json run tag differs from the manifest")

    figure_provenance = _load_json(RESULTS_DIR / "figure_provenance.json")
    _validate_figure_source_identity(figure_provenance, manifest)
    if figure_provenance.get("run_tag") != expected_run_tag:
        _fail("figure_provenance.json run_tag differs from the manifest")
    figure_seed_panel = figure_provenance.get("seed_panel")
    if not isinstance(figure_seed_panel, list) or len(figure_seed_panel) != len(
        EXPECTED_SEEDS
    ) or set(figure_seed_panel) != set(EXPECTED_SEEDS):
        _fail("figure_provenance.json does not name the exact seed panel")
    if figure_provenance.get("illustrative_seed") != 42:
        _fail("figure_provenance.json must identify predeclared seed 42")
    if figure_provenance.get("n_seed_envelopes_loaded") != len(EXPECTED_SEEDS):
        _fail("figure_provenance.json does not declare all 20 loaded envelopes")
    normalized_seed_root = str(figure_provenance.get("seed_root", "")).replace(
        "\\", "/",
    ).rstrip("/")
    if not normalized_seed_root.endswith(
        "/mvp/simulation/results/benchmark_seeds"
    ):
        _fail(
            "figure_provenance.json was not rendered from the canonical flat "
            "archived seed directory"
        )
    seed_input_records = figure_provenance.get("seed_input_artifacts")
    if not isinstance(seed_input_records, list):
        _fail("figure_provenance.json lacks seed-input byte records")
    declared_seed_records: dict[str, dict[str, Any]] = {}
    for record in seed_input_records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            _fail("figure_provenance.json has a malformed seed-input record")
        name = str(record["file"])
        if name in declared_seed_records:
            _fail("figure_provenance.json repeats a seed-input record")
        declared_seed_records[name] = record
    expected_seed_names = {
        f"benchmark_seeds/seed_{seed}.json" for seed in EXPECTED_SEEDS
    }
    if set(declared_seed_records) != expected_seed_names:
        _fail("figure_provenance.json does not bind the exact seed-input panel")
    manifest_records = {
        str(record.get("file")): record
        for record in manifest.get("artifacts", [])
        if isinstance(record, dict)
    }
    for seed in EXPECTED_SEEDS:
        name = f"benchmark_seeds/seed_{seed}.json"
        path = RESULTS_DIR / name
        actual = {
            "file": name,
            "seed": seed,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if declared_seed_records[name] != actual:
            _fail(
                f"figure_provenance.json seed-input bytes differ from {name}"
            )
        manifest_record = manifest_records.get(name)
        if not isinstance(manifest_record, dict) or any(
            manifest_record.get(key) != actual[key]
            for key in ("file", "bytes", "sha256")
        ):
            _fail(f"figure input {name} is not identically bound by the manifest")
    aggregate_input_records = figure_provenance.get(
        "aggregate_input_artifacts"
    )
    if not isinstance(aggregate_input_records, list):
        _fail("figure_provenance.json lacks aggregate-input byte records")
    declared_aggregate_records: dict[str, dict[str, Any]] = {}
    for record in aggregate_input_records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            _fail("figure_provenance.json has a malformed aggregate-input record")
        name = str(record["file"])
        if name in declared_aggregate_records:
            _fail("figure_provenance.json repeats an aggregate-input record")
        declared_aggregate_records[name] = record
    if set(declared_aggregate_records) != set(EXPECTED_FIGURE_AGGREGATE_INPUTS):
        _fail(
            "figure_provenance.json does not bind the exact aggregate inputs "
            "read by the renderer"
        )
    for name in EXPECTED_FIGURE_AGGREGATE_INPUTS:
        path = RESULTS_DIR / name
        actual = {
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if declared_aggregate_records[name] != actual:
            _fail(
                f"figure_provenance.json aggregate-input bytes differ from {name}"
            )
        manifest_record = manifest_records.get(name)
        if not isinstance(manifest_record, dict) or any(
            manifest_record.get(key) != actual[key]
            for key in ("file", "bytes", "sha256")
        ):
            _fail(f"figure aggregate input {name} is not identically manifested")
    if figure_provenance.get("render_input_isolated_snapshot") is not True:
        _fail("figure_provenance.json lacks isolated-render input evidence")
    panels = figure_provenance.get("panels")
    required_panel_groups = {
        "heatwave", "overproduction", "cyber_outage",
        "adaptive_pricing", "cross_scenario_and_secondary",
    }
    if not isinstance(panels, dict) or set(panels) != required_panel_groups:
        _fail("figure_provenance.json has an incomplete panel inventory")
    expected_panel_n = {
        "heatwave": {"a": 1, "b": 1, "c": 1, "d": 20},
        "overproduction": {"a": 1, "b": 1, "c": 1, "d": 20},
        "cyber_outage": {"a": 1, "b": 20, "c": 20, "d": 20},
        "adaptive_pricing": {"a": 1, "b": 1, "c": 1, "d": 1},
    }
    for group_name, expected_entries in expected_panel_n.items():
        group = panels.get(group_name)
        if not isinstance(group, dict) or set(group) != set(expected_entries):
            _fail(f"figure_provenance.json has an incomplete {group_name} panel")
        for panel_name, expected_n in expected_entries.items():
            entry = group.get(panel_name)
            if not isinstance(entry, dict) or entry.get("n_seeds") != expected_n:
                _fail(
                    f"figure_provenance.json {group_name}/{panel_name} has an "
                    "incorrect actual seed count"
                )
    cross_group = panels.get("cross_scenario_and_secondary")
    if (
        not isinstance(cross_group, dict)
        or cross_group.get("n_seeds") != 20
        or cross_group.get("fields") != list(EXPECTED_FIGURE_AGGREGATE_INPUTS)
    ):
        _fail("figure_provenance.json cross-scenario group is not 20-seed evidence")
    seed42 = _load_json(RESULTS_DIR / "benchmark_seeds" / "seed_42.json")
    trace_panel = seed42.get("traces") if isinstance(seed42, dict) else None
    available_trace_fields: set[str] = set()
    if isinstance(trace_panel, dict):
        for scenario_modes in trace_panel.values():
            if not isinstance(scenario_modes, dict):
                continue
            for trace_cell in scenario_modes.values():
                if isinstance(trace_cell, dict):
                    available_trace_fields.update(trace_cell)
    declared_trace_fields: set[str] = set()
    for group in panels.values():
        entries = group.values() if isinstance(group, dict) else ()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            fields = entry.get("fields", [])
            if not isinstance(fields, list) or not entry.get("aggregation"):
                _fail("figure_provenance.json has a malformed panel record")
            declared_trace_fields.update(
                field for field in fields
                if (
                    isinstance(field, str)
                    and field not in EXPECTED_FIGURE_AGGREGATE_INPUTS
                )
            )
    missing_fields = declared_trace_fields - available_trace_fields
    if missing_fields:
        _fail(
            "figure_provenance.json names unavailable raw fields: "
            f"{sorted(missing_fields)}"
        )
    print("[PASS] staged benchmark/stress/secondary-analysis run provenance")


def _validate_exact_figure_inventory() -> None:
    """Reject missing, stale, extra, unparseable, or unbound figures."""

    from mvp.simulation.validation.figure_artifacts import (
        validate_figure_directory,
    )

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    source_commit = str(manifest.get("simulation_source_commit", ""))
    run_tag = str(manifest.get("artifact_run_tag", ""))
    try:
        validate_figure_directory(
            RESULTS_DIR,
            source_commit=source_commit,
            run_tag=run_tag,
        )
    except ValueError as exc:
        _fail(str(exc))
    print("[PASS] exact decodable 10-figure PNG/PDF inventory")


def _load_hpc_validator(module_name: str, filename: str):
    path = REPO_ROOT / "hpc" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load HPC evidence validator: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(f"cannot import HPC evidence validator {filename}: {exc}")
    return module


def _validate_core_submission_receipt() -> None:
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    run_tag = str(manifest.get("artifact_run_tag", ""))
    source_commit = str(manifest.get("simulation_source_commit", ""))
    receipt_path = RESULTS_DIR / "core_submission_receipts" / f"{run_tag}.json"
    module = _load_hpc_validator(
        "agribrain_core_submission_receipt", "core_submission_receipt.py",
    )
    try:
        module.validate_receipt_file(
            receipt_path,
            expected_run_tag=run_tag,
            expected_source_commit=source_commit,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"invalid core Slurm submission receipt: {exc}")
    print("[PASS] hash-bound core Slurm submission DAG receipt")


def _validate_raw_seed_inputs() -> None:
    """Rerun the full raw retained-seed and trace contract at final gate."""

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    run_tag = str(manifest.get("artifact_run_tag", ""))
    source_commit = str(manifest.get("simulation_source_commit", ""))
    module = _load_hpc_validator(
        "agribrain_raw_seed_inputs", "validate_raw_publication_inputs.py",
    )
    submission_receipt = _load_json(
        RESULTS_DIR / "core_submission_receipts" / f"{run_tag}.json"
    )
    try:
        module.validate_seed_inputs(
            RESULTS_DIR / "benchmark_seeds",
            source_commit=source_commit,
            run_tag=run_tag,
            submission_receipt=submission_receipt,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"raw retained seed/trace validation failed: {exc}")
    print(
        "[PASS] exact 20 retained seed envelopes, episode design, streams, "
        "and complete trace contract"
    )


def _validate_raw_h3_inputs() -> None:
    """Rerun the raw H3 dose/ledger gate inside the final receipt pass."""

    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    run_tag = str(manifest.get("artifact_run_tag", ""))
    source_commit = str(manifest.get("simulation_source_commit", ""))
    module = _load_hpc_validator(
        "agribrain_raw_publication_inputs", "validate_raw_publication_inputs.py",
    )
    submission_receipt = _load_json(
        RESULTS_DIR / "core_submission_receipts" / f"{run_tag}.json"
    )
    try:
        module.validate_stress_inputs(
            RESULTS_DIR / "stress_runs" / run_tag,
            seed_root=RESULTS_DIR / "benchmark_seeds",
            source_commit=source_commit,
            run_tag=run_tag,
            h3_ledger_root=RESULTS_DIR / "decision_ledger_h3" / run_tag,
            primary_ledger_root=(
                RESULTS_DIR / "decision_ledger_per_seed" / run_tag
            ),
            submission_receipt=submission_receipt,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"raw H3 input/ledger validation failed: {exc}")
    print(
        "[PASS] final raw H3 dose/endpoints, 500 stressed ledgers, "
        "and 100 reused primary-ledger references"
    )


def _validate_complete_decision_ledgers() -> None:
    """Re-run the exact 1,100-ledger structural and Merkle gate."""
    manifest = _load_json(RESULTS_DIR / "artifact_manifest.json")
    run_tag = str(manifest.get("artifact_run_tag", "")).strip()
    ledger_root = RESULTS_DIR / "decision_ledger_per_seed" / run_tag
    validator_path = REPO_ROOT / "hpc" / "validate_decision_ledgers.py"
    spec = importlib.util.spec_from_file_location(
        "agribrain_validate_decision_ledgers", validator_path,
    )
    if spec is None or spec.loader is None:
        _fail(f"cannot load decision-ledger validator: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.validate_inventory(
            ledger_root, RESULTS_DIR / "benchmark_seeds",
        )
    except Exception as exc:
        _fail(f"decision-ledger validation failed: {exc}")
    print("[PASS] exact 1,100-ledger inventory, schema, and Merkle integrity")


def _validate_threshold_assertions() -> None:
    """Validate numeric integrity and presence of declared contrasts.

    The previous schema check only verified field *presence*; a run
    that produced all-null effects passed. This adds explicit numeric
    This gate encodes no preferred direction, minimum performance, or
    hypothesis outcome. Statistical conclusions are read from the generated
    estimates and tests rather than forced by validation code.

    This compatibility check follows stricter exact-panel checks earlier in
    the validator; a canonical run has 20 paired seeds and populated tests.
    """
    sig = _load_json(RESULTS_DIR / "benchmark_significance.json")
    if isinstance(sig, dict) and isinstance(sig.get("significance"), dict):
        sig = sig["significance"]
    summary = _load_json(RESULTS_DIR / "benchmark_summary.json")
    if isinstance(summary, dict) and "summary" in summary and isinstance(summary["summary"], dict):
        summary = summary["summary"]

    failures = []
    for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
        # Plausibility bound on agribrain ARI mean.
        try:
            ari_mean = float(summary[sc]["agribrain"]["ari"]["mean"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{sc}/agribrain ari mean missing or non-numeric")
            continue
        if not (0.0 <= ari_mean <= 1.0):
            failures.append(f"{sc}/agribrain ARI mean {ari_mean} out of [0,1]")

        # ------------------------------------------------------------
        # Primary H1 contrast: agribrain vs no_context
        # ------------------------------------------------------------
        _check_contrast_record(sig, sc, "agribrain_vs_no_context",
                               failures, require_p=True)

        # ------------------------------------------------------------
        # Confirmatory H2: all four directional contrasts belong to one
        # 20-test Holm family. Presence is required, but no preferred outcome
        # is imposed by this integrity check.
        # ------------------------------------------------------------
        for comp_name in (
            "mcp_only_vs_no_context", "pirag_only_vs_no_context",
            "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
        ):
            rec = sig.get(sc, {}).get(comp_name, {}).get("ari")
            if not isinstance(rec, dict):
                failures.append(
                    f"{sc}/{comp_name}/ari record missing — H2 family incomplete"
                )
                continue
            _check_contrast_record(sig, sc, comp_name, failures,
                                   require_p=True)
            if "p_value_adj_holm_h2_directional" not in rec:
                failures.append(
                    f"{sc}/{comp_name}/ari missing "
                    "p_value_adj_holm_h2_directional"
                )

    if failures:
        _fail("numeric-integrity assertions failed:\n  - " + "\n  - ".join(failures[:20]))
    print("[PASS] numeric integrity + declared contrast presence")


def _check_contrast_record(sig: dict, sc: str, comp_name: str,
                           failures: list, require_p: bool) -> None:
    """Shared consistency check for any (scenario, comparison) ARI record.

    Validates: record exists; mean_diff present and finite; p_value
    in [0, 1] when ``require_p``; effect-size CI bracketed correctly.
    The canonical caller has already required a 20-seed populated test; the
    conditional field handling remains only for reuse by diagnostic callers.
    """
    import math as _m
    rec = sig.get(sc, {}).get(comp_name, {}).get("ari")
    if not isinstance(rec, dict):
        failures.append(f"{sc}/{comp_name}/ari record missing")
        return
    try:
        md = float(rec["mean_diff"])
    except (KeyError, TypeError, ValueError):
        failures.append(f"{sc}/{comp_name} ari mean_diff missing or non-numeric")
        return
    if not _m.isfinite(md):
        failures.append(f"{sc}/{comp_name} ari mean_diff non-finite ({md})")
    if require_p and "p_value" in rec and rec["p_value"] is not None:
        try:
            p = float(rec["p_value"])
            if not (0.0 <= p <= 1.0):
                failures.append(f"{sc}/{comp_name} ari p_value {p} out of [0,1]")
        except (TypeError, ValueError):
            failures.append(f"{sc}/{comp_name} ari p_value non-numeric")
    lo = rec.get("effect_size_ci_low")
    hi = rec.get("effect_size_ci_high")
    if lo is not None and hi is not None and float(lo) > float(hi):
        failures.append(
            f"{sc}/{comp_name} ari effect-size CI inverted: "
            f"low={lo} > high={hi}"
        )


def validate_full_publication_release(
    results_dir: Path,
    *,
    repo_root: Path,
    recovery_receipt: Path | None = None,
) -> None:
    """Run the complete semantic validator in an isolated interpreter.

    The subprocess avoids mutating this module's path globals in an API
    process and ensures archive/API/full-evidence consumers rerun the same
    endpoint, ledger, figure, environment, DAG, and receipt gates as the HPC
    publisher. A contract-valid receipt alone is never treated as evidence.
    """

    results_dir = results_dir.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    script = (
        repo_root / "mvp" / "simulation" / "validation"
        / "validate_publication_artifacts.py"
    )
    if not script.is_file() or script.is_symlink():
        raise ValueError("canonical publication validator source is unavailable")
    command = [
            sys.executable,
            str(script),
            "--results-dir",
            str(results_dir),
            "--repo-root",
            str(repo_root),
    ]
    if recovery_receipt is not None:
        command.extend(["--recovery-receipt", str(recovery_receipt.resolve())])
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
        raise ValueError(
            "full publication semantic validation failed: " + diagnostic[-8000:]
        )


def main(argv: list[str] | None = None) -> None:
    global RESULTS_DIR, REPO_ROOT, RECOVERY_RECEIPT_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Publication results directory to validate.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Source repository whose protocol and validators are authoritative.",
    )
    parser.add_argument(
        "--write-receipt",
        action="store_true",
        help=(
            "After every semantic gate passes, write the immutable validation "
            "receipt. Rebuild the manifest and run this validator again "
            "without this flag to verify the receipt's manifested bytes."
        ),
    )
    parser.add_argument(
        "--recovery-receipt",
        type=Path,
        help=(
            "Exact canonical recovery authorization required for any "
            "dual-provenance semantic validation."
        ),
    )
    args = parser.parse_args(argv)
    raw_results_dir = args.results_dir
    raw_repo_root = args.repo_root
    if raw_results_dir.is_symlink() or raw_repo_root.is_symlink():
        _fail("results/repository validation roots must not be symlinks")
    try:
        RESULTS_DIR = raw_results_dir.resolve(strict=True)
        REPO_ROOT = raw_repo_root.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve validation roots: {exc}")
    if not RESULTS_DIR.is_dir() or not REPO_ROOT.is_dir():
        _fail("results/repository validation roots must be directories")
    RECOVERY_RECEIPT_PATH = args.recovery_receipt
    # Establish the exact safe literal-byte inventory before any semantic
    # parser opens a manifested payload.
    _validate_manifest(receipt_expected=not args.write_receipt)
    _validate_significance()
    _validate_benchmark_summary()
    _validate_paper_benchmark_table()
    _validate_tables_against_summary()
    _validate_channel_saturation()
    _validate_h1_h2_against_raw()
    _validate_reaggregated_core_statistics()
    _validate_channel_saturation_against_raw()
    _validate_stress_passfail()
    _validate_h3_test()
    _validate_core_submission_receipt()
    _validate_publication_environment()
    _validate_forecast_receipt()
    _validate_run_provenance()
    _validate_evidence_scope_metadata()
    _validate_exact_figure_inventory()
    _validate_raw_seed_inputs()
    _validate_complete_decision_ledgers()
    _validate_raw_h3_inputs()
    _validate_derived_evidence_replay()
    _validate_h3_aggregation_replay()
    _validate_threshold_assertions()
    if args.write_receipt:
        _write_publication_validation_receipt()
    else:
        _validate_publication_validation_receipt()
    print("[PASS] publication artifact validation complete")


if __name__ == "__main__":
    main()
