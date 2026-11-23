"""Recompute the primary Static / No-Context / AGRI-BRAIN statistics from data/primary and check the reported tables.

The comparison uses 20 seeds and five scenarios. Needs numpy and scipy; the versions used for the study are pinned in
reproduction/source/agribrain/backend/requirements-lock.txt.

    python analysis/primary_statistics.py                # print the H1 summary and check the reported tables
    python analysis/primary_statistics.py --output DIR   # also write primary_h1.json and primary_h1.csv

Procedure, as used for the reported tables: every interval is a 95% BCa interval from 10,000 resamples, drawn with a
generator seeded from a BLAKE2b hash of the cell key; a paired difference resamples whole seeds. H1 uses one-sided exact
Wilcoxon signed-rank tests per scenario, Holm-adjusted across the five scenarios. The check exits with status 1 if any
recomputed value differs from the reported tables by more than 1e-9.
"""
import argparse
import csv
import hashlib
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "primary"
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["static", "no_context", "agribrain"]
METRICS = ["ari", "waste", "carbon", "slca", "equity", "rle"]
N_BOOT = 10_000
TOLERANCE = 1e-9
PRACTICAL_MARGIN = 0.005


def cell_seed(scope, key):
    payload = "::".join((scope,) + tuple(str(part) for part in key))
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest(), "big")


def bca_interval(boots, theta, jacks, alpha=0.05):
    """BCa interval; the percentile interval is used when the correction is undefined. Returns (low, high, method)."""
    percentile = (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)), "percentile")
    p0 = float((np.count_nonzero(boots < theta) + 0.5 * np.count_nonzero(boots == theta)) / len(boots))
    if p0 <= 0.0 or p0 >= 1.0:
        return percentile
    z0, z_low, z_high = float(ndtri(p0)), float(ndtri(alpha / 2)), float(ndtri(1.0 - alpha / 2))
    centre = float(np.mean(jacks))
    spread = float(np.sum((centre - jacks) ** 2))
    if spread <= 0.0 or not np.isfinite(spread):
        return percentile
    acceleration = float(np.sum((centre - jacks) ** 3)) / (6.0 * spread ** 1.5)

    def adjust(z):
        denominator = 1.0 - acceleration * (z0 + z)
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return None
        return 0.5 * (1.0 + erf((z0 + (z0 + z) / denominator) / sqrt(2.0)))

    low, high = adjust(z_low), adjust(z_high)
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high) or low > high:
        return percentile
    low, high = max(min(low, 1.0 - 1e-9), 1e-9), max(min(high, 1.0 - 1e-9), 1e-9)
    return float(np.quantile(boots, low)), float(np.quantile(boots, high)), "BCa"


def mean_interval(values, key):
    x = np.asarray(values, dtype=float)
    theta = float(np.mean(x))
    if float(np.std(x, ddof=1)) == 0.0:
        return theta, theta, "deterministic"
    rng = np.random.default_rng(cell_seed("bootstrap_ci", key))
    boots = np.array([float(np.mean(rng.choice(x, len(x), replace=True))) for _ in range(N_BOOT)])
    jacks = np.array([float(np.mean(np.delete(x, i))) for i in range(len(x))])
    return bca_interval(boots, theta, jacks)


def paired_interval(a, b, key):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    theta = float(np.mean(x - y))
    if float(np.std(x - y, ddof=1)) == 0.0:
        return theta, theta, "deterministic"
    rng = np.random.default_rng(cell_seed("bootstrap_diff_ci", key))
    index = np.arange(len(x))
    boots = []
    for _ in range(N_BOOT):
        sample = rng.choice(index, size=len(index), replace=True)
        boots.append(float(np.mean(x[sample] - y[sample])))
    jacks = np.array([float(np.mean(np.delete(x, i) - np.delete(y, i))) for i in range(len(x))])
    return bca_interval(np.asarray(boots, dtype=float), theta, jacks)


def summarize(values, key):
    x = np.asarray(values, dtype=float)
    low, high, method = mean_interval(x, key)
    return {"mean": float(x.mean()), "low": low, "high": high, "n": len(x), "sd": float(x.std(ddof=1)), "ci_method": method}


def holm_adjust(p_values):
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - rank)))
        adjusted[name] = running
    return adjusted


def load_endpoints():
    with (DATA / "three_mode_endpoints.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seeds = sorted({int(row["seed"]) for row in rows})
    table = {(row["scenario"], row["mode"], int(row["seed"])): row for row in rows}
    if len(table) != len(rows) or len(rows) != len(SCENARIOS) * len(MODES) * len(seeds):
        raise ValueError("data/primary/three_mode_endpoints.csv is not a complete seed-scenario-mode panel")

    def vector(scenario, mode, metric):
        return np.array([float(table[(scenario, mode, seed)][metric]) for seed in seeds])

    return seeds, vector


def overall_vectors(seeds, vector, metric, scale=1.0):
    """Per-seed equal-scenario means, as in the overall outcomes."""
    per_scenario = {mode: [vector(sc, mode, metric) for sc in SCENARIOS] for mode in MODES}
    return {mode: np.array([np.mean([column[i] for column in per_scenario[mode]]) * scale for i in range(len(seeds))])
            for mode in MODES}


def h1_by_scenario(vector):
    results, raw = {}, {}
    for scenario in SCENARIOS:
        agribrain, no_context = vector(scenario, "agribrain", "ari"), vector(scenario, "no_context", "ari")
        gain = agribrain - no_context
        low, high, method = paired_interval(agribrain, no_context, (scenario, "no_context", "ari"))
        raw[scenario] = float(wilcoxon(gain.tolist(), alternative="greater", method="exact").pvalue)
        results[scenario] = {
            "no_context_ari": float(no_context.mean()), "agribrain_ari": float(agribrain.mean()), "mean_gain": float(gain.mean()),
            "low": low, "high": high, "ci_method": method,
            "relative_percent": float(100 * (np.mean(agribrain) - np.mean(no_context)) / np.mean(no_context)),
            "seeds_positive": int((gain > 0).sum()), "n": int(len(gain)), "p_one_sided": raw[scenario]}
    holm = holm_adjust(raw)
    for scenario, row in results.items():
        row["p_holm"] = holm[scenario]
        row["h1_supported"] = bool(holm[scenario] < 0.05 and row["mean_gain"] > 0)
        row["practical_margin_supported"] = bool(row["ci_method"] == "BCa" and row["low"] > PRACTICAL_MARGIN)
    return results


class Comparison:
    """Collects every recomputed-versus-reported difference."""

    def __init__(self):
        self.count, self.worst, self.failures = 0, 0.0, []

    def numbers(self, label, got, want):
        for name, reported in want.items():
            if reported in ("", None):
                continue
            difference = abs(float(got[name]) - float(reported))
            self.count += 1
            self.worst = max(self.worst, difference)
            if difference > TOLERANCE:
                self.failures.append(f"{label} {name}: recomputed {got[name]!r}, reported {reported!r}")

    def text(self, label, got, want):
        self.count += 1
        if got != want:
            self.failures.append(f"{label}: recomputed {got!r}, reported {want!r}")


def check_reported(seeds, vector, h1):
    check = Comparison()
    with (DATA / "plotted_estimates.csv").open(encoding="utf-8", newline="") as handle:
        plotted = list(csv.DictReader(handle))
    not_recomputable = 0
    for row in plotted:
        scenario, series, metric = row["scenario"], row["series"], row["metric"]
        if row["figure"] == "2/3" and metric in METRICS:
            got = summarize(vector(scenario, series, metric), (scenario, series, metric))
            label = f"plotted_estimates {scenario}/{series}/{metric}"
            check.numbers(label, got, {k: row[k] for k in ("mean", "low", "high", "sd", "n")})
            check.text(label + " ci_method", got["ci_method"], row["ci_method"])
        elif row["figure"] == "2" and series == "paired_gain":
            got = h1[scenario]
            check.numbers(f"plotted_estimates paired_gain {scenario}",
                          {"mean": got["mean_gain"], "low": got["low"], "high": got["high"], "relative_percent": got["relative_percent"]},
                          {k: row[k] for k in ("mean", "low", "high", "relative_percent")})
        else:
            not_recomputable += 1
    with (DATA / "paired_seed_ARI.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario, index = row["scenario"], seeds.index(int(row["seed"]))
            a, b = vector(scenario, "agribrain", "ari")[index], vector(scenario, "no_context", "ari")[index]
            check.numbers(f"paired_seed_ARI {row['seed']}/{scenario}", {"agribrain": a, "without_context": b, "gain": a - b},
                          {k: row[k] for k in ("agribrain", "without_context", "gain")})
    table7 = json.loads((DATA / "overall_outcomes.json").read_text(encoding="utf-8"))
    check.text("Overall outcomes seeds", table7["seeds"], seeds)
    scales = {"waste": 100.0, "constraint_violation_rate": 100.0}
    for metric, report in table7["metrics"].items():
        stored = {mode: np.array(table7["seed_values"][metric][mode]) for mode in MODES}
        if metric in METRICS:
            derived = overall_vectors(seeds, vector, metric, scales.get(metric, 1.0))
            for mode in MODES:
                check.numbers(f"Overall outcomes {metric}/{mode} seed values", dict(enumerate(derived[mode])), dict(enumerate(stored[mode])))
        results = {}
        for mode in MODES:
            twin = next((m for m in results if np.array_equal(stored[m], stored[mode])), None)
            results[mode] = dict(results[twin]) if twin else summarize(stored[mode], ("table7", "overall", mode, metric))
            check.numbers(f"Overall outcomes {metric}/{mode}", results[mode], {k: report[mode][k] for k in ("mean", "low", "high", "sd", "n")})
            check.text(f"Overall outcomes {metric}/{mode} ci_method", results[mode]["ci_method"], report[mode]["ci_method"])
        difference = summarize(stored["agribrain"] - stored["no_context"], ("table7", "overall", "paired_gain", metric))
        check.numbers(f"Overall outcomes {metric}/difference", difference, {k: report["difference"][k] for k in ("mean", "low", "high", "sd", "n")})
        check.text(f"Overall outcomes {metric}/difference ci_method", difference["ci_method"], report["difference"]["ci_method"])
    for scenario, reported in table7["scenario_ari_one_sided_wilcoxon_p"].items():
        check.numbers(f"Overall outcomes Wilcoxon p {scenario}", {"p": h1[scenario]["p_one_sided"]}, {"p": reported})
    return check, not_recomputable


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="directory for primary_h1.json and primary_h1.csv")
    parser.add_argument("--no-check", action="store_true", help="skip the comparison with the reported tables")
    args = parser.parse_args()

    seeds, vector = load_endpoints()
    h1 = h1_by_scenario(vector)
    ari = overall_vectors(seeds, vector, "ari")
    gain = ari["agribrain"] - ari["no_context"]
    low, high, method = mean_interval(gain, ("table7", "overall", "paired_gain", "ari"))
    overall = {"mean_gain": float(gain.mean()), "low": low, "high": high, "ci_method": method,
               "relative_percent": float(100 * gain.mean() / ari["no_context"].mean()), "seeds_positive": int((gain > 0).sum()),
               **{f"{mode}_ari": float(ari[mode].mean()) for mode in MODES}}
    print(f"{'scenario':18s}{'No-Context':>11s}{'AGRI-BRAIN':>11s}{'gain':>9s}{'95% BCa':>19s}{'rel %':>7s}{'seeds+':>7s}{'p (Holm)':>11s}  H1")
    for scenario, row in h1.items():
        print(f"{scenario:18s}{row['no_context_ari']:11.4f}{row['agribrain_ari']:11.4f}{row['mean_gain']:9.4f}"
              f"   [{row['low']:.4f}, {row['high']:.4f}]{row['relative_percent']:7.2f}{row['seeds_positive']:5d}/{row['n']}{row['p_holm']:11.2e}  "
              f"{'supported' if row['h1_supported'] else 'not supported'}")
    print(f"{'overall':18s}{overall['no_context_ari']:11.4f}{overall['agribrain_ari']:11.4f}{overall['mean_gain']:9.4f}"
          f"   [{overall['low']:.4f}, {overall['high']:.4f}]{overall['relative_percent']:7.2f}{overall['seeds_positive']:5d}/{len(seeds)}")
    status = 0
    report = {"h1_by_scenario": h1, "overall": overall, "seeds": seeds}
    if not args.no_check:
        check, skipped = check_reported(seeds, vector, h1)
        report["check"] = {"values_compared": check.count, "largest_difference": check.worst, "failures": check.failures,
                           "tolerance": TOLERANCE, "rows_without_per_seed_data": skipped}
        print(f"\nCompared {check.count} values with the reported tables; largest difference {check.worst:.3g} "
              f"(tolerance {TOLERANCE:g}); {skipped} reported rows (latency, routing, audit) have no per-seed data in data/primary "
              "and were not recomputed.")
        for failure in check.failures[:20]:
            print("MISMATCH", failure)
        status = 1 if check.failures else 0
        print("CHECK PASSED" if status == 0 else f"CHECK FAILED ({len(check.failures)} mismatches)")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "primary_h1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (args.output / "primary_h1.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario"] + list(next(iter(h1.values()))))
            writer.writeheader()
            for scenario, row in h1.items():
                writer.writerow({"scenario": scenario, **row})
    return status


if __name__ == "__main__":
    sys.exit(main())
