"""Regenerate the ARI performance and weight sensitivity figures from the supplied data, after checking the values to be plotted.

    python analysis/make_figures.py --output DIR

writes ari_performance_by_scenario.{png,pdf} and weight_sensitivity.{png,pdf}.

The ARI performance figure is recomputed from data/primary/three_mode_endpoints.csv with the procedure of
analysis/primary_statistics.py and compared with data/primary/plotted_estimates.csv before it is drawn. The weight sensitivity
figure is recomputed from data/sensitivity/seed_endpoints.csv, using the equal-scenario seed means, the paired gain over
No-Context and the scipy.stats.bootstrap BCa interval (10,000 resamples, random_state 20260924) of reproduction/analyze.py, and
compared with data/sensitivity/paired_ari_sensitivity.csv. Drawing stops with an error if any value differs by more than 1e-9.

The other figures in figures/ and the architecture diagram are not regenerated here. The output is a re-rendering of the data,
not a byte copy of figures/. Needs numpy, scipy and matplotlib (see analysis/requirements.txt).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

import primary_statistics as ps

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY = ROOT / "data" / "sensitivity"
TOLERANCE = 1e-9
COLORS = {"static": "#595959", "no_context": "#0072B2", "agribrain": "#009E73"}
HATCH = {"static": "", "no_context": "..", "agribrain": "xx"}
NAMES = {"static": "Static", "no_context": "No-Context", "agribrain": "AGRI-BRAIN"}
SCENARIO_NAMES = {"heatwave": "Heatwave", "overproduction": "Overproduction", "cyber_outage": "Cyber outage",
                  "adaptive_pricing": "Adaptive pricing", "baseline": "Baseline"}
SETTING_NAMES = {"nominal": "Nominal", "base_policy": "Base prior", "mcp_prior": "Tool prior", "retrieval_prior": "Retrieval prior",
                 "w_c": "Carbon weight", "w_l": "Labour weight", "w_r": "Community weight", "w_p": "Price weight",
                 "eta": "Waste penalty", "eta_rho": "Risk penalty"}


def require_match(label, computed, reported):
    worst = max(abs(float(computed[name]) - float(reported[name])) for name in reported if reported[name] not in ("", None))
    if worst > TOLERANCE:
        raise SystemExit(f"{label}: recomputed values differ from the supplied table by {worst:.3g}; not drawing")


def figure2_data():
    seeds, vector = ps.load_endpoints()
    bars = {(sc, mode): ps.summarize(vector(sc, mode, "ari"), (sc, mode, "ari")) for sc in ps.SCENARIOS for mode in ps.MODES}
    gains = ps.h1_by_scenario(vector)
    with (ps.DATA / "plotted_estimates.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["figure"] == "2/3" and row["metric"] == "ari":
                require_match(f"ARI performance, mean ARI {row['scenario']}/{row['series']}", bars[(row["scenario"], row["series"])],
                              {k: row[k] for k in ("mean", "low", "high")})
            elif row["figure"] == "2" and row["series"] == "paired_gain":
                got = gains[row["scenario"]]
                require_match(f"ARI performance, paired gain {row['scenario']}", {"mean": got["mean_gain"], "low": got["low"], "high": got["high"],
                                                              "relative_percent": got["relative_percent"]},
                              {k: row[k] for k in ("mean", "low", "high", "relative_percent")})
    return bars, gains


def seed_interval(values):
    values = np.asarray(values, dtype=float)
    if np.ptp(values) < 1e-14:
        return float(values.mean()), float(values.mean()), "constant"
    result = bootstrap((values,), np.mean, n_resamples=10000, confidence_level=0.95, method="BCa", random_state=20260924)
    return float(result.confidence_interval.low), float(result.confidence_interval.high), "BCa"


def figure_s2_data():
    with (SENSITIVITY / "seed_endpoints.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seeds = list(dict.fromkeys(int(r["seed"]) for r in rows))
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    ari = {(r["setting"], int(r["seed"]), r["scenario"], r["mode"]): float(r["ari"]) for r in rows}

    def seed_means(setting, mode):
        return np.array([np.mean([ari[(setting, s, sc, mode)] for sc in ps.SCENARIOS]) for s in seeds])

    nominal = np.array([np.mean([ari[("nominal", s, sc, "agribrain")] - ari[("nominal", s, sc, "no_context")] for sc in ps.SCENARIOS])
                        for s in seeds])
    with (SENSITIVITY / "paired_ari_sensitivity.csv").open(encoding="utf-8", newline="") as handle:
        reported = {r["setting"]: r for r in csv.DictReader(handle) if r["scenario"] == "overall" and r["comparator"] == "no_context"}
    result = []
    for setting in settings:
        gain = seed_means(setting, "agribrain") - seed_means(setting, "no_context")
        change = gain - nominal
        low, high, _ = seed_interval(gain)
        change_low, change_high, _ = seed_interval(change)
        row = {"setting": setting, "mean_gain": float(gain.mean()), "ci_low": low, "ci_high": high,
               "change_from_nominal": float(change.mean()), "change_ci_low": change_low, "change_ci_high": change_high}
        require_match(f"Weight sensitivity {setting}", row, {k: reported[setting][k] for k in row if k != "setting"})
        result.append(row)
    return result


def setting_label(setting):
    if setting.startswith("joint"):
        return f"Joint {setting[5:7]}% {setting[-1]}"
    base, _, level = setting.rpartition("_")
    return setting_label_for(base, level) if base else SETTING_NAMES.get(setting, setting)


def setting_label_for(base, level):
    sign = "−20%" if level == "minus20" else "+20%"
    return f"{SETTING_NAMES[base]} {sign}"


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                         "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.titlesize": 18, "figure.titleweight": "bold"})
    return plt


def draw_figure2(bars, gains, output):
    plt = configure_matplotlib()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    width = 0.26
    x = np.arange(len(ps.SCENARIOS))
    for k, mode in enumerate(ps.MODES):
        stats = [bars[(sc, mode)] for sc in ps.SCENARIOS]
        mean = np.array([s["mean"] for s in stats])
        err = np.array([[s["mean"] - s["low"] for s in stats], [s["high"] - s["mean"] for s in stats]])
        left.bar(x + (k - 1) * width, mean, width, yerr=err, color=COLORS[mode], hatch=HATCH[mode], edgecolor="black", linewidth=0.6,
                 error_kw={"elinewidth": 1.1, "capsize": 2.5}, label=NAMES[mode])
    left.set_xticks(x, [SCENARIO_NAMES[sc] for sc in ps.SCENARIOS], rotation=30, ha="right")
    left.set_ylabel("Mean ARI")
    left.set_title("(a) Resilience performance under disruption", loc="left")
    left.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    y = np.arange(len(ps.SCENARIOS))
    mean = np.array([gains[sc]["mean_gain"] for sc in ps.SCENARIOS])
    err = np.array([[gains[sc]["mean_gain"] - gains[sc]["low"] for sc in ps.SCENARIOS], [gains[sc]["high"] - gains[sc]["mean_gain"] for sc in ps.SCENARIOS]])
    right.barh(y, mean, 0.55, xerr=err, color=COLORS["agribrain"], hatch="xx", edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.1, "capsize": 2.5})
    for yi, sc in zip(y, ps.SCENARIOS):
        right.text(gains[sc]["high"] + 0.0008, yi, f"+{gains[sc]['relative_percent']:.2f}%", va="center", fontweight="bold")
    right.axvline(0.01, color="black", linestyle="--", linewidth=1.2)
    right.text(0.0105, -0.72, "0.01 reference", va="center", fontsize=11, fontweight="bold")
    right.set_yticks(y, [SCENARIO_NAMES[sc] for sc in ps.SCENARIOS], fontweight="bold")
    right.set_ylim(len(ps.SCENARIOS) - 0.4, -1.0)
    right.set_xlim(0, 0.043)
    right.set_xticks([0, 0.01, 0.02, 0.03])
    right.set_xlabel("Paired ARI gain")
    right.set_title("(b) AGRI-BRAIN’s ARI gain over No-Context", loc="left")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", color="0.9")
    left.set_axisbelow(True)
    fig.suptitle("ARI performance across scenarios", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"ari_performance_by_scenario.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_figure_s2(rows, output):
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 9.5), sharey=True)
    y = np.arange(len(rows))
    panels = [(axes[0], "mean_gain", "ci_low", "ci_high", "#009E73", "o"), (axes[1], "change_from_nominal", "change_ci_low", "change_ci_high", "#595959", "s")]
    for ax, mean, low, high, color, marker in panels:
        for yi, row in zip(y, rows):
            ax.hlines(yi, row[low], row[high], color=color, linewidth=1.5)
            ax.plot([row[low], row[high]], [yi, yi], marker="|", linestyle="", color=color)
            ax.plot(row[mean], yi, marker="D" if row["setting"] == "nominal" else marker, color=color, markersize=5)
        ax.axvline(0, color="0.3", linestyle="--", linewidth=1)
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvline(0.01, color="0.3", linestyle="--", linewidth=1)
    axes[0].text(0.0102, -0.9, "0.01 reference", fontsize=10, color="0.35", fontweight="bold")
    axes[0].set_xlim(0, 0.031)
    axes[0].set_xticks([0, 0.01, 0.02, 0.03])
    groups = [row["setting"].startswith(("base", "mcp", "retrieval")) for row in rows]
    kinds = ["nominal" if r["setting"] == "nominal" else "prior" if g else "joint" if r["setting"].startswith("joint")
             else "penalty" if r["setting"].startswith("eta") else "weight" for r, g in zip(rows, groups)]
    for i in range(1, len(rows)):
        if kinds[i] != kinds[i - 1]:
            for ax in axes:
                ax.axhline(i - 0.5, color="0.8", linewidth=0.8)
    axes[0].set_yticks(y, [setting_label(r["setting"]) for r in rows], fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_title("(a) ARI gain over No-Context", loc="left")
    axes[0].set_xlabel("Paired ARI gain")
    axes[1].set_title("(b) Change from nominal", loc="left")
    axes[1].set_xlabel("Change in paired ARI gain")
    fig.suptitle("Sensitivity to weight magnitudes", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"weight_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="directory for the regenerated figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bars, gains = figure2_data()
    print("ARI performance values match data/primary/plotted_estimates.csv")
    draw_figure2(bars, gains, args.output)
    rows = figure_s2_data()
    print(f"Weight sensitivity values for {len(rows)} settings match data/sensitivity/paired_ari_sensitivity.csv")
    draw_figure_s2(rows, args.output)
    print(f"Wrote ari_performance_by_scenario and weight_sensitivity (png, pdf) to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
