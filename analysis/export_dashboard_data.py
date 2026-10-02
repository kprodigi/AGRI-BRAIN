"""Write the summary data shown on the dashboard's results page.

Every value is read from the files under data/; nothing is recomputed. With
--check the script compares the existing output file with a fresh export and
exits with an error if they differ.
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agribrain/frontend/src/data/studyResults.json"
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["static", "no_context", "agribrain"]
METRICS = ["ari", "waste", "carbon", "slca", "rle", "mean_decision_latency_ms"]


def r6(x):
    return round(float(x), 6) + 0.0


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def interval(row, mean="mean", low="low", high="high"):
    return {"mean": r6(row[mean]), "low": r6(row[low]), "high": r6(row[high])}


def build():
    table7 = json.loads((ROOT / "data/primary/overall_outcomes.json").read_text(encoding="utf-8"))
    overall = {}
    for metric in METRICS:
        entry = table7["metrics"][metric]
        overall[metric] = {"label": entry["label"], **{m: {k: r6(entry[m][k]) for k in ("mean", "low", "high")} for m in MODES}}

    plotted = rows("data/primary/plotted_estimates.csv")
    scenarios = {s: {"metrics": {}} for s in SCENARIOS}
    for row in plotted:
        s = scenarios[row["scenario"]]
        if row["figure"] == "2/3":
            s["metrics"].setdefault(row["metric"], {})[row["series"]] = interval(row)
        elif row["figure"] == "2":
            s["paired_gain"] = interval(row)
        elif row["figure"] == "4":
            s["different_actions_percent"] = interval(row)

    sensitivity = rows("data/sensitivity/paired_ari_sensitivity.csv")
    for row in sensitivity:
        if row["setting"] == "nominal" and row["comparator"] == "no_context" and row["scenario"] in scenarios:
            scenarios[row["scenario"]]["relative_gain_percent"] = r6(row["relative_gain_percent"])
    settings = [
        {
            "setting": row["setting"], "family": row["family"],
            "gain": r6(row["mean_gain"]), "low": r6(row["ci_low"]), "high": r6(row["ci_high"]),
            "change": r6(row["change_from_nominal"]),
        }
        for row in sensitivity
        if row["scenario"].lower() == "overall" and row["comparator"] == "no_context"
    ]

    routing = json.loads((ROOT / "data/primary/routing_time_analysis.json").read_text(encoding="utf-8"))["routing"]
    routes = {}
    for mode in ("no_context", "agribrain"):
        bins = [b for sc in routing.values() for b in sc[mode]["mean_percent"]]
        routes[mode] = [round(sum(b[i] for b in bins) / len(bins), 1) for i in range(3)]

    controls = json.loads((ROOT / "data/primary/validation_20260922/control_summary.json").read_text(encoding="utf-8"))["aggregate"]
    controls = {k: {"mean": r6(v["mean"]), "low": r6(v["ci95"][0]), "high": r6(v["ci95"][1])} for k, v in controls.items()}

    summary = json.loads((ROOT / "data/sensitivity/summary.json").read_text(encoding="utf-8"))["counts"]
    return {
        "meta": {
            "seeds": 20, "scenarios": SCENARIOS, "modes": MODES,
            "primary_evaluations": 300, "sensitivity": summary,
            "intervals": "95% BCa across seeds; sensitivity intervals are pointwise",
        },
        "overall": overall,
        "scenarios": scenarios,
        "sensitivity": settings,
        "routes_percent": {"order": ["cold_chain", "local_redistribution", "recovery"], **routes},
        "frozen_controls": controls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the output file is out of date")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if not OUT.is_file() or OUT.read_bytes().decode("utf-8") != text:
            sys.exit("Dashboard data is out of date; run analysis/export_dashboard_data.py")
        print("Dashboard data matches data/")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("utf-8"))
    print("Wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
