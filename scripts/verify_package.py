"""Verify the distributed files and the published results without running simulations."""
from pathlib import Path
import ast, csv, hashlib, json, re, sys
from collections import defaultdict

R = Path(__file__).resolve().parents[1]
STUDY = R / "data" / "study"
PRIMARY = ("static", "no_context", "agribrain")
MAIN_MODES = PRIMARY + ("adjustment_only", "reservation_only", "agribrain_no_peer", "feature_inputs_prior",
                        "agribrain_eligibility_no_peer", "no_pinn", "rule_always_redistribute")
SEEDS = range(5001, 5021)
SCENARIOS = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline")


def load(p): return json.loads((R / p).read_text(encoding="utf-8"))
def rows(p):
    with (R / p).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256((R / p).read_bytes()).hexdigest()
def check(ok, message):
    if not ok: raise AssertionError(message)


def check_links():
    """Every relative link and image in the README and the documentation pages must exist."""
    pages = [R / "README.md", *sorted((R / "docs").glob("*.md")), R / "figures" / "README.md", STUDY / "README.md"]
    pattern = re.compile(r'\]\(([^)\s]+)\)|src="([^"]+)"')
    for page in pages:
        for markdown, html in pattern.findall(page.read_text(encoding="utf-8")):
            target = (markdown or html).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")): continue
            check((page.parent / target).exists(), "Broken link in " + page.relative_to(R).as_posix() + ": " + target)


def seed_means(ends, setting, mode):
    """Equal-scenario mean ARI of each seed."""
    per = defaultdict(list)
    for x in ends:
        if x["setting"] == setting and x["mode"] == mode: per[int(x["seed"])].append(float(x["ari"]))
    check(sorted(per) == list(SEEDS) and all(len(v) == 5 for v in per.values()), "Coverage: " + setting + " " + mode)
    return [sum(per[s]) / 5 for s in SEEDS]


def main():
    manifest = load("FILE_HASHES.json")
    for name, digest in manifest.items():
        check(sha(name) == digest, "Checksum mismatch: " + name)
    published = load("data/study/MANIFEST.json")["files"]
    for name, info in published.items():
        check(sha("data/study/" + name) == info["published_sha256"], "Published table differs from its manifest: " + name)
        check(len(rows("data/study/" + name)) == info["published_rows"], "Row count: " + name)

    ends = rows("data/study/seed_endpoints.csv")
    cells = {(x["setting"], x["seed"], x["scenario"], x["mode"]) for x in ends}
    check(len(cells) == len(ends), "Duplicate endpoint")
    settings = {x["setting"] for x in ends}
    sens = {x["setting"] for x in rows("data/study/sensitivity_settings.csv")}
    check(len(sens) == 40 and settings == sens | {"main_fail_closed", "main_fail_open"}, "Published settings")
    for setting in settings:
        modes = MAIN_MODES if setting == "main_fail_closed" else PRIMARY
        check({x["mode"] for x in ends if x["setting"] == setting} == set(modes), "Modes in " + setting)
        for mode in modes: seed_means(ends, setting, mode)
    check(len(ends) == 1000 + 300 + 40 * 300, "Endpoint count")

    # the principal comparison and the adjustment at the guard, recomputed from the endpoints
    a, n = seed_means(ends, "main_fail_closed", "agribrain"), seed_means(ends, "main_fail_closed", "no_context")
    gain = sum(x - y for x, y in zip(a, n)) / 20
    principal = next(x for x in rows("data/study/principal_comparisons.csv") if x["condition"] == "fail-closed")
    check(abs(gain - float(principal["mean"])) < 1e-9, "Principal comparison")
    check(all(x > y for x, y in zip(a, n)), "AGRI-BRAIN ahead of No-Context on every seed")
    g, r = seed_means(ends, "main_fail_closed", "agribrain_no_peer"), seed_means(ends, "main_fail_closed", "reservation_only")
    at_guard = sum(x - y for x, y in zip(g, r)) / 20
    row = next(x for x in rows("data/study/guard_by_adjustment.csv") if x["scenario"] == "overall" and x["guard"] == "reservation")
    check(abs(at_guard - float(row["adjustment_effect"])) < 1e-9, "Adjustment at the guard")
    # every sensitivity gain, recomputed from the endpoints
    for x in rows("data/study/sensitivity_settings.csv"):
        a, n = seed_means(ends, x["setting"], "agribrain"), seed_means(ends, x["setting"], "no_context")
        check(abs(sum(p - q for p, q in zip(a, n)) / 20 - float(x["agribrain_minus_no_context"])) < 1e-9,
              "Sensitivity gain: " + x["setting"])

    for p in R.rglob("*.py"):
        if "node_modules" not in p.parts: ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    figures = load("figures/figure_index.json")
    check(len(figures) == 5 and len({x["file"] for x in figures}) == 5, "Figure coverage")
    check(all((R / "figures" / x["file"]).is_file() for x in figures), "Missing figure")
    check(all((R / d).exists() for x in figures for d in x["data"]), "Missing figure data")
    records = sorted(p.name for p in (R / "data/figure_inputs").rglob("*.json.gz"))
    check(len(records) == 200, "Figure input records")
    check((R / "docs/images/architecture.png").is_file(), "Missing pipeline diagram")
    check_links()
    print(json.dumps({"status": "PASS", "files_verified": len(manifest), "published_tables": len(published),
                      "endpoints": len(ends), "settings": len(settings), "overall_paired_ari_gain": gain,
                      "adjustment_at_the_guard": at_guard, "figures": len(figures), "simulation_rerun": False}, indent=2))


if __name__ == "__main__":
    main()
