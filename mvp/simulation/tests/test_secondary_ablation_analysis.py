import csv
import json
from pathlib import Path

import pytest

from mvp.simulation.analysis.export_secondary_ablations import (
    COMPARISONS,
    SCENARIOS,
    SEEDS,
    analyse,
    main,
)


def _write_panel(root: Path) -> None:
    root.mkdir()
    for index, seed in enumerate(SEEDS):
        scenarios = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            base = 0.70 + scenario_index * 0.01 + index * 0.0007
            records = {"agribrain": {"ari": base}}
            for comparison_index, (_, mode) in enumerate(COMPARISONS, start=1):
                records[mode] = {"ari": base - comparison_index * 0.002 - (index % 4) * 0.0001}
            scenarios[scenario] = records
        payload = {
            "_meta": {"source_commit": "a" * 40, "run_tag": "aaaaaaa_20260828_010101"},
            "seed": seed,
            "scenarios": scenarios,
        }
        (root / f"seed_{seed}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cli_is_byte_deterministic_and_exports_exact_families(tmp_path):
    seed_root = tmp_path / "seeds"
    _write_panel(seed_root)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["--seed-root", str(seed_root), "--output-dir", str(out_a)]) == 0
    assert main(["--seed-root", str(seed_root), "--output-dir", str(out_b)]) == 0
    for name in ("secondary_ablation_analysis.json", "secondary_ablation_analysis.csv"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    payload = json.loads((out_a / "secondary_ablation_analysis.json").read_text())
    assert set(payload["by_scenario"]) == set(SCENARIOS)
    assert all(len(records) == 3 for records in payload["by_scenario"].values())
    assert all(
        0 <= record["p_value_adj_by_within_scenario"] <= 1
        for records in payload["by_scenario"].values() for record in records
    )
    with (out_a / "secondary_ablation_analysis.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(SCENARIOS) * 3


def test_analysis_fails_closed_on_missing_comparator(tmp_path):
    seed_root = tmp_path / "seeds"
    _write_panel(seed_root)
    path = seed_root / f"seed_{SEEDS[0]}.json"
    payload = json.loads(path.read_text())
    del payload["scenarios"][SCENARIOS[0]][COMPARISONS[0][1]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing/non-finite ARI"):
        analyse(seed_root)


def test_analysis_fails_closed_on_extra_seed_file(tmp_path):
    seed_root = tmp_path / "seeds"
    _write_panel(seed_root)
    (seed_root / "seed_999.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        analyse(seed_root)
