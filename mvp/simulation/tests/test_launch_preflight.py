from pathlib import Path

from hpc.validate_launch_preflight import CORE_CONTRACT, validate_contract

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_live_core_and_structural_launch_contracts_are_complete():
    assert validate_contract(REPO_ROOT, "all") == []


def test_preflight_fails_closed_when_evidence_flag_is_removed(tmp_path):
    for relative in CORE_CONTRACT:
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        if relative == "hpc/publication_env.sh":
            text = text.replace("export FULL_EVIDENCE_CAPTURE=1", "export FULL_EVIDENCE_CAPTURE=0")
        target.write_text(text, encoding="utf-8")
    failures = validate_contract(tmp_path, "core")
    assert any("FULL_EVIDENCE_CAPTURE=1" in failure for failure in failures)


def test_preflight_fails_closed_when_worker_wrapper_is_unwired(tmp_path):
    for relative in CORE_CONTRACT:
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        if relative == "hpc/hpc_seed.sh":
            text = text.replace("python hpc/run_with_resource_receipt.py", "python run_seed_directly.py")
        target.write_text(text, encoding="utf-8")
    failures = validate_contract(tmp_path, "core")
    assert any("run_with_resource_receipt.py" in failure for failure in failures)
