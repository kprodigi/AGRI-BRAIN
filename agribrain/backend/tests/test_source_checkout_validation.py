"""Focused tests for the publication checkout identity/cleanliness gate."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "hpc" / "validate_source_checkout.py"
SPEC = importlib.util.spec_from_file_location("validate_source_checkout", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

COMMIT = "a" * 40


def _fake_git_output(*, head: str = COMMIT, status: bytes = b""):
    def fake_output(git: str, repo_root: Path, args: list[str]) -> bytes:
        assert git == "/usr/bin/git"
        assert repo_root == MODULE.REPO_ROOT
        if args == ["rev-parse", "--show-toplevel"]:
            return f"{MODULE.REPO_ROOT}\n".encode()
        if args == ["rev-parse", "HEAD"]:
            return f"{head}\n".encode()
        if args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]:
            return status
        raise AssertionError(f"unexpected git invocation: {args}")

    return fake_output


class SourceCheckoutValidationTests(unittest.TestCase):
    def _validate(self, env, *, head=COMMIT, status=b"", allow=False):
        with (
            patch.object(MODULE.shutil, "which", return_value="/usr/bin/git"),
            patch.object(
                MODULE, "_git_output", side_effect=_fake_git_output(head=head, status=status)
            ),
        ):
            return MODULE.validation_errors(env, allow_run_artifacts=allow)

    def test_clean_checkout_with_exact_full_commit_is_accepted(self):
        self.assertEqual(self._validate({"AGRIBRAIN_GIT_COMMIT": COMMIT}), [])

    def test_git_is_mandatory(self):
        with patch.object(MODULE.shutil, "which", return_value=None):
            errors = MODULE.validation_errors({"AGRIBRAIN_GIT_COMMIT": COMMIT})
        self.assertEqual(errors, ["git executable is unavailable on PATH"])

    def test_missing_or_mismatched_commit_is_rejected(self):
        missing = self._validate({}, head="b" * 40)
        mismatch = self._validate({"AGRIBRAIN_GIT_COMMIT": COMMIT}, head="b" * 40)
        self.assertTrue(any("must be a full lowercase" in error for error in missing))
        self.assertTrue(any("does not equal checkout HEAD" in error for error in mismatch))

    def test_source_change_is_rejected(self):
        errors = self._validate(
            {"AGRIBRAIN_GIT_COMMIT": COMMIT},
            status=b" M mvp/simulation/generate_results.py\0",
            allow=True,
        )
        self.assertTrue(any("generate_results.py" in error for error in errors))

    def test_parallel_result_output_requires_explicit_worker_allowance(self):
        status = (
            b"?? mvp/simulation/results/benchmark_seeds/run/seed_42.json\0"
            b" M mvp/simulation/results/benchmark_summary.json\0"
        )
        strict = self._validate({"AGRIBRAIN_GIT_COMMIT": COMMIT}, status=status)
        worker = self._validate(
            {"AGRIBRAIN_GIT_COMMIT": COMMIT}, status=status, allow=True
        )
        self.assertTrue(any("uncommitted non-output changes" in error for error in strict))
        self.assertEqual(worker, [])

    def test_rename_crossing_output_boundary_is_rejected(self):
        # Porcelain -z places a rename/copy's second path in the next NUL field.
        status = b"R  mvp/simulation/results/generated.json\0README.md\0"
        errors = self._validate(
            {"AGRIBRAIN_GIT_COMMIT": COMMIT}, status=status, allow=True
        )
        self.assertTrue(any("README.md" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
