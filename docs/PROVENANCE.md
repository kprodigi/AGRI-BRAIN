# Provenance

The adaptive primary evidence uses base revision `83c91ce8592c1d40234e5a6bb2a8ce342412f383` and run `context_pair_20260914_193548_83c91ce`. These identifiers belong in the reproducibility record even where other descriptions use a shorter form.

The primary study combines 200 adaptive frozen evaluations with 100 verified historical Static evaluations. All 100 seed–scenario environmental identities matched, and all 28,800 Static steps were checked under the current outcome equations. This is documented reuse, not a claim that all three modes were originally executed together. The completed sensitivity study independently executes all three modes at every setting.

`reproduction/source` holds the computational source needed by the study, without historical public interfaces or results, and restricts its public registry to three modes. The retained capability expressions match the original source. The repository root also holds the application source (backend, frontend, Solidity contracts) and the evaluated-commit files that the primary run uses, unmodified and with the full mode registry; earlier-study results, figures, publication tooling and its documentation are not distributed. Unreachable internal compatibility branches and ledger field names remain to avoid unrelated numerical refactoring. The original source hash manifest and the changed-file hashes are retained. No evaluated result was regenerated or edited when this repository was assembled.

Compact evidence receipts may retain original run names and filenames. Any author-local absolute paths in JSON metadata are replaced with `[LOCAL_PATH]/<filename>` for portability and privacy; original hashes and a sanitization receipt identify these changes. Numerical values are not changed. The repository-wide manifest (`FILE_HASHES.json`) verifies the distributed files, whereas historical receipts refer to the original artifacts.

The scripts in `analysis/` were written for this repository and are not part of the evaluated source. They reimplement the interval and test procedure used for the reported tables and reproduce those tables to within 1e-9; the check reports every difference and exits with an error above that tolerance.

Figure images and four data files that previously carried figure or table numbers were renamed to descriptive names; their contents are byte-identical. `provenance/metadata_sanitization.json` records the previous path of each renamed receipt entry.
