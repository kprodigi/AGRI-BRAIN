#!/usr/bin/env bash
# set_branch_protection.sh
#
# One-time helper to install GitHub branch-protection rules on
# AGRI-BRAIN's main branch via the REST API. The rules required for
# publication-grade CI gating are:
#
#   1. Require these status checks to pass before merging:
#        - artifact-validation
#        - slow-tests (ubuntu / py3.11)
#        - backend-tests (ubuntu-latest / py3.11)
#        - backend-tests (windows-latest / py3.11)
#        - backend-tests (macos-latest / py3.11)
#        - python-lint (ruff)
#        - contract-tests
#        - contract-analysis
#        - frontend-build
#   2. Require branches to be up-to-date before merging.
#   3. Disallow force-pushes and deletions.
#   4. Linear history (no merge commits without rebase).
#
# Usage:
#   GH_TOKEN="<your-classic-or-fine-grained-PAT-with-repo-admin>" \
#       bash hpc/set_branch_protection.sh
#
# The PAT needs ``Administration: Read and write`` on this repository
# (fine-grained) or the legacy ``repo`` scope (classic). Tokens scoped
# to a single repo are preferred; do not use a tokenless / read-only
# token -- the API call is a PUT and requires write access.
#
# This is a one-time setup. Once the rules are installed they persist
# until manually changed in Settings -> Branches.
set -euo pipefail

OWNER="${OWNER:-kprodigi}"
REPO="${REPO:-AGRI-BRAIN}"
BRANCH="${BRANCH:-main}"

if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "ERROR: GH_TOKEN env var is required (PAT with admin:repo scope)." >&2
  echo "" >&2
  echo "  GH_TOKEN=ghp_... bash hpc/set_branch_protection.sh" >&2
  exit 1
fi

# Status checks the PR must pass. Names must match the ``name:`` fields
# in .github/workflows/ci.yml exactly. ``backend-tests (...)`` includes
# the matrix expansion.
read -r -d '' BODY <<'JSON' || true
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "artifact-validation",
      "slow-tests (ubuntu / py3.11)",
      "backend-tests (ubuntu-latest / py3.11)",
      "backend-tests (windows-latest / py3.11)",
      "backend-tests (macos-latest / py3.11)",
      "python-lint (ruff)",
      "contract-tests",
      "contract-analysis",
      "frontend-build"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

echo "Installing branch protection on ${OWNER}/${REPO}@${BRANCH}..."
RESPONSE=$(curl -sS -X PUT \
  -H "Authorization: Bearer ${GH_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${OWNER}/${REPO}/branches/${BRANCH}/protection" \
  -d "${BODY}")

# Heuristic success check: GitHub returns the protection object with
# url field on success and a message field on error.
if echo "${RESPONSE}" | grep -q '"url"'; then
  echo "OK: branch protection installed on ${BRANCH}."
  echo ""
  echo "Verify:"
  echo "  curl -sS -H 'Authorization: Bearer \$GH_TOKEN' \\"
  echo "       https://api.github.com/repos/${OWNER}/${REPO}/branches/${BRANCH}/protection"
else
  echo "FAILED. API response:" >&2
  echo "${RESPONSE}" >&2
  exit 1
fi
