# AGRI-BRAIN

This directory holds the implementation of the AGRI-BRAIN system:

- `backend/` -- FastAPI backend (port 8100), MCP / piRAG layer, decision engine, chain client.
- `frontend/` -- React + Vite dashboard (port 5173).
- `contracts/` -- Solidity smart contracts (Hardhat).
- `agents/` -- multi-episode agent runner.

## Where to start

The canonical README and run guide live at the repository root:

- [Repository README](../README.md) -- architecture overview, claims, screenshots.
- [HOW_TO_RUN.md](../HOW_TO_RUN.md) -- complete setup, environment variables, and walk-throughs.
- [docs/RELEASE.md](../docs/RELEASE.md) -- release procedure (version bumps, lockfile, DOI, tagging).
- [docs/METHODS_REPRO_APPENDIX.md](../docs/METHODS_REPRO_APPENDIX.md) -- the canonical reproduction recipe.
- [docs/CLAIMS_TO_EVIDENCE.md](../docs/CLAIMS_TO_EVIDENCE.md) -- claim-to-artifact crosswalk.

This stub replaces a previous duplicate quick-start that drifted out of sync
with the root README. Edit the root README only.
