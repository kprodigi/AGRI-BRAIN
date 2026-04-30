# AGRI-BRAIN Full-Stack Pre-HPC Audit

> **HISTORICAL SNAPSHOT.** This audit was run against repo HEAD
> ``1bdc602``. Many of the findings below were addressed in the fix
> packs that followed; the Post-fix update section at the end of the
> file records which items were closed. Do not treat this report as
> current truth; for the current codebase status refresh against
> ``HEAD`` (see root ``HOW_TO_RUN.md``).

**Auditor:** repository audit (read-only; no product attribution)  
**Date:** 2026-04-23  
**Repository HEAD:** 1bdc60242a7b3b518de41ff0d41c71a992b666d8  
**Auditor instruction:** PRE_HPC_FULLSTACK_AUDIT_PROMPT.md  
**Mode:** read-only

## 0. Executive summary

- Total files inspected: approximately 240 text and code paths enumerated (135 Python files under the repo, plus shell, YAML, Solidity, JSON manifests, frontend JSX, Markdown); architecture-critical and Path-B touchpoints read in full; remaining files covered via targeted reads and cross-repo pattern scans. See Section 7.
- Total findings: 36  
  - CRITICAL: 16  
  - HIGH: 8  
  - MEDIUM: 7  
  - LOW: 3  
  - OBSERVATION: 2  
- Recommended next action: **NO-GO** for a submission bundle that must be defensible as internally consistent (counterfactual evaluation, figure pipeline, and silent exception sites need resolution first). For **HPC seed JSON generation only**, the repository is in **conditional GO** shape: `THETA` / `THETA_CONTEXT` shape checks are present in `hpc_*.sh`, and `run_single_seed.py` iterates 8 modes consistent with `generate_results.MOD`.

## 1. Findings by severity

### 1.1 CRITICAL

#### F-001 (CRITICAL): `except Exception: pass` swallows errors in `context_to_logits` temporal path
- File: `agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py:199`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Failures when applying temporal modulation to the context modifier are ignored. The model can run with partially applied or wrong temporal scaling without surfacing a fault, breaking traceability and debugging on HPC or in production.
- Recommended action: Replace with logging at warning or error level and a defined fallback (for example, skip temporal scaling only when the failure mode is known), or re-raise in strict modes.
- References: F-002 to F-015 (same pattern class).

#### F-002 (CRITICAL): `except Exception: pass` in cooperative overlay blending
- File: `agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py:448`
- Layer: `backend.src.agents`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Any failure in the cooperative agent MCP or piRAG path is silenced. The primary agent may proceed with a modifier that is not the intended blend, without visibility.
- Recommended action: Log the exception with role and hour; consider degrading to non-cooperative modifier explicitly.

#### F-003 (CRITICAL): `except Exception: pass` in RAG import path for `decide` in `app.py`
- File: `agri-brain-mvp-1.0.0/backend/src/app.py:573`
- Layer: `backend.src`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Any failure in `get_policy_context` is invisible. The API returns decisions that may omit context that operators assume was applied.
- Recommended action: Log at minimum; return a field in the response when context loading failed.

#### F-004 (CRITICAL): `except Exception: pass` in `generate_results` RAG path
- File: `mvp/simulation/generate_results.py:322`
- Layer: `mvp.simulation`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Legacy RAG path failures are silent. Mixed success across seeds or scenarios can skew comparisons without a single log line.
- Recommended action: Log once per episode or at debug level with scenario and mode.

#### F-005 (CRITICAL): `except Exception: pass` in `trace_exporter`
- File: `agri-brain-mvp-1.0.0/backend/pirag/trace_exporter.py:185`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Paper evidence export can silently drop fields. Reproducibility and supplementary material integrity suffer.
- Recommended action: Log and count skipped export segments.

#### F-006 (CRITICAL): `except Exception: pass` in `agent_pipeline`
- File: `agri-brain-mvp-1.0.0/backend/pirag/agent_pipeline.py:58`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Pipeline stages can fail without propagating, producing empty or partial retrieval for downstream guards.
- Recommended action: Log with stage identifier; fail closed when guards require non-empty evidence.

#### F-007 (CRITICAL): `except Exception: pass` in `explain_decision`
- File: `agri-brain-mvp-1.0.0/backend/pirag/explain_decision.py:106`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Explainability text can be silently empty while the rest of the system reports success.
- Recommended action: Log and set a user-visible fallback string that states explanation generation failed.

#### F-008 (CRITICAL): `except Exception: pass` in `context_builder`
- File: `agri-brain-mvp-1.0.0/backend/pirag/context_builder.py:119`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Partial context construction is discarded without trace. Affects `guards_passed` and piRAG honesty assumptions.
- Recommended action: Structured logging and propagation of a failure flag on `rag_context`.

#### F-009 (CRITICAL): `except Exception: pass` at end of `retrieve_role_context` in `context_builder`
- File: `agri-brain-mvp-1.0.0/backend/pirag/context_builder.py:349`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Top-level failure of retrieval returns whatever partial state was built, with no signal that the function exited through the error path.
- Recommended action: Return a dict with `error` set, or re-raise after logging.

#### F-010 (CRITICAL): `except Exception: pass` in `context_provider`
- File: `agri-brain-mvp-1.0.0/backend/pirag/context_provider.py:107`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Context enrichment for policy hooks fails silently.
- Recommended action: Log with stack trace in dev; surface degraded mode in return payload.

#### F-011 (CRITICAL): `except Exception: pass` in outer handler in `context_provider`
- File: `agri-brain-mvp-1.0.0/backend/pirag/context_provider.py:174`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Same class as F-010 at outer scope; callers get `{}` with no error bit.
- Recommended action: Unify with a single error-reporting return shape.

#### F-012 (CRITICAL): `except Exception: pass` in `unit_guard`
- File: `agri-brain-mvp-1.0.0/backend/pirag/guards/unit_guard.py:23`
- Layer: `backend.pirag.guards`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Unit validation can short-circuit without recording why. Physics or regulatory checks may appear to pass.
- Recommended action: Return False with a reason string or log the exception.

#### F-013 (CRITICAL): `except Exception: pass` in `dynamic_knowledge` ingestion
- File: `agri-brain-mvp-1.0.0/backend/pirag/dynamic_knowledge.py:116`
- Layer: `backend.pirag`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Knowledge-base updates from simulation history can fail without notification; future retrieval drift is unexplained.
- Recommended action: Log ingestion failures and skip count.

#### F-014 (CRITICAL): `except Exception: pass` in `ingest_corpus` script
- File: `agri-brain-mvp-1.0.0/backend/pirag/scripts/ingest_corpus.py:11`
- Layer: `backend.pirag.scripts`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Operator runs can report success while individual documents fail to ingest.
- Recommended action: Per-file error logging and non-zero exit if any file fails in strict mode.

#### F-015 (CRITICAL): `except Exception: pass` in `build_artifact_manifest`
- File: `mvp/simulation/analysis/build_artifact_manifest.py:45`
- Layer: `mvp.simulation.analysis`
- Evidence:
  ```
  except Exception:
      pass
  ```
- Why this matters: Manifest generation can miss artifacts silently; paper submission packages may be incomplete.
- Recommended action: Log each skipped path with reason.

#### F-016 (CRITICAL): Invariant "no `no_yield` outside `docs/path_b/`" is false
- File: `mvp/simulation/generate_figures.py:133` (and lines `145`, `157`, `169`, `831`, `839`)
- Layer: `mvp.simulation`
- Evidence:
  ```
  "no_yield":   "#8D6E63",   # warm brown (Path B ablation)
  ```
- Why this matters: The audit charter states `no_yield` must not appear in production code, simulator pipelines, or user-facing figure tooling outside historical `docs/path_b/`. The figure module still encodes the dropped mode in palettes and comments, so the repository violates the stated invariant.
- Recommended action: Remove `no_yield` keys and comments; align `fig7` copy with the 8-mode-only story.
- References: Section 2.2 of the audit prompt.

### 1.2 HIGH

#### F-017 (HIGH): Counterfactual `select_action` omits supply and demand forecast kwargs
- File: `agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py:555`
- Layer: `backend.src.agents`
- Evidence:
  ```
  action_without, probs_without = _sa(
      mode="agribrain", rho=obs.rho, inv=obs.inv,
      y_hat=obs.y_hat, temp=obs.temp, tau=obs.tau,
  ```
- Why this matters: The live path passes `supply_hat`, `supply_std`, and `demand_std` from `obs.raw` into `select_action` in `step()`. The counterfactual path does not, so `phi[6:9]` default to zero while the real decision used a full 9D `phi`. The recorded "action without context" conflates removal of the context modifier with removal of forecast features, which invalidates attribution metrics and evaluator summaries.
- Recommended action: Forward the same `supply_hat`, `supply_std`, `demand_std` as in `step()` when calling `_sa` for the counterfactual, keeping only `context_modifier=None` as the controlled difference.

#### F-018 (HIGH): `DemoPage` still displays `Θ(3×6)` and `φ(6D)` in policy math
- File: `agri-brain-mvp-1.0.0/frontend/src/pages/DemoPage.jsx:304`
- Layer: `frontend`
- Evidence:
  ```
  logits = Θ(3×6) × φ(6D) + γ·τ + SLCA_bonus + role_bias
  ```
- Why this matters: User-facing and demo text contradicts the shipped 3×9 and 9D state, creating manuscript and product inconsistency.
- Recommended action: Update the template string to 3×9 and 9D and align subscripts with the paper.

#### F-019 (HIGH): `run_benchmark_suite.py` module docstring still says nine modes
- File: `mvp/simulation/benchmarks/run_benchmark_suite.py:7`
- Layer: `mvp.simulation.benchmarks`
- Evidence:
  ```
  all nine simulation modes.
  ```
- Why this matters: Stage 3 `MODES` tuple has eight entries. Readers following the docstring will mis-plan seeds and paper counts.
- Recommended action: Replace "nine" with "eight" and reconcile with `run_single_seed.py` and `generate_results.MOD`.

#### F-020 (HIGH): `yield_query` `ToolSpec` description still references `psi_5`
- File: `agri-brain-mvp-1.0.0/backend/pirag/mcp/registry.py:307`
- Layer: `backend.pirag.mcp`
- Evidence:
  ```
  description="Holt-Winters yield/supply forecast with normalised supply uncertainty (psi_5)",
  ```
- Why this matters: Supply uncertainty is no longer a context feature; the description misstates the architecture and confuses tool consumers.
- Recommended action: Reword to state CV / residual-std and state-vector consumption.

#### F-021 (HIGH): `hpc_run.sh` uses unpinned `pip install` for backend and pytest
- File: `hpc_run.sh:34`–`hpc_run.sh:35`
- Layer: repository root / HPC
- Evidence:
  ```
  pip install -e agri-brain-mvp-1.0.0/backend --quiet
  pip install pytest --quiet
  ```
- Why this matters: Reproducibility across login-node installs and HPC job nodes is weakened if dependency resolution drifts between submission dates.
- Recommended action: Install from a locked requirements export or pin `pytest` to a version hash recorded next to the paper.

#### F-022 (HIGH): `decide` path uses an unseeded `numpy` RNG for `select_action`
- File: `agri-brain-mvp-1.0.0/backend/src/app.py:577`
- Layer: `backend.src`
- Evidence:
  ```
  rng = np.random.default_rng()
  ```
- Why this matters: Stochastic action draws from the REST `decide` path are not reproducible from seed alone, unlike the simulation where RNGs are threaded from `run_all`. Comparing API samples to HPC results is not straightforward.
- Recommended action: Accept an optional `seed` in the request body or derive a deterministic sub-seed from `step` and agent id; document the behaviour in OpenAPI.

#### F-023 (HIGH): `generate_figures.py` comment block still claims `no_yield` for fig7 and significance JSON
- File: `mvp/simulation/generate_figures.py:831`–`mvp/simulation/generate_figures.py:835`
- Layer: `mvp.simulation`
- Evidence:
  ```
  The no_yield mode stays
  in the simulator and the significance JSON but is not plotted here:
  ```
- Why this matters: `generate_results.MOD` and benchmark JSONs no longer include `no_yield`; the comment is stale and misleads figure maintainers.
- Recommended action: Rewrite the docstring to describe 8 publication modes only.

#### F-024 (HIGH): `pyproject.toml` uses open lower bounds on core numerical stack
- File: `agri-brain-mvp-1.0.0/backend/pyproject.toml:10`–`agri-brain-mvp-1.0.0/backend/pyproject.toml:22`
- Layer: dependency posture
- Evidence:
  ```
  'numpy>=2.1,<3',
  'pandas>=2.2,<3',
  ```
- Why this matters: Any minor release inside the range can change numerics or dtype behaviour; long HPC runs are harder to bit-match across years.
- Recommended action: Record a `pip freeze` or lockfile in-repo for paper runs and cite it in `HOW_TO_RUN.md`.

### 1.3 MEDIUM

#### F-025 (MEDIUM): No test asserts LSTM `std` differs from `series_std` on a nontrivial series
- File: `agri-brain-mvp-1.0.0/backend/tests/test_path_b_integration.py:258`
- Layer: `backend.tests`
- Evidence:
  ```
  assert "std" in out
  assert "series_std" in out
  assert out["std"] >= 0.0
  ```
- Why this matters: The prompt asked for a regression that residual-std and series-std differ when both are well defined. Nonnegativity alone allows a bug where `std` aliases `series_std`.
- Recommended action: Add a case with clear residual structure and `assert out["std"] != out["series_std"]` within tolerance, or assert inequality when variance is nonzero.

#### F-026 (MEDIUM): `compat.py` swallows all JSON body parse failures with `pass`
- File: `agri-brain-mvp-1.0.0/backend/src/routers/compat.py:28`
- Layer: `backend.src.routers`
- Evidence:
  ```
  except (json.JSONDecodeError, ValueError): pass
  ```
- Why this matters: Malformed bodies silently become `{}`; clients get decisions based on defaults without an error, which hampers debugging and security review.
- Recommended action: Return 400 on parse failure for POST bodies.

#### F-027 (MEDIUM): Mixed `print` and `logging` in simulation orchestrator
- File: `mvp/simulation/generate_results.py:473` and scattered `print` in `run_all`
- Layer: `mvp.simulation`
- Evidence:
  ```
  print(f"  Policy weights updated via REINFORCE (delta norm: {delta_norm:.6f})")
  ```
- Why this matters: HPC logs mix unstructured prints with other tools; `logging` with levels is easier to filter in Slurm.
- Recommended action: Route through `logging` for seed and aggregate jobs.

#### F-028 (MEDIUM): `registry.py` uses MD5 for cache key hashing
- File: `agri-brain-mvp-1.0.0/backend/pirag/mcp/registry.py:77`
- Layer: `backend.pirag.mcp`
- Evidence:
  ```
  hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
  ```
- Why this matters: Not a collision attack surface here, but reviewers may question MD5. Blake2 or SHA-256 would avoid review friction.
- Recommended action: Swap to a modern hash for cache keys if policy requires it.

#### F-029 (MEDIUM): `lstm_demand_forecast` passes `tail=8` into `in_sample_residual_std` as keyword
- File: `agri-brain-mvp-1.0.0/backend/src/models/lstm_demand.py:336`
- Layer: `backend.src.models`
- Evidence:
  ```
  residual_std = model.in_sample_residual_std(tail, tail=8)
  ```
- Why this matters: The second `tail=8` overrides the name `tail` in the first positional for the parameter also named `tail` in the method signature, which is easy to mis-edit later.
- Recommended action: Use `in_sample_residual_std(tail, tail=8)` as explicit `series=tail, tail=8` style by renaming the parameter in the public API to `window` to avoid the duplicate name.

#### F-030 (MEDIUM): `test_path_b` docstring still says "unchanged behaviour" for yield_query while architecture moved
- File: `agri-brain-mvp-1.0.0/backend/tests/test_path_b_integration.py:22`
- Layer: `backend.tests`
- Evidence:
  ```
  1. yield_query MCP tool (unchanged behaviour)
  ```
- Why this matters: Behaviour relative to `psi_5` changed; the test header is stale.
- Recommended action: Update the section title to reflect state-vector consumption.

#### F-031 (MEDIUM): CI workflow uses floating tags `v4` / `v5` on GitHub Actions
- File: `.github/workflows/ci.yml:15`–`.github/workflows/ci.yml:16`
- Layer: CI
- Evidence:
  ```
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
  ```
- Why this matters: Moving major tags can change build behaviour; supply-chain hardening often prefers a pinned commit SHA.
- Recommended action: Pin to immutable SHAs as allowed by your org policy.

### 1.4 LOW

#### F-032 (LOW): Extra literature name in `THETA` inline comment for `phi_6` without a matching full citation block for Schoenherr
- File: `agri-brain-mvp-1.0.0/backend/src/models/action_selection.py:180`
- Layer: `backend.src.models`
- Evidence:
  ```
  redistribution pathways (Fisher 1997; Schoenherr & Swink 2012).
  ```
- Why this matters: The module `References` list does not include Schohenerr, so a reader cannot resolve the source from the same block as Fisher and Chopra.
- Recommended action: Add the bibliographic line to the `References` section or drop the in-comment name.

#### F-033 (LOW): `rng = np.random.default_rng()` in `app.py` `decide` is unseeded
- File: `agri-brain-mvp-1.0.0/backend/src/app.py:577`
- Layer: `backend.src`
- Evidence:
  ```
  rng = np.random.default_rng()
  ```
- Why this matters: Stochastic `select_action` calls are not reproducible across requests. Low if only deterministic mode is used in practice; higher if not.
- Recommended action: Optional seed in request body.

#### F-034 (LOW): `compat.py` one-line `except` reduces readability
- File: `agri-brain-mvp-1.0.0/backend/src/routers/compat.py:28`
- Layer: `backend.src.routers`
- Evidence:
  ```
  except (json.JSONDecodeError, ValueError): pass
  ```
- Why this matters: Style and static analysis tools often flag one-line `except` bodies.
- Recommended action: Expand to a block (same semantic as F-026).

### 1.5 OBSERVATION

#### F-035 (OBSERVATION): Physics reranker docstring uses "Physics plausibility re-ranking"
- File: `agri-brain-mvp-1.0.0/backend/pirag/physics_reranker.py:9`
- Layer: `backend.pirag`
- Evidence:
  ```
  2. **Physics plausibility re-ranking**:
  ```
- Why this matters: Wording is close to but not identical to the figure copy mentioned in the audit prompt. Worth aligning for editorial consistency.
- Recommended action: Pick one phrasing for paper, code, and figures.

#### F-036 (OBSERVATION): `test_path_b` notes residual-std tests but Holt-Winters volatility test is the main inequality test for supply
- File: `agri-brain-mvp-1.0.0/backend/tests/test_path_b_integration.py:281`
- Layer: `backend.tests`
- Why this matters: LSTM side lacks a volatility contrast test mirroring `test_holt_winters_std_grows_with_volatility`.
- Recommended action: Optional parity test for demand series.

## 2. Findings by layer

### 2.1 Repository hygiene
- F-021, F-024, F-031

### 2.2 Dependency posture
- F-021, F-024

### 2.3 Backend src
- F-002, F-003, F-004, F-017, F-022, F-026, F-033, F-034 (coordinator, app, decide path, compat)

### 2.4 Backend pirag
- F-001, F-005, F-006, F-007, F-008, F-009, F-010, F-011, F-012, F-013, F-014, F-020, F-028, F-035

### 2.5 Backend tests
- F-025, F-030, F-036

### 2.6 Simulation pipeline
- F-004, F-015, F-016, F-019, F-023, F-027

### 2.7 Frontend
- F-018

### 2.8 Smart contracts
- No issues logged in this pass beyond CI smoke via Hardhat; full formal security review of six contracts is out of scope for this file-read sweep.

### 2.9 Knowledge base
- No finding; 20 `.txt` files present under `agri-brain-mvp-1.0.0/backend/pirag/knowledge_base/`.

### 2.10 Documentation
- F-018, F-019, F-020, F-023; root `README.md` aligns with 9D `phi` and 5D `psi` in the architecture highlights (spot-checked).

### 2.11 Configuration / CI
- F-021, F-031; `settings.py` default `CORS_ORIGINS` to `*` in dev (acceptable for local dev; production should set explicit origins).

### 2.12 Cross-cutting
- F-001 through F-016 (error-handling class)

### 2.13 Security
- F-003, F-022, F-026, F-028 (error hiding, CORS in dev, MD5 in cache keys for review comfort)

### 2.14 Reproducibility
- F-021, F-022, F-024, F-033

### 2.15 Symmetric forecast channel (1bdc602 specific)
- F-017, F-025; core `action_selection` / forecasters / `generate_results` plumbing verified in read-through (see Section 3)

## 3. Architectural invariant verification

| Invariant | Verified | Evidence (file:line) |
|---|---|---|
| `THETA.shape == (3, 9)` | yes | `agri-brain-mvp-1.0.0/backend/src/models/action_selection.py:201`–`action_selection.py:205` |
| `THETA_CONTEXT.shape == (3, 5)` | yes | `agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py:52`–`context_to_logits.py:57` |
| `_MCP_FEATURE_MASK == [1.0, 1.0, 0.0, 0.0, 1.0]` | yes | `context_to_logits.py:78` |
| `_PIRAG_FEATURE_MASK == [0.0, 0.0, 1.0, 1.0, 0.0]` | yes | `context_to_logits.py:79` |
| `_MODIFIER_CLAMP == 1.0` | yes | `context_to_logits.py:39` |
| `MODIFIER_RULES` length 5 | yes | `context_to_logits.py:211`–`context_to_logits.py:217` |
| `VALID_MODES` = 8 entries, no `no_yield` | yes | `action_selection.py:127`–`action_selection.py:130` |
| `INV_BASELINE` defined in action_selection.py | yes | `action_selection.py:149` |
| `build_feature_vector` accepts 3 forecast kwargs | yes | `action_selection.py:304`–`action_selection.py:306` |
| Static registry count = 13 | yes (when optional imports succeed) | `registry.py:132`–`registry.py:320` (7 unconditional + 6 in `try` blocks) |
| Runtime registry count = 18 | yes (13 + 5, by code inspection) | `registry.py` plus `mcp/agent_capabilities.py:155`–`agent_capabilities.py:173` (5 roles) |
| `yield_query` in processor/coop/dist workflows | yes | `tool_dispatch.py:165`, `tool_dispatch.py:173`, `tool_dispatch.py:182` |
| `recovery_capacity_check` in DISTRIBUTOR_WORKFLOW | yes | `tool_dispatch.py:180` |
| `lstm_demand_forecast` returns residual-std under `std` | yes | `lstm_demand.py:351` |
| `yield_supply_forecast` returns residual-std under `std` | yes | `yield_forecast.py:121` |
| `env_state[...]` populated | yes | `generate_results.py:336`–`generate_results.py:344` |
| SLURM scripts assert THETA and THETA_CONTEXT | yes | `hpc_seed.sh:50`–`hpc_seed.sh:52`, `hpc_aggregate.sh:48`–`hpc_aggregate.sh:50`, `hpc_run.sh:47`–`hpc_run.sh:49` |
| Frontend `FEATURE_LABELS` has 5 entries | yes | `frontend/src/components/explainability/ExplainabilityPanel.jsx:16`–`ExplainabilityPanel.jsx:22` (other pages: `McpPiragPage.jsx:32` area) |
| No `no_yield` outside `docs/path_b/` | no | F-016 `generate_figures.py:133` |
| 20 knowledge base .txt files | yes | `knowledge_base` glob (20 files) |
| 6 Solidity contracts present and tested | not executed | Contracts exist under `contracts/hardhat/contracts/`; `ci.yml:47`–`ci.yml:48` runs `npx hardhat test` (this audit did not re-run) |

## 4. HPC submission readiness

- **Section 3 invariants** largely hold in code for tensors, modes, and `env_state` plumbing. The explicit exception is the **`no_yield` residue in `generate_figures.py`**, which breaks the prompt's "no `no_yield`" rule (F-016), not the numerical engine.
- **`hpc_seed.sh` → `run_single_seed.py`**: entry point matches `hpc_seed.sh:56`; `--output-dir` and seed list match the charter.
- **`hpc_aggregate.sh`**: runs Stage 1 `generate_results.py`, Stage 3 `run_benchmark_suite.py`, and downstream stages; `RUN_TAG` gating is consistent with `hpc_run.sh:54`–`hpc_run.sh:55`.
- **Path-shape checks**: all three shell scripts embed the Python assert block for `THETA` and `THETA_CONTEXT` (see Section 3 table).
- **Wall time**: this audit did not time `run_all`; a conservative operator should still profile one seed on the target cluster after venv install.
- **Output isolation**: `hpc_seed.sh:36`–`hpc_seed.sh:37` and `hpc_run.sh:54` support hash-tagged directories as specified.

**Verdict for raw benchmark JSON generation:** conditional **GO** with logging discipline. **Verdict for a paper-ready, internally consistent bundle:** **NO-GO** until F-016, F-017, F-018, F-019, and F-023 are triaged.

## 5. Manuscript-vs-code drift

| Paper claim (from prompt Section 2 or live UI) | Code reality | Severity | Reference |
|---|---|---|---|
| `no_yield` absent outside historical docs | `generate_figures.py` still names `no_yield` in style dicts and text | F-016 | `generate_figures.py:133` |
| Policy uses Θ(3,9) and 9D `phi` in UI copy | `DemoPage` still shows 3×6 and 6D | F-018 | `DemoPage.jsx:304` |
| `yield_query` help text matches architecture | Registry description still says `psi_5` | F-020 | `registry.py:307` |
| Stage 3 mode count in prose | Docstring says nine modes, tuple has eight | F-019 | `run_benchmark_suite.py:7` and `39`–`42` |
| `no_yield` in fig7 narrative | Comment claims simulator still has the mode | F-023 | `generate_figures.py:831` |
| "Action without context" is context-only ablation | Counterfactual omits forecast kwargs | F-017 | `coordinator.py:555` |

## 6. Suggested triage order (no work performed)

1. **F-016, F-023, F-019** so pipelines, docstrings, and figure code agree on 8 modes and no stray `no_yield` in product paths.  
2. **F-017** because it affects all context-attribution and evaluator statistics.  
3. **F-018, F-020** for immediate user-visible consistency with the paper.  
4. **F-001 through F-015** in batch by subsystem (coordinator, context pipeline, app), replacing silent `pass` with logging or explicit degraded flags.  
5. **F-021, F-024** when you freeze the paper's environment.  
6. **F-025** to lock the residual-std contract in tests.  

## 7. Audit scope notes

- **Execution scope:** No unit tests, benchmarks, Hardhat, or HPC jobs were run, per charter. `pytest --collect-only` reported `98/113 tests collected (15 deselected)`, matching the stated baseline. Runtime registry tool count (18) was verified by static inspection of `get_default_registry` and `register_all_agent_capabilities`, not by executing Python.
- **Solidity coverage:** No line-by-line reentrancy or access-control proof was performed on the six contracts; CI runs `hardhat test` but this audit did not re-read every `.sol` file line in full in favor of time-boxed focus on 1bdc602 state and Python paths.
- **File coverage:** The prompt asked for a full read of every file in the repository. In practice, this pass prioritised all backend `src` and `pirag` modules material to the symmetric forecast change, all `mvp/simulation` Python, HPC scripts, CI, root `README`, frontend pages tied to `FEATURE_LABELS` and `DemoPage`, and pattern scans (for example, `no_yield`, `except Exception`). Lockfiles and minified build JSON were not line-audited.
- **Prior report:** No pre-existing `audit/codex_full_stack_audit.md` was present; this file is the first in `audit/`.

## 8. Methodology notes

The audit used `git status`, `git log`, and `git rev-parse` on a clean tree at `1bdc60242a7b3b518de41ff0d41c71a992b666d8`. The `audit/` directory was created. Files were inspected with editor reads and `rg`/`grep` for invariants, banned symbols, and failure patterns. `pytest --collect-only` was used once to reconcile test inventory with the documented 98/113 split. No repository files were modified except creating this report under `audit/`. Approximate wall time for the review session: 90 to 120 minutes of focused tool-assisted reading.

---

## Post-fix update — 2026-04-23

The following findings from the Codex audit were addressed in the
pre-HPC fix pack:

- F-016: `no_yield` residue removed from `generate_figures.py`. Commit B.
- F-017: counterfactual `select_action` now passes the three forecast
  kwargs (`supply_hat`, `supply_std`, `demand_std`) via stored
  `self._step_*` attributes. Commit A. This was the critical
  methodological fix that preserves attribution integrity for paper
  Tables 5/6/7.
- F-018: `DemoPage.jsx` policy math updated to `Θ(3×9) × φ(9D)` with
  the `Δz` context modifier term. Commit C.
- F-019: `run_benchmark_suite.py` docstring updated from "nine modes"
  to "eight modes". Commit B.
- F-020: `yield_query` `ToolSpec` description rewritten without the
  `psi_5` reference; surrounding comment marks the tool as
  informational. Commit C.
- F-023: `generate_figures.py` fig7 comment block rewritten for the
  8-mode story; the `fig7_modes` filter no longer excludes `no_yield`
  because `MODES` no longer contains it. Commit B.

The following findings are deferred to a post-HPC cleanup pass:

- F-001 through F-015 (silent `except Exception: pass` sites): real
  observability issues but do not change benchmark numerics.
- F-021, F-024 (lockfile gap): documented separately.
- F-022, F-025, F-026 through F-036 (test gaps, lint, style, security
  hardening): non-blocking.

Test suite: `98 passed, 15 deselected` post-fix (unchanged from Codex
baseline). 8-mode coverage test: `15 passed, 4 deselected`.
Architectural invariants all PASS. Counterfactual verification confirms
the buggy CF delta in phi[6:9] was non-zero and is now zero after the
fix.

Verdict: **GO for HPC submission.**
