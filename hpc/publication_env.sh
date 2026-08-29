#!/bin/bash
# Canonical, submission-grade environment for every AGRI-BRAIN publication
# task.  The Slurm submit inherits the user's login environment, so every
# setting that can alter the simulated treatment is pinned here and optional
# external integrations are disabled explicitly.

export APP_ENV=dev
export FORECAST_METHOD=holt_linear
export SUPPLY_FORECAST_METHOD=persistence
export ONLINE_LEARNING=false
unset DATA_CSV
unset DECISION_LEDGER_DIR

export LLM_PROVIDER=template
unset LLM_API_URL LLM_MODEL LLM_API_KEY
export SIM_API_BASE=

export DETERMINISTIC_MODE=false
export STOCH_TEMP_STD_C=2.5
export STOCH_RH_STD=7.0
export STOCH_DEMAND_FRAC_STD=0.25
export STOCH_INVENTORY_FRAC_STD=0.22
export STOCH_TRANSPORT_KM_STD=0.22
export STOCH_K_REF_STD=0.20
export STOCH_EA_R_STD=0.14
export STOCH_ONSET_JITTER_H=6.0
export STOCH_THETA_NOISE_STD=0.15
export STOCH_POLICY_TEMP_STD=0.0
export STOCH_DELAY_PROB=0.10

export FAILURE_INJECTION=false
export MCP_RELIABILITY=false
export MCP_QOS_ROUTING=false
export PIRAG_COUNTERFACTUAL=false
export PHYSICS_CONSISTENCY_GATE=false
export HETEROGENEOUS_PROFILES=false
export RESEARCH_METRICS=false
export DYNAMIC_KB_FEEDBACK=false
# Experimental treatment calls tools through the in-process JSON-RPC server.
# Wall-clock token buckets are a deployment-boundary control, not a scientific
# factor; disable them so action selection cannot depend on task order, node
# speed, or elapsed wall time. Public API deployments retain the default
# MCP_RATE_LIMITS=transport posture outside this publication environment.
export MCP_RATE_LIMITS=disabled
export PROTOCOL_MAX_RECORDS=4096

export CHAIN_SUBMIT=0
unset CHAIN_CFG_JSON CHAIN_RPC CHAIN_PRIVKEY POLICY_URI
unset APP_API_KEY WS_API_KEY GOVERNANCE_API_KEY CHAIN_API_KEY PHASE_API_KEY MCP_API_KEY

export BENCHMARK_USE_TABLES=false
export BENCHMARK_WRITE_COMPAT=false
export EXPORT_LEGACY_SINGLE_RUN_TRACES=0
export STRICT_VALIDATION=1
# Mandatory lossless archive of every executed adaptation and evaluation
# episode.  Publication workers fail closed if this is missing, so a completed
# HPC run cannot silently retain only the final episode.
export FULL_EVIDENCE_CAPTURE=1
unset AGRIBRAIN_ALLOW_DIRTY

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export MPLBACKEND=Agg

# One numerical thread per Slurm task keeps BLAS/OpenMP reductions and runtime
# oversubscription independent of node library defaults. The Python simulation
# remains stochastic by seed; this only removes ambient parallel reduction order.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
