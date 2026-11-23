#!/usr/bin/env python3
"""Generate the versioned synthetic quality trajectories used by the PINN.

This generator is an independent data-generating process (DGP), not a call to
the residual model and not a relabeling of its mechanistic baseline.  The DGP
integrates a distinct latent rate that adds declared packaging, handling, and
humidity-transient effects.  The mechanistic baseline is computed separately
and retained only for residual-model input and diagnostic comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "agribrain" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.models.spoilage import arrhenius_k, compute_spoilage  # noqa: E402


SCHEMA_VERSION = 1
DATASET_VERSION = "synthetic_spoilage_residual_v1"
MASTER_SEED = 20260829
N_TRAJECTORIES = 36
N_STEPS = 288
STEP_HOURS = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(
    trajectory_id: str,
    seed: int,
    thermal_regime: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time_h = np.arange(N_STEPS, dtype=np.float64) * STEP_HOURS
    phase = rng.uniform(0.0, 2.0 * np.pi)
    if thermal_regime == "cold_chain":
        center, amplitude, pulse = 4.5, 1.8, 2.0
    elif thermal_regime == "heat_excursion":
        center, amplitude, pulse = 8.0, 4.0, 8.0
    elif thermal_regime == "oscillatory":
        center, amplitude, pulse = 6.0, 5.0, 4.5
    else:
        raise ValueError(f"unknown thermal regime: {thermal_regime}")

    temp = center + amplitude * np.sin(2.0 * np.pi * time_h / 24.0 + phase)
    pulse_start = rng.uniform(15.0, 49.0)
    pulse_width = rng.uniform(4.0, 12.0)
    pulse_shape = np.exp(-0.5 * ((time_h - pulse_start) / pulse_width) ** 2)
    temp += pulse * pulse_shape + rng.normal(0.0, 0.28, size=N_STEPS)
    ambient = temp + rng.uniform(6.0, 14.0) + 2.0 * np.sin(
        2.0 * np.pi * time_h / 24.0 + phase + 0.6
    )

    rh = 89.0 - 0.55 * (temp - center)
    rh += 4.5 * np.sin(2.0 * np.pi * time_h / 18.0 + phase / 2.0)
    rh += rng.normal(0.0, 1.0, size=N_STEPS)
    rh = np.clip(rh, 65.0, 99.5)

    shock = np.abs(rng.normal(0.018, 0.009, size=N_STEPS))
    handling_centers = rng.choice(np.arange(24, N_STEPS - 24), size=3, replace=False)
    for center_index in handling_centers:
        shock[center_index:center_index + 3] += rng.uniform(0.04, 0.11)
    shock = np.clip(shock, 0.0, 0.20)

    packaging_index = float(rng.uniform(0.12, 0.88))
    k_ref = float(rng.uniform(0.00155, 0.00275))
    ea_over_r = float(rng.uniform(6500.0, 9500.0))
    timestamps = pd.Timestamp("2025-01-10T06:00:00") + pd.to_timedelta(time_h, unit="h")
    base = pd.DataFrame({
        "timestamp": timestamps,
        "tempC": temp,
        "RH": rh,
        "shockG": shock,
        "ambientC": ambient,
    })
    mechanistic = compute_spoilage(base, k_ref=k_ref, Ea_R=ea_over_r)
    mechanistic_quality = mechanistic["shelf_left"].to_numpy(dtype=np.float64)

    # Independent latent DGP.  It integrates its own state under an augmented
    # rate law.  Packaging index is centered: values below 0.5 are protective,
    # while higher values increase permeability. Handling shocks and rapid
    # humidity changes add separate nonnegative stress.  None of these target
    # values is obtained by training a model against C_mech or its ODE residual.
    latent_quality = np.ones(N_STEPS, dtype=np.float64)
    rh_transient = np.zeros(N_STEPS, dtype=np.float64)
    rh_transient[1:] = np.abs(np.diff(rh)) / STEP_HOURS
    for index in range(1, N_STEPS):
        mid_time = 0.5 * (time_h[index] + time_h[index - 1])
        mid_temp = 0.5 * (temp[index] + temp[index - 1])
        mid_rh = 0.005 * (rh[index] + rh[index - 1])
        base_rate = float(arrhenius_k(
            mid_temp,
            k_ref=k_ref,
            Ea_R=ea_over_r,
            rh_frac=mid_rh,
        ))
        alpha = mid_time / (mid_time + 12.0)
        handling = 0.5 * (shock[index] + shock[index - 1])
        transient = 0.5 * (rh_transient[index] + rh_transient[index - 1])
        log_multiplier = (
            0.44 * (packaging_index - 0.50)
            + 0.80 * handling
            + 0.0040 * transient
        )
        latent_rate = base_rate * alpha * float(np.exp(log_multiplier))
        latent_quality[index] = latent_quality[index - 1] * np.exp(
            -latent_rate * STEP_HOURS
        )

    observation_noise = rng.normal(0.0, 0.0015, size=N_STEPS)
    observation_noise[0] = 0.0
    observed_quality = np.clip(latent_quality + observation_noise, 0.0, 1.0)
    latent_correction = latent_quality - mechanistic_quality
    if np.max(np.abs(latent_correction)) > 0.08:
        raise RuntimeError(
            f"DGP correction exceeds the declared residual bound in {trajectory_id}"
        )

    return pd.DataFrame({
        "dataset_version": DATASET_VERSION,
        "trajectory_id": trajectory_id,
        "trajectory_seed": int(seed),
        "thermal_regime": thermal_regime,
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_h": time_h,
        "tempC": temp,
        "RH": rh,
        "shockG": shock,
        "ambientC": ambient,
        "packaging_index": packaging_index,
        "rh_transient_per_h": rh_transient,
        "k_ref_per_h": k_ref,
        "ea_over_r_kelvin": ea_over_r,
        "mechanistic_quality_fraction": mechanistic_quality,
        "latent_quality_fraction": latent_quality,
        "observed_quality_fraction": observed_quality,
        "latent_correction_fraction": latent_correction,
    })


def generate(output_dir: Path, *, force: bool = False) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "synthetic_spoilage_residual_v1.csv"
    manifest_path = output_dir / "synthetic_spoilage_residual_v1_manifest.json"
    if not force and (dataset_path.exists() or manifest_path.exists()):
        raise FileExistsError("synthetic PINN artifacts already exist; pass --force to replace")

    master = np.random.default_rng(MASTER_SEED)
    seeds = master.integers(1, 2**31 - 1, size=N_TRAJECTORIES, dtype=np.int64)
    regimes = ("cold_chain", "heat_excursion", "oscillatory")
    frames = []
    ids = []
    for index, seed in enumerate(seeds):
        trajectory_id = f"trajectory_{index + 1:03d}"
        ids.append(trajectory_id)
        frames.append(_trajectory(trajectory_id, int(seed), regimes[index % 3]))
    dataset = pd.concat(frames, ignore_index=True)

    # Split at trajectory level.  Test IDs are fixed before any model fitting
    # and are never used for initialization or checkpoint selection.
    split_rng = np.random.default_rng(MASTER_SEED + 1)
    permutation = split_rng.permutation(np.asarray(ids, dtype=str)).tolist()
    splits = {
        "train": permutation[:24],
        "validation": permutation[24:30],
        "test": permutation[30:36],
    }
    split_map = {
        trajectory: split
        for split, trajectories in splits.items()
        for trajectory in trajectories
    }
    dataset.insert(4, "split", dataset["trajectory_id"].map(split_map))
    dataset.to_csv(dataset_path, index=False, float_format="%.12g", lineterminator="\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "target_origin": "independent_synthetic_dgp",
        "synthetic_only": True,
        "external_validation": False,
        "empirical_claims_permitted": False,
        "master_seed": MASTER_SEED,
        "split_seed": MASTER_SEED + 1,
        "n_trajectories": N_TRAJECTORIES,
        "n_rows_per_trajectory": N_STEPS,
        "n_rows": int(len(dataset)),
        "step_hours": STEP_HOURS,
        "trajectory_seeds": {
            trajectory: int(seed) for trajectory, seed in zip(ids, seeds)
        },
        "trajectory_splits": splits,
        "target_column": "observed_quality_fraction",
        "noise_free_diagnostic_column": "latent_quality_fraction",
        "baseline_column": "mechanistic_quality_fraction",
        "known_latent_correction_column": "latent_correction_fraction",
        "correction_bound": 0.08,
        "dgp": {
            "state_equation": "dC_true/dt = -k_base*alpha*exp(u)*C_true",
            "u": "0.44*(packaging_index-0.50)+0.80*handling_shock_G+0.0040*abs_dRH_dt",
            "observation_noise_sd": 0.0015,
            "quality_unit": "dimensionless fraction",
            "independence_statement": (
                "Targets are integrated from the augmented latent DGP before "
                "the residual model is trained; they are not copied from the "
                "mechanistic baseline or generated from its numerical residual."
            ),
        },
        "dataset_file": dataset_path.name,
        "dataset_sha256": sha256(dataset_path),
        "generator_file": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
        "generator_sha256": sha256(Path(__file__)),
        "limitations": [
            "All targets are synthetic; this is internal simulation validation only.",
            "The dataset cannot support a claim of empirical spinach shelf-life accuracy.",
            "Packaging and handling coefficients are declared DGP assumptions, not field estimates.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return dataset_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset, manifest = generate(args.output_dir.resolve(), force=args.force)
    print(f"wrote {dataset}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
