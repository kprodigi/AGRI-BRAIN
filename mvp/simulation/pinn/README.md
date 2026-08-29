# Synthetic spoilage residual evidence

This directory contains a fully synthetic, versioned benchmark used to train
the frozen physics-informed residual in `src.models.pinn_residual`. It does not
contain observed spinach shelf-life labels and cannot support an empirical
validation claim.

`generate_synthetic_spoilage_data.py` creates 36 independent trajectories from
a declared latent data-generating process. Trajectories—not rows—are assigned
to fixed train (24), validation (6), and untouched test (6) groups.
`train_spoilage_pinn.py` fits three deterministic initializations with an exact
analytic Jacobian for every loss term, selects only by validation RMSE, then
reports the test split once. The generated dataset, manifests, frozen
checkpoint, training history, seeds, splits, hyperparameters, and metrics live
under `artifacts/` and are cryptographically bound by SHA-256.

The raw network output is retained for diagnostics and is not claimed to be
physically valid by itself.  Runtime deployment clips
`C_mech + delta_C` to `[0,1]` and then takes the cumulative minimum within each
trajectory.  The checkpoint manifest reports raw and deployed holdout metrics
separately, including constraint-violation counts.

Regeneration is an explicit source-changing operation and is not part of an
HPC episode. The confirmatory simulator only loads the frozen checkpoint.
