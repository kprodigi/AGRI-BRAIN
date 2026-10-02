# Data

All numerical results belong to the Static / No-Context / AGRI-BRAIN study. Results from earlier configurations are not included, and `static_only_evidence.json.gz` holds historical Static evaluations only.

## Primary comparison (`data/primary/`)

| File | Contents |
|---|---|
| `three_mode_endpoints.csv` | 300 seed–scenario–mode rows: ARI, waste fraction, RLE, social-performance proxy, modeled carbon and equity |
| `paired_seed_ARI.csv` | 100 paired AGRI-BRAIN/No-Context comparisons |
| `overall_outcomes.json` | Reported aggregate estimates and uncertainty, with the per-seed values they were computed from |
| `plotted_estimates.csv` | Estimates and intervals underlying the primary result plots. Its `figure` column uses legacy identifiers, not current figure labels |
| `time_series_estimates.csv` | Scenario time-series estimates |
| `routing_time_analysis.json` | Executed routing analysis |
| `pricing_analysis.json` | Pricing-scenario analysis |
| `selected_decision_examples.json` | Four retained individual decision records |
| `audit_summary.json` | Ledger reconstruction and explanation checks |
| `directional_disagreements.csv` | Disagreeing dominant-factor summaries, retained for transparency |
| `static_compatibility.json` | Checks that the retained Static evaluations match the current environmental identities and outcome equations |
| `static_only_evidence.json.gz` | Historical Static evaluations, by seed |
| `source_manifest.json` | Source identity of the primary run |
| `spoilage_pinn_v1_manifest.json` | Manifest of the frozen spoilage predictor |

### Validation (`data/primary/validation_20260922/`)

| File | Contents |
|---|---|
| `protocol_validation.json` | Interface-invariance tests across the tested transport paths, fault checks and the temperature sweep |
| `control_summary.json`, `control_integrity.json` | Summary and integrity checks of the frozen-policy controls |
| `control_endpoints.csv`, `frozen_control_decisions.csv.gz` | Per-seed control endpoints and per-step control decisions |
| `role_activity.json` | Per-role activity summary |
| `validation_provenance.json` | Date, environment and source identity of the validation run |

## Weight sensitivity (`data/sensitivity/`)

| File | Contents |
|---|---|
| `seed_endpoints.csv` | 6,900 evaluation endpoints across 23 settings |
| `paired_ari_sensitivity.csv` | Scenario and overall paired gains and changes from nominal, with pointwise 95% BCa intervals |
| `effective_parameters.csv` | Actual parameters after normalization |
| `route_changes.csv` | Routing differences under perturbations |
| `summary.json` | Study counts and headline summary |

## Conventions

ARI and proxy scores are dimensionless; waste is a fraction, not a percent. See the outcome contracts and the source for precise definitions. Carbon is the modeled episode transport and cooling emissions indicator in kg CO2-equivalent. Do not infer calibrated real-world performance from these synthetic endpoints.

Overall primary comparisons average the five scenarios within each seed before summarizing the 20 seed means. Confidence intervals in the supplied primary summaries are 95% BCa intervals; sensitivity intervals are pointwise. A 0.01 ARI reference is descriptive, not a prespecified hypothesis. The decision-explanation figure illustrates single retained decisions, not a population estimate that would need error bars.

## Figures

[`figures/README.md`](../figures/README.md) lists every figure with its caption and data. The images are the final figures of the study, and `figures/figure_index.json` carries the same information in machine-readable form. The architecture diagram is `docs/images/architecture.jpg`. The ARI performance and weight sensitivity figures can be regenerated from the supplied data with `analysis/make_figures.py`; the others are supplied as final images with their data and are not regenerated here.

## Raw records

Compact evidence supports inspection of the reported results without downloading the full ledgers. The primary adaptation and evaluation archives and the complete sensitivity ledger archive (about 17 GB) are not hosted in this repository and are available on request from the corresponding author; their checksums are in `provenance/raw_archives.json`. The compact sensitivity analysis is distributed as `data/sensitivity/`. No DOI is claimed for any of these assets.
