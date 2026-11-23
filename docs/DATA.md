# Data

`data/study` holds the published results of the study. Each table is the subset of one frozen analysis output that the paper, its supplementary material or the response letter shows: values are copied unchanged, and rows and columns the paper does not show are left out. [`data/study/MANIFEST.json`](../data/study/MANIFEST.json) records, for every table, the hash and row count of the source output, the rows kept, the number of columns left out and the hash of the published file, and `scripts/verify_package.py` checks each published file against it. The design, the comparisons and the analysis plan are in [study/PROTOCOL.md](../study/PROTOCOL.md).

All outcomes are modeled under the declared assumptions of a synthetic cold chain; they are not field measurements.

## Modes and settings

| Code in the tables | Mode in the paper |
|---|---|
| `static` | Static |
| `no_context` | No-Context |
| `adjustment_only` | Adjustment only |
| `reservation_only` | Guard only |
| `agribrain_no_peer` | Adjustment and guard |
| `feature_inputs_prior` | Feature inputs |
| `agribrain_eligibility_no_peer` | The bounded adjustment at the guard of Feature inputs, the reference of the implementation check |
| `no_pinn` | Mechanistic model |
| `rule_always_redistribute` | Always redistribute |
| `agribrain` | AGRI-BRAIN |

| Setting | Meaning |
|---|---|
| `main_fail_closed` | Main setting: a diversion whose confirmation is missing is removed |
| `main_fail_open` | Main setting with that diversion left to the destination |
| all others | The 36 sensitivity settings and four acceptance-document checks (names beginning `acceptance_document`); `settings.csv` gives the parameters of each |

## Tables

| File | Contents |
|---|---|
| `seed_endpoints.csv` | One row per setting, seed, scenario and mode: the per-seed values behind every reported mean (ARI, waste, RLE, social performance, emissions indicator, route shares, refused and diverted volume, guard activity and decision latency) |
| `modes.csv` | Means and 95% BCa intervals the paper shows: Table 8, Table S4 and the route shares for Static, No-Context and AGRI-BRAIN; the guard activity of AGRI-BRAIN; the rule that always redistributes; and the refused share of AGRI-BRAIN with diversions left available |
| `principal_comparisons.csv` | AGRI-BRAIN minus No-Context, overall, with the exact Wilcoxon test and the Holm adjustment |
| `comparisons_by_scenario.csv` | The same comparison in each scenario |
| `effects.csv` | The overall ARI effects reported in Section 5 |
| `guard_by_adjustment.csv` | Effect of the context adjustment without a guard and at the guard (Table 7) |
| `sensitivity_settings.csv`, `sensitivity_by_scenario.csv` | Gain over No-Context in each sensitivity setting and document check, overall and by scenario |
| `settings.csv` | Parameters of every published setting |
| `document_cutoff.csv`, `document_identifier.csv`, `document_absent.csv` | Acceptance-document checks: a changed acceptance cutoff, the document under another identifier, and no document |
| `guard_activity.csv` | How often the guard removed a route and for which reason, and how often it changed the route |
| `gain_decomposition.csv` | The overall gain over No-Context split into refusal and merit parts |
| `gain_by_disruption_window.csv` | The gain over No-Context before, during and after the disruption |
| `break_even_share.csv` | Median share of a diversion the destination must accept for the diversion to be worth making |
| `rescoring_contrasts.csv` | The gain over No-Context with the outcomes scored under other weights, normalizers and aggregations |
| `loss_and_recovery.csv`, `loss_and_recovery_contrasts.csv` | Loss of resilience and the number of seeds that recover within the episode |
| `peer_influence.csv` | Decisions on which peer messages changed the route |
| `implementation_check.csv`, `constraint_activity.csv`, `score_influence.csv` | Feature inputs against the bounded adjustment: routing identity, activity of the bounds, and how often the adjustment changes the most probable route |
| `control_contrasts.csv` | Frozen AGRI-BRAIN policies: the live adjustment against no adjustment, a constant adjustment and a shuffled adjustment |
| `scenario_identifier.csv` | The submitted configuration rerun without its scenario terms |

Intervals are 95% BCa intervals of the seed values, computed by `study/inference.py`; a paired difference resamples whole seeds.

## Figure inputs

`data/figure_inputs/agribrain` and `data/figure_inputs/no_context` hold the per-decision records of the 100 seed–scenario evaluations of each mode in the main setting, limited to the fields the figures read: the route and its probabilities, the guard's record with its reason, the MCP term of the adjustment, the disturbance signals and the physical-state features, and every recorded quantity of the decision Fig. 5 explains. The figures and the dashboard's decision difference are computed from them.
