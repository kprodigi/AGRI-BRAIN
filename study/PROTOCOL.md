# Protocol of the revised study

This study was designed during the revision of the manuscript, after the results of the submitted study were known. A first version of the design was reviewed independently before any adaptive mode had been run on evaluation seeds, and the design was changed in response (Section 10). The design, the comparisons and the analysis are fixed in the tagged commit that produced the results, and were fixed before any mode was run on the final evaluation seeds. No positive conclusion is required for a task to pass, and every mode, comparison and setting listed here is reported.

## 1. Why the design changed

In the submitted outcome model, local redistribution is the best-scoring route in every state. A policy that always redistributes therefore scores above every adaptive mode, and the comparison of adaptive modes measures how often each one redistributes. The submitted context adjustment also carried little condition-dependent information: its retrieval term barely varied, and one component followed the role schedule. The revised study corrects both. The outcome model gains two constraints under which the best route depends on conditions, and the context carries the information those constraints depend on.

What the revised study can show:

- whether information held by the destinations improves an adaptive routing policy when it reaches the decision through the protocol;
- how much of that comes from a guard that applies the information as a constraint and how much from the score adjustment, at each of two guard levels;
- whether a condition-dependent adjustment does better than a constant adjustment learned over the same episodes;
- where the system stands against rules given the same acceptance criteria and capacity reports, available under the same conditions. The rules read these values directly rather than through retrieval and the tool channel, and the greedy rule also knows the outcome equations.

What it cannot show: that the learned policy is the best use of the information. Product quality stays exogenous, so routing changes destination capacity and later feasibility, not the quality trajectory. All parameters remain synthetic, and the rule that reserves room for a whole dispatch is an operating policy, not a physical constraint.

## 2. Outcome model

Unchanged: the waste, emissions-indicator, social-performance and ARI equations, their coefficients, the five scenarios and the generation of the latent trajectories. Cold-chain outcomes are identical to the submitted model, so Static is numerically unchanged on a given seed.

Added: destination capacity and acceptance criteria. For a decision step with latent inventory I, demand D and spoilage risk ρ:

| Element | Definition |
|---|---|
| Dispatch | one dispatch opportunity every 15 minutes, of size q = I / 12,000 |
| Capacity of destination a (redistribution, recovery) | B_a, in reference dispatches; starts full at B_a,max |
| Acceptance, redistribution | the product is marketable: ρ ≤ 0.50 |
| Acceptance, recovery | the product is at risk: ρ > 0.10 |
| Accepted share | α = (1 if accepted, else 0) × min(1, B_a / q); cold chain α = 1; a dispatch of size zero is accepted in full |
| Accepted share is scored as | route a, by the unchanged equations |
| Refused share | continues by cold chain after the diverted leg: waste w_raw (1 + d_a / d_CC), emissions E(d_a) + E(d_CC), cold-chain social scores with the combined-distance emissions term |
| Step outcome | share-weighted waste, emissions and social score; ARI form unchanged |
| Capacity update | B_a ← min(B_a,max, B_a − α q + r_a), with r_redistribution = 0.5 D / 15 and r_recovery = 0.25 per step |
| RLE | credit counted on the accepted share only |

Buffers are 8 reference dispatches for redistribution and 4 for recovery. Both acceptance thresholds are the ones already declared for the at-risk and marketability conditions. The refusal penalty adds no fitted coefficient: the refused share continues by cold chain after the diverted leg, and its loss and emissions grow in proportion to the extra distance travelled. That proportionality is a modelling assumption. With unlimited capacity and no acceptance criteria these equations are the submitted model.

A destination takes what it can of a dispatch; the dispatch is not split in advance. Eligibility is binary. Waste and social performance are linear in the accepted share and ARI is the product (1 − waste) × social performance × (1 − ρ), so a partly accepted diversion can still be worth more than the cold chain. At the nominal state (4 °C, reference inventory, no stress) redistribution accepted at share a is worth (1 − w(1.375 − 0.825a))(0.5762 + 0.2454a) against (1 − w) 0.6086 for the cold chain, where w is the raw waste fraction; at w = 0.10 the two are equal at a = 0.193. The analysis reports the distribution of this break-even share over the recorded steps (Section 8).

## 3. Rule policies on the seeds of the submitted study

The capacity values and the treatment of a missing capacity report were examined with rule policies on the 20 seeds of the submitted study, in the simulator. No adaptive mode was run or replayed on those seeds under the revised outcome model before the design was fixed. Because these seeds informed the design, they are not evaluation seeds of the revised study (Section 7). The tables below are from the code of this commit; rule decisions and outcomes were validated step by step (Section 7).

| Rule | ARI, fail-closed | ARI, fail-open | Cold chain / redistribution / recovery, % of steps | Diverted volume refused |
|---|---|---|---|---|
| Always cold chain (Static) | 0.4560 | 0.4560 | 100 / 0 / 0 | n/a |
| Always redistribute | 0.5164 | 0.5164 | 0 / 100 / 0 | 62% |
| Always recovery | 0.3923 | 0.3923 | 0 / 0 / 100 | 92% |
| Risk threshold, no capacity information | 0.4668 | 0.4668 | 62 / 37 / 0.4 | 68% |
| Capacity-informed priority rule | 0.5444 | 0.5406 | 52 / 42 / 6 | 2.8% |
| Model-informed greedy rule | 0.5355 | 0.5316 | 33 / 59 / 8 | 34% |

Route shares and refused volume are those under fail-closed; the refused volume is pooled over the cells of a seed. The two missing-confirmation rules differ only for the two informed rules and only in the cyber outage, where the priority rule scores 0.488 under fail-closed and 0.469 under fail-open, and its refused volume over all scenarios rises from 2.8% to 22%.

Four screening criteria were set after a stricter variant had been examined (redistribution limited to at-risk product, now a sensitivity setting) and before the adopted capacity values were evaluated. They describe whether the benchmark separates informed from uninformed routing; they are diagnostics, not a validation of the model.

| Criterion | Fail-closed | Fail-open |
|---|---|---|
| V1. The priority rule exceeds every fixed route in every scenario by at least 0.01 ARI | smallest margin 0.019; met | smallest margin 0.006, in the cyber outage; not met |
| V2. Under the priority rule each route is used on at least 5% of steps | smallest share 5.9%; met | 5.9%; met |
| V3. Under always-redistribute, 20% to 80% of the diverted volume is refused in every scenario | 38% to 78%; met | 38% to 78%; met |
| V4. The priority rule exceeds the best uninformed rule by at least 0.01 ARI in at least 18 of 20 seeds | +0.028, 20 of 20; met | +0.024, 20 of 20; met |

The criteria are recomputed from the rule references in every setting of the study.

The same rules can be replayed on the recorded environments with the capacity of the destinations following each rule's own decisions. The replay reproduces the simulator's routes exactly for the risk-threshold, priority and greedy rules. It shows how the value of a diversion depends on the share of the dispatch that the destination accepts:

| Rule | ARI, fail-closed | Full reservation minus rule, fail-closed | ARI, fail-open |
|---|---|---|---|
| Priority rule that diverts when the destination would accept any share | 0.5242 | +0.0202 (0.0199 to 0.0206) | 0.5203 |
| at least 20% of the dispatch | 0.5323 | +0.0121 (0.0118 to 0.0123) | 0.5285 |
| at least 40% | 0.5385 | +0.0058 (0.0057 to 0.0060) | 0.5347 |
| at least 60% | 0.5421 | +0.0023 (0.0022 to 0.0023) | 0.5383 |
| at least 80% | 0.5438 | +0.0005 (0.0005 to 0.0006) | 0.5400 |
| 100%: the capacity-informed priority rule | 0.5444 | | 0.5406 |
| Greedy rule, partial acceptance allowed | 0.5355 | +0.0089 (0.0087 to 0.0091) | 0.5316 |
| Always redistribute | 0.5164 | +0.0280 (0.0275 to 0.0285) | 0.5164 |

Differences are paired by seed, with 95% BCa intervals, and are positive in 20 of 20 seeds in every row. Over the recorded steps the break-even accepted share has median 0.19 for redistribution (5th to 95th percentile 0.16 to 0.24) and 0.46 for recovery (0.39 to 0.49). A rule that diverts for any share above break-even gains at that step and still scores below full reservation over the episode: the step value is convex in the accepted share, being a product of two increasing linear terms, so a given amount of capacity is worth more on a dispatch it can take whole than spread over partly accepted dispatches. This covers the threshold rules and the greedy rule that were tested. It does not show that a policy free to accept partly must do worse, because such a policy can also wait.

How often each acceptance criterion is reached matters for what the study can show. On these seeds the latent risk is at or below the at-risk threshold on 63% of steps, where recovery does not accept the product. It exceeds the marketability cutoff on 0.35% of steps (100 of 28,800, all in one heatwave cell), so redistribution is almost always eligible. In the main setting the criteria of the acceptance document therefore act mostly through the recovery threshold of the guard; the component ψ3 (Section 4) is active on very few decisions, and a score adjustment that uses retrieval alone is close to none. Risk lies between 0.40 and 0.50 on 6.7% of heatwave steps and on none in the other scenarios, so the settings with a cutoff of 0.40, applied or stated in the document, bind in the heatwave only. The analysis reports, for every setting with decision records, how often each destination accepts the product and how often each context component is active.

Three choices rest on these seeds and are disclosed here.

1. *Capacity values.* They were kept after the screening criteria were met; no value was chosen by how an adaptive mode then scored.
2. *Reserving room for the whole dispatch.* Among the rules tested, the priority rule that diverts only when the destination confirms room for the whole dispatch has the highest mean. The rules that wait for 60% or 80% differ from it by less than the practical threshold, so 100% is not shown to be an optimal value. It is kept because it needs no tuning value, as the reservation guard of the full system, and it is compared with a guard that does not reserve (Section 4). Reserved shares of 50% and 80% are sensitivity settings.
3. *A capacity report that cannot be read.* The tool channel is unavailable from hour 24 of the cyber outage. Whether a diversion is then still attempted changes the rule results in that scenario, as the two columns show. The first version of the design blocked the diversion because the screening criteria were otherwise not met in that scenario. Both behaviours are now main conditions of the study, each run with the full set of modes (Section 6).

## 4. Context, guard and missing information

```
physical state φ(s) ──► base route scores ─────────────┐
tools, retrieval ──► context ψ ──► bounded adjustment ──┼─► route scores ──► guard ──► probabilities ──► route
peer messages ──► bounded bias ────────────────────────┘                (eligibility, capacity)
```

| | Component | Definition | Source |
|---|---|---|---|
| ψ0 | Operating-envelope severity | unchanged | envelope tool |
| ψ1 | Forecast urgency | unchanged | forecast tool |
| ψ2 | Redistribution shortfall | 1 − min(1, B_redistribution / q), from the outlet's capacity report; 0 without a report | capacity tool |
| ψ3 | Redistribution ineligibility | 1 when the estimated risk is outside the redistribution acceptance band stated in the retrieved acceptance document; needs the retrieval gate | retrieval |
| ψ4 | Recovery shortfall | 1 − min(1, B_recovery / q), from the facility's capacity report; 0 without a report | capacity tool |

The weight columns of ψ0, ψ1 and ψ4 keep their submitted values. The two redistribution columns mirror the recovery column with the two routes exchanged (cold chain +0.12, redistribution −0.20, recovery +0.08), so no new magnitude is introduced. The clip at ±1, the sign constraints, the change bounds and the learning rule are unchanged. No weight was enlarged to make the adjustment more influential.

The guard replaces the probability-gap override of the submitted design. Before sampling, it removes routes from the choice set and renormalizes the probabilities over the remaining routes; the cold chain is always available. Two guard levels are compared.

| Guard level | Removes a diversion when |
|---|---|
| Eligibility | the estimated risk is outside the acceptance band stated in the retrieved acceptance document, or the destination reports no room at all for a dispatch of positive size |
| Reservation | as above, or the destination's latest report shows room for less than the whole dispatch |

Under the eligibility guard a partly accepted diversion remains possible and capacity enters the decision through the score adjustment. Under the reservation guard the capacity components ψ2 and ψ4 are active only where the guard has already removed the route, so they have little opportunity to contribute; the eligibility level is therefore the more informative test of the score adjustment. It does not guarantee a positive effect.

Missing information is of two kinds, and they are kept apart.

- *Withheld by the definition of a mode.* A mode that never requests a piece of information has the corresponding check switched off. Without tools there is no capacity check and the guard applies the document's criteria only. Without retrieval there is no eligibility check; the capacity check remains, and the destination still applies its own criteria.
- *Requested but unavailable.* The capacity report is unavailable while the tool channel is down, and the criteria are unavailable when the acceptance document is not retrieved. The missing-confirmation rule decides: **fail-closed** removes the diversion whose confirmation is missing; **fail-open** leaves it, and the destination enforces. The eligibility guard needs no capacity confirmation, so a missing report does not change its choice set; its capacity components are then zero under either rule.

One acceptance document is added to the corpus, stating both destinations' criteria as value-unit lines. Each role retrieves it at the start of its stage through the existing gate, and the criteria are read from the content of the top-ranked passage, whatever its identifier. Retrieval no longer adds a standing routing bias. The scenario identifier and its search terms are removed from every query: the scenario slot of the query template is fixed to its baseline value in all scenarios. The recovery agent's capacity message reports remaining capacity.

The decision record keeps the pre-guard probabilities, the removed routes, the reason for each, and whether the criteria and the capacity report were available, unavailable or withheld. Learning uses the post-guard probabilities.

## 5. Modes

| Mode | Score adjustment | Guard | Peer messages | Purpose |
|---|---|---|---|---|
| Static (`static`) | no | none | no | fixed cold chain |
| No-Context (`no_context`) | no | none | no | adaptive policy without protocol information |
| Adjustment only (`adjustment_only`) | yes | none | no | score adjustment alone |
| Eligibility guard only (`eligibility_only`) | no | eligibility | no | guard level alone |
| Adjustment + eligibility guard (`agribrain_eligibility_no_peer`) | yes | eligibility | no | both entry points, partial acceptance possible |
| Reservation guard only (`reservation_only`) | no | reservation | no | guard level alone |
| Adjustment + reservation guard (`agribrain_no_peer`) | yes | reservation | no | both entry points |
| AGRI-BRAIN (`agribrain`) | yes | reservation | yes | full system |
| AGRI-BRAIN with the eligibility guard (`agribrain_eligibility`) | yes | eligibility | yes | full system without reservation |
| Without guard (`agribrain_no_guard`) | yes | none | yes | full system minus the guard |
| Without tools (`pirag_only`) | retrieval component only, which is rarely active (Section 3) | document criteria only; capacity withheld | yes | joint information-and-enforcement comparison |
| Without retrieval (`mcp_only`) | tool components only | capacity reservation only; criteria withheld | yes | joint information-and-enforcement comparison |
| Fixed context weights (`agribrain_fixed_context`) | yes, not adapted | reservation | yes | adaptation of the context weights |
| Sign-unconstrained (`agribrain_sign_unconstrained`) | yes, signs free | reservation | yes | effect of the sign constraint |
| Feature inputs, prior-initialized (`feature_inputs_prior`) | ψ enters an unconstrained linear term that starts at the prescribed weights | eligibility | no | same information, conventional interface |
| Feature inputs, zero-initialized (`feature_inputs_zero`) | same, starting at zero | eligibility | no | value of the prescribed weights |
| Learned constant adjustment (`constant_adjustment`) | a constant vector adapted over the same episodes | eligibility | no | sustained adjustment without condition dependence |
| Mechanistic-only estimator (`no_pinn`) | yes | reservation | yes | effect of the spoilage estimator on decisions |

No-Context, adjustment only, the two guard-only modes and the two adjustment + guard modes form a 3 × 2 design of guard level by score adjustment, without peer messages. The two full-system modes differ only in the guard level. The feature-input modes and the learned constant differ from adjustment + eligibility guard only in the interface (no clip, no sign constraint, no change bound) or in the input (a constant in place of ψ); they are placed at the eligibility level, where the capacity components are not redundant with the guard.

Removing a channel removes information and the check that depends on it. The modes without tools and without retrieval are therefore reported as joint information-and-enforcement comparisons. Information and enforcement are separated by the 3 × 2 design, not by these two modes.

With the change bounds in force, a context weight can reach zero only if its initial magnitude is at most 0.10. The sign-unconstrained mode can therefore differ from the full system only through the three weights of magnitude 0.08.

Rule references, all given the information the guard is given (the criteria stated in the acceptance document, the capacity reports while the tool channel is up, the dispatch size implied by the dispatcher's own inventory observation, and the missing-confirmation rule):

| Rule | Decision |
|---|---|
| Always redistribute (`rule_always_redistribute`) | redistribution at every step |
| Always recovery (`rule_always_recovery`) | recovery at every step |
| Risk threshold (`rule_risk_threshold`) | cold chain while the estimated risk is at or below the at-risk threshold, redistribution up to the marketability cutoff, recovery above it; no capacity information |
| Capacity-informed priority rule (`rule_capacity_informed`) | redistribution when the reservation test allows it, otherwise recovery when the test allows it, otherwise cold chain |
| Model-informed greedy rule (`rule_greedy_informed`) | the route with the highest step ARI predicted by the outcome equations on the observed state, expecting each destination to accept min(1, reported room / dispatch size); partial acceptance allowed |

The priority rule is a baseline, not an upper bound: it is one fixed ordering of the routes. The greedy rule knows the outcome equations and the nominal spoilage kinetics, which no learned mode is given.

## 6. Settings

| Family | Settings | Seeds | Modes per task | Purpose |
|---|---|---|---|---|
| Primary | 2 | final | the 18 modes and 5 rule references | the main setting under fail-closed and under fail-open: principal comparisons and estimated effects |
| Submitted | 2 | submitted study | Static, No-Context, AGRI-BRAIN, always redistribute, always recovery; and AGRI-BRAIN without scenario terms | the submitted configuration with its fixed-route references; influence of the scenario identifier |
| Diagnostic | 2 | submitted study | as primary; and the sensitivity modes without destination constraints | the main fail-closed setting on the seeds that informed the design, labelled diagnostic; identity of the revised model without destination constraints with the submitted model |
| Weights | 22 | final | Static, No-Context, AGRI-BRAIN, AGRI-BRAIN with the eligibility guard, 5 rule references | nine weight groups at 0.8 and 1.2 times nominal, and four joint profiles |
| Structural | 31 | final | same | one-factor changes to the structural assumptions |
| Content | 4 | final | same | changes to the acceptance document |

Weight groups: base-policy prior, tool-context prior, retrieval-context prior, the four social-score weights (changed before normalization), and the two reward penalties. Joint profiles A and B use opposite alternating signs across the nine groups, at 10% and 20%.

Structural settings:

| Assumption | Settings |
|---|---|
| Capacity report age | reports one hour old |
| Redistribution capacity | × 0.25, × 0.5, × 2, × 4 |
| Recovery capacity | × 0.5, × 2 |
| Buffers | × 0.5, × 2 |
| At-risk threshold | 0.05, 0.15 |
| Marketability cutoff | 0.40, 0.60 |
| Refusal penalty | × 0.5, × 2, × 4; the refused share is lost |
| Share of the dispatch reserved | 50%, 80% |
| Waste-saving coefficients | × 0.8, × 1.2, equalized |
| Route social scores | premium halved; all routes equal |
| Route distances | redistribution 80 km; all routes 120 km |
| Context-weight signs | condition columns reversed |
| Route merit order | redistribution and recovery coefficients exchanged |
| Learning objective | reward replaced by step performance |
| Redistribution limited to at-risk product | strict variant |
| Unlimited capacity, no acceptance criteria | the submitted outcome model |

The settings for the refusal penalty, for redistribution capacity and for the reserved share form three graded series around the main fail-closed setting. Where the refused share is lost, it returns by cold chain as in the main setting but none of it is sold: its waste fraction is 1.

Content settings:

| Setting | What changes |
|---|---|
| Stated cutoff 0.40 | the document states a marketability cutoff of 0.40 while the destination applies 0.50 |
| Other identifier | the same document is stored under another identifier and source label |
| Document absent, fail-closed | the corpus holds no acceptance document |
| Document absent, fail-open | the same, under the other missing-confirmation rule |

In these settings only the document changes. The validator checks at every step that the criteria used by the guard are the values stated in the retrieved document, so the first setting shows whether the guard follows the content of the document or a value fixed in the code, and the second whether it follows the content or the identifier.

Totals: **63 settings; 6,300 tasks; 128,100 executed episodes; 68,400 adaptation episodes; 59,700 evaluation episodes; 17,193,600 evaluation routing choices.** The frozen-policy controls add 1,600 evaluation episodes. The independent repeat used for quality assurance adds 85 episodes and is excluded from analysis.

## 7. Seeds and execution

| Seed set | Seeds | Use |
|---|---|---|
| Final evaluation | 5001 to 5020 | primary, weights, structural and content settings; not run before the design was fixed in the tagged commit, and refused by the harness in a development runtime |
| Submitted study | the 20 seeds of the submitted study | the submitted configuration and the diagnostic settings; rule policies were run on these seeds while the outcome model was designed |
| Development | 3 and 11 | software checks and rehearsal of the whole chain |

Seeds and the five scenarios are fully crossed with the settings. A task is one setting × seed × scenario cell and executes every mode of its setting on the same latent environment and the same recorded random draws. Static and the rule references have one evaluation episode; each adaptive mode has three adaptation episodes followed by one frozen evaluation. Nothing is reused between settings. Each planned cell contributes one accepted result: a task that has one is verified and skipped, not executed again. Two pilot cells are executed a second time for quality assurance; that repeat is kept apart and excluded from analysis. A failed attempt may be followed by another; attempts that did not complete are kept and never enter the analysis.

Every task is checked before it is accepted:

- pairing, frozen evaluation, finite metrics, valid probabilities, and complete episode evidence;
- the ledger validator (`study/validate.py`) re-derives, from the recorded quantities alone: that the capacity parameters in the ledger are those of the setting; the step outcomes including the refused share; the capacity of both destinations and each accepted share; that a capacity report reaching the decision side states the reconstructed capacity and is present exactly when the mode requests it and the tool channel is up; the routes the guard removed and their reasons, at its guard level and with the information it held; that the executed route was never a removed one; that the guard's criteria are the values stated in the retrieved document; the three context components read from the capacity report and the acceptance document; the route of every rule reference, from what the rule may read; that no scenario term entered a query where the setting removes it; and that the submitted override never fired under the revised design;
- the endpoints stored with the result are reconciled with the same ledger: every quantity that the analysis reads and that the decision records determine is rebuilt from the records and must equal the stored value: an episode quantity within 1e-12, taken relative to the rebuilt value where that exceeds one in magnitude; each value of a step trace and each context weight within 1e-12; the quantities of the ledger header and the count of sign reversals exactly. These are the six episode metrics (ARI, modeled waste, social performance, the emissions indicator, RLE and the stability score), the diverted and refused volume and their ratio, the two guard fractions, the mean and 95th-percentile decision latency, every step trace (outcomes, routes, probabilities, accepted shares, capacities, the environment and the observations), the quantities carried in the ledger header, and the context weights of the frozen policy;
- in the submitted setting, routing must be identical to the stored results of the submitted study and metrics, probabilities and trajectories must agree within 1e-10.

The capacity, acceptance, refusal, guard and rule arithmetic of the validator is written independently of the simulator. The unchanged base equations are imported from the model package; unit tests check them, and the validator's outcome model, against values worked by hand.

Fifteen pilot cells run first, two of them a second time independently. The pilot check verifies each of them from its retained ledgers and requires: identical results on repetition (1e-12); reproduction of the submitted results; identical Static outcomes under both outcome models; identical No-Context and fixed-route outcomes without destination constraints and under the submitted model; under fail-closed, no diversion from hour 24 of the cyber outage by any mode whose reservation guard requested the report or by either informed rule, and under fail-open some diversion; guard removals that follow the cutoff stated in the document; identical results when the document is stored under another identifier; and, without the document under fail-closed, no diversion by a guarded mode or by a rule that needs the criteria. The bulk of the study waits for this check. A resumed study recovers any missing pilot cell or repeat and runs the check again before any other task is released, and the worker script refuses a task outside the pilot without a passed check of its plan.

Primary and submitted tasks retain all adaptation and evaluation ledgers and complete episode archives. Diagnostic and sensitivity tasks retain their evaluation ledgers, compressed, with endpoints, traces, validation counts and the hashes of the rest of their evidence. The reconciliation runs when a task is accepted and again in every full verification, which validates the retained ledgers of the task once more: the pilot check verifies the pilot cells in full and the analysis every task, whatever the task retains. The lighter check made when a completed task is skipped on resumption, when missing tasks are listed and when the controls load a primary task confirms the recorded hash of every retained file, the identity of the evidence and, where the episode archives are retained, that the stored metrics and traces are those of the archives; it does not repeat the reconciliation. A task of the study that carries no record of the reconciliation is refused. Results are written outside the repository. Failed attempts are kept; only validated tasks enter the analysis, and the analysis refuses an incomplete study. The study requires a clean checkout, Python 3.11 and the exact dependency lock.

## 8. Comparisons and analysis

The unit is the seed (n = 20). Overall values are the equal-scenario mean of each seed. Intervals are 95% BCa intervals from 10,000 resamples of whole seeds, drawn with a generator seeded from a hash of the cell key, as in the submitted study. They are pointwise, not simultaneous.

**Principal comparisons.** Five paired differences in overall ARI, each under fail-closed and under fail-open: ten tests.

| | Comparison | Modes | Question |
|---|---|---|---|
| 1 | AGRI-BRAIN minus No-Context | `agribrain` − `no_context` | does protocol information improve the adaptive policy |
| 2 | Reservation guard minus eligibility guard, both with adjustment | `agribrain` − `agribrain_eligibility` | does reserving room for the whole dispatch do better than allowing partial acceptance |
| 3 | Adjustment + eligibility guard minus eligibility guard only | `agribrain_eligibility_no_peer` − `eligibility_only` | does the score adjustment add to the guard |
| 4 | Adjustment + eligibility guard minus learned constant + eligibility guard | `agribrain_eligibility_no_peer` − `constant_adjustment` | does condition dependence add to a sustained adjustment |
| 5 | AGRI-BRAIN minus the capacity-informed priority rule | `agribrain` − `rule_capacity_informed` | does the learned system do better than a fixed rule given the same criteria and capacity reports |

Each is tested with a two-sided exact Wilcoxon signed-rank test, Holm-adjusted across the ten tests, and is called significant when the adjusted p is below 0.05. No direction is assumed. No other test is made in this study.

The size of a difference is read from its interval against the practical threshold of 0.005 ARI, the margin of the original protocol: *larger than the threshold* when the interval lies entirely beyond +0.005 or −0.005; *smaller than the threshold* when it lies entirely within ±0.005; *unresolved* when it contains a threshold. An unresolved comparison is reported as unresolved, not as absent.

With 20 seeds the smallest attainable two-sided p is 1.9 × 10⁻⁶, and 1.9 × 10⁻⁵ after adjustment. A small p says that a difference has a consistent sign across seeds; it says nothing about its size or about how precisely it is estimated. Every comparison is therefore reported with its interval, its half-width and the seed-level standard deviation of the paired difference.

**Reported with intervals, without a test,** for each main setting and for the diagnostic setting:

- the principal comparisons in each scenario, and comparison 2 without peer messages (`agribrain_no_peer` − `agribrain_eligibility_no_peer`);
- ARI by guard level without and with the score adjustment (the 3 × 2 design), the adjustment effect at each level, the guard effects, and the two interactions;
- the peer effect at each guard level;
- the full system minus each of: without guard, without tools, without retrieval (the last two as joint information-and-enforcement comparisons), fixed context weights, sign-unconstrained, mechanistic-only estimator;
- the value of the prescribed weights (prior- minus zero-initialized feature inputs);
- the full system, with each guard level, minus each rule reference; the priority rule minus the greedy rule and minus always redistribute;
- the influence of the scenario identifier in the submitted study, and the submitted configuration against its fixed-route references.

**Implementation check.** Adjustment + eligibility guard and prior-initialized feature inputs compute the same adjustment while the clip, the sign constraint and the change bound are inactive. The analysis reports their difference, in how many cells their routing is identical, and how often each constraint was active: decisions with the clip active; context weights held at zero by the sign constraint; context weights at their change bound; sign reversals in the unconstrained mode. Agreement of the two modes is a check of the implementation, not evidence about the interface.

**Secondary endpoints:** share of the diverted volume refused, modeled waste, the route-distance emissions indicator per routing opportunity, social performance, RLE, route shares, decision latency, and guard activity (share of decisions with a route removed, by route and reason; share where the pre-guard sample differed; availability of the capacity report and of the acceptance document).

**Computed from the decision records,** without further runs of an adaptive mode:

- *Gain decomposition.* For each step, let F(a) be the ARI of route a if the destination took the whole dispatch. The paired difference of two modes is split exactly into route merit, F(a_mode) − F(a_reference), and refusal avoided, [ARI_mode − F(a_mode)] − [ARI_reference − F(a_reference)]. The same difference is also tabulated by route switch.
- *Gain before, during and after the disruption.* The paired gain per step in each window of the heatwave (hours 24 to 48 after the onset offset), overproduction (12 to 60) and the cyber outage (from hour 24, or from the onset of its environmental disturbance if that is earlier).
- *Context components.* The share of the applied adjustment carried by each component; how far the physical state predicts each component (the R² of a linear fit, and the out-of-sample R² of a 10-nearest-neighbour prediction that leaves one seed out); and, for each capacity and eligibility component, how often it is active on a route the guard has already removed.
- *Influence of the adjustment on the scores.* Over all decisions of a mode: the distribution of the gap between the two highest base scores, of the adjustment, and of the change in routing probabilities; how often the adjustment changes the highest-scoring route, among all routes and among the routes the guard leaves.
- *Peer messages.* For each guard level, the decisions on which the modes with and without peer messages have different probabilities, the sum of those differences (the expected number of changed decisions), and the number of decisions with a different route.
- *Index parameters.* The executed decisions re-scored under equal social weights, each social weight doubled and renormalized, the emissions normalizer halved and doubled, and additive and geometric aggregation of the three factors; each factor is also reported separately. The routing decisions are not re-made.
- *Loss and recovery.* For each mode, seed and disturbed scenario: loss is the step mean of the baseline-scenario ARI minus the scenario ARI; the retained fraction is the ratio of the episode means; the largest two-hour shortfall is one minus the smallest ratio of two-hour sums. Recovery time is defined for the heatwave and overproduction scenarios, whose disturbance ends within the episode (a decaying temperature tail follows): the hours from that end until performance over two hours stays at or above 95% of the baseline-scenario performance for two consecutive hours. A seed that does not reach this within the episode is counted as not recovered.
- *Value of partial acceptance.* The rule replays of Section 3 on the recorded environments of each main setting and of the diagnostic setting: a priority rule that diverts when the destination would accept at least a given share of the dispatch, the greedy rule and always redistribute, each paired with full reservation. How often each destination accepts the product, and the distribution of the break-even accepted share over those steps.

**Frozen-policy controls** (`study/controls.py`). For each primary cell and for both guard levels of the full system, the evaluation episode is executed again from its frozen checkpoint with the adjustment as computed, replaced by the mean adjustment of the adaptation episodes, permuted within the episode, or removed. The guard, the peer messages, the weights and the environment are unchanged. The first execution must reproduce the stored evaluation. Contrasts are reported with 95% intervals. These are executions of one frozen policy, not separately adapted modes.

**Sensitivity.** For every weight, structural and content setting: four paired differences per scenario and overall (comparisons 1, 2 and 5, and the full system with the eligibility guard minus No-Context), each with its interval, its reading against the threshold and its change from the main setting with the same missing-confirmation rule on the same seeds; the refused shares; and the screening criteria of Section 3 as diagnostics. The three graded series are also reported together. Counts of settings by reading are descriptive, not a corrected global test.

**Diagnostic family.** The main fail-closed setting on the seeds of the submitted study is reported with the same tables and labelled diagnostic; it enters no test. Without destination constraints, Static, No-Context and the fixed routes must take the routes they take under the submitted model.

## 9. Reported whatever the outcome

- All modes, rule references, principal comparisons, estimated effects and sensitivity settings listed above, including any setting where a gain disappears or reverses.
- The submitted configuration with its fixed-route references, as the record of why the model was corrected.
- Whether a rule reference scores above AGRI-BRAIN.
- Which entry point carries the gain: the guard or the score adjustment; and whether a learned constant matches the condition-dependent adjustment.
- Whether peer messages change any decision.
- The robustness statement is limited to what holds.

## 10. Disclosures

- *Seeds of the submitted study.* Rule policies were run on them while the outcome model was designed (Section 3). They are used only for the submitted configuration and for results labelled diagnostic.
- *Development runs.* Adaptive modes were run under earlier versions of this design on seeds 3 and 11 only. What was seen there shaped the design: the bounded pathway and the feature-input mode agreed in every cell, so their comparison became an implementation check; under the reservation guard the score adjustment added little and its capacity components were redundant with the guard, so the eligibility guard level was added and comparisons 3 and 4 are made there; a rule given the same information scored above the learned system, so comparison 5 is principal and two-sided. Two seeds do not support any estimate, and none is reported from them.
- *Independent review.* A review of the first version led to: the two guard levels; both missing-confirmation rules as main conditions; rule references given the information the guard is given, and the greedy rule; reading the acceptance criteria from the content of the retrieved document; the validator checks on the capacity parameters, the capacity reports and the rules; retained evaluation ledgers for every task; the final evaluation seeds; and the single family of ten tests.
- *Settings added after development runs.* The graded series for the refusal penalty and for redistribution capacity were extended; the reserved shares of 50% and 80%, the document under another identifier and the absent document were added.
- *The first version of the design* specified one-sided tests of AGRI-BRAIN against No-Context in each scenario and an equivalence test of the two interfaces. They are replaced by the comparisons of Section 8.
- *Correction after the design was fixed.* A review of the tagged package (`revised-study`) found that the script resuming an interrupted study released the remaining tasks without the pilot check and did not recover the pilot's repeat, and that the collection could be read as finished before its archives were written. The scripts were corrected, with tests, before any task of the study was run, and a failed submission now stops the chain. Three statements of Section 7 were made exact at the same time: what the endpoint reconciliation tolerates, which verification repeats it, and that each planned cell contributes one accepted result. A second review of the corrected package (`revised-study-launch-correction`) found that a resumption went ahead when the queue could not be read, and that a failed reanalysis left the completion record of an earlier analysis in place, so that the stopped stage was misreported. Both were corrected, with tests (`revised-study-launch-correction-2`). The design, the settings, the seeds, the comparisons and the code that produces, validates and analyses the results are those of `revised-study`.

## 11. Running the study

On a Slurm cluster, from a clean checkout of the tagged commit with Python 3.11 available:

```bash
bash study/setup.sh
```

```bash
bash study/submit.sh --partition=<partition>
```

The chain is pilot cells and their repeat → pilot check → study blocks → controls → collection. The analysis is complete when `COMPLETE.json` exists in the results folder. The collection is complete when `<results>_COLLECTED.sha256` exists beside the results folder: it is written last, after both archives and their checksums, so neither the end of the jobs nor `COMPLETE.json` is the signal to download.

If the jobs end before that, find the stage that stopped: a pilot cell or its repeat, the pilot check, a task, a control, the analysis or an archive. `bash study/status.sh <results>` names it from the results folder and points to its logs in `<results>/logs`; it reads only. A pilot check that fails states the requirement that was not met, and resuming does not change that. Otherwise, once no job of the study is left in the queue, source the `run.env` of the results folder and run `study/retry.sh` from a compute allocation. It refuses to resume if the queue cannot be read or still holds a job of the study. It recovers the stages in their order, runs the pilot check again, and releases nothing outside the pilot unless that check passes. A collection withdraws the pilot pass, the completion record of the analysis and the record of an earlier collection before it starts.

The analysis writes its tables to `analysis/` in the results folder. `summary.json` holds the principal comparisons, the implementation check, the screening criteria and the sensitivity summary; the CSV files hold every row.

For development on another Python version, pass `--development-runtime <empty folder>` and `prepare --development-seeds`. Development results are marked as such and cannot be mixed with the study; a development runtime refuses the final evaluation seeds.
