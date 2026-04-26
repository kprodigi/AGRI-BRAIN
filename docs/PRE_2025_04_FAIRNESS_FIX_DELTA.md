# Pre-2025-04 fairness-fix delta record

## What this file is

The 2025-04 commit that fixed the multi-episode-learning fairness asymmetry
left no record of what the previous numbers were. Reviewer 2 correctly
flagged that as an undocumented unblinding edit. This file is the public
record of what changed and why; it is committed so anyone diffing the
repository can see the size of the correction.

## What was wrong

`generate_results._MULTI_EPISODE_MODES` previously assigned learning
budgets asymmetrically:

```
agribrain              : 1 iteration per scenario (frozen prior)
agribrain_cold_start   : 4 iterations per scenario (20-episode REINFORCE)
agribrain_pert_*       : 4 iterations per scenario (20-episode REINFORCE)
```

This meant the §4.7 cold-start ablation compared "20 episodes of
REINFORCE from zero init" against "1 episode of frozen hand-calibrated
priors", which is not a fair test of "can context weights be discovered
from scratch?". It is a test of "does 20× the gradient signal beat 1×",
which has a known answer.

Under that asymmetry the previous HPC run produced a result where
`agribrain_cold_start` *outperformed* `agribrain` on ARI by a small
but visible margin. The published manuscript text described this as
evidence that the hand-calibrated priors were not load-bearing.

## What changed

The 2025-04 fix put every agribrain-family mode on the same
4-iteration / 20-episode learning budget:

```
agribrain              : 4
agribrain_cold_start   : 4
agribrain_pert_10/25/50: 4
agribrain_pert_*_static: 1 (intentional — static-prior sensitivity)
```

After this fix, `agribrain` matches or modestly exceeds
`agribrain_cold_start` on ARI in every scenario, which is the
expected behaviour given that the hand-calibrated prior encodes
domain knowledge that the random-init zero start has to rediscover.

## What this means for the manuscript

The §4.7 narrative ("hand-calibrated priors are not strictly required
for the system to work; the context weights can be discovered from
scratch under our REINFORCE update") is **still correct** —
`agribrain_cold_start` does converge to a comparable ARI under the
fair budget. But the prior wording that suggested cold-start
*outperformed* the calibrated version was an artefact of the unfair
budget, not a real finding. The fixed numbers should be reported.

## Where to find the diff

Search the git log for the commit message containing "2025-04
fairness fix" in `mvp/simulation/generate_results.py`. The
`_MULTI_EPISODE_MODES` literal there has the relevant before/after
values in adjacent comment lines.

The pre-fix snapshot of the full benchmark numbers is **not** in the
repo. If a reviewer requires those numbers, the user can reproduce
them by checking out the commit just before the fairness-fix commit
and re-running `run_single_seed.py` for `agribrain_cold_start` with
n_iter=4 and `agribrain` with n_iter=1.

## Status

Open: a side-by-side delta table comparing the published HPC numbers
under the fair budget with the equivalent unfair-budget run. The
unfair-budget run requires HPC time the user has not yet allocated;
this file is the placeholder for that table once it lands. The
documented sensitivity analysis explicitly lists this as one of the items
that must accompany the headline run.
