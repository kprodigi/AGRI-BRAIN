"""Tests of the inference helpers, the record-based analyses and the tables.

    python -m pytest study -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study

study.configure()

import analyze
import inference
import records
import validate
from src.models import slca, waste
from src.models.policy import Policy

N = records.STEPS


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def test_interval_is_the_one_of_the_submitted_analysis() -> None:
    values = np.random.default_rng(1).normal(0.02, 0.01, 20)
    key = ('scenario', 'mode', 'ari')
    assert inference.mean_interval(values, key) == inference.submitted.mean_interval(values, key)
    low95, high95, method = inference.mean_interval(values, key)
    low90, high90, _ = inference.mean_interval(values, key, alpha=0.10)
    assert method == 'BCa' and low95 < low90 < values.mean() < high90 < high95
    assert inference.mean_interval(np.full(20, 0.25), key) == (0.25, 0.25, 'deterministic')
    with pytest.raises(ValueError):
        inference.mean_interval([0.1, float('nan')], key)


def test_estimate_reports_the_mean_the_spread_and_the_signs() -> None:
    result = inference.estimate([0.2, -0.1, 0.0, 0.3], ('x',))
    assert result['mean'] == pytest.approx(0.1) and result['n'] == 4
    assert result['sd'] == pytest.approx(np.std([0.2, -0.1, 0.0, 0.3], ddof=1))
    assert (result['positive_seeds'], result['negative_seeds'], result['zero_seeds']) == (2, 1, 1)
    assert result['low'] <= result['mean'] <= result['high']
    assert inference.estimate([0.2, -0.1, 0.0, 0.3], ('x',), alpha=0.10)['confidence'] == 0.90


def enumerated(differences, alternative):
    d = np.asarray([x for x in differences if x != 0], dtype=float)
    ranks = rankdata(np.abs(d))
    observed = ranks[d > 0].sum()
    sums = [sum(r for r, s in zip(ranks, signs, strict=True) if s) for signs in
            itertools.product([0, 1], repeat=len(d))]
    upper = np.mean([s >= observed - 1e-9 for s in sums])
    lower = np.mean([s <= observed + 1e-9 for s in sums])
    return {'greater': upper, 'less': lower, 'two-sided': min(1.0, 2 * min(upper, lower))}[alternative]


@pytest.mark.parametrize('alternative', ['greater', 'less', 'two-sided'])
def test_signed_rank(alternative) -> None:
    untied = np.random.default_rng(2).normal(0.3, 1.0, 12)
    p, method = inference.signed_rank(untied, alternative)
    assert method == 'exact'
    assert p == wilcoxon(untied.tolist(), alternative=alternative, method='exact').pvalue
    assert p == pytest.approx(enumerated(untied, alternative), abs=1e-12)
    tied = [0.5, -0.5, 1.0, 1.0, 0.0, 2.0, -0.25, 0.0, 3.0]
    p, method = inference.signed_rank(tied, alternative)
    assert method != 'exact' and p == pytest.approx(enumerated(tied, alternative), abs=1e-12)
    assert inference.signed_rank([0.0, 0.0, 0.0], alternative) == (1.0, 'all differences zero')


def test_all_positive_differences_reach_the_floor_of_the_exact_test() -> None:
    differences = np.arange(1, 21) / 100.0
    assert inference.signed_rank(differences, 'greater')[0] == pytest.approx(0.5 ** 20)
    assert inference.signed_rank(differences, 'two-sided')[0] == pytest.approx(2 * 0.5 ** 20)


# ---------------------------------------------------------------------------
# Principal comparisons and readings
# ---------------------------------------------------------------------------
def test_reading_of_an_interval_against_the_threshold() -> None:
    def read(low, high):
        return analyze.reading({'low': low, 'high': high})
    assert read(0.006, 0.010) == 'larger than the threshold, positive'
    assert read(-0.010, -0.006) == 'larger than the threshold, negative'
    assert read(-0.002, 0.004) == 'smaller than the threshold'
    assert read(0.0, 0.0) == 'smaller than the threshold'
    # An interval that contains a threshold does not settle the question.
    assert read(0.004, 0.010) == 'unresolved'
    assert read(-0.006, 0.001) == 'unresolved'
    assert read(0.005, 0.010) == 'unresolved'
    assert analyze.PRACTICAL_THRESHOLD == 0.005


def synthetic_data(shift):
    """Endpoints of the main settings with given mean differences by mode."""
    rng = np.random.default_rng(7)
    modes = {m for _, a, b, _ in analyze.PRINCIPAL for m in (a, b)}
    cells = {}
    for setting in analyze.CONDITIONS:
        for seed in study.FINAL_SEEDS:
            for scenario in study.SCENARIOS:
                base = 0.5 + rng.normal(0.0, 0.01)
                cells[(setting, seed, scenario)] = {
                    mode: {'ari': base + shift.get((setting, mode), 0.0) + rng.normal(0.0, 0.0005),
                           'diverted_volume': 10.0, 'refused_volume': 2.5 + seed % 2,
                           'routing_sha256': f'{mode}{seed % 3}'}
                    for mode in modes}
    return analyze.Data(cells, {}, study.SEED_SETS)


def test_principal_comparisons_are_ten_tests_with_one_holm_family() -> None:
    shift = {('main_fail_closed', 'agribrain'): 0.02, ('main_fail_open', 'agribrain'): 0.02,
             ('main_fail_closed', 'rule_capacity_informed'): 0.05}
    rows = analyze.principal_comparisons(synthetic_data(shift))
    assert len(rows) == 10 and {r['tests_in_family'] for r in rows} == {10}
    assert [r['comparison'] for r in rows[:5]] == [name for name, *_ in analyze.PRINCIPAL]
    assert {r['condition'] for r in rows} == {'fail-closed', 'fail-open'}
    by_key = {(r['setting'], r['comparison']): r for r in rows}
    gain = by_key[('main_fail_closed', 'agribrain_minus_no_context')]
    assert gain['mean'] == pytest.approx(0.02, abs=1e-3) and gain['positive_seeds'] == 20
    assert gain['reading'] == 'larger than the threshold, positive'
    # Twenty differences of one sign give the smallest attainable p-value, and
    # the adjusted value carries the size of the family.
    floor = 2 * 0.5 ** 20
    assert gain['p_two_sided'] == pytest.approx(floor) == gain['smallest_attainable_p']
    assert gain['p_holm'] >= 5 * floor and gain['significant']
    rule = by_key[('main_fail_closed', 'agribrain_minus_capacity_informed_rule')]
    assert rule['mean'] == pytest.approx(-0.03, abs=1e-3)
    assert rule['reading'] == 'larger than the threshold, negative'
    null = by_key[('main_fail_closed', 'adjustment_at_eligibility_guard')]
    assert abs(null['mean']) < 1e-3 and null['reading'] == 'smaller than the threshold'
    assert not null['significant']
    for row in rows:
        assert row['p_holm'] >= row['p_two_sided']
        assert row['half_width'] == pytest.approx(0.5 * (row['high'] - row['low']))
        assert row['sd'] > 0 and row['threshold'] == 0.005
    # The adjustment is the Holm step-down over all ten p-values.
    ordered = sorted(rows, key=lambda r: r['p_two_sided'])
    running = 0.0
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (10 - rank) * row['p_two_sided']))
        assert row['p_holm'] == pytest.approx(running)


def test_data_pools_refused_volume_and_compares_routing() -> None:
    data = synthetic_data({})
    refused = data.refused_share('main_fail_closed', 'agribrain', 'overall')
    assert refused.tolist() == [0.35 if seed % 2 else 0.25 for seed in study.FINAL_SEEDS]
    assert data.metric('main_fail_closed', 'agribrain', 'refused_share_of_diverted_volume',
                       'heatwave').tolist() == refused.tolist()
    assert data.same_routing('main_fail_closed', 'main_fail_open', 'agribrain') == 100
    assert len(data.keys('main_fail_open', 'overall')) == 100
    assert len(data.keys('main_fail_open', 'baseline')) == 20
    assert data.seeds['submitted'] == study.ORIGINAL_SEEDS
    assert analyze.reference_setting({'policy': {'missing_confirmation_closed': False}}) == (
        'main_fail_open')
    assert analyze.reference_setting({'policy': {}}) == 'main_fail_closed'
    assert analyze.endpoint({**dict.fromkeys(study.METRICS, 0.5), 'action_trace': [0, 1, 2, 1]})[
        'share_redistribution'] == 0.5


def test_effects_are_contrasts_of_modes_of_the_primary_setting() -> None:
    for name, (coefficients, meaning) in analyze.EFFECTS.items():
        assert set(coefficients) <= set(study.PRIMARY_MODES), name
        assert sum(coefficients.values()) == pytest.approx(0.0, abs=1e-12), name
        assert meaning
    for contrasts in (analyze.SENSITIVITY_CONTRASTS, analyze.GRADED_CONTRASTS):
        for mode, reference in contrasts.values():
            assert {mode, reference} <= set(study.SENSITIVITY_MODES)
    assert set(analyze.CONDITIONS) == {s['id'] for s in study.design() if s['family'] == 'primary'}
    for without, adjusted in analyze.GUARD_LEVELS.values():
        assert {without, adjusted} <= set(study.PRIMARY_MODES)


# ---------------------------------------------------------------------------
# Record-based analyses
# ---------------------------------------------------------------------------
POLICY = Policy(destination_capacity_enabled=True, revised_context_design=True).model_dump()
PARAMETERS = {'policy': POLICY, 'saving_coefficients': dict(waste.SAVE_FLOOR),
              'route_social_scores': {k: dict(v) for k, v in slca._ACTION_BASES.items()}}
OBSERVED_AS = {'rho_policy_observed_trace': 'rho_outcome_environmental_trace',
               'inventory_policy_observed_trace': 'inventory_outcome_environmental_trace',
               'temp_policy_observed_trace': 'temp_outcome_environmental_trace',
               'rh_policy_observed_trace': 'rh_outcome_environmental_trace'}


def random_environment(seed):
    rng = np.random.default_rng(seed)
    return {
        'rho_outcome_environmental_trace': rng.uniform(0.05, 0.6, N).tolist(),
        'inventory_outcome_environmental_trace': rng.uniform(9e3, 2e4, N).tolist(),
        'temp_outcome_environmental_trace': rng.uniform(3.0, 20.0, N).tolist(),
        'rh_outcome_environmental_trace': rng.uniform(80.0, 99.0, N).tolist(),
        'transport_multiplier_outcome_environmental_trace': rng.uniform(0.8, 1.3, N).tolist(),
        'demand_outcome_environmental_trace': rng.uniform(5.0, 25.0, N).tolist()}


def constant_environment(rho=0.30):
    """Nominal state at every step: a dispatch of size 1 and demand at its reference."""
    return {
        'rho_outcome_environmental_trace': [rho] * N,
        'inventory_outcome_environmental_trace': [12_000.0] * N,
        'temp_outcome_environmental_trace': [4.0] * N,
        'rh_outcome_environmental_trace': [90.0] * N,
        'transport_multiplier_outcome_environmental_trace': [1.0] * N,
        'demand_outcome_environmental_trace': [15.0] * N}


def record(routes, seed=1, scenario='heatwave', environment=None, onset=0.0, learners=None):
    """A cell whose modes follow the given (action, accepted share) sequences."""
    environment = random_environment(seed) if environment is None else environment
    model = validate.OutcomeModel(POLICY, PARAMETERS, POLICY['k_ref'], POLICY['Ea_R'], 'agribrain')
    rows = [dict(zip(records.ENVIRONMENT, values, strict=True)) for values in zip(
        *(environment[t] for t in records.ENVIRONMENT.values()), strict=True)]
    results = {}
    for mode, (actions, shares) in routes.items():
        results[mode] = {
            **environment, **{name: environment[source] for name, source in OBSERVED_AS.items()},
            'hours': (np.arange(N) / 4.0).tolist(),
            'action_trace': [int(a) for a in actions], 'accepted_share_trace': list(shares),
            'ari_trace': [model.step(row, validate.ACTIONS[a], s)['ari']
                          for row, a, s in zip(rows, actions, shares, strict=True)],
            'effective_k_ref': POLICY['k_ref'], 'effective_Ea_R': POLICY['Ea_R'],
            'scenario_onset_offset_hours': onset,
            'learner_summary': (learners or {}).get(mode),
            'destination_capacity_model': {'refusal_penalty_scale': 1.0,
                                           'refused_share_lost': False}}
    return {'task': {'seed': seed, 'scenario': scenario}, 'actual_parameters': PARAMETERS,
            'results': results}


def routes(seed):
    rng = np.random.default_rng(seed)
    return {
        'no_context': (rng.integers(0, 3, N), np.where(rng.random(N) < 0.5, 1.0, rng.random(N))),
        'agribrain': (rng.integers(0, 3, N), np.where(rng.random(N) < 0.8, 1.0, rng.random(N))),
        'static': (np.zeros(N, dtype=int), np.ones(N))}


def test_cell_reconstructs_and_rejects_unpaired_environments() -> None:
    cell = records.Cell(record(routes(3)))
    assert cell.check_reconstruction('agribrain') < 1e-12
    broken = record(routes(3))
    trace = list(broken['results']['agribrain']['temp_outcome_environmental_trace'])
    trace[0] += 1.0
    broken['results']['agribrain']['temp_outcome_environmental_trace'] = trace
    with pytest.raises(ValueError, match='environment'):
        records.Cell(broken)


def test_gain_decomposition_is_exact() -> None:
    cell = records.Cell(record(routes(4)))
    split = records.decompose(cell, 'agribrain', 'no_context')
    a, b = cell.traces['agribrain']['ari'], cell.traces['no_context']['ari']
    assert split['gain'] == pytest.approx(float((a - b).mean()), abs=1e-12)
    assert split['merit'] + split['refusal'] == pytest.approx(split['gain'], abs=1e-12)
    assert split['switch'][:, :, 0].sum() == pytest.approx(1.0)
    assert split['switch'][:, :, 1].sum() == pytest.approx(split['merit'], abs=1e-12)
    assert split['switch'][:, :, 2].sum() == pytest.approx(split['refusal'], abs=1e-12)
    # With every dispatch accepted in full, the whole gain is route merit.
    full = {mode: (actions, np.ones(N)) for mode, (actions, _) in routes(4).items()}
    split = records.decompose(records.Cell(record(full)), 'agribrain', 'no_context')
    assert split['refusal'] == pytest.approx(0.0, abs=1e-12)
    # The same route on every step leaves only the refusal part.
    same = routes(4)
    same['agribrain'] = (same['no_context'][0], np.ones(N))
    split = records.decompose(records.Cell(record(same)), 'agribrain', 'no_context')
    assert split['merit'] == pytest.approx(0.0, abs=1e-12) and split['refusal'] > 0


def test_disruption_windows_partition_the_episode() -> None:
    heatwave = records.Cell(record(routes(5), scenario='heatwave', onset=1.5))
    windows = records.disruption_windows(heatwave)
    assert list(windows) == ['before', 'during', 'after']
    # Hours 25.5 to 49.5 inclusive, at four decisions per hour.
    assert [int(windows[w].sum()) for w in windows] == [102, 97, 89]
    assert sum(windows.values()).tolist() == [1] * N
    outage = records.disruption_windows(records.Cell(record(routes(5), scenario='cyber_outage')))
    assert list(outage) == ['before', 'during'] and int(outage['before'].sum()) == 96
    # The tool channel goes down at hour 24 whatever the offset of the environment.
    for onset, before in ((2.0, 96), (-2.0, 88)):
        late = records.disruption_windows(
            records.Cell(record(routes(5), scenario='cyber_outage', onset=onset)))
        assert int(late['before'].sum()) == before
    assert records.disruption_windows(records.Cell(record(routes(5), scenario='baseline'))) is None
    assert records.disruption_windows(
        records.Cell(record(routes(5), scenario='adaptive_pricing'))) is None
    # The window gains, weighted by their length, give the episode gain.
    seeds = [1, 2]
    cells = {(seed, scenario): records.Cell(record(routes(seed), seed, scenario))
             for seed in seeds for scenario in study.SCENARIOS}
    rows = records.window_table(cells, seeds, [('agribrain', 'no_context')], 'x')
    assert {(r['scenario'], r['window']) for r in rows} == {
        ('heatwave', 'before'), ('heatwave', 'during'), ('heatwave', 'after'),
        ('overproduction', 'before'), ('overproduction', 'during'), ('overproduction', 'after'),
        ('cyber_outage', 'before'), ('cyber_outage', 'during')}
    weights = {'before': 96, 'during': 97, 'after': 95}
    total = sum(r['mean'] * weights[r['window']] for r in rows if r['scenario'] == 'heatwave') / N
    expected = np.mean([records.decompose(cells[(seed, 'heatwave')], 'agribrain', 'no_context')['gain']
                        for seed in seeds])
    assert total == pytest.approx(expected, abs=1e-12)


def test_alternative_index_parameters() -> None:
    variants = records.alternatives(POLICY)
    assert set(variants) == {
        'declared', 'equal_social_weights', 'emissions_weight_doubled', 'labour_weight_doubled',
        'community_weight_doubled', 'price_information_weight_doubled',
        'emissions_normalizer_halved', 'emissions_normalizer_doubled'}
    for variant in variants.values():
        if 'weights' in variant:
            assert sum(variant['weights'].values()) == pytest.approx(1.0)
    doubled = variants['emissions_weight_doubled']['weights']
    assert doubled['w_c'] / doubled['w_l'] == pytest.approx(2 * POLICY['w_c'] / POLICY['w_l'])
    cell = records.Cell(record(routes(5)))
    scores = records.rescore_cell(cell, ['agribrain', 'static'])
    assert scores[('declared', 'agribrain')] == pytest.approx(
        float(cell.traces['agribrain']['ari'].mean()), abs=1e-12)
    for mode in ('agribrain', 'static'):
        product = scores[('declared', mode)]
        assert product < scores[('geometric_aggregation', mode)] < scores[('additive_aggregation', mode)]
        assert scores[('emissions_normalizer_halved', mode)] < product < scores[
            ('emissions_normalizer_doubled', mode)]
    assert scores[('factor_one_minus_risk', 'agribrain')] == scores[('factor_one_minus_risk', 'static')]


def test_overlap_measures() -> None:
    rng = np.random.default_rng(6)
    features = rng.normal(size=(2000, 4))
    groups = np.repeat(np.arange(20), 100)
    threshold = (features[:, 0] > 0.3).astype(float)
    linear = features @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
    noise = rng.normal(size=2000)
    assert records.linear_r2(features, linear) == pytest.approx(1.0)
    assert 0.3 < records.linear_r2(features, threshold) < 0.8
    assert records.neighbour_r2(features[:, :1], threshold, groups) > 0.9
    assert records.neighbour_r2(features, noise, groups) < 0.05
    assert abs(records.linear_r2(features, noise)) < 0.05
    assert records.linear_r2(features, np.ones(2000)) is None
    assert records.neighbour_r2(features, np.ones(2000), groups) is None


def test_recovery_time() -> None:
    baseline = np.ones(N)
    disturbed = np.ones(N)
    disturbed[96:200] = 0.5                      # hours 24 to 50
    assert records.recovery_hours(disturbed, baseline, 48.0) == pytest.approx(2.0)
    assert records.recovery_hours(baseline, baseline, 48.0) == 0.0
    never = disturbed.copy()
    never[200:] = 0.9
    assert records.recovery_hours(never, baseline, 48.0) is None
    late = np.ones(N)
    late[96:] = 0.5
    assert records.recovery_hours(late, baseline, 60.0) is None
    assert records.rolling_ratio(disturbed, baseline).min() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Rule replays and the value of partial acceptance
# ---------------------------------------------------------------------------
def constant_cell(scenario='baseline', rho=0.30):
    fixed = {'static': (np.zeros(N, dtype=int), np.ones(N))}
    return records.Cell(record(fixed, scenario=scenario, environment=constant_environment(rho)))


def test_rule_replay_follows_the_capacity_worked_by_hand() -> None:
    """A dispatch of size 1 per step; redistribution holds 8 and regains 0.5 per
    step, recovery holds 4 and regains 0.25 per step."""
    cell = constant_cell()
    ari, actions, diverted, refused = records.replay_rule(cell, 'always_redistribute')
    # Room 8 - 0.5 t: taken whole for 15 steps, then half of every dispatch.
    assert set(actions) == {1} and diverted == pytest.approx(288.0)
    assert refused == pytest.approx(0.5 * 273)
    # Priority rule with reservation: redistribution for 15 steps, then it
    # alternates with recovery until recovery is drawn down, and settles on
    # redistribution, cold chain, redistribution, recovery. Nothing is refused.
    ari_reserved, actions, diverted, refused = records.replay_rule(cell, 'capacity_informed')
    assert actions[:15] == [1] * 15 and actions[15:28] == [2, 1] * 6 + [2]
    assert actions[28:36] == [1, 0, 1, 2, 1, 0, 1, 2] and actions[-4:] == [1, 0, 1, 2]
    assert refused == 0.0 and diverted == pytest.approx(sum(a != 0 for a in actions))
    assert ari_reserved > ari
    # Reserving half of the dispatch: the rule always redistributes here, and
    # half of each dispatch after the buffer is used is refused.
    _, actions, _, refused = records.replay_rule(cell, 'capacity_informed', reservation_share=0.5)
    assert set(actions) == {1} and refused == pytest.approx(0.5 * 273)
    # The greedy rule starts on redistribution and keeps diverting.
    _, actions, _, _ = records.replay_rule(cell, 'greedy_informed')
    assert actions[:15] == [1] * 15 and 0 not in actions[:15]
    # Below the at-risk threshold recovery does not accept; above the cutoff
    # redistribution does not.
    assert set(records.replay_rule(constant_cell(rho=0.05), 'risk_threshold')[1]) == {0}
    assert set(records.replay_rule(constant_cell(rho=0.60), 'risk_threshold')[1]) == {2}
    assert records.replay_rule(constant_cell(rho=0.60), 'always_redistribute')[3] == pytest.approx(288.0)


def test_rule_replay_without_a_capacity_confirmation() -> None:
    cell = constant_cell('cyber_outage')
    for rule in ('capacity_informed', 'greedy_informed'):
        closed = records.replay_rule(cell, rule, closed=True)[1]
        opened = records.replay_rule(cell, rule, closed=False)[1]
        assert closed[:96] == opened[:96] and set(closed[96:]) == {0}
        assert set(opened[96:]) == {1}
    # Rules that do not read the report are unaffected.
    assert records.replay_rule(cell, 'risk_threshold', closed=True)[1] == records.replay_rule(
        constant_cell(), 'risk_threshold')[1]


def test_reservation_tradeoff_table() -> None:
    seeds = [1, 2]
    cells = {(seed, scenario): constant_cell(scenario) for seed in seeds
             for scenario in study.SCENARIOS}
    rows = {row['rule']: row for row in records.reservation_tradeoff_table(cells, seeds, 'x')}
    assert len(rows) == len(records.RESERVATION_SHARES) + 2
    reserved = rows['priority rule, diverts with 100%']
    assert reserved['refused_share_of_diverted_volume'] == 0.0
    assert 'full_reservation_minus_rule' not in reserved
    always = rows['always redistribute']
    assert always['share_local_redistribute'] == 1.0
    assert always['full_reservation_minus_rule'] == pytest.approx(reserved['ari'] - always['ari'])
    assert always['seeds_full_reservation_higher'] == 2
    for row in rows.values():
        assert sum(row[f'share_{route}'] for route in records.ROUTES) == pytest.approx(1.0)
        assert row['ari'] == pytest.approx(np.mean([row[f'ari_{s}'] for s in study.SCENARIOS]))


class LinearCell:
    """A cell whose waste and social score are given at full and zero acceptance."""

    def __init__(self, cold, full, none):
        self.values = {0: (cold, cold), 1: (full, none)}

    def outcome(self, step, action, share):
        full, none = self.values[action]
        return {'waste': none[0] + (full[0] - none[0]) * share,
                'slca': none[1] + (full[1] - none[1]) * share}


def test_break_even_accepted_share() -> None:
    # Raw waste 0.10 at the nominal state: redistribution is worth
    # (0.8625 + 0.0825 a)(0.5762 + 0.2454 a) against 0.9 x 0.6086.
    cell = LinearCell((0.10, 0.6086), (0.055, 0.8216), (0.1375, 0.5762))
    share = records.break_even_share(cell, 0, 1)
    assert (0.8625 + 0.0825 * share) * (0.5762 + 0.2454 * share) == pytest.approx(0.54774, abs=1e-12)
    roots = np.roots([0.0825 * 0.2454, 0.8625 * 0.2454 + 0.0825 * 0.5762,
                      0.8625 * 0.5762 - 0.9 * 0.6086])
    assert share == pytest.approx(float(roots[(roots > 0) & (roots < 1)][0]), abs=1e-9)
    assert round(share, 3) == 0.193
    # A diversion that is worth the cold chain even when refused in full.
    assert records.break_even_share(LinearCell((0.10, 0.60), (0.05, 0.80), (0.10, 0.60)), 0, 1) == 0.0
    # A diversion that is never worth the cold chain.
    assert records.break_even_share(LinearCell((0.10, 0.60), (0.10, 0.59), (0.20, 0.50)), 0, 1) is None
    # On the recorded environment of a cell.
    real = constant_cell()
    share = records.break_even_share(real, 0, 1)
    assert 0.0 < share < 1.0
    assert real.outcome(0, 1, share)['ari'] == pytest.approx(real.outcome(0, 0, 1.0)['ari'], abs=1e-12)
    table = records.break_even_table({(1, s): real for s in study.SCENARIOS}, [1], 'x')
    row = next(r for r in table if r['scenario'] == 'overall' and r['route'] == 'local_redistribute')
    assert row['eligible_steps'] == row['steps'] == 5 * N
    assert row['destination_accepts_share'] == 1.0
    assert row['never_worth_the_diversion_share'] == 0.0
    # Above the marketability cutoff redistribution accepts nothing; recovery does.
    spoiled = records.break_even_table(
        {(1, s): constant_cell(rho=0.60) for s in study.SCENARIOS}, [1], 'x')
    accepts = {r['route']: r['destination_accepts_share'] for r in spoiled
               if r['scenario'] == 'overall'}
    assert accepts == {'local_redistribute': 0.0, 'recovery': 1.0}
    assert row['break_even_share_p50'] == pytest.approx(share)


# ---------------------------------------------------------------------------
# Decision records
# ---------------------------------------------------------------------------
def ledger_row(step, base, modifier, removed=(), clip=(1.0, 1.0, 1.0), probs=(0.2, 0.5, 0.3),
               action=1, bias=(0.0, 0.0, 0.0), read=0):
    contributions = np.zeros((3, 5))
    contributions[:, 0] = modifier
    return {
        'step_index': step, 'role': 'distributor', 'psi': [1.0, 0.0, 0.0, 0.0, 0.0],
        'phi': [0.1] * 10, 'context_feature_contributions': contributions.tolist(),
        'context_nonfeature_residual': [0.0, 0.0, 0.0], 'context_modifier': list(modifier),
        'base_logits': list(base), 'policy_temperature': 1.0,
        'guard': {'removed_routes': list(removed)}, 'peer_message_bias': list(bias),
        'primary_mcp_tools_invoked_step': ['a', 'b'], 'action_idx': action, 'probs': list(probs),
        'context_integration': {'primary': {'clip_derivative': [1.0, 1.0, 1.0]},
                                'cooperative': None,
                                'composition': {'clip_derivative': list(clip)}},
        'step_channel_evidence': {'peer': {'consumed_count': read},
                                  'cooperative': {'active': False}}}


def test_score_influence_and_clip_activity() -> None:
    rows = [
        # The adjustment changes the leading route (0.5 against a gap of 0.2).
        ledger_row(0, (1.0, 0.8, 0.0), (0.0, 0.5, 0.0)),
        # It does not (0.1 against a gap of 1.0).
        ledger_row(1, (2.0, 1.0, 0.0), (0.0, 0.1, 0.0)),
        # The leading route changes, but the guard has removed the new leader.
        ledger_row(2, (1.0, 0.8, 0.0), (0.0, 0.5, 0.0), removed=[1]),
        # The clip is active, and the leading route changes.
        ledger_row(3, (0.9, 0.5, 0.0), (-1.0, 0.0, 0.0), clip=(0.0, 1.0, 1.0)),
    ]
    arrays = records.ledger_arrays(rows)
    assert arrays['clipped'].tolist() == [False, False, False, True]
    assert arrays['removed'][2].tolist() == [False, True, False]
    cells = {(seed, scenario): arrays for seed in (1, 2) for scenario in study.SCENARIOS}
    table = records.score_influence_table({'agribrain': cells}, [1, 2], 'x')
    row = next(r for r in table if r['scenario'] == 'overall')
    assert row['decisions'] == 40 and row['clip_active_decisions'] == 10
    # Decisions 0, 2 and 3 change the leading route; with the guard, 0 and 3.
    assert row['leading_route_changed_share'] == 0.75
    assert row['leading_available_route_changed_share'] == 0.5
    assert row['base_score_gap_p50'] == pytest.approx(0.3)          # gaps 0.2, 1.0, 0.2, 0.4
    assert row['largest_adjustment_p95'] == pytest.approx(1.0)
    assert 0.0 < row['probability_change_p50'] < 0.5
    activity = records.role_activity(cells)
    assert activity['distributor']['decisions'] == 40
    assert activity['distributor']['tool_calls'] == 80
    assert activity['distributor']['decisions_with_route_removed'] == 10


def test_peer_influence_counts_changed_probabilities_and_routes() -> None:
    def cells(probs, actions, bias):
        rows = [ledger_row(i, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), probs=p, action=a, bias=b,
                           read=int(any(b)))
                for i, (p, a, b) in enumerate(zip(probs, actions, bias, strict=True))]
        arrays = records.ledger_arrays(rows)
        return {(seed, scenario): arrays for seed in (1, 2) for scenario in study.SCENARIOS}

    quiet = [(0.0, 0.0, 0.0)] * 3
    without = cells([(0.2, 0.5, 0.3)] * 3, [1, 1, 2], quiet)
    with_peers = cells([(0.2, 0.5, 0.3), (0.1, 0.6, 0.3), (0.2, 0.3, 0.5)], [1, 1, 1],
                       [(0.0, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, -0.2)])
    ledgers = {'agribrain': with_peers, 'agribrain_no_peer': without,
               'agribrain_eligibility': without, 'agribrain_eligibility_no_peer': without}
    table = records.peer_influence_table(ledgers, [1, 2], 'x')
    row = next(r for r in table if r['mode'] == 'agribrain' and r['scenario'] == 'heatwave')
    assert row['decisions'] == 6 and row['decisions_with_a_peer_bias'] == 4
    assert row['decisions_with_a_peer_message_read'] == 4
    assert row['largest_peer_bias'] == pytest.approx(0.3)
    assert row['decisions_with_different_probabilities'] == 4
    assert row['largest_probability_change'] == pytest.approx(0.2)
    assert row['expected_changed_decisions'] == pytest.approx(2 * (0.1 + 0.2))
    assert row['decisions_with_a_different_route'] == 2
    same = next(r for r in table if r['mode'] == 'agribrain_eligibility'
                and r['scenario'] == 'overall')
    assert same['expected_changed_decisions'] == 0.0 and same['decisions_with_a_different_route'] == 0


def learner(final, constrained=True, mode='relative_delta', value=0.5, reversals=0):
    initial = [[-0.4, 0.12], [0.3, -0.08]]
    return {'initial_theta': initial, 'final_theta': final, 'sign_constrained': constrained,
            'sign_reversal_count': reversals, 'magnitude_cap_mode': mode,
            'magnitude_cap_value': value, 'magnitude_cap_abs_floor': 0.10, 'n_updates': 864}


def test_constraint_activity_after_adaptation() -> None:
    # Bounds: 0.20, 0.10, 0.15 and 0.10 around the initial weights.
    np.testing.assert_allclose(records.change_bound(learner(None)), [[0.20, 0.10], [0.15, 0.10]])
    learners = {
        # One weight at its bound (-0.4 - 0.2), one held at zero by the sign constraint.
        'agribrain': learner([[-0.6, 0.15], [0.3, 0.0]]),
        # Without the constraint the same weight changes sign.
        'agribrain_sign_unconstrained': learner([[-0.6, 0.15], [0.3, 0.02]], False, reversals=1),
        'no_context': None,
    }
    fixed = {mode: (np.zeros(N, dtype=int), np.ones(N)) for mode in learners}
    cells = {(seed, scenario): records.Cell(record(fixed, seed, scenario, learners=learners))
             for seed in (1, 2) for scenario in study.SCENARIOS}
    rows = {row['mode']: row for row in records.constraint_activity_table(cells, [1, 2], 'x')}
    assert set(rows) == {'agribrain', 'agribrain_sign_unconstrained'}
    bounded = rows['agribrain']
    assert bounded['cells'] == 10 and bounded['weights'] == 40
    assert bounded['weights_held_at_zero_by_the_sign_constraint'] == 10
    assert bounded['weights_at_change_bound'] == 10 and bounded['cells_with_a_weight_at_bound'] == 10
    assert bounded['sign_reversals_in_final_weights'] == 0
    assert bounded['largest_change_as_share_of_bound'] == pytest.approx(1.0)
    assert bounded['largest_weight_change'] == pytest.approx(0.2)
    free = rows['agribrain_sign_unconstrained']
    assert free['weights_held_at_zero_by_the_sign_constraint'] == 0
    assert free['sign_reversals_in_final_weights'] == 10
    # The 0.08 weight reaches its bound of 0.10 on the way through zero.
    assert free['weights_at_change_bound'] == 20
