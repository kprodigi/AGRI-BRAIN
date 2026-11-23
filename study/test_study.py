"""Tests of the study design, the harness and the ledger validator.

Run from the repository root, separately from the backend tests because the
harness pins the publication environment:

    python -m pytest study -q
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate

import study

study.configure()

from src.models import action_selection, slca, waste
from src.models.mode_capabilities import capabilities_for
from src.models.policy import Policy

FAMILIES = {'primary': 2, 'submitted': 2, 'diagnostic': 2, 'weights': 22, 'structural': 31,
            'content': 4}
MAIN = 'main_fail_closed'


def setting(setting_id):
    return next(s for s in study.design() if s['id'] == setting_id)


def changed(setting_id, reference=MAIN):
    """Policy fields in which a setting differs from the reference setting."""
    base = setting(reference)['policy']
    return {k: v for k, v in setting(setting_id)['policy'].items() if base.get(k) != v}


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def test_design_is_a_complete_crossing() -> None:
    settings, tasks = study.design(), study.tasks()
    assert len(settings) == 63 and len(tasks) == 6300
    assert {f: sum(s['family'] == f for s in settings) for f in FAMILIES} == FAMILIES
    assert set(FAMILIES) == set(study.RETAINED_EVIDENCE)
    assert len({(t['setting'], t['seed'], t['scenario']) for t in tasks}) == 6300
    assert [t['index'] for t in tasks] == list(range(6300))
    for s in settings:
        cells = [(t['seed'], t['scenario']) for t in tasks if t['setting'] == s['id']]
        assert sorted(cells) == sorted(
            (n, c) for n in study.SEED_SETS[s['seeds']] for c in study.SCENARIOS)


def test_seed_sets_are_disjoint_and_assigned_by_family() -> None:
    final, original, development = (set(study.FINAL_SEEDS), set(study.ORIGINAL_SEEDS),
                                    set(study.DEVELOPMENT_SEEDS))
    assert len(final) == len(original) == 20 and study.FINAL_SEEDS == list(range(5001, 5021))
    assert not final & original and not final & development and not original & development
    for s in study.design():
        expected = 'original' if s['family'] in ('submitted', 'diagnostic') else 'final'
        assert s['seeds'] == expected, s['id']
        assert study.setting_seeds(s['id']) == study.SEED_SETS[expected]


def test_final_seeds_are_not_run_in_development() -> None:
    for seed in study.FINAL_SEEDS:
        with pytest.raises(ValueError, match='final evaluation seeds'):
            study.refuse_final_seeds_in_development(seed, True)
        study.refuse_final_seeds_in_development(seed, False)
    for seed in [*study.DEVELOPMENT_SEEDS, *study.ORIGINAL_SEEDS]:
        study.refuse_final_seeds_in_development(seed, True)
    # A run of the study cannot be prepared on the development seeds.
    with pytest.raises(ValueError, match='development'):
        study.prepare(Path(tempfile.gettempdir()) / 'not_created', False, development_seeds=True)
    development = dict.fromkeys(study.SEED_SETS, study.DEVELOPMENT_SEEDS)
    assert len(study.tasks(development)) == 630
    assert not {t['seed'] for t in study.tasks(development)} & set(study.FINAL_SEEDS)


def test_task_and_pilot_indices() -> None:
    tasks = study.tasks()
    for index in (0, 137, 2502, 6299):
        t = tasks[index]
        assert study.task_index(t['setting'], t['seed'], t['scenario']) == index
    for index, (setting_id, scenario) in zip(
            study.pilot_indices(), study.PILOT_CELLS, strict=True):
        assert (tasks[index]['setting'], tasks[index]['seed'], tasks[index]['scenario']) == (
            setting_id, study.setting_seeds(setting_id)[0], scenario)
    assert set(study.REPEATED_PILOT_CELLS) <= set(study.PILOT_CELLS)
    assert {setting(s)['family'] for s, _ in study.PILOT_CELLS} == set(FAMILIES)
    # The cluster worker addresses the controls by the first task indices.
    import controls
    assert controls.primary_indices() == list(range(200))
    # The cyber outage is piloted under both missing-confirmation rules.
    assert {('main_fail_closed', 'cyber_outage'), ('main_fail_open', 'cyber_outage')} <= set(
        study.PILOT_CELLS)


def test_episode_totals() -> None:
    per_task = {s['id']: study.episode_counts(s['modes'])['episodes'] for s in study.design()}
    assert per_task['main_fail_closed'] == per_task['main_fail_open'] == 74
    assert per_task['diagnostic_fail_closed'] == 74
    assert per_task['diagnostic_unlimited_capacity'] == 18
    assert per_task['submitted'] == 11 and per_task['submitted_without_scenario_terms'] == 4
    assert {per_task[s['id']] for s in study.design()
            if s['family'] in study.SENSITIVITY_FAMILIES} == {18}
    assert sum(per_task[t['setting']] for t in study.tasks()) == 128_100


def test_every_setting_is_a_valid_policy_and_retention_is_declared() -> None:
    fields = set(Policy.model_fields)
    for s in study.design():
        assert set(s['policy']) <= fields, s['id']
        Policy(**s['policy'])
        assert s['family'] in study.RETAINED_EVIDENCE
        assert set(s['patches']) <= {'base_policy_scale', 'tool_prior_scale',
                                     'retrieval_prior_scale', 'saving', 'social',
                                     'context_sign_reversal'}
    assert study.RETAINED_EVIDENCE['primary'] == study.RETAINED_EVIDENCE['submitted'] == 'full'


def test_main_settings_differ_only_in_the_missing_confirmation_rule() -> None:
    closed, opened = setting('main_fail_closed'), setting('main_fail_open')
    assert closed['modes'] == opened['modes'] == study.PRIMARY_MODES
    assert closed['policy'] == study.REVISED and not closed['patches'] and not opened['patches']
    assert changed('main_fail_open') == {'missing_confirmation_closed': False}
    assert Policy(**closed['policy']).missing_confirmation_closed is True
    assert Policy(**closed['policy']).reservation_share == 1.0
    assert setting('diagnostic_fail_closed')['policy'] == closed['policy']
    assert setting('diagnostic_fail_closed')['modes'] == study.PRIMARY_MODES


def test_weight_settings() -> None:
    weights = [s for s in study.design() if s['family'] == 'weights']
    for s in weights:
        policy = s['policy']
        assert sum(policy[k] for k in ['w_c', 'w_l', 'w_r', 'w_p']) == pytest.approx(1.0, abs=1e-14)
        different = [k for k, v in s['factors'].items() if v != 1.0]
        assert len(different) == (9 if s['id'].startswith('joint') else 1)
        assert all(0.8 <= v <= 1.2 for v in s['factors'].values())
    for amplitude in ('10', '20'):
        a, b = setting(f'joint{amplitude}_A')['factors'], setting(f'joint{amplitude}_B')['factors']
        assert all(a[k] + b[k] == pytest.approx(2.0) for k in a)
    with pytest.raises(ValueError):
        study.weight_setting('x', {})
    with pytest.raises(ValueError):
        study.weight_setting('x', {**dict.fromkeys(study.WEIGHT_GROUPS, 1.0), 'eta': 1.5})


def test_sensitivity_settings_differ_from_the_main_setting_and_from_each_other() -> None:
    main = setting(MAIN)
    seen = {}
    for s in study.design():
        if s['family'] not in study.SENSITIVITY_FAMILIES:
            continue
        assert s['modes'] == study.SENSITIVITY_MODES and s['seeds'] == 'final'
        assert s['policy']['destination_capacity_enabled'] and s['policy']['revised_context_design']
        key = repr((sorted(s['policy'].items()), sorted(s['patches'].items(), key=str)))
        assert key not in seen, (s['id'], seen.get(key))
        seen[key] = s['id']
        assert s['policy'] != main['policy'] or any(
            v != 1.0 for v in s['patches'].values()), s['id']
    assert set(study.SENSITIVITY_MODES) <= set(study.PRIMARY_MODES)


def test_document_settings() -> None:
    assert changed('acceptance_document_cutoff_040') == {'published_rho_cut': 0.40}
    stated = Policy(**setting('acceptance_document_cutoff_040')['policy'])
    assert stated.acceptance_rho_cut == 0.50          # the destination still applies 0.50
    assert validate.published_criteria(stated.model_dump())['rho_cut'] == 0.40
    assert changed('acceptance_document_renamed') == {
        'acceptance_document_id': 'outlet_intake_rules'}
    assert changed('acceptance_document_absent') == {'acceptance_document_present': False}
    assert changed('acceptance_document_absent_fail_open') == {
        'acceptance_document_present': False, 'missing_confirmation_closed': False}


def test_graded_series_are_ordered_one_factor_changes() -> None:
    main = setting(MAIN)['policy']
    levels = {'refusal_penalty': ('refusal_penalty_scale', [0.5, 1.0, 2.0, 4.0]),
              'redistribution_capacity': ('capacity_kappa_lr', [0.125, 0.25, 0.5, 1.0, 2.0]),
              'reservation_share': ('reservation_share', [0.5, 0.8, 1.0])}
    assert set(levels) == set(study.GRADED_SERIES)
    for series, (field, values) in levels.items():
        ids = [s for s, _ in study.GRADED_SERIES[series]]
        assert ids.count(MAIN) == 1
        graded = ids[:len(values)]
        assert [Policy(**setting(s)['policy']).model_dump()[field] for s in graded] == values
        for s in graded:
            assert set(changed(s)) <= {field} and not setting(s)['patches']
            assert setting(s)['policy'] == {**main, **changed(s)}
    assert changed('refused_share_lost') == {'refused_share_lost': True}
    assert study.GRADED_SERIES['refusal_penalty'][-1][0] == 'refused_share_lost'
    assert study.GRADED_SERIES['redistribution_capacity'][-1][0] == 'unlimited_capacity_no_criteria'


def test_submitted_settings_leave_the_submitted_model_in_place() -> None:
    assert setting('submitted')['policy'] == {}
    assert setting('submitted')['modes'] == [
        'static', 'no_context', 'agribrain', 'rule_always_redistribute', 'rule_always_recovery']
    assert setting('submitted_without_scenario_terms')['modes'] == ['agribrain']
    assert setting('submitted_without_scenario_terms')['policy'] == {'query_scenario_terms': False}
    default = Policy()
    assert not default.destination_capacity_enabled and not default.revised_context_design
    assert default.query_scenario_terms and default.reward_objective == 'linear'
    unlimited = Policy(**setting('diagnostic_unlimited_capacity')['policy'])
    assert not unlimited.destination_acceptance_enabled and unlimited.capacity_bmax_lr == 1.0e9


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def adjusts(mode):
    caps = capabilities_for(mode)
    return caps.context_kind is not None and caps.context_adjustment


def differing(mode, reference):
    a, b = (dataclasses.asdict(capabilities_for(m)) for m in (mode, reference))
    return {key for key in a if a[key] != b[key]}


def test_entry_points_of_the_comparison_modes() -> None:
    table = {  # mode: (score adjustment, guard, peer messages)
        'no_context': (False, 'none', False), 'adjustment_only': (True, 'none', False),
        'eligibility_only': (False, 'eligibility', False),
        'agribrain_eligibility_no_peer': (True, 'eligibility', False),
        'reservation_only': (False, 'reservation', False),
        'agribrain_no_peer': (True, 'reservation', False),
        'agribrain': (True, 'reservation', True),
        'agribrain_eligibility': (True, 'eligibility', True),
        'agribrain_no_guard': (True, 'none', True), 'no_pinn': (True, 'reservation', True),
        'pirag_only': (True, 'reservation', True), 'mcp_only': (True, 'reservation', True),
        'agribrain_fixed_context': (True, 'reservation', True),
        'agribrain_sign_unconstrained': (True, 'reservation', True),
        'feature_inputs_prior': (True, 'eligibility', False),
        'feature_inputs_zero': (True, 'eligibility', False),
        'constant_adjustment': (True, 'eligibility', False),
    }
    assert set(table) | {'static', *study.RULES} == set(study.PRIMARY_MODES)
    for mode, (adjustment, guard, peers) in table.items():
        caps = capabilities_for(mode)
        assert (adjusts(mode), caps.guard, caps.peer_messages) == (adjustment, guard, peers), mode
        assert caps.learned and caps.episode_count == 4 and caps.fixed_rule is None
    assert capabilities_for('pirag_only').context_kind == 'pirag_only'
    assert capabilities_for('mcp_only').context_kind == 'mcp_only'
    assert not capabilities_for('agribrain_fixed_context').context_matrix_learning
    assert not capabilities_for('agribrain_sign_unconstrained').sign_constrained_learning
    assert not capabilities_for('no_pinn').spoilage_residual
    for mode, init in (('feature_inputs_prior', 'prior'), ('feature_inputs_zero', 'zero')):
        caps = capabilities_for(mode)
        assert (caps.context_interface, caps.context_weight_init) == ('feature', init)
    assert capabilities_for('constant_adjustment').constant_adjustment
    for mode in ['static', *study.RULES]:
        caps = capabilities_for(mode)
        assert not caps.learned and caps.episode_count == 1 and caps.guard == 'none'
    assert [capabilities_for(m).fixed_rule for m in study.RULES] == [
        'always_redistribute', 'always_recovery', 'risk_threshold', 'capacity_informed',
        'greedy_informed']
    # A mode in this list requests the capacity report that its guard needs.
    for mode in study.RESERVATION_MODES:
        caps = capabilities_for(mode)
        assert caps.guard == 'reservation' and caps.context_kind in ('full', 'mcp_only')


def test_compared_modes_differ_in_the_factor_that_is_compared() -> None:
    adjustment = {'context_adjustment', 'context_matrix_learning'}
    assert differing('agribrain', 'agribrain_eligibility') == {'guard'}
    assert differing('agribrain_no_peer', 'agribrain_eligibility_no_peer') == {'guard'}
    assert differing('reservation_only', 'eligibility_only') == {'guard'}
    assert differing('agribrain_eligibility_no_peer', 'eligibility_only') == adjustment
    assert differing('agribrain_no_peer', 'reservation_only') == adjustment
    assert differing('agribrain_eligibility_no_peer', 'constant_adjustment') == {
        'constant_adjustment', 'context_weight_init'}
    assert differing('agribrain_eligibility_no_peer', 'feature_inputs_prior') == {
        'context_interface'}
    assert differing('agribrain', 'agribrain_no_peer') == {'peer_messages'}
    assert differing('agribrain_eligibility', 'agribrain_eligibility_no_peer') == {'peer_messages'}
    assert differing('agribrain', 'agribrain_no_guard') == {'guard'}
    assert differing('agribrain', 'pirag_only') == {'context_kind'}
    assert differing('agribrain', 'mcp_only') == {'context_kind'}


# ---------------------------------------------------------------------------
# Applying a setting
# ---------------------------------------------------------------------------
def test_settings_apply_and_are_restored_after_an_exception() -> None:
    from pirag import context_design

    from mvp.simulation import generate_results as gr

    original = (gr.Policy, gr.SCENARIOS, gr.RESULTS_DIR, action_selection.THETA.copy(),
                context_design.THETA_CONTEXT_REVISED.copy(), dict(waste.SAVE_FLOOR),
                {k: dict(v) for k, v in slca._ACTION_BASES.items()})
    saving_object, bases_object = waste.SAVE_FLOOR, slca._ACTION_BASES
    output = Path(tempfile.gettempdir())

    def applied(setting_id, check):
        with pytest.raises(RuntimeError, match='restoration'):
            with study.overridden(gr, setting(setting_id), 'heatwave', output) as actual:
                check(actual)
                raise RuntimeError('restoration')
        assert gr.Policy is original[0] and gr.SCENARIOS is original[1]
        assert gr.RESULTS_DIR is original[2]
        np.testing.assert_array_equal(action_selection.THETA, original[3])
        np.testing.assert_array_equal(context_design.THETA_CONTEXT_REVISED, original[4])
        assert waste.SAVE_FLOOR is saving_object and dict(waste.SAVE_FLOOR) == original[5]
        assert slca._ACTION_BASES is bases_object
        assert {k: dict(v) for k, v in slca._ACTION_BASES.items()} == original[6]

    def joint(actual):
        policy = gr.Policy()
        assert policy.eta == pytest.approx(0.4) and policy.eta_rho == pytest.approx(0.6)
        assert gr.SCENARIOS == ['heatwave']
        np.testing.assert_array_equal(action_selection.THETA, original[3] * 1.2)
        prior = context_design.THETA_CONTEXT_REVISED
        np.testing.assert_array_equal(prior[:, [0, 1, 2, 4]], original[4][:, [0, 1, 2, 4]] * 0.8)
        np.testing.assert_array_equal(prior[:, [3]], original[4][:, [3]] * 1.2)
        assert actual['initial_context_matrix'] == prior.tolist()

    def saving(actual):
        assert waste.SAVE_FLOOR['local_redistribute'] == pytest.approx(0.36)
        assert waste.SAVE_FLOOR['recovery'] == pytest.approx(0.20)
        assert waste.SAVE_FLOOR['cold_chain'] == 0.0
        assert actual['saving_coefficients'] == dict(waste.SAVE_FLOOR)

    def social(actual):
        cold = slca._ACTION_BASES['cold_chain']
        assert all(dict(v) == dict(cold) for v in slca._ACTION_BASES.values())
        assert actual['route_social_scores']['recovery'] == dict(cold)

    def signs(actual):
        prior = context_design.THETA_CONTEXT_REVISED
        np.testing.assert_array_equal(prior[:, [0, 1]], -original[4][:, [0, 1]])
        np.testing.assert_array_equal(prior[:, [2, 3, 4]], original[4][:, [2, 3, 4]])

    def merit(actual):
        policy = gr.Policy()
        assert (policy.km_local, policy.km_recovery) == (80.0, 45.0)
        assert waste.SAVE_FLOOR['recovery'] == 0.45
        assert slca._ACTION_BASES['recovery'] == original[6]['local_redistribute']
        assert slca._ACTION_BASES['local_redistribute'] == original[6]['recovery']

    def reservation(actual):
        assert gr.Policy().reservation_share == 0.5
        assert actual['policy']['reservation_share'] == 0.5

    applied('joint20_A', joint)
    applied('saving_minus20', saving)
    applied('social_scores_equal', social)
    applied('condition_signs_reversed', signs)
    applied('merit_order_exchanged', merit)
    applied('reservation_share_050', reservation)


# ---------------------------------------------------------------------------
# Ledger validator
# ---------------------------------------------------------------------------
CONFIG = {'acceptance_enabled': True, 'rho_at': 0.10, 'rho_cut': 0.50,
          'strict_redistribution': False}
POLICY = Policy(destination_capacity_enabled=True, revised_context_design=True).model_dump()
PARAMETERS = {'saving_coefficients': dict(waste.SAVE_FLOOR),
              'route_social_scores': {k: dict(v) for k, v in slca._ACTION_BASES.items()}}
CRITERIA = {'rho_at': 0.10, 'rho_cut': 0.50, 'strict_redistribution': False}
ROOM = {1: 2.0, 2: 2.0}
# Nominal state: 4 degrees C, inventory at its baseline, unit transport multiplier.
NOMINAL = {'rho_outcome_environmental': 0.30, 'inventory_outcome_environmental': 12_000.0,
           'temp_outcome_environmental': 4.0, 'rh_outcome_environmental': 90.0,
           'transport_multiplier_outcome_environmental': 1.0}


def outcome_model(**options):
    return validate.OutcomeModel(POLICY, PARAMETERS, POLICY['k_ref'], POLICY['Ea_R'],
                                 'agribrain', **options)


def test_destination_acceptance() -> None:
    accepts = validate.accepted_by_destination
    assert accepts('cold_chain', 0.99, CONFIG)
    assert accepts('local_redistribute', 0.50, CONFIG)
    assert not accepts('local_redistribute', 0.51, CONFIG)
    assert not accepts('recovery', 0.10, CONFIG) and accepts('recovery', 0.11, CONFIG)
    strict = {**CONFIG, 'strict_redistribution': True}
    assert not accepts('local_redistribute', 0.05, strict)
    assert accepts('recovery', 0.0, {**CONFIG, 'acceptance_enabled': False})


@pytest.mark.parametrize(('kind', 'rho', 'criteria', 'room', 'size', 'share', 'closed', 'expected'), [
    # Inside the band with room for the whole dispatch: nothing is removed.
    ('reservation', 0.30, CRITERIA, ROOM, 1.0, 1.0, True, {}),
    ('eligibility', 0.30, CRITERIA, ROOM, 1.0, 1.0, True, {}),
    # Outside the band stated in the document.
    ('reservation', 0.05, CRITERIA, ROOM, 1.0, 1.0, True, {2: 'acceptance'}),
    ('eligibility', 0.60, CRITERIA, ROOM, 1.0, 1.0, True, {1: 'acceptance'}),
    ('eligibility', 0.05, {**CRITERIA, 'strict_redistribution': True}, ROOM, 1.0, 1.0, True,
     {1: 'acceptance', 2: 'acceptance'}),
    # The criteria are those of the document, whatever the destination applies.
    ('reservation', 0.45, {**CRITERIA, 'rho_cut': 0.40}, ROOM, 1.0, 1.0, True, {1: 'acceptance'}),
    # Room for part of the dispatch: the reservation guard removes the route,
    # the eligibility guard leaves it.
    ('reservation', 0.30, CRITERIA, {1: 0.5, 2: 2.0}, 1.0, 1.0, True, {1: 'capacity'}),
    ('eligibility', 0.30, CRITERIA, {1: 0.5, 2: 2.0}, 1.0, 1.0, True, {}),
    # A smaller reserved share.
    ('reservation', 0.30, CRITERIA, {1: 0.5, 2: 2.0}, 1.0, 0.5, True, {}),
    ('reservation', 0.30, CRITERIA, {1: 0.49, 2: 2.0}, 1.0, 0.5, True, {1: 'capacity'}),
    # No room at all: both guards remove the route, for a dispatch of positive size.
    ('eligibility', 0.30, CRITERIA, {1: 0.0, 2: 2.0}, 1.0, 1.0, True, {1: 'no_capacity'}),
    ('eligibility', 0.30, CRITERIA, {1: 0.0, 2: 0.0}, 0.0, 1.0, True, {}),
    ('reservation', 0.30, CRITERIA, {1: 0.0, 2: 0.0}, 0.0, 1.0, True, {}),
    # The acceptance reason comes first.
    ('reservation', 0.60, CRITERIA, {1: 0.5, 2: 2.0}, 1.0, 1.0, True, {1: 'acceptance'}),
    # A capacity report that was requested and is unavailable.
    ('reservation', 0.30, CRITERIA, 'unavailable', 1.0, 1.0, True,
     {1: 'no_capacity_report', 2: 'no_capacity_report'}),
    ('reservation', 0.30, CRITERIA, 'unavailable', 1.0, 1.0, False, {}),
    ('eligibility', 0.30, CRITERIA, 'unavailable', 1.0, 1.0, True, {}),
    # Criteria that were requested and are unavailable.
    ('reservation', 0.30, 'unavailable', ROOM, 1.0, 1.0, True,
     {1: 'no_acceptance_criteria', 2: 'no_acceptance_criteria'}),
    ('eligibility', 0.30, 'unavailable', ROOM, 1.0, 1.0, True,
     {1: 'no_acceptance_criteria', 2: 'no_acceptance_criteria'}),
    ('reservation', 0.60, 'unavailable', ROOM, 1.0, 1.0, False, {}),
    # Information the mode never requests switches the check off under either rule.
    ('reservation', 0.30, CRITERIA, 'withheld', 1.0, 1.0, True, {}),
    ('reservation', 0.60, 'withheld', ROOM, 1.0, 1.0, True, {}),
    ('reservation', 0.60, 'withheld', {1: 0.5, 2: 2.0}, 1.0, 1.0, True, {1: 'capacity'}),
])
def test_routes_a_guard_removes(kind, rho, criteria, room, size, share, closed, expected) -> None:
    assert validate.removed_routes(kind, rho, criteria, room, size, share, closed) == expected
    # The simulator's own test gives the same routes.
    from src.models.destination_capacity import route_feasibility
    information = {'guard_kind': kind, 'dispatch_size': size, 'reservation_share': share,
                   'missing_confirmation_closed': closed,
                   'criteria': validate.state_name(criteria),
                   'capacity': validate.state_name(room)}
    if isinstance(criteria, dict):
        information.update(criteria)
    if isinstance(room, dict):
        information.update(redistribution_capacity=room[1], recovery_capacity=room[2])
    feasible, reasons = route_feasibility(rho, information)
    assert {i for i in (1, 2) if not feasible[i]} == set(expected)
    assert all(reasons[i] == reason for i, reason in expected.items())


def test_information_a_guard_holds() -> None:
    report = {'dispatch_size': 1.0, 'redistribution_capacity': 0.7, 'recovery_capacity': 1.5}
    record = {'available': True, **CRITERIA, 'doc_id': 'x'}
    states = validate.information_states
    assert states(capabilities_for('agribrain'), report, record) == (CRITERIA, {1: 0.7, 2: 1.5})
    assert states(capabilities_for('agribrain'), None, record) == (CRITERIA, 'unavailable')
    assert states(capabilities_for('agribrain'), report, {'available': False})[0] == 'unavailable'
    assert states(capabilities_for('agribrain'), report, None)[0] == 'unavailable'
    # Without tools the capacity report is never requested; without retrieval
    # the criteria are never requested.
    assert states(capabilities_for('pirag_only'), None, record) == (CRITERIA, 'withheld')
    assert states(capabilities_for('mcp_only'), report, None) == ('withheld', {1: 0.7, 2: 1.5})
    assert validate.state_name(CRITERIA) == 'available'
    assert validate.state_name('withheld') == 'withheld'


def test_published_criteria_and_shortfall() -> None:
    assert validate.published_criteria(POLICY) == CRITERIA
    assert validate.published_criteria({**POLICY, 'published_rho_cut': 0.40})['rho_cut'] == 0.40
    report = {'dispatch_size': 1.0, 'redistribution_capacity': 2.0, 'recovery_capacity': 2.0}
    assert validate.shortfall(None, 'recovery_capacity') == 0.0
    assert validate.shortfall(report, 'recovery_capacity') == 0.0
    assert validate.shortfall({**report, 'recovery_capacity': 0.25}, 'recovery_capacity') == 0.75
    assert validate.shortfall({**report, 'dispatch_size': 0.0}, 'recovery_capacity') == 0.0


def test_outcome_model_gives_the_values_worked_by_hand() -> None:
    """Nominal state: emissions are 0.12 kg per km over 120, 45 and 80 km; the
    social composite is 0.30 (1 - emissions / 50) plus the weighted route
    scores; a diversion keeps 0.55 or 0.75 of the raw waste; a refused share
    travels the diverted leg and then the cold chain."""
    model = outcome_model()
    cold = model.step(NOMINAL, 'cold_chain', 1.0)
    raw = cold['waste']
    expected = {  # route: (emissions, composite, waste / raw waste), accepted in full
        'cold_chain': (14.4, 0.6086, 1.0),
        'local_redistribute': (5.4, 0.8216, 0.55),
        'recovery': (9.6, 0.7324, 0.75)}
    for route, (emissions, composite, kept) in expected.items():
        step = model.step(NOMINAL, route, 1.0)
        assert step['carbon'] == pytest.approx(emissions, abs=1e-12)
        assert step['slca'] == pytest.approx(composite, abs=1e-12)
        assert step['waste'] == pytest.approx(kept * raw, abs=1e-15)
        assert step['ari'] == pytest.approx((1 - kept * raw) * composite * 0.70, abs=1e-12)
    refused = {  # route: (emissions, composite, waste / raw waste), refused in full
        'local_redistribute': (19.8, 0.5762, 1.375), 'recovery': (24.0, 0.5510, 5.0 / 3.0)}
    for route, (emissions, composite, lost) in refused.items():
        step = model.step(NOMINAL, route, 0.0)
        assert step['carbon'] == pytest.approx(emissions, abs=1e-12)
        assert step['slca'] == pytest.approx(composite, abs=1e-12)
        assert step['waste'] == pytest.approx(lost * raw, abs=1e-15)
    # With raw waste w, redistribution accepted at share a is worth
    # (1 - w(1.375 - 0.825 a)) (0.5762 + 0.2454 a) against (1 - w) 0.6086.
    for share in (0.0, 0.2, 0.5, 1.0):
        step = model.step(NOMINAL, 'local_redistribute', share)
        assert step['ari'] / 0.70 == pytest.approx(
            (1 - raw * (1.375 - 0.825 * share)) * (0.5762 + 0.2454 * share), abs=1e-12)
    # The penalty on the refused share scales the diverted leg.
    doubled = outcome_model(penalty_scale=2.0).step(NOMINAL, 'local_redistribute', 0.0)
    assert doubled['carbon'] == pytest.approx(2 * 5.4 + 14.4, abs=1e-12)
    assert doubled['waste'] == pytest.approx(raw * (1 + 2 * 45 / 120), abs=1e-15)


def test_outcome_model_mixes_the_refused_share_linearly() -> None:
    model = outcome_model()
    row = {'rho_outcome_environmental': 0.30, 'inventory_outcome_environmental': 13_000.0,
           'temp_outcome_environmental': 9.0, 'rh_outcome_environmental': 88.0,
           'transport_multiplier_outcome_environmental': 1.1}
    cold = model.step(row, 'cold_chain', 1.0)
    for route in ('local_redistribute', 'recovery'):
        full, none, half = (model.step(row, route, share) for share in (1.0, 0.0, 0.5))
        assert full['waste'] < cold['waste'] < none['waste']
        assert none['carbon'] > max(full['carbon'], cold['carbon'])
        assert none['slca'] < cold['slca']
        for key in ('waste', 'carbon', 'slca'):
            assert half[key] == pytest.approx(0.5 * (full[key] + none[key]), abs=1e-12)
        assert half['ari'] == pytest.approx(
            (1 - half['waste']) * half['slca'] * (1 - 0.30), abs=1e-12)
    # The refused share is the cold chain after the diverted leg.
    ratio = POLICY['km_local'] / POLICY['km_coldchain']
    refused = model.step(row, 'local_redistribute', 0.0)
    assert refused['waste'] == pytest.approx(cold['waste'] * (1 + ratio), abs=1e-12)
    assert refused['carbon'] == pytest.approx(
        cold['carbon'] + model.step(row, 'local_redistribute', 1.0)['carbon'], abs=1e-12)
    # In the variant where the refused share is lost, all of it is waste and
    # nothing else about it changes.
    lost = outcome_model(refused_share_lost=True)
    for route in ('local_redistribute', 'recovery'):
        assert lost.step(row, route, 1.0) == model.step(row, route, 1.0)
        none, half = lost.step(row, route, 0.0), lost.step(row, route, 0.5)
        assert none['waste'] == 1.0 and none['ari'] == 0.0
        assert (none['carbon'], none['slca']) == (
            model.step(row, route, 0.0)['carbon'], model.step(row, route, 0.0)['slca'])
        assert half['waste'] == pytest.approx(
            0.5 * (model.step(row, route, 1.0)['waste'] + 1.0), abs=1e-12)
    # Alternative index parameters change the score, not the physical outcome.
    equal = dict.fromkeys(('w_c', 'w_l', 'w_r', 'w_p'), 0.25)
    rescored = model.step(row, 'recovery', 1.0, weights=equal, carbon_cap=25.0)
    base = model.step(row, 'recovery', 1.0)
    assert (rescored['waste'], rescored['carbon']) == (base['waste'], base['carbon'])
    assert rescored['slca'] != base['slca']


def observed_row(rho, inventory=12_000.0, temperature=4.0, humidity=90.0):
    return {'rho_policy_observed': rho, 'inventory_policy_observed': inventory,
            'temp_policy_observed': temperature, 'rh_policy_observed': humidity}


def test_rules_reconstructed_by_the_validator() -> None:
    predictor = outcome_model()

    def rule(name, rho, criteria=CRITERIA, room=ROOM, closed=True, share=1.0, size=1.0):
        return validate.rule_action(name, observed_row(rho), criteria, room, size, share, closed,
                                    predictor)

    assert rule('always_redistribute', 0.9) == 1 and rule('always_recovery', 0.0) == 2
    assert [rule('risk_threshold', rho) for rho in (0.05, 0.10, 0.30, 0.50, 0.60)] == [0, 0, 1, 1, 2]
    assert rule('risk_threshold', 0.30, criteria='unavailable') == 0
    # Priority rule with reservation.
    assert rule('capacity_informed', 0.30) == 1
    assert rule('capacity_informed', 0.30, room={1: 0.9, 2: 2.0}) == 2
    assert rule('capacity_informed', 0.30, room={1: 0.9, 2: 0.9}) == 0
    assert rule('capacity_informed', 0.30, room={1: 0.9, 2: 0.9}, share=0.8) == 1
    assert rule('capacity_informed', 0.60) == 2 and rule('capacity_informed', 0.05) == 1
    assert rule('capacity_informed', 0.30, room='unavailable') == 0
    assert rule('capacity_informed', 0.30, room='unavailable', closed=False) == 1
    assert rule('capacity_informed', 0.30, criteria='unavailable') == 0
    # Greedy rule: partial acceptance is allowed and valued by the outcome equations.
    assert rule('greedy_informed', 0.30) == 1
    assert rule('greedy_informed', 0.30, room={1: 0.5, 2: 2.0}) == 2
    assert rule('greedy_informed', 0.30, room={1: 0.5, 2: 0.0}) == 1
    assert rule('greedy_informed', 0.30, room={1: 0.1, 2: 0.0}) == 0     # below break-even
    assert rule('greedy_informed', 0.30, room={1: 0.0, 2: 0.0}) == 0
    assert rule('greedy_informed', 0.60) == 2 and rule('greedy_informed', 0.60, room={1: 2.0, 2: 0.0}) == 0
    assert rule('greedy_informed', 0.30, room='unavailable') == 0
    assert rule('greedy_informed', 0.30, room='unavailable', closed=False) == 1
    assert rule('greedy_informed', 0.30, criteria='unavailable') == 0
    with pytest.raises(ValueError):
        rule('unknown', 0.3)


def test_validator_rules_agree_with_the_simulator_rules() -> None:
    """The validator's reconstruction and the simulator's rule give the same
    route over a grid of states, capacities and information states."""
    policy = Policy(destination_capacity_enabled=True, revised_context_design=True)
    predictor = outcome_model()
    rng = np.random.default_rng(20261112)
    compared = 0
    for _ in range(400):
        rho = float(rng.uniform(0.0, 0.8))
        inventory = float(rng.uniform(6_000.0, 20_000.0))
        temperature, humidity = float(rng.uniform(2.0, 30.0)), float(rng.uniform(70.0, 99.0))
        size = max(inventory, 0.0) / 12_000.0
        room = {1: float(rng.choice([0.0, 0.3, 0.8, 1.5, 3.0])),
                2: float(rng.choice([0.0, 0.3, 0.8, 1.5, 3.0]))}
        capacity = str(rng.choice(['available', 'available', 'unavailable']))
        criteria = str(rng.choice(['available', 'available', 'unavailable']))
        closed, share = bool(rng.integers(0, 2)), float(rng.choice([0.5, 0.8, 1.0]))
        information = {'criteria': criteria, 'capacity': capacity, 'dispatch_size': size,
                       'reservation_share': share, 'missing_confirmation_closed': closed,
                       'observed': {'inventory': inventory, 'temperature': temperature,
                                    'humidity': humidity}}
        if criteria == 'available':
            information.update(CRITERIA)
        if capacity == 'available':
            information.update(redistribution_capacity=room[1], recovery_capacity=room[2])
        row = observed_row(rho, inventory, temperature, humidity)
        for name in ('risk_threshold', 'capacity_informed', 'greedy_informed'):
            expected = action_selection.fixed_rule_action(name, rho, information, policy)
            assert validate.rule_action(
                name, row, CRITERIA if criteria == 'available' else 'unavailable',
                room if capacity == 'available' else 'unavailable', size, share, closed,
                predictor) == expected, (name, information)
            compared += 1
    assert compared == 1200


# ---------------------------------------------------------------------------
# Endpoints reconciled with the ledger
# ---------------------------------------------------------------------------
def evaluation(steps=40, capacity=True):
    """A ledger and the result the simulator would store for it.

    The result is computed with the simulator's own definitions (the RLE
    tracker, the equity function, sequential sums), not with the validator.
    """
    from src.models.resilience import RLETracker, compute_equity
    rng = np.random.default_rng(20261113)
    prior = [[-0.4, -0.3, 0.12, 0.12, 0.12], [0.3, 0.25, -0.2, -0.2, 0.08],
             [0.15, 0.1, 0.08, 0.08, -0.2]]
    final = (np.asarray(prior) + rng.uniform(-0.05, 0.05, (3, 5))).tolist()
    rows, tracker, carbon, diverted, refused = [], RLETracker(), 0.0, 0.0, 0.0
    for step in range(steps):
        action = int(rng.integers(0, 3))
        share = 1.0 if action == 0 else float(rng.choice([1.0, 0.6, 0.0]))
        rho = float(rng.uniform(0.02, 0.7))
        size = float(rng.uniform(0.6, 1.8))
        probs = rng.dirichlet([1.0, 1.0, 1.0]).tolist()
        row = {
            'step_index': step, 'hour': step / 4.0, 'action': validate.ACTIONS[action],
            'action_idx': action, 'ari': float(rng.uniform(0.2, 0.7)),
            'waste': float(rng.uniform(0.01, 0.2)), 'slca': float(rng.uniform(0.4, 0.9)),
            'carbon_kg': float(rng.uniform(5.0, 25.0)), 'probs': probs,
            'rho': rho + 0.01, 'rho_policy_observed': rho + 0.01,
            'rho_outcome_environmental': rho,
            'inventory_outcome_environmental': 12_000.0 * size,
            'inventory_policy_observed': 11_900.0 * size,
            'temp_outcome_environmental': float(rng.uniform(3.0, 25.0)),
            'temp_policy_observed': float(rng.uniform(3.0, 25.0)),
            'rh_outcome_environmental': float(rng.uniform(80.0, 99.0)),
            'rh_policy_observed': float(rng.uniform(80.0, 99.0)),
            'demand_outcome_environmental': float(rng.uniform(5.0, 25.0)),
            'transport_multiplier_outcome_environmental': float(rng.uniform(0.8, 1.3)),
            'decision_latency_ms': float(rng.uniform(1.0, 30.0)),
            'guard': {'removed_routes': [1] if step % 5 == 0 else [],
                      'sampled_action_pre_guard': 1 if step % 7 == 0 else action},
            'effective_context_theta': final}
        if capacity:
            row.update({'accepted_share': share, 'dispatch_size': size,
                        'redistribution_capacity_before': float(rng.uniform(0.0, 8.0)),
                        'recovery_capacity_before': float(rng.uniform(0.0, 4.0))})
        rows.append(row)
        tracker.update(rho, row['action'], share if capacity else 1.0)
        carbon += row['carbon_kg']
        if capacity and action != 0:
            diverted += size
            refused += (1.0 - share) * size
    latency = np.array([row['decision_latency_ms'] for row in rows])
    social = [row['slca'] for row in rows]
    metadata = {'latent_environment_sha256': 'a' * 64, 'effective_k_ref': 0.0123,
                'effective_Ea_R': 8100.0, 'scenario_onset_offset_hours': 1.5}
    result = {
        'ari': float(np.mean([row['ari'] for row in rows])),
        'waste': float(np.mean([row['waste'] for row in rows])),
        'slca': float(np.mean(social)), 'carbon': float(carbon), 'rle': float(tracker.rle),
        'equity': float(compute_equity(social)),
        'mean_decision_latency_ms': float(np.mean(latency)),
        'p95_decision_latency_ms': float(np.percentile(latency, 95)),
        **{name: [row[field] for row in rows]
           for name, field in validate.RECONCILED_TRACES.items() if field in rows[0]},
        **metadata,
        'learner_summary': {'initial_theta': prior, 'final_theta': final,
                            'sign_reversal_count': 0}}
    if capacity:
        model = {'enabled': True, 'kappa_lr': 0.5, 'refused_share_lost': False}
        metadata['destination_capacity_model'] = dict(model)
        result.update({
            'destination_capacity_model': dict(model), 'diverted_volume': float(diverted),
            'refused_volume': float(refused),
            'refused_share_of_diverted_volume': float(refused / diverted),
            'guard_removed_step_fraction': sum(bool(r['guard']['removed_routes']) for r in rows)
            / steps,
            'guard_changed_action_fraction': sum(
                r['guard']['sampled_action_pre_guard'] != r['action_idx'] for r in rows) / steps})
    return metadata, rows, result, {'initial_context_matrix': prior}


def reconcile(metadata, rows, result, parameters, mode='agribrain'):
    return validate.reconcile_endpoints(metadata, rows, result, mode, parameters, 'ledger')


def test_endpoints_are_rebuilt_from_the_ledger() -> None:
    metadata, rows, result, parameters = evaluation()
    summary = reconcile(metadata, rows, result, parameters)
    assert summary['metrics'] == [
        'ari', 'waste', 'slca', 'equity', 'carbon', 'rle', 'diverted_volume', 'refused_volume',
        'refused_share_of_diverted_volume', 'guard_removed_step_fraction',
        'guard_changed_action_fraction', 'mean_decision_latency_ms', 'p95_decision_latency_ms']
    assert summary['traces'] == list(validate.RECONCILED_TRACES)
    assert summary['header_fields'] == [*validate.RECONCILED_HEADER, 'initial_context_weights']
    assert summary['trace_values'] == 40 * (len(validate.RECONCILED_TRACES) + 2)   # probs are 3-vectors
    assert summary['context_weight_records'] == 40
    assert summary['largest_metric_difference'] <= 1e-15
    # The submitted configuration has no destination constraints, no guard record
    # that the result summarizes, and no initial matrix recorded with the setting.
    metadata, rows, result, _ = evaluation(capacity=False)
    summary = reconcile(metadata, rows, result, {'initial_context_matrix': None})
    assert summary['metrics'] == ['ari', 'waste', 'slca', 'equity', 'carbon', 'rle',
                                  'mean_decision_latency_ms', 'p95_decision_latency_ms']
    assert 'accepted_share_trace' not in summary['traces'] and len(summary['traces']) == 18
    assert summary['header_fields'] == validate.RECONCILED_HEADER[:4]
    # What is compared between the run and a later verification leaves out the
    # size of the largest difference.
    total = {'ledgers': 1, 'metrics': 13, 'largest_metric_difference': 1e-16,
             'by_mode': {'agribrain': summary}}
    structure = validate.reconciled_structure(total)
    assert 'largest_metric_difference' not in structure
    assert 'largest_metric_difference' not in structure['by_mode']['agribrain']
    assert structure['by_mode']['agribrain']['metrics'] == summary['metrics']


@pytest.mark.parametrize('name', [
    'ari', 'waste', 'slca', 'equity', 'carbon', 'rle', 'diverted_volume', 'refused_volume',
    'refused_share_of_diverted_volume', 'guard_removed_step_fraction',
    'guard_changed_action_fraction', 'mean_decision_latency_ms', 'p95_decision_latency_ms'])
def test_a_changed_metric_is_detected(name) -> None:
    metadata, rows, result, parameters = evaluation()
    # A difference far below anything reported is still a mismatch.
    with pytest.raises(ValueError, match=f'ledger: {name} is '):
        reconcile(metadata, rows, {**result, name: result[name] + 1e-9}, parameters)
    with pytest.raises(ValueError, match=f'ledger: {name} is '):
        reconcile(metadata, rows, {**result, name: result[name] * (1 - 1e-9) - 1e-10}, parameters)
    # Rounding in the last place is not.
    reconcile(metadata, rows, {**result, name: result[name] * (1 + 2e-16)}, parameters)


@pytest.mark.parametrize('name', list(validate.RECONCILED_TRACES))
def test_a_changed_trace_is_detected(name) -> None:
    metadata, rows, result, parameters = evaluation()
    changed = [list(v) if isinstance(v, list) else v for v in result[name]]
    if name == 'action_trace':
        changed[17] = (changed[17] + 1) % 3
    elif name == 'prob_trace':
        changed[17][0] += 1e-9
    else:
        changed[17] += 1e-9
    with pytest.raises(ValueError, match=f'{name} is not the series of the ledger'):
        reconcile(metadata, rows, {**result, name: changed}, parameters)
    with pytest.raises(ValueError, match=f'{name} is not the series of the ledger'):
        reconcile(metadata, rows, {**result, name: result[name][:-1]}, parameters)
    # A trace the result carries must be recorded in the ledger.
    field = validate.RECONCILED_TRACES[name]
    stripped = [{k: v for k, v in row.items() if k != field} for row in rows]
    if field in ('accepted_share', 'dispatch_size', 'slca', 'ari', 'waste', 'carbon_kg',
                 'rho_outcome_environmental', 'action_idx'):
        return  # these fields also feed a metric, which fails first
    with pytest.raises(ValueError, match=f'does not record {field}'):
        reconcile(metadata, stripped, result, parameters)


def test_changed_header_fields_and_context_weights_are_detected() -> None:
    metadata, rows, result, parameters = evaluation()
    for key, value in (('latent_environment_sha256', 'b' * 64), ('effective_k_ref', 0.0124),
                       ('effective_Ea_R', 8100.5), ('scenario_onset_offset_hours', 0.0),
                       ('destination_capacity_model', {'enabled': True, 'kappa_lr': 0.25,
                                                       'refused_share_lost': False})):
        with pytest.raises(ValueError, match=f'{key} differs from the header'):
            reconcile(metadata, rows, {**result, key: value}, parameters)
        with pytest.raises(ValueError, match=f'{key} differs from the header'):
            reconcile({k: v for k, v in metadata.items() if k != key}, rows, result, parameters)
    learner = result['learner_summary']
    final = [list(row) for row in learner['final_theta']]
    final[1][2] += 1e-9
    with pytest.raises(ValueError, match='context weights of the decision'):
        reconcile(metadata, rows, {**result, 'learner_summary': {**learner, 'final_theta': final}},
                  parameters)
    other = [list(row) for row in learner['initial_theta']]
    other[0][0] = -0.41
    with pytest.raises(ValueError, match='initial context weights'):
        reconcile(metadata, rows, result, {'initial_context_matrix': other})
    with pytest.raises(ValueError, match='sign reversal count'):
        reconcile(metadata, rows,
                  {**result, 'learner_summary': {**learner, 'sign_reversal_count': 2}}, parameters)
    # A mode that starts from zero weights is not compared with the prior of the setting.
    zero = {**learner, 'initial_theta': np.zeros((3, 5)).tolist(), 'sign_reversal_count': 4}
    summary = reconcile(metadata, rows, {**result, 'learner_summary': zero}, parameters,
                        mode='feature_inputs_zero')
    assert 'initial_context_weights' not in summary['header_fields']
    assert summary['context_weight_records'] == 40


def test_stored_reference_of_the_submitted_configuration_is_complete() -> None:
    reference = study.load(study.REPO / 'reproduction/primary_reference.json')['values']
    assert set(reference) == set(map(str, study.ORIGINAL_SEEDS))
    for cells in reference.values():
        assert set(cells) == set(study.SCENARIOS)
        for modes in cells.values():
            assert {'no_context', 'agribrain'} <= set(modes)
            assert all(len(modes[m]['action_trace']) == 288 for m in ('no_context', 'agribrain'))
