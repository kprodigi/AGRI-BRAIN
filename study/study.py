"""Study of the revised design.

One design file defines every run reported for the revised manuscript:

primary      the comparison modes and rule references under the revised
             context design and the outcome model with destination capacity
             and acceptance criteria, under both missing-confirmation rules
             (fail-closed and fail-open), on the final evaluation seeds
submitted    the submitted configuration on its own seeds, reproduced as a
             regression check against the stored primary results and run with
             the fixed-route references, and the same configuration without
             scenario terms in the retrieval query
diagnostic   the revised design on the seeds of the submitted study, which
             informed the design and are therefore not evaluation seeds
weights      the weight-magnitude settings of the submitted sensitivity design,
             applied to the revised design
structural   one-factor changes to the structural assumptions
content      changes to the acceptance document: another cutoff than the one
             the destination applies, another identifier, no document

A task is one (setting, seed, scenario) cell and runs every mode of its setting
on the same latent environment. Commands: prepare, task, verify, pilot-check,
missing.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Final evaluation seeds. They are not run before the design is fixed in a
# tagged commit, and never in a development runtime.
FINAL_SEEDS = list(range(5001, 5021))
# Seeds of the submitted study. Rule policies were run on them while the
# outcome model was designed, so they are not evaluation seeds.
ORIGINAL_SEEDS = [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505, 606, 707, 808, 909,
                  1010, 1111, 1212, 1313, 1414, 1515]
DEVELOPMENT_SEEDS = [3, 11]
SEED_SETS = {'final': FINAL_SEEDS, 'original': ORIGINAL_SEEDS}
SCENARIOS = ['heatwave', 'overproduction', 'cyber_outage', 'adaptive_pricing', 'baseline']
RULES = ['rule_always_redistribute', 'rule_always_recovery', 'rule_risk_threshold',
         'rule_capacity_informed', 'rule_greedy_informed']
PRIMARY_MODES = [
    'static', 'no_context', 'adjustment_only',
    'eligibility_only', 'agribrain_eligibility_no_peer',
    'reservation_only', 'agribrain_no_peer',
    'agribrain', 'agribrain_eligibility', 'agribrain_no_guard',
    'pirag_only', 'mcp_only', 'agribrain_fixed_context', 'agribrain_sign_unconstrained',
    'feature_inputs_prior', 'feature_inputs_zero', 'constant_adjustment', 'no_pinn',
    *RULES,
]
SENSITIVITY_MODES = ['static', 'no_context', 'agribrain', 'agribrain_eligibility', *RULES]
SUBMITTED_MODES = ['static', 'no_context', 'agribrain',
                   'rule_always_redistribute', 'rule_always_recovery']
# Modes whose guard needs a capacity confirmation that they request.
RESERVATION_MODES = ['reservation_only', 'agribrain_no_peer', 'agribrain', 'mcp_only',
                     'agribrain_fixed_context', 'agribrain_sign_unconstrained', 'no_pinn']
WEIGHT_GROUPS = ['base_policy', 'tool_prior', 'retrieval_prior',
                 'w_c', 'w_l', 'w_r', 'w_p', 'eta', 'eta_rho']
DEFAULT_SOCIAL = [0.30, 0.20, 0.25, 0.25]
METRICS = ['ari', 'waste', 'rle', 'slca', 'carbon', 'equity']
TRACES = ['ari_trace', 'waste_trace', 'rho_trace', 'action_trace', 'prob_trace',
          'carbon_trace', 'slca_trace']
ENV_TRACES = ['hours', 'temp_outcome_environmental_trace', 'rh_outcome_environmental_trace',
              'rho_outcome_environmental_trace', 'inventory_outcome_environmental_trace',
              'demand_outcome_environmental_trace',
              'transport_multiplier_outcome_environmental_trace']
REVISED = {'destination_capacity_enabled': True, 'revised_context_design': True}
FAIL_OPEN = {'missing_confirmation_closed': False}
# ``full`` keeps every adaptation and evaluation ledger and episode archive;
# ``evaluation`` keeps the compressed evaluation ledgers.
RETAINED_EVIDENCE = {'primary': 'full', 'submitted': 'full', 'diagnostic': 'evaluation',
                     'weights': 'evaluation', 'structural': 'evaluation',
                     'content': 'evaluation'}
SENSITIVITY_FAMILIES = ['weights', 'structural', 'content']
# Cells executed and checked before the bulk of the study is released. Each is
# a cell of the study at the first seed of its setting.
PILOT_CELLS = [
    ('main_fail_closed', 'heatwave'), ('main_fail_closed', 'cyber_outage'),
    ('main_fail_open', 'cyber_outage'),
    ('submitted', 'heatwave'), ('submitted_without_scenario_terms', 'heatwave'),
    ('diagnostic_fail_closed', 'heatwave'), ('diagnostic_unlimited_capacity', 'heatwave'),
    ('base_policy_minus20', 'heatwave'), ('merit_order_exchanged', 'heatwave'),
    ('learning_objective_step_performance', 'heatwave'),
    ('refused_share_lost', 'heatwave'), ('reservation_share_050', 'heatwave'),
    ('acceptance_document_cutoff_040', 'heatwave'),
    ('acceptance_document_renamed', 'heatwave'),
    ('acceptance_document_absent', 'heatwave'),
]
# Pilot cells executed a second time, independently, to check determinism.
REPEATED_PILOT_CELLS = [('main_fail_closed', 'heatwave'), ('submitted', 'heatwave')]
# Graded series around the fail-closed main setting: (setting, level), in
# increasing order.
GRADED_SERIES = {
    'refusal_penalty': [
        ('refusal_penalty_half', '0.5'), ('main_fail_closed', '1'),
        ('refusal_penalty_double', '2'), ('refusal_penalty_quadruple', '4'),
        ('refused_share_lost', 'refused share lost')],
    'redistribution_capacity': [
        ('redistribution_capacity_quarter', '0.25'), ('redistribution_capacity_half', '0.5'),
        ('main_fail_closed', '1'), ('redistribution_capacity_double', '2'),
        ('redistribution_capacity_quadruple', '4'),
        ('unlimited_capacity_no_criteria', 'unlimited, no acceptance criteria')],
    'reservation_share': [
        ('reservation_share_050', '0.5'), ('reservation_share_080', '0.8'),
        ('main_fail_closed', '1')],
}
SAVING_DEFAULT = {'local_redistribute': 0.45, 'recovery': 0.25}
SOCIAL_DEFAULT = {
    'cold_chain': {'L': 0.60, 'R': 0.55, 'P': 0.55},
    'local_redistribute': {'L': 0.82, 'R': 0.78, 'P': 0.78},
    'recovery': {'L': 0.70, 'R': 0.72, 'P': 0.68},
}
UNLIMITED = {'capacity_kappa_lr': 1.0e9, 'capacity_kappa_rec': 1.0e9,
             'capacity_bmax_lr': 1.0e9, 'capacity_bmax_rec': 1.0e9,
             'destination_acceptance_enabled': False,
             'published_rho_at': 0.0, 'published_rho_cut': 1.0}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def social_weights(factors):
    names = ['w_c', 'w_l', 'w_r', 'w_p']
    weights = [v * factors[k] for v, k in zip(DEFAULT_SOCIAL, names, strict=True)]
    total = sum(weights)
    return dict(zip(names, [v / total for v in weights], strict=True))


def weight_setting(setting_id, factors):
    if set(factors) != set(WEIGHT_GROUPS):
        raise ValueError('Incorrect factor set')
    if any(not 0.8 <= float(x) <= 1.2 for x in factors.values()):
        raise ValueError('Factors out of bounds')
    policy = {**REVISED, **social_weights(factors),
              'eta': 0.5 * factors['eta'], 'eta_rho': 0.5 * factors['eta_rho']}
    return {'id': setting_id, 'family': 'weights', 'seeds': 'final',
            'modes': SENSITIVITY_MODES, 'policy': policy, 'factors': dict(factors),
            'patches': {'base_policy_scale': factors['base_policy'],
                        'tool_prior_scale': factors['tool_prior'],
                        'retrieval_prior_scale': factors['retrieval_prior']}}


def sensitivity_setting(family, setting_id, assumption, policy=None, patches=None):
    return {'id': setting_id, 'family': family, 'seeds': 'final',
            'modes': SENSITIVITY_MODES, 'assumption': assumption,
            'policy': {**REVISED, **(policy or {})}, 'patches': dict(patches or {})}


def structural_setting(setting_id, assumption, policy=None, patches=None):
    return sensitivity_setting('structural', setting_id, assumption, policy, patches)


def blended_social(share):
    """Route social scores moved toward the cold-chain values by ``share``."""
    base = SOCIAL_DEFAULT['cold_chain']
    return {route: {k: round(v + share * (base[k] - v), 6) for k, v in scores.items()}
            for route, scores in SOCIAL_DEFAULT.items()}


def design():
    settings = [
        {'id': 'main_fail_closed', 'family': 'primary', 'seeds': 'final',
         'modes': PRIMARY_MODES, 'policy': dict(REVISED), 'patches': {}},
        {'id': 'main_fail_open', 'family': 'primary', 'seeds': 'final',
         'modes': PRIMARY_MODES, 'policy': {**REVISED, **FAIL_OPEN}, 'patches': {}},
        {'id': 'submitted', 'family': 'submitted', 'seeds': 'original',
         'modes': SUBMITTED_MODES, 'policy': {}, 'patches': {}},
        {'id': 'submitted_without_scenario_terms', 'family': 'submitted', 'seeds': 'original',
         'modes': ['agribrain'], 'policy': {'query_scenario_terms': False}, 'patches': {}},
        {'id': 'diagnostic_fail_closed', 'family': 'diagnostic', 'seeds': 'original',
         'modes': PRIMARY_MODES, 'policy': dict(REVISED), 'patches': {}},
        {'id': 'diagnostic_unlimited_capacity', 'family': 'diagnostic', 'seeds': 'original',
         'modes': SENSITIVITY_MODES, 'policy': {**REVISED, **UNLIMITED}, 'patches': {}},
    ]
    for group in WEIGHT_GROUPS:
        for factor, label in [(0.8, 'minus20'), (1.2, 'plus20')]:
            values = dict.fromkeys(WEIGHT_GROUPS, 1.0)
            values[group] = factor
            settings.append(weight_setting(f'{group}_{label}', values))
    # Fixed opposing profiles, declared before any result; not a distribution.
    signs = [1, -1, 1, -1, 1, -1, 1, -1, 1]
    for amplitude in [0.10, 0.20]:
        for direction, label in [(1, 'A'), (-1, 'B')]:
            values = {k: round(1 + amplitude * direction * s, 6)
                      for k, s in zip(WEIGHT_GROUPS, signs, strict=True)}
            settings.append(weight_setting(f'joint{round(amplitude * 100)}_{label}', values))
    s = structural_setting
    settings += [
        s('capacity_report_one_hour_old', 'capacity report age',
          {'capacity_report_age_steps': 4}),
        s('redistribution_capacity_quarter', 'redistribution capacity',
          {'capacity_kappa_lr': 0.125}),
        s('redistribution_capacity_half', 'redistribution capacity', {'capacity_kappa_lr': 0.25}),
        s('redistribution_capacity_double', 'redistribution capacity', {'capacity_kappa_lr': 1.0}),
        s('redistribution_capacity_quadruple', 'redistribution capacity',
          {'capacity_kappa_lr': 2.0}),
        s('recovery_capacity_half', 'recovery capacity', {'capacity_kappa_rec': 0.125}),
        s('recovery_capacity_double', 'recovery capacity', {'capacity_kappa_rec': 0.5}),
        s('buffers_half', 'buffers', {'capacity_bmax_lr': 4.0, 'capacity_bmax_rec': 2.0}),
        s('buffers_double', 'buffers', {'capacity_bmax_lr': 16.0, 'capacity_bmax_rec': 8.0}),
        s('at_risk_threshold_005', 'at-risk threshold', {'acceptance_rho_at': 0.05}),
        s('at_risk_threshold_015', 'at-risk threshold', {'acceptance_rho_at': 0.15}),
        s('marketability_cutoff_040', 'marketability cutoff', {'acceptance_rho_cut': 0.40}),
        s('marketability_cutoff_060', 'marketability cutoff', {'acceptance_rho_cut': 0.60}),
        s('refusal_penalty_half', 'refusal penalty', {'refusal_penalty_scale': 0.5}),
        s('refusal_penalty_double', 'refusal penalty', {'refusal_penalty_scale': 2.0}),
        s('refusal_penalty_quadruple', 'refusal penalty', {'refusal_penalty_scale': 4.0}),
        s('refused_share_lost', 'refusal penalty', {'refused_share_lost': True}),
        s('reservation_share_050', 'share of the dispatch reserved', {'reservation_share': 0.5}),
        s('reservation_share_080', 'share of the dispatch reserved', {'reservation_share': 0.8}),
        s('saving_minus20', 'waste-saving coefficients', None,
          {'saving': {k: round(0.8 * v, 6) for k, v in SAVING_DEFAULT.items()}}),
        s('saving_plus20', 'waste-saving coefficients', None,
          {'saving': {k: round(1.2 * v, 6) for k, v in SAVING_DEFAULT.items()}}),
        s('saving_equalized', 'waste-saving coefficients', None,
          {'saving': {'local_redistribute': 0.35, 'recovery': 0.35}}),
        s('social_premium_half', 'route social scores', None, {'social': blended_social(0.5)}),
        s('social_scores_equal', 'route social scores', None, {'social': blended_social(1.0)}),
        s('redistribution_distance_80', 'route distances', {'km_local': 80.0}),
        s('distances_equal', 'route distances', {'km_local': 120.0, 'km_recovery': 120.0}),
        s('condition_signs_reversed', 'context-weight signs', None,
          {'context_sign_reversal': [0, 1]}),
        s('merit_order_exchanged', 'route merit order',
          {'km_local': 80.0, 'km_recovery': 45.0},
          {'saving': {'local_redistribute': 0.25, 'recovery': 0.45},
           'social': {'cold_chain': SOCIAL_DEFAULT['cold_chain'],
                      'local_redistribute': SOCIAL_DEFAULT['recovery'],
                      'recovery': SOCIAL_DEFAULT['local_redistribute']}}),
        s('learning_objective_step_performance', 'learning objective',
          {'reward_objective': 'step_performance'}),
        s('strict_redistribution', 'redistribution limited to at-risk product',
          {'acceptance_strict_redistribution': True}),
        s('unlimited_capacity_no_criteria', 'no destination constraints', UNLIMITED),
    ]
    c = sensitivity_setting
    settings += [
        # The document states 0.40; the destination still applies 0.50.
        c('content', 'acceptance_document_cutoff_040',
          'marketability cutoff stated in the acceptance document',
          {'published_rho_cut': 0.40}),
        # The same content under another identifier and source label.
        c('content', 'acceptance_document_renamed', 'identifier of the acceptance document',
          {'acceptance_document_id': 'outlet_intake_rules'}),
        # No acceptance document in the corpus, under each missing-confirmation rule.
        c('content', 'acceptance_document_absent', 'acceptance document absent, fail-closed',
          {'acceptance_document_present': False}),
        c('content', 'acceptance_document_absent_fail_open',
          'acceptance document absent, fail-open',
          {'acceptance_document_present': False, **FAIL_OPEN}),
    ]
    ids = [setting['id'] for setting in settings]
    counts = {family: sum(x['family'] == family for x in settings)
              for family in RETAINED_EVIDENCE}
    assert len(ids) == len(set(ids)) == 63
    assert counts == {'primary': 2, 'submitted': 2, 'diagnostic': 2, 'weights': 22,
                      'structural': 31, 'content': 4}
    return settings


def tasks(seed_sets=None):
    seed_sets = SEED_SETS if seed_sets is None else seed_sets
    return [{'index': i, 'setting': setting['id'], 'seed': seed, 'scenario': scenario}
            for i, (setting, seed, scenario) in enumerate(
                (s, n, c) for s in design() for n in seed_sets[s['seeds']] for c in SCENARIOS)]


def setting_seeds(setting_id, seed_sets=None):
    seed_sets = SEED_SETS if seed_sets is None else seed_sets
    return list(seed_sets[next(s for s in design() if s['id'] == setting_id)['seeds']])


def task_index(setting_id, seed, scenario, seed_sets=None):
    for task in tasks(seed_sets):
        if (task['setting'], task['seed'], task['scenario']) == (setting_id, seed, scenario):
            return task['index']
    raise ValueError('No such task')


def pilot_indices(seed_sets=None, cells=None):
    return [task_index(setting_id, setting_seeds(setting_id, seed_sets)[0], scenario, seed_sets)
            for setting_id, scenario in (PILOT_CELLS if cells is None else cells)]


def episode_counts(modes):
    from src.models.mode_capabilities import capabilities_for
    caps = [capabilities_for(m) for m in modes]
    return {'groups': len(modes),
            'episodes': sum(c.episode_count for c in caps),
            'adaptation': sum(c.adaptation_episode_count for c in caps),
            'final': len(modes)}


# ---------------------------------------------------------------------------
# Source identity and environment
# ---------------------------------------------------------------------------
def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def source_identity(development):
    commit = git('rev-parse', 'HEAD')
    # Concurrent tasks read the same checkout; none may take the index lock.
    dirty = git('--no-optional-locks', 'status', '--porcelain')
    if dirty and not development:
        raise ValueError('The study must run from a clean checkout of a commit')
    return {'commit': commit, 'clean': not bool(dirty),
            'harness_sha256': canonical({n: sha(HERE / n) for n in ['study.py', 'validate.py']})}


def configure(development_runtime=None):
    for p in [REPO, REPO / 'agribrain/backend', REPO / 'hpc']:
        sys.path.insert(0, str(p))
    if development_runtime:
        sys.path.insert(0, str(Path(development_runtime).resolve()))
    from hpc.validate_publication_env import EXPECTED, MUST_BE_UNSET
    for key in MUST_BE_UNSET:
        os.environ.pop(key, None)
    os.environ.update(EXPECTED)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    return development_runtime is not None


def runtime_receipt(development):
    from hpc.capture_publication_environment import (
        _core_identity,
        _installed_distribution_pairs,
        _locked_versions,
        _validate_distribution_set,
    )
    pairs = _installed_distribution_pairs()
    receipt = {'python': sys.version, 'platform': platform.platform(),
               'development_only': development,
               'distributions': sorted(f'{n}=={v}' for n, v in pairs)}
    if not development:
        if sys.version_info[:2] != (3, 11):
            raise ValueError('The study requires Python 3.11')
        if sys.prefix == sys.base_prefix:
            raise ValueError('The study requires an isolated environment')
        report, errors = _validate_distribution_set(pairs, _locked_versions(), _core_identity())
        if errors:
            raise ValueError('; '.join(errors))
        receipt['lock_validation'] = report
    return receipt


def refuse_final_seeds_in_development(seed, development):
    """The final evaluation seeds are run only by the study itself."""
    if development and seed in FINAL_SEEDS:
        raise ValueError('The final evaluation seeds are not run in a development runtime')


def prepare(output, development, development_seeds=False):
    output = Path(output).resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError('Results must be outside the repository')
    if development_seeds and not development:
        raise ValueError('Only a development run may use the development seeds')
    seed_sets = (dict.fromkeys(SEED_SETS, DEVELOPMENT_SEEDS) if development_seeds
                 else {name: list(seeds) for name, seeds in SEED_SETS.items()})
    identity = source_identity(development)
    receipt = runtime_receipt(development)
    output.mkdir(parents=True, exist_ok=False)
    settings = design()
    all_tasks = tasks(seed_sets)
    save(output / 'settings.json', settings)
    save(output / 'tasks.json', all_tasks)
    (output / 'pilot_indices.txt').write_text(
        ''.join(f'{i}\n' for i in pilot_indices(seed_sets)), encoding='utf-8', newline='\n')
    (output / 'repeat_indices.txt').write_text(
        ''.join(f'{i}\n' for i in pilot_indices(seed_sets, REPEATED_PILOT_CELLS)),
        encoding='utf-8', newline='\n')
    counts = {'settings': len(settings), 'tasks': len(all_tasks), 'episodes': 0}
    by_setting = {s['id']: s for s in settings}
    for task in all_tasks:
        counts['episodes'] += episode_counts(by_setting[task['setting']]['modes'])['episodes']
    plan = {'run_tag': output.name, 'source': identity, 'development_only': development,
            'seed_sets': seed_sets,
            'settings_sha256': sha(output / 'settings.json'),
            'tasks_sha256': sha(output / 'tasks.json'),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'retained_evidence': RETAINED_EVIDENCE, 'counts': counts}
    save(output / 'plan.json', plan)
    save(output / 'setup_environment.json', receipt)
    print(json.dumps(counts))


def read_plan(output, development):
    output = Path(output).resolve()
    plan = load(output / 'plan.json')
    if plan['development_only'] != development:
        raise ValueError('Cannot mix development results and results of the study')
    identity = source_identity(development)
    if not development and identity != plan['source']:
        raise ValueError('Source or harness changed after preparation')
    for filename, key in [('settings.json', 'settings_sha256'), ('tasks.json', 'tasks_sha256')]:
        if sha(output / filename) != plan[key]:
            raise ValueError('Changed design or tasks')
    if (load(output / 'settings.json') != design()
            or load(output / 'tasks.json') != tasks(plan['seed_sets'])):
        raise ValueError('Unrecognised design or tasks')
    if not development and plan['seed_sets'] != SEED_SETS:
        raise ValueError('Seeds of the plan are not those of the design')
    return plan


# ---------------------------------------------------------------------------
# Applying a setting
# ---------------------------------------------------------------------------
@contextmanager
def overridden(gr, setting, scenario, output):
    """Apply a setting to the simulator for one task and restore it afterwards.

    The setting changes process-wide model state, so each task runs in its own
    process.
    """
    import numpy as np
    from pirag import context_design
    from src.models import action_selection, slca, waste
    from src.models.policy import Policy
    patches = setting['patches']
    revised = bool(setting['policy'].get('revised_context_design', False))
    old = (gr.Policy, gr.SCENARIOS, gr.RESULTS_DIR, action_selection.THETA.copy(),
           context_design.THETA_CONTEXT_REVISED.copy(), dict(waste.SAVE_FLOOR),
           deepcopy(slca._ACTION_BASES))
    prior = old[4].copy()
    tool = list(context_design.TOOL_FEATURE_INDICES)
    retrieval = list(context_design.RETRIEVAL_FEATURE_INDICES)
    prior[:, tool] *= patches.get('tool_prior_scale', 1.0)
    prior[:, retrieval] *= patches.get('retrieval_prior_scale', 1.0)
    reversed_columns = list(patches.get('context_sign_reversal', []))
    prior[:, reversed_columns] *= -1.0
    try:
        gr.Policy = lambda: Policy(**setting['policy'])
        gr.SCENARIOS = [scenario]
        gr.RESULTS_DIR = output / 'auxiliary'
        action_selection.THETA = old[3] * patches.get('base_policy_scale', 1.0)
        context_design.THETA_CONTEXT_REVISED = prior
        # In-place updates, so that every module holding a reference sees them.
        waste.SAVE_FLOOR.update(patches.get('saving', {}))
        for route, scores in patches.get('social', {}).items():
            slca._ACTION_BASES[route].update(scores)
        unchanged = [c for c in range(5) if c not in reversed_columns]
        assert np.array_equal(np.sign(prior[:, unchanged]), np.sign(old[4][:, unchanged]))
        yield {'policy': Policy(**setting['policy']).model_dump(),
               'initial_context_matrix': prior.tolist() if revised else None,
               'base_policy_matrix_before_seed_perturbation': action_selection.THETA.tolist(),
               'saving_coefficients': dict(waste.SAVE_FLOOR),
               'route_social_scores': deepcopy(slca._ACTION_BASES)}
    finally:
        gr.Policy, gr.SCENARIOS, gr.RESULTS_DIR = old[0], old[1], old[2]
        action_selection.THETA = old[3]
        context_design.THETA_CONTEXT_REVISED = old[4]
        waste.SAVE_FLOOR.clear()
        waste.SAVE_FLOOR.update(old[5])
        for route, scores in old[6].items():
            slca._ACTION_BASES[route].clear()
            slca._ACTION_BASES[route].update(scores)


# ---------------------------------------------------------------------------
# Checks on a completed task
# ---------------------------------------------------------------------------
def check_results(results, modes):
    import numpy as np
    from src.models.mode_capabilities import capabilities_for
    if set(results) != set(modes):
        raise ValueError('Mode set differs')
    if len({r['latent_environment_sha256'] for r in results.values()}) != 1:
        raise ValueError('Unpaired environments')
    for mode, r in results.items():
        caps = capabilities_for(mode)
        if r['episode_index'] != 3 or r['learning_enabled']:
            raise ValueError('Evaluation not frozen')
        if caps.learned and not r['learner_freeze_summary']['learners_frozen']:
            raise ValueError('Learner not frozen')
        if not all(np.isfinite(r[k]) for k in METRICS):
            raise ValueError('Nonfinite metric')
        if len(r['action_trace']) != 288:
            raise ValueError('Incorrect number of decisions')
        if abs(np.mean(r['ari_trace']) - r['ari']) > 1e-12:
            raise ValueError('ARI/trace disagreement')
        probs = np.array(r['prob_trace'])
        if (probs.shape != (288, 3) or not np.isfinite(probs).all() or np.any(probs < 0)
                or not np.allclose(probs.sum(1), 1, atol=1e-10, rtol=0)):
            raise ValueError('Invalid routing probabilities')
    if 'static' in results and set(results['static']['action_trace']) != {0}:
        raise ValueError('Static not fixed cold chain')
    for mode, action in (('rule_always_redistribute', 1), ('rule_always_recovery', 2)):
        if mode in results and set(results[mode]['action_trace']) != {action}:
            raise ValueError(f'{mode} is not a fixed route')


def reference_check(results, task):
    """The submitted configuration must reproduce the stored primary results."""
    import numpy as np
    reference = load(REPO / 'reproduction/primary_reference.json')['values'].get(
        str(task['seed']), {}).get(task['scenario'])
    if task['setting'] != 'submitted' or reference is None:
        return {'available': False}
    errors = {}
    for mode in ['no_context', 'agribrain']:
        ref, r = reference[mode], results[mode]
        if r['action_trace'] != ref['action_trace']:
            raise ValueError('Submitted configuration: routing differs from the stored primary run')
        keys = METRICS + ['ari_trace', 'prob_trace'] + ENV_TRACES
        errors[mode] = max(float(np.max(np.abs(np.asarray(r[k]) - np.asarray(ref[k])))) for k in keys)
        if errors[mode] > 1e-10:
            raise ValueError(f'Submitted configuration: output differs from the stored primary run: {errors}')
    return {'available': True, 'max_abs_error': errors, 'all_576_adaptive_actions_equal': True}


def retain_evaluation_ledgers(attempt, modes, scenario):
    """Keep the evaluation ledgers, compressed, and drop the other evidence files.

    The hashes of everything that is dropped stay in the evidence manifest.
    """
    kept = attempt / 'evaluation_ledgers'
    kept.mkdir()
    for mode in modes:
        source = attempt / 'decision_ledgers' / f'{mode}__{scenario}.jsonl'
        with source.open('rb') as plain, gzip.GzipFile(
                kept / f'{source.name}.gz', 'wb', compresslevel=6, mtime=0) as packed:
            shutil.copyfileobj(plain, packed)
    shutil.rmtree(attempt / 'decision_ledgers')
    shutil.rmtree(attempt / 'auxiliary', ignore_errors=True)


def run_task(output, index, development):
    import validate as revised_validation

    from hpc.validate_complete_episode_evidence import validate_complete_evidence
    from mvp.simulation import generate_results as gr
    from mvp.simulation.benchmarks.episode_archive import to_json_native
    output = Path(output).resolve()
    plan = read_plan(output, development)
    all_tasks = tasks(plan['seed_sets'])
    if not 0 <= index < len(all_tasks):
        raise ValueError('Task index outside design')
    task = all_tasks[index]
    refuse_final_seeds_in_development(task['seed'], development)
    setting = next(s for s in design() if s['id'] == task['setting'])
    modes = setting['modes']
    dest = output / 'tasks' / f'{index:04d}'
    if (dest / 'result.json').exists():
        verify_result(output, index)
        print(f'Already complete and verified: {index}')
        return
    dest.mkdir(parents=True, exist_ok=True)
    attempt = dest / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{os.getpid()}"
    attempt.mkdir()
    save(attempt / 'environment.json', runtime_receipt(development))
    os.environ['RUN_TAG'] = plan['run_tag']
    os.environ['AGRIBRAIN_GIT_COMMIT'] = plan['source']['commit']
    started = time.monotonic()
    with (attempt / 'simulation.log').open('w', encoding='utf-8') as log, \
            redirect_stdout(log), redirect_stderr(log):
        with overridden(gr, setting, task['scenario'], attempt) as actual:
            with gr.decision_ledger_scope(attempt / 'decision_ledgers', reset=True):
                data = gr.run_all(task['seed'], modes=modes)
        results = data['results'][task['scenario']]
        check_results(results, modes)
        expected = episode_counts(modes)
        counts = validate_complete_evidence(
            attempt / 'decision_ledgers', expected_groups=expected['groups'],
            expected_episodes=expected['episodes'],
            expected_adaptation_ledgers=expected['adaptation'],
            expected_final_ledgers=expected['final'],
            manifest_path=attempt / 'evidence_manifest.json')['counts']
        reference = reference_check(results, task)
        validation, reconciliation = revised_validation.validate_task(
            attempt / 'decision_ledgers', task['scenario'], modes, actual, results)
    retained = {m: to_json_native({k: v for k, v in r.items() if not k.startswith('_')})
                for m, r in results.items()}
    retention = plan['retained_evidence'][setting['family']]
    if retention == 'evaluation':
        retain_evaluation_ledgers(attempt, modes, task['scenario'])
    manifest = {p.relative_to(attempt).as_posix(): sha(p)
                for p in attempt.rglob('*') if p.is_file()}
    record = {'task': task, 'setting': setting, 'actual_parameters': actual,
              'plan_sha256': sha(output / 'plan.json'), 'source': plan['source'],
              'attempt_directory': attempt.name, 'retained_evidence': retention,
              'files': manifest, 'results': retained, 'evidence_counts': counts,
              'submitted_reproduction': reference, 'revised_validation': validation,
              'endpoint_reconciliation': reconciliation,
              'development_only': development, 'seconds': time.monotonic() - started,
              'bytes': sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file())}
    save(dest / 'result.json', record)
    print(f"COMPLETE {index} {task['setting']} seed {task['seed']} {task['scenario']}: "
          f"{record['seconds']:.1f}s; ARI " + ', '.join(
              f'{m}={results[m]["ari"]:.6f}' for m in modes[:4]))


def evaluation_ledger(attempt, retention, mode, scenario):
    """Bytes of the retained evaluation ledger of one mode."""
    if retention == 'full':
        return (attempt / 'decision_ledgers' / f'{mode}__{scenario}.jsonl').read_bytes()
    return gzip.decompress(
        (attempt / 'evaluation_ledgers' / f'{mode}__{scenario}.jsonl.gz').read_bytes())


def verify_result(output, index, deep=False):
    """Check a completed task against its retained evidence.

    The retained files must have the recorded hashes and the endpoints must
    carry the archive hash of the evidence manifest. With ``deep`` the retained
    evaluation ledgers, whose bytes must have the hash recorded with the
    result, are validated again and every endpoint they determine is rebuilt
    from them: the episode metrics, the step traces, the guard fractions, the
    volumes and the context weights must equal the stored values, and the
    counts must be those recorded when the task ran. The returned record
    carries the reconciliation.
    """
    output = Path(output)
    dest = output / 'tasks' / f'{index:04d}'
    r = load(dest / 'result.json')
    plan = load(output / 'plan.json')
    if (r['task'] != tasks(plan['seed_sets'])[index]
            or r['plan_sha256'] != sha(output / 'plan.json')):
        raise ValueError('Task/plan mismatch')
    setting = next(s for s in design() if s['id'] == r['task']['setting'])
    if r['setting'] != setting or r['retained_evidence'] != RETAINED_EVIDENCE[setting['family']]:
        raise ValueError('Setting mismatch')
    if Path(r['attempt_directory']).name != r['attempt_directory']:
        raise ValueError('Unsafe attempt path')
    attempt = dest / r['attempt_directory']
    for name, digest in r['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe evidence path')
        if sha(attempt / name) != digest:
            raise ValueError(f'Changed evidence {index}: {name}')
    final = {a['identity']['mode']: a
             for a in load(attempt / 'evidence_manifest.json')['artifacts']
             if a['identity']['episode_index'] == 3}
    if set(final) != set(setting['modes']):
        raise ValueError('Evidence manifest does not cover the modes of the setting')
    scenario = r['task']['scenario']
    for mode, artifact in final.items():
        identity, endpoint = artifact['identity'], r['results'][mode]
        if identity['benchmark_seed'] != r['task']['seed'] or identity['scenario'] != scenario:
            raise ValueError('Evidence identity mismatch')
        if endpoint['episode_evidence_canonical_sha256'] != artifact['archive_canonical_sha256']:
            raise ValueError('Endpoint/archive mismatch')
        if endpoint['decision_ledger_sha256'] != artifact['ledger_literal_sha256']:
            raise ValueError('Endpoint/ledger mismatch')
        if r['retained_evidence'] == 'full':
            archived = json.loads(gzip.decompress(
                (attempt / 'decision_ledgers' / artifact['archive']).read_bytes()))['episode_result']
            for key in METRICS + TRACES + ['latent_environment_sha256']:
                if archived[key] != endpoint[key]:
                    raise ValueError(f'Endpoint/archive mismatch: {mode} {key}')
        if deep:
            data = evaluation_ledger(attempt, r['retained_evidence'], mode, scenario)
            if hashlib.sha256(data).hexdigest() != endpoint['decision_ledger_sha256']:
                raise ValueError(f'Retained ledger of {mode} does not have the recorded hash')
            rows = [json.loads(line) for line in data.decode('utf-8').splitlines()[1:] if line]
            rows.sort(key=lambda row: row['step_index'])
            if [row['action_idx'] for row in rows] != endpoint['action_trace']:
                raise ValueError(f'Retained ledger of {mode} does not give the recorded routing')
            if abs(sum(row['ari'] for row in rows) / len(rows) - endpoint['ari']) > 1e-12:
                raise ValueError(f'Retained ledger of {mode} does not give the recorded ARI')
    if deep:
        import validate as revised_validation
        directory, suffix = ((attempt / 'decision_ledgers', '') if r['retained_evidence'] == 'full'
                             else (attempt / 'evaluation_ledgers', '.gz'))
        again, reconciled = revised_validation.validate_task(
            directory, scenario, setting['modes'], r['actual_parameters'], r['results'], suffix)
        if json.loads(json.dumps(again)) != r['revised_validation']:
            raise ValueError(f'Retained ledgers of task {index} do not validate as recorded')
        reconciled = json.loads(json.dumps(reconciled))
        stored = r.get('endpoint_reconciliation')
        if stored is None:
            # Results of development runs made before the reconciliation existed.
            if not r['development_only']:
                raise ValueError(f'Task {index} carries no endpoint reconciliation')
        elif (revised_validation.reconciled_structure(stored)
              != revised_validation.reconciled_structure(reconciled)):
            raise ValueError(f'Endpoints of task {index} do not reconcile as recorded')
        r['endpoint_reconciliation'] = reconciled
    check_results(r['results'], setting['modes'])
    return r


def largest_difference(a, b, modes, keys):
    import numpy as np
    return max(float(np.max(np.abs(np.asarray(a[m][k], dtype=float) - np.asarray(b[m][k], dtype=float))))
               for m in modes for k in keys)


def diversions_after(result, hour):
    """Number of diverted dispatches from ``hour`` on."""
    return sum(action != 0 for action, h in zip(
        result['action_trace'], result['hours'], strict=True) if h >= hour)


def check_pilot(output, development):
    """Checks on the pilot cells that must pass before the study is released."""
    output = Path(output).resolve()
    plan = read_plan(output, development)
    seed_sets = plan['seed_sets']
    records = {cell: verify_result(output, index, deep=True)
               for cell, index in zip(PILOT_CELLS, pilot_indices(seed_sets), strict=True)}
    report = {'plan_sha256': sha(output / 'plan.json'), 'indices': pilot_indices(seed_sets)}
    keys = METRICS + TRACES

    def results(setting_id, scenario='heatwave'):
        return records[(setting_id, scenario)]['results']

    # 1. An independent second execution gives the same results.
    repeat = output / 'reproducibility_repeat'
    read_plan(repeat, development)
    report['repeat_largest_difference'] = {}
    for cell, index in zip(REPEATED_PILOT_CELLS, pilot_indices(seed_sets, REPEATED_PILOT_CELLS),
                           strict=True):
        again = verify_result(repeat, index, deep=True)
        modes = list(records[cell]['results'])
        difference = largest_difference(records[cell]['results'], again['results'], modes, keys)
        if difference > 1e-12:
            raise ValueError(f'Repeat of {cell} differs by {difference}')
        report['repeat_largest_difference']['/'.join(cell)] = difference

    # 2. The submitted configuration reproduces the stored primary results.
    reproduction = records[('submitted', 'heatwave')]['submitted_reproduction']
    if not development and not reproduction.get('available'):
        raise ValueError('Stored reference of the submitted configuration was not checked')
    report['submitted_reproduction'] = reproduction

    # 3. On the seeds of the submitted study: cold-chain outcomes are those of
    #    the submitted model, and without destination constraints No-Context
    #    and the fixed routes are those of the submitted configuration.
    submitted = results('submitted')
    difference = largest_difference(results('diagnostic_fail_closed'), submitted, ['static'], keys)
    if difference > 1e-12:
        raise ValueError(f'Static differs between the outcome models by {difference}')
    report['static_difference_between_outcome_models'] = difference
    unlimited = results('diagnostic_unlimited_capacity')
    if unlimited['no_context']['action_trace'] != submitted['no_context']['action_trace']:
        raise ValueError('No-Context routing differs without destination constraints')
    same = ['no_context', 'rule_always_redistribute', 'rule_always_recovery']
    difference = largest_difference(unlimited, submitted, same, METRICS + ['ari_trace'])
    if difference > 1e-10:
        raise ValueError(f'Outcomes differ without destination constraints by {difference}')
    if any(unlimited[m]['refused_volume'] != 0.0 for m in unlimited):
        raise ValueError('Refused volume without destination constraints')
    report['difference_without_destination_constraints'] = difference

    # 4. Missing capacity confirmation during the cyber outage (tool channel
    #    down from hour 24). Fail-closed: a mode whose reservation guard
    #    requested the report, and both informed rules, divert nothing. The
    #    mode without tools never requests the report and is not blocked by it.
    closed, opened = results('main_fail_closed', 'cyber_outage'), results('main_fail_open', 'cyber_outage')
    blocked = [*RESERVATION_MODES, 'rule_capacity_informed', 'rule_greedy_informed']
    for mode in blocked:
        if diversions_after(closed[mode], 24.0):
            raise ValueError(f'{mode} diverted without a capacity confirmation under fail-closed')
    report['diversions_after_hour_24_of_the_cyber_outage'] = {
        'fail_closed': {m: diversions_after(closed[m], 24.0) for m in closed},
        'fail_open': {m: diversions_after(opened[m], 24.0) for m in opened}}
    if all(diversions_after(opened[m], 24.0) == 0 for m in blocked):
        raise ValueError('Fail-open left no diversion available during the outage')

    # 5. The guard followed the cutoff stated in the document, not the one applied.
    content = records[('acceptance_document_cutoff_040', 'heatwave')]
    counts = content['revised_validation']['by_mode']['agribrain']
    estimates = content['results']['agribrain']['rho_policy_observed_trace']
    stated = sum(x > 0.40 for x in estimates)
    if counts['acceptance_policy_available_steps'] == 288 and (
            counts['guard_removed_by_reason']['local_redistribute']['acceptance'] != stated):
        raise ValueError('Guard removals do not follow the cutoff stated in the document')
    report['document_cutoff'] = {
        'steps_with_estimated_risk_above_stated_cutoff': stated,
        'steps_with_estimated_risk_above_applied_cutoff': sum(x > 0.50 for x in estimates),
        'redistribution_removed_for_acceptance':
            counts['guard_removed_by_reason']['local_redistribute']['acceptance'],
        'acceptance_policy_available_steps': counts['acceptance_policy_available_steps']}

    # 6. The same document under another identifier gives the same decisions.
    main, renamed = results('main_fail_closed'), results('acceptance_document_renamed')
    difference = largest_difference(main, renamed, list(renamed), keys)
    if difference > 1e-12:
        raise ValueError(f'Renaming the acceptance document changed the results by {difference}')
    report['difference_after_renaming_the_document'] = difference

    # 7. Without the document, under fail-closed, no guarded mode and no rule
    #    that needs the criteria diverts.
    absent = results('acceptance_document_absent')
    for mode in ['agribrain', 'agribrain_eligibility', 'rule_risk_threshold',
                 'rule_capacity_informed', 'rule_greedy_informed']:
        if set(absent[mode]['action_trace']) != {0}:
            raise ValueError(f'{mode} diverted without acceptance criteria under fail-closed')

    # 8. What each sensitivity pilot cell changed; a zero is a valid result.
    report['influence'] = []
    for (setting_id, scenario), record in records.items():
        if record['setting']['family'] not in SENSITIVITY_FAMILIES:
            continue
        a = record['results']['agribrain']
        b = records[('main_fail_closed', scenario)]['results']['agribrain']
        report['influence'].append({
            'setting': setting_id, 'scenario': scenario,
            'changed_actions': sum(x != y for x, y in zip(
                a['action_trace'], b['action_trace'], strict=True)),
            'ari_change': a['ari'] - b['ari']})
    save(output / 'PILOT_PASSED.json', report)
    print('Pilot passed: ' + json.dumps({k: v for k, v in report.items() if k != 'influence'}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--development-runtime', type=Path)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ['prepare', 'task', 'verify', 'pilot-check', 'missing']:
        parser = sub.add_parser(cmd)
        parser.add_argument('--output', required=True, type=Path)
        if cmd in ['task', 'verify']:
            parser.add_argument('--index', required=True, type=int)
        if cmd == 'prepare':
            parser.add_argument('--development-seeds', action='store_true',
                                help='development runs only: use the development seeds')
    args = p.parse_args()
    development = configure(args.development_runtime)
    sys.path.insert(0, str(HERE))
    if args.command == 'prepare':
        prepare(args.output, development, args.development_seeds)
    elif args.command == 'task':
        run_task(args.output, args.index, development)
    elif args.command == 'verify':
        read_plan(args.output, development)
        verify_result(args.output, args.index, deep=True)
        print('Verified')
    elif args.command == 'pilot-check':
        check_pilot(args.output, development)
    else:
        plan = read_plan(args.output, development)
        missing = []
        for task in tasks(plan['seed_sets']):
            try:
                verify_result(args.output, task['index'])
            except (OSError, ValueError, KeyError):
                missing.append(task['index'])
        print(','.join(map(str, missing)))


if __name__ == '__main__':
    main()
