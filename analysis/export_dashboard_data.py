"""Write the summary data shown on the dashboard's results page.

Values are read from the result tables in data/study; the decision difference of each scenario and the latency
intervals are computed here with the study's own interval function (study/inference.py), and the decision difference
from the records in data/figure_inputs. With --check the script compares the existing output file with a fresh export and
exits with an error if they differ.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study.inference import estimate  # noqa: E402

OUT = ROOT / 'agribrain/frontend/src/data/studyResults.json'
STUDY = ROOT / 'data/study'
RECORDS = ROOT / 'data/figure_inputs'
SCENARIOS = ['heatwave', 'overproduction', 'cyber_outage', 'adaptive_pricing', 'baseline']
MODES = ['static', 'no_context', 'agribrain']
SEEDS = range(5001, 5021)
STEPS = 288
METRICS = [  # metric, label, scale
    ('ari', 'Adaptive Resilience Index', 1.0),
    ('waste', 'Food waste (%)', 100.0),
    ('carbon', 'Emissions indicator (kg CO2-eq per routing opportunity)', 1.0 / STEPS),
    ('slca', 'Social performance', 1.0),
    ('rle', 'Reverse Logistics Efficiency', 1.0),
    ('refused_share_of_diverted_volume', 'Diverted volume refused (%)', 100.0),
]
CONTROLS = {'live_minus_fixed': 'fixed', 'live_minus_shuffled': 'shuffled', 'live_minus_zero': 'zero'}


def r6(x):
    return round(float(x), 6) + 0.0


def rows(name):
    with (STUDY / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def interval(row, scale=1.0):
    return {k: r6(float(row[k]) * scale) for k in ('mean', 'low', 'high')}


def actions(path):
    return [r['action'] for r in json.load(gzip.open(path))['rows']]


def build():
    modes = [r for r in rows('modes.csv') if r['setting'] == 'main_fail_closed']
    by = {(r['mode'], r['scenario'], r['metric']): r for r in modes}

    overall = {}
    for metric, label, scale in METRICS:
        entry = {'label': label}
        for m in MODES:
            row = by.get((m, 'overall', metric))
            if row is not None and row['low'] != '':
                entry[m] = interval(row, scale)
        overall[metric] = entry
    ends = [r for r in rows('seed_endpoints.csv') if r['setting'] == 'main_fail_closed' and r['mode'] in MODES]
    entry = {'label': 'Decision latency (ms)'}
    for m in MODES:
        per_seed = [np.mean([float(r['mean_decision_latency_ms']) for r in ends if r['mode'] == m and int(r['seed']) == s])
                    for s in SEEDS]
        e = estimate(per_seed, ('dashboard_latency', m))
        entry[m] = {k: r6(e[k]) for k in ('mean', 'low', 'high')}
    overall['mean_decision_latency_ms'] = entry

    gains = {r['scenario']: r for r in rows('comparisons_by_scenario.csv') if r['condition'] == 'fail-closed'}
    scenarios = {}
    for s in SCENARIOS:
        metrics = {'ari': {m: interval(by[(m, s, 'ari')]) for m in MODES}}
        diff = np.array([np.mean(np.array(actions(RECORDS / 'no_context' / f'no_context_{seed}_{s}.json.gz'))
                                 != np.array(actions(RECORDS / 'agribrain' / f'direct_{seed}_{s}.json.gz')))
                         for seed in SEEDS])
        ci = bootstrap((diff,), np.mean, method='BCa', confidence_level=0.95, n_resamples=9999,
                       random_state=np.random.default_rng(20261122)).confidence_interval
        scenarios[s] = {'metrics': metrics, 'paired_gain': interval(gains[s]),
                        'different_actions_percent': {'mean': r6(100 * diff.mean()), 'low': r6(100 * ci.low),
                                                      'high': r6(100 * ci.high)}}

    principal = next(r for r in rows('principal_comparisons.csv') if r['condition'] == 'fail-closed')
    settings = [{'setting': 'nominal', 'family': 'weights', 'gain': r6(principal['mean']), 'low': r6(principal['low']),
                 'high': r6(principal['high']), 'change': 0.0}]
    for r in rows('sensitivity_settings.csv'):
        if r['family'] == 'weights':
            settings.append({'setting': r['setting'], 'family': 'weights', 'gain': r6(r['agribrain_minus_no_context']),
                             'low': r6(r['agribrain_minus_no_context_low']),
                             'high': r6(r['agribrain_minus_no_context_high']),
                             'change': r6(r['agribrain_minus_no_context_change_from_reference'])})

    routes = {m: [round(100 * float(by[(m, 'overall', k)]['mean']), 1)
                  for k in ('share_cold_chain', 'share_redistribution', 'share_recovery')]
              for m in ('no_context', 'agribrain')}
    controls = {CONTROLS[r['contrast']]: interval(r) for r in rows('control_contrasts.csv')
                if r['scenario'] == 'overall' and r['contrast'] in CONTROLS}
    families = {}
    for r in rows('sensitivity_settings.csv'):
        families[r['family']] = families.get(r['family'], 0) + 1
    return {
        'meta': {
            'seeds': len(SEEDS), 'scenarios': SCENARIOS, 'modes': MODES, 'primary_evaluations': 300,
            'sensitivity': {'weight_settings': families.get('weights', 0),
                            'structural_settings': families.get('structural', 0),
                            'acceptance_document_checks': families.get('content', 0)},
            'intervals': '95% BCa across seeds; sensitivity intervals are pointwise',
        },
        'overall': overall,
        'scenarios': scenarios,
        'sensitivity': settings,
        'routes_percent': {'order': ['cold_chain', 'local_redistribution', 'recovery'], **routes},
        'frozen_controls': controls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='fail if the output file is out of date')
    args = parser.parse_args()
    text = json.dumps(build(), indent=1) + '\n'
    if args.check:
        if not OUT.is_file() or OUT.read_bytes().decode('utf-8') != text:
            sys.exit('Dashboard data is out of date; run analysis/export_dashboard_data.py')
        print('Dashboard data matches data/study')
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode('utf-8'))
    print('Wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
