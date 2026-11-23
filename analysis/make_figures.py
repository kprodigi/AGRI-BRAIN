"""Regenerate every study figure from the files in this repository.

The figure scripts in analysis/figures read the result tables in data/study and the decision records in
data/figure_inputs. Each figure is written to OUTPUT under its name in figures/. With --compare, the script also reports
whether each new image is byte-identical to the one in figures/; that holds with the library versions in
analysis/requirements.txt and the Arial font, and a different environment changes only the rendering.

usage: python analysis/make_figures.py --output DIR [--compare]
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'analysis' / 'figures'
STUDY = ROOT / 'data' / 'study'
RECORDS = ROOT / 'data' / 'figure_inputs'
JOBS = [  # script, inputs; each script writes the figure of its own name
    ('ari_performance_by_scenario', [STUDY]),
    ('executed_routing_choices', [RECORDS / 'agribrain', RECORDS / 'no_context', STUDY]),
    ('routing_decisions_and_information', [RECORDS / 'agribrain']),
    ('routing_decision_explanation', [RECORDS / 'agribrain']),
    ('weight_sensitivity', [STUDY]),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare', action='store_true', help='report byte identity with figures/')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for script, inputs in JOBS:
            name = script + '.png'
            subprocess.run([sys.executable, script + '.py', *map(str, inputs), str(tmp / script)], cwd=SCRIPTS,
                           check=True, stdout=subprocess.DEVNULL)
            shutil.copyfile(tmp / name, args.output / name)
            line = f'{name}: written'
            if args.compare:
                same = sha(args.output / name) == sha(ROOT / 'figures' / name)
                line += ', identical to figures/' if same else ', differs from figures/ (rendering environment)'
            print(line)


if __name__ == '__main__':
    main()
