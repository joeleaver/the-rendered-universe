"""The epistemic firewall.

The physicist package must never import the engine or the renderer.
If this test fails, the physicist has seen God, and every 'discovery'
in this repo is worthless.
"""
import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(r'^\s*(from|import)\s+(engine|render)\b', re.MULTILINE)


def check():
    root = Path(__file__).resolve().parent.parent
    bad = []
    for path in (root / 'physicist').glob('*.py'):
        if FORBIDDEN.search(path.read_text()):
            bad.append(path.name)
    return bad


if __name__ == '__main__':
    bad = check()
    if bad:
        print(f'FIREWALL BREACHED: {bad}')
        sys.exit(1)
    print('firewall intact: physicist imports neither engine nor render')
