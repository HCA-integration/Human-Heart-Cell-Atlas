#!/usr/bin/env python
"""Print the configured roots as shell exports:  eval "$(python tools/print_env.py)" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hhca import paths  # noqa: E402

for name, val in paths.roots().items():
    if val:
        print(f'export HHCA_{name}="{val}"')
print(f'export HHCA_REPO="{paths.REPO}"')
