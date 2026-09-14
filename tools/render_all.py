#!/usr/bin/env python
"""Render every panel: figures/<Fig>/panel_*.py -> $HHCA_OUT/<Fig>/panel_<x>.png|pdf.

    python tools/render_all.py [Fig4 ...] [--timeout 3600]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figures", nargs="*")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    names = a.figures or sorted(p.name for p in FIGS.iterdir() if p.is_dir() and p.name != "_lib")
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(FIGS / "_lib")]))
    failed = []
    for fig in names:
        for s in sorted((FIGS / fig).glob("panel_*.py")):
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, s.name], cwd=s.parent, env=env,
                                   capture_output=True, text=True, timeout=a.timeout)
                ok = r.returncode == 0
                err = "" if ok else (r.stderr.strip().splitlines() or ["?"])[-1][:160]
            except subprocess.TimeoutExpired:
                ok, err = False, f"timeout {a.timeout}s"
            print(f"{'OK ' if ok else 'ERR'} {fig}/{s.name} {time.time() - t0:.0f}s {err}", flush=True)
            if not ok:
                failed.append(f"{fig}/{s.name}")
    if failed:
        print("failed:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
