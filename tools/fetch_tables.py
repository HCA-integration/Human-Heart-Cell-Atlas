#!/usr/bin/env python
"""Download or verify the processed-table bundle that the figure notebooks read.

    python tools/fetch_tables.py --url https://<host>/<bundle>/   # download every file in MANIFEST.csv
    python tools/fetch_tables.py --verify                          # md5-check the local bundle

The bundle root is $HHCA_TABLES (or `tables:` in config/paths.yaml).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hhca.paths import TABLES  # noqa: E402


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(manifest: Path):
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f))


def verify(root: Path) -> int:
    bad = 0
    for r in rows(root / "MANIFEST.csv"):
        p = root / r["path"]
        if not p.exists():
            print(f"MISSING {r['path']}"); bad += 1
        elif p.stat().st_size != int(r["bytes"]) or md5(p) != r["md5"]:
            print(f"CORRUPT {r['path']}"); bad += 1
    print("bundle OK" if not bad else f"{bad} problems")
    return 1 if bad else 0


def fetch(root: Path, base: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    base = base.rstrip("/") + "/"
    urllib.request.urlretrieve(base + "MANIFEST.csv", root / "MANIFEST.csv")
    for r in rows(root / "MANIFEST.csv"):
        p = root / r["path"]
        if p.exists() and p.stat().st_size == int(r["bytes"]) and md5(p) == r["md5"]:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"get {r['path']} ({int(r['bytes'])/1e6:.1f} MB)")
        urllib.request.urlretrieve(base + r["path"], p)
    return verify(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of the published bundle")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dest", default=None, help="bundle root (default: $HHCA_TABLES)")
    a = ap.parse_args()
    root = Path(a.dest) if a.dest else Path(TABLES)
    if a.url:
        sys.exit(fetch(root, a.url))
    if a.verify:
        sys.exit(verify(root))
    ap.error("--url or --verify required")


if __name__ == "__main__":
    main()
