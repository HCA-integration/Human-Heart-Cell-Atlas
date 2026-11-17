"""Point the data roots in config/ and workflow/hpc/ at your own machine.

The configs and SLURM scripts use the placeholder roots listed in config/roots.yaml. Given a copy of that file
with real paths, every occurrence of each root is replaced, and config/roots.yaml is updated to the new values
so the command can be run again later.

Usage: python tools/set_roots.py --roots my_roots.yaml [--dry-run]
"""

import argparse
import glob
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS_FILE = os.path.join(REPO, "config/roots.yaml")
TARGETS = ["config/*.yaml", "workflow/hpc/*.sbatch", "workflow/hpc/*.sh", "scripts/*/*.sh", "scripts/*/*/*.sh"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", required=True, help="your copy of config/roots.yaml with real paths")
    parser.add_argument("--dry-run", action="store_true", help="list the files that would change, write nothing")
    args = parser.parse_args()

    with open(ROOTS_FILE) as fh:
        current = yaml.safe_load(fh)
    with open(args.roots) as fh:
        wanted = yaml.safe_load(fh)
    # longest roots first, so a root nested inside another one is replaced before its parent
    pairs = sorted(
        ((current[k].rstrip("/"), wanted[k].rstrip("/")) for k in current if k in wanted and wanted[k] != current[k]),
        key=lambda pair: -len(pair[0]),
    )
    files = sorted({f for pattern in TARGETS for f in glob.glob(os.path.join(REPO, pattern))} - {ROOTS_FILE})

    n_changed = 0
    for path in files:
        with open(path) as fh:
            text = fh.read()
        new_text = text
        for old, new in pairs:
            new_text = new_text.replace(old, new)
        if new_text == text:
            continue
        n_changed += 1
        n_hits = sum(text.count(old) for old, _ in pairs)
        print(f"{'would update' if args.dry_run else 'updated'} {os.path.relpath(path, REPO)} ({n_hits} paths)")
        if not args.dry_run:
            with open(path, "w") as fh:
                fh.write(new_text)
    if not args.dry_run and pairs:
        with open(ROOTS_FILE, "w") as fh:
            yaml.safe_dump({**current, **{k: wanted[k] for k in current if k in wanted}}, fh, sort_keys=False)
    print(f"{n_changed} files {'to update' if args.dry_run else 'updated'}")


if __name__ == "__main__":
    main()
