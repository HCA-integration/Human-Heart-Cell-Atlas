#!/usr/bin/env python
"""Resolve slide lists and paths of config/vascular.yaml for run_vascular.sh.

decode, h5ad_2um and factor_order come from the `substrate` block (relative to the repository root); every other
key is a top-level path template filled with the `roots`. HHCA_<ROOT> environment variables override a root.

Inputs: config/vascular.yaml.
Outputs: the requested value on stdout.

Usage:
  python paths.py CONFIG slides berlin|ah|all|raas_pool_order|common_from
  python paths.py CONFIG path KEY [SLIDE]     # e.g. decode, h5ad_2um, factor_order, spaceranger
  python paths.py CONFIG get a.b.c            # raw value (lists joined by spaces)
"""

import argparse

from hhca.config import get, load_config, section_path


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="config/vascular.yaml")
    parser.add_argument("what", choices=["slides", "path", "get"], help="kind of query")
    parser.add_argument("args", nargs="*", help="slide group, path key and slide, or dotted config key")
    return parser.parse_args()


def main():
    """Print the slide list, path or config value asked for."""
    args = parse_args()
    cfg = load_config(args.config)
    query = args.args
    if args.what == "slides":
        groups = cfg["slides"]
        slides = groups["berlin"] + groups["ah"] if query[0] == "all" else groups[query[0]]
        print(" ".join(slides))
    elif args.what == "path":
        print(section_path(cfg, query[0], query[1] if len(query) > 1 else None))
    else:
        value = get(cfg, query[0])
        print(" ".join(map(str, value)) if isinstance(value, list) else value)


if __name__ == "__main__":
    main()
