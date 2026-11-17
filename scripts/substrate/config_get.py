#!/usr/bin/env python
"""Print values of config/substrate.yaml as shell assignments for the sbatch files.

Usage:
  config_get.py CONFIG slides                -> SLIDES="a b c"
  config_get.py CONFIG slide NAME            -> SITE=..; BUILDER=..; THREAD=..; SEED=..; REGION=..; SQ2=..; SQ8=..
  config_get.py CONFIG key a.b.c             -> value (lists joined by spaces)
  config_get.py CONFIG json a.b              -> value as JSON (a block for the scripts of the certainty environment)
HHCA_<ROOT> environment variables override the roots (e.g. HHCA_SPACERANGER_AH).
"""

import argparse
import json
import os
import shlex
import sys

from hhca.config import get, load_config


def root(cfg, name):
    """A root of the config, or its HHCA_<ROOT> override."""
    return os.environ.get("HHCA_" + name.upper(), cfg["roots"][name])


def slide_assignments(cfg, name):
    """Shell assignments with the settings and the Space Ranger folders of one slide."""
    slide = cfg["slides"][name]
    base = root(cfg, "spaceranger_berlin" if slide["site"] == "berlin" else "spaceranger_ah")
    outs = f"{base}/{name}_GRCh38-2020-A/outs/binned_outputs"
    outs_008um = f"{root(cfg, 'spaceranger_008um')}/{name}_GRCh38-2020-A/outs/binned_outputs"
    values = {
        "SITE": slide["site"],
        "BUILDER": slide["builder"],
        "THREAD": slide["thread"],
        "SEED": slide["seed"],
        "REGION": slide["region"],
        "SQ2": f"{outs}/square_002um",
        "SQ8": f"{outs_008um}/square_008um",
    }
    return "; ".join(f"{variable}={shlex.quote(str(value))}" for variable, value in values.items())


def key_value(cfg, dotted_key):
    """Value of a dotted key; `roots.<name>` with the HHCA_<ROOT> override applied."""
    value = get(cfg, dotted_key)
    last = dotted_key.split(".")[-1]
    if last in cfg.get("roots", {}) and dotted_key.startswith("roots."):
        value = root(cfg, last)
    return value


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="path of config/substrate.yaml")
    parser.add_argument("query", help="slides, slide, key or json")
    parser.add_argument("argument", nargs="?", help="slide name (query slide) or dotted key (query key)")
    return parser.parse_args()


def main():
    """Answer one query."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.query == "slides":
        print("SLIDES=" + shlex.quote(" ".join(cfg["slides"])))
    elif args.query == "slide":
        print(slide_assignments(cfg, args.argument))
    elif args.query == "key":
        value = key_value(cfg, args.argument)
        print(" ".join(map(str, value)) if isinstance(value, list) else value)
    elif args.query == "json":
        print(json.dumps(get(cfg, args.argument)))
    else:
        sys.exit(f"unknown query {args.query}")


if __name__ == "__main__":
    main()
