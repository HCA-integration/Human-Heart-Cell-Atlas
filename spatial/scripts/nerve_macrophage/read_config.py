#!/usr/bin/env python
"""Read config/nerve_macrophage.yaml for the SLURM job and the pipeline scripts.

Writes the value of a dotted key to stdout with the data roots ({heart}, {inputs}) filled in; lists come
space-separated and the per-slide placeholder {s} is kept. HHCA_<ROOT> environment variables override a root.

Inputs: config/nerve_macrophage.yaml.
Outputs: the value on stdout.

Usage: python scripts/nerve_macrophage/read_config.py substrate.decode
       python scripts/nerve_macrophage/read_config.py slides.berlin
"""

import argparse
import os

from hhca import config_path
from hhca.config import get, load_config

CONFIG = config_path("nerve_macrophage")


def load(path=CONFIG):
    """Parse the pipeline config."""
    return load_config(path)


def roots(cfg):
    """Data roots of the config, each overridable by HHCA_<ROOT>."""
    return {name: os.environ.get(f"HHCA_{name.upper()}", path) for name, path in cfg["roots"].items()}


def fill(template, cfg, **values):
    """Fill the root names of a path template; {s} and {slide} stay unless given."""
    values = {**roots(cfg), "s": "{s}", "slide": "{slide}", **values}
    return str(template).format(**values)


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("key", help="dotted key of the config, e.g. substrate.decode")
    return parser.parse_args()


def main():
    """Print the value of the requested key."""
    args = parse_args()
    cfg = load()
    value = get(cfg, args.key)
    if isinstance(value, list):
        print(" ".join(fill(x, cfg) if isinstance(x, str) else str(x) for x in value))
    else:
        print(fill(value, cfg) if isinstance(value, str) else value)


if __name__ == "__main__":
    main()
