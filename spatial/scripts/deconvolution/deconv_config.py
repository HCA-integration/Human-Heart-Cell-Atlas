#!/usr/bin/env python
"""Read config/deconvolution.yaml for the deconvolution scripts and jobs.

As a module it provides load() and add_argument() to the scripts of this folder. The NNLS environment has no YAML
reader, so the jobs that run in it first write the config as JSON with the Python of another environment and
pass that file as --config.

Inputs:  the config file (pipeline name, or a .yaml / .json path).
Outputs: the config as JSON on stdout.

Usage: deconv_config.py [CONFIG] > deconvolution.json
"""

import argparse
import json


def load(config="deconvolution"):
    """Return the parsed config, given as pipeline name or as the path of a .yaml or .json file."""
    if str(config).endswith(".json"):
        with open(config) as handle:
            return json.load(handle)
    from hhca.config import load_config

    return load_config(config)


def add_argument(parser):
    """Add the --config argument to the parser of a script."""
    parser.add_argument("--config", default="deconvolution", help="config/deconvolution.yaml, or a JSON copy of it")
    return parser


def main():
    """Print the config as JSON."""
    parser = argparse.ArgumentParser(description="config/deconvolution.yaml as JSON")
    parser.add_argument("config", nargs="?", default="deconvolution", help="pipeline name or config file")
    print(json.dumps(load(parser.parse_args().config)))


if __name__ == "__main__":
    main()
