#!/usr/bin/env python
"""Read config/cardiomyocyte.yaml for the shell driver and the cardiomyocyte scripts.

Path templates are filled with the roots and the slide id; HHCA_<ROOT> environment variables override a root.
As a module it provides load(), path() and slides() to the scripts of this folder.

Inputs: the config file (path, or pipeline name).
Outputs: the requested value on stdout.

Usage: cm_config.py CONFIG path KEY [SLIDE]         paths.KEY with {roots} and {slide} filled
       cm_config.py CONFIG slides berlin|ah|all     slide lists
       cm_config.py CONFIG endo                     endocardial sections, comma-joined
       cm_config.py CONFIG get a.b.c [SLIDE]        any value (lists space-joined, dicts as JSON)
"""

import argparse
import json
import sys

from hhca import config as hhca_config


def load(config):
    """Return the parsed config and its roots (environment overrides applied)."""
    cfg = hhca_config.load_config(config)
    return cfg, hhca_config.resolve_roots(cfg)


def fill(template, roots, slide=None, **extra):
    """Fill a path template; {base} is the slide id without its VS_ prefix."""
    values = dict(extra)
    if slide is not None:
        values["slide"] = slide
        values["base"] = slide[3:] if slide.startswith("VS_") else slide
    return hhca_config.fill(template, roots, **values)


def slides(cfg, which):
    """Slide ids of the list `which` (berlin, ah or all)."""
    slide_lists = cfg["slides"]
    if which == "berlin":
        return list(slide_lists["berlin"])
    if which == "ah":
        return list(slide_lists["ah"])
    if which == "all":
        return list(slide_lists["berlin"]) + list(slide_lists["ah"])
    sys.exit(f"unknown slide list {which}")


def path(cfg, key, slide=None):
    """Filled path template paths.<key> of the config."""
    return hhca_config.section_path(cfg, key, slide)


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Query config/cardiomyocyte.yaml from the shell")
    parser.add_argument("config", help="config/cardiomyocyte.yaml")
    parser.add_argument("query", help="path | slides | endo | get")
    parser.add_argument("args", nargs="*", help="KEY [SLIDE] for path and get; berlin|ah|all for slides")
    return parser.parse_args()


def main():
    """Print the requested value."""
    args = parse_args()
    cfg, roots = load(args.config)
    query, rest = args.query, args.args
    if query == "path":
        print(path(cfg, rest[0], rest[1] if len(rest) > 1 else None))
    elif query == "slides":
        print(" ".join(slides(cfg, rest[0])))
    elif query == "endo":
        print(",".join(cfg["slides"]["endo_sections"]))
    elif query == "get":
        value = hhca_config.get(cfg, rest[0])
        if isinstance(value, list):
            print(" ".join(map(str, value)))
        elif isinstance(value, dict):
            print(json.dumps(value))
        else:
            print(fill(value, roots, rest[1] if len(rest) > 1 else None))
    else:
        sys.exit(f"unknown query {query}")


if __name__ == "__main__":
    main()
