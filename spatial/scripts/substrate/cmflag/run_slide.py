#!/usr/bin/env python
"""Run CMFLAG for one slide with the inputs listed in config/substrate_cmflag.yaml.

The script named by the slide's `producer` (cmflag_tif.py for the Berlin slides, cmflag_ah.py for the AH slides)
registers the H&E image, saved in <results>/cmflag/registration/, and writes the flags to <results>/cmflag/flags/.

Usage: python run_slide.py --config config/substrate_cmflag.yaml --slide SLIDE [--dry-run]
"""

import argparse
import os
import subprocess
import sys

from hhca.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="config/substrate_cmflag.yaml")
    parser.add_argument("--slide", required=True, help="slide id (a key of `slides` in the config)")
    parser.add_argument("--dry-run", action="store_true", help="print the command and stop")
    return parser.parse_args()


def main():
    """Build the command of the slide from the config and run it."""
    args = parse_args()
    cfg = load_config(args.config)
    roots = dict(cfg["roots"])
    slide_cfg = dict(cfg["slides"][args.slide])
    base = slide_cfg.get("base", "")

    def fill(value):
        return value.format(slide=args.slide, base=base, **roots) if isinstance(value, str) else value

    shared = {key: fill(value) for key, value in cfg["shared"].items()}
    slide_cfg = {key: fill(value) for key, value in slide_cfg.items()}
    results = roots["results"]
    outdir = f"{results}/cmflag/flags"
    common = [
        "--factor-order",
        shared["factor_order"],
        "--quantiles",
        shared["quantiles"],
        "--tau",
        shared["tau"],
        "--cm-fit",
        shared["cm_fit"],
        "--vsmc-fit",
        shared["vsmc_fit"],
        "--outdir",
        outdir,
        "--force",
    ]
    if slide_cfg.get("structs"):
        common += ["--structs", slide_cfg["structs"], "--cells", slide_cfg["cells"]]
    reg_cache = f"{results}/cmflag/registration/{args.slide}_align_grids.npz"
    os.makedirs(os.path.dirname(reg_cache), exist_ok=True)
    if slide_cfg["producer"] == "cmflag_tif":
        checkpoint = f"{shared['gcert_dir']}/{args.slide}.features.npz"
        cmd = (
            [
                sys.executable,
                f"{HERE}/cmflag_tif.py",
                args.slide,
                "--decode",
                slide_cfg["decode"],
                "--gcert-checkpoint",
                checkpoint,
                "--spaceranger",
                slide_cfg["spaceranger"],
                "--tif",
                slide_cfg["tif"],
                "--reg-cache",
                reg_cache,
            ]
            + common
            + ["--force-reg"]
        )
    else:
        cmd = [
            sys.executable,
            f"{HERE}/cmflag_ah.py",
            args.slide,
            "--he-source",
            slide_cfg["he_source"],
            "--h5ad",
            slide_cfg["h5ad"],
            "--spaceranger",
            slide_cfg["spaceranger"],
            "--hires",
            slide_cfg["hires"],
            "--reg-cache",
            reg_cache,
            "--force-reg",
        ] + common
        if slide_cfg.get("czi"):
            cmd += ["--czi", slide_cfg["czi"]]
        if slide_cfg.get("micro_dims"):
            cmd += ["--micro-dims", slide_cfg["micro_dims"]]
    print("command:", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    os.makedirs(outdir, exist_ok=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
