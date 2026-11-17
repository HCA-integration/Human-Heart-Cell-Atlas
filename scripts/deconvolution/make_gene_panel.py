#!/usr/bin/env python
"""Build the gene panel shared by cell2location and stereoscope.

Panel = the top-N highly variable genes of the reference (scanpy dispersion-based HVGs on log1p CP10k) among the
genes present in every slide, plus a curated set of cardiac markers (hhca.tl.gene_panel; config methods.gene_panel
and markers). The R methods and spotiphy select their own
genes and do not read this file.

Inputs:  the shared reference h5ad, <data>/<slide>/genes.txt
Outputs: the panel, one gene per line (--panel)

Usage: make_gene_panel.py --ref REF.h5ad --data DIR --panel OUT.txt --celltypes CT.txt [--n-hvg 3000]
           [--config CONFIG]
"""

import argparse

import bench_common as bc
import deconv_config
import hhca


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]), slide=False)
    parser.add_argument("--n-hvg", type=int, default=None, help="number of highly variable genes (default: config)")
    parser.add_argument("--slides", default="", help="comma-separated slide names (default: the benchmark slides)")
    return deconv_config.add_argument(parser).parse_args()


def main():
    """Select the highly variable genes shared by the reference and all slides and write the panel."""
    args = parse_args()
    bc.configure(args)
    cfg = deconv_config.load(args.config)
    settings = cfg["methods"]["gene_panel"]
    slides = [slide for slide in args.slides.split(",") if slide] or cfg["slides"]["benchmark"]
    panel = hhca.tl.gene_panel(
        bc.load_ref(),
        [bc.read_lines(f"{bc.DATA}/{slide}/genes.txt") for slide in slides],
        args.n_hvg or settings["n_hvg"],
        cfg["markers"]["cardiac"] + cfg["markers"]["lymphatic"],
        settings["min_cells"],
    )
    with open(bc.PANEL, "w") as handle:
        handle.write("\n".join(panel) + "\n")
    print(f"panel size = {len(panel)} (highly variable genes + markers); wrote {bc.PANEL}", flush=True)


if __name__ == "__main__":
    main()
