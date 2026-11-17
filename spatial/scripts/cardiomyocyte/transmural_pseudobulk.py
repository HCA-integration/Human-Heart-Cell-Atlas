"""Fig 3 l: composition and pseudobulk per transmural section (epi / myo / endo) from the CMFLAG-clean 2 um h5ads.

Raw counts are summed over the ventricular-cardiomyocyte bins and over all bins of each section
(hhca.tl.section_pseudobulk). --donors and --layers set the sections and their order (default: the donors of
config transmural.donors, epi -> endo); sections with another gene space (AH10) are matched by name to the genes
of the first section (missing genes = 0).

Inputs: the 2 um h5ads of the sections (config paths.h5ad_2um; obs L3_label, raw counts).
Outputs: in --out-dir
  composition_counts_by_sample.csv   Level-3 bin counts per section
  cm_pseudobulk_counts.csv           genes x section, raw counts summed over the ventricular-CM bins
  all_pseudobulk_counts.csv          genes x section, raw counts summed over all bins
  sample_meta.csv                    bins, CM bins and total counts per section

Usage: transmural_pseudobulk.py --out-dir DIR [--donors D ...] [--layers L ...]
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1))

import argparse

import anndata as ad

import hhca
from hhca.config import load_config, section_path
from hhca.log import get_logger

log = get_logger("pseudobulk")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Transmural composition and pseudobulk per section (Fig 3 l)")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--donors", nargs="+", default=None, help="donors (default: config transmural.donors)")
    parser.add_argument("--layers", nargs="+", default=None, help="order of the layers (default: epi myo endo)")
    return parser.parse_args()


def main():
    """Sum the counts of every section and write the composition, pseudobulk and sample tables."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    os.makedirs(args.out_dir, exist_ok=True)
    donors = args.donors or cfg["transmural"]["donors"]
    layers = args.layers or cfg["transmural"]["layers"]
    slide_of = {(section["donor"], section["layer"]): slide for slide, section in cfg["sections"].items()}
    samples = {
        f"{donor}_{layer}": (donor, layer, slide_of[donor, layer])
        for donor in donors
        for layer in layers
        if (donor, layer) in slide_of
    }
    pseudobulk = hhca.tl.section_pseudobulk(
        {
            sample: ad.read_h5ad(section_path(cfg, "h5ad_2um", slide), backed="r")
            for sample, (_, _, slide) in samples.items()
        },
        cfg["cell_types"]["ventricular"],
        cfg["fig3l"]["chunk"],
        log,
    )
    pseudobulk["composition"].to_csv(f"{args.out_dir}/composition_counts_by_sample.csv")
    pseudobulk["cell_type"].to_csv(f"{args.out_dir}/cm_pseudobulk_counts.csv")
    pseudobulk["all"].to_csv(f"{args.out_dir}/all_pseudobulk_counts.csv")
    meta = pseudobulk["meta"]
    meta.insert(1, "donor", [samples[sample][0] for sample in meta["sample"]])
    meta.insert(2, "layer", [samples[sample][1] for sample in meta["sample"]])
    meta.insert(3, "slide", [samples[sample][2] for sample in meta["sample"]])
    meta.insert(5, "n_clean", meta["n_bins"])  # all bins of a CMFLAG-clean h5ad are kept
    meta = meta.rename(
        columns={"n_cell_type": "n_cm_clean", "cell_type_counts": "cm_total_counts", "all_counts": "all_total_counts"}
    )
    meta.to_csv(f"{args.out_dir}/sample_meta.csv", index=False)
    log("done, wrote to", args.out_dir)


if __name__ == "__main__":
    main()
