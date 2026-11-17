"""huCIRA cytokine-programme gene sets on the Xenium panel.

Programme per cytokine = union over the 11 huCIRA immune cell types of the up genes in cytokine_programs.json
(top-50, log_fc > 0.5, adj_p < 0.05), intersected with the 477-gene panel (hhca.tl.programme_genes). The 18
cytokines of Fig 6 h / ED9 are listed in config params.cytokine.programmes.

Inputs:
    huCIRA programme JSON (config inputs.cytokine_programs)
    genes.txt of the extraction folder (config inputs.extracted)

Outputs:
    <out>/program_panel_genes.csv (cytokine, label, n_union, n_on_panel, genes), read by core_features.py and
    celltype_cytokine_scores.py

Usage: python scripts/xenium/downstream/program_genes.py [--programs JSON] [--genes genes.txt] [--out DIR]
"""

import argparse
import json
import os

import downstream_common as common
import hhca


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--programs", default=cfg["inputs"]["cytokine_programs"], help="huCIRA programme JSON")
    parser.add_argument("--genes", default=os.path.join(common.extracted(cfg), "genes.txt"), help="panel gene list")
    parser.add_argument("--out", default=common.out_dir(cfg, "shared"), help="output folder")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    cytokines = cfg["params"]["cytokine"]["programmes"]

    with open(args.programs) as handle:
        programmes = json.load(handle)["programs"]
    table = hhca.tl.programme_genes(programmes, common.read_lines(args.genes), cytokines)
    table.to_csv(os.path.join(args.out, "program_panel_genes.csv"), index=False)
    print("cytokines with >= 3 on-panel genes:", int((table.n_on_panel >= 3).sum()), "of", len(cytokines))
    print("wrote", os.path.join(args.out, "program_panel_genes.csv"))


if __name__ == "__main__":
    main()
