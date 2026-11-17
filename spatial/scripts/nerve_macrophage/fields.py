#!/usr/bin/env python
"""Fig 5 b, c, f, j field extracts (H&E crop + 2 um bins) for the figure scripts.

Per field (config/nerve_macrophage.yaml `fields`, slide VS_BO_HDZ150319_34_myo, coordinates in bin micrometres):
  registration  CMFLAG registration M0, t0 (results/substrate/cmflag/registration, SIFT + ECC) plus a local
                phase-correlation offset against the CytAssist image, as CMFLAG does per field (160 um half
                tile, 2 um grid; kept if err <= 0.45 and |offset| <= 110 um; hhca.tl.registration_offset)
  he            microscopy H&E crop at native resolution: px = ([x, y] + offset) @ M0.T + t0
  bins          2 um bins of the CMFLAG-clean h5ad in the field (x, y, Level-3 label; hhca.tl.registered_field)
  unassigned    in-tissue SpaceRanger 2 um bins without a bin in the clean h5ad (no decode pixel, or removed by
                the four CMFLAG spillover flags)
  f only        myelin display score per neural bin = MBP+PLP1+MOG+MPZ+PMP22 counts of the 2 um bins within
                display_radius_um of the bin (per-bin smoothing of the member-bin counts; hhca.tl.local_myelin)
Fig 5 c: the 800 um window on the registered microscopy H&E (offset taken at the window centre) with the
clean-h5ad bins, and the neural / vascular / other bin counts in the window.

Inputs: config `fields` (H&E TIF, CytAssist image, registration), the CMFLAG-clean 2 um h5ad and the
SpaceRanger 2 um tissue positions of the slide; for panel f also <run-dir>/nerves/<slide>_neural_bins.csv.gz
(nerves.py) when present.
Outputs: <run-dir>/fields/<panel>_<id>.npz and <run-dir>/fields/fields.json.

Usage: python fields.py --run-dir results/nerve_macrophage [--config config/nerve_macrophage.yaml]
"""

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import hhca
import read_config

Image.MAX_IMAGE_PIXELS = None


class Slide:
    """The bins of the slide of the fields, its registered H&E and the CytAssist image."""

    def __init__(self, cfg, run_dir):
        settings, substrate = cfg["fields"], cfg["substrate"]
        self.name = settings["slide"]
        self.cfg = cfg

        def fill(template):
            return read_config.fill(template, cfg).replace("{s}", self.name)

        self.h5ad, self.registration, self.tif = (
            fill(substrate["h5ad_2um"]),
            fill(substrate["registration"]),
            fill(settings["tif"]),
        )
        self.adata = ad.read_h5ad(self.h5ad, backed="r")
        self.tissue_xy = hhca.io.read_tissue_positions(fill(substrate["spaceranger"]))
        registration = np.load(self.registration)
        self.um_to_px = (registration["M0"], registration["t0"])
        self.image = tifffile.memmap(self.tif, mode="r")
        cytassist_dir = fill(settings["cytassist_dir"])
        with open(f"{cytassist_dir}/scalefactors_json.json") as handle:
            scale = json.load(handle)
        self.cytassist_um_per_px = float(scale["microns_per_pixel"]) / float(scale["tissue_hires_scalef"])
        self.cytassist = np.asarray(Image.open(f"{cytassist_dir}/tissue_hires_image.png").convert("RGB"))
        neural_bins = f"{run_dir}/nerves/{self.name}_neural_bins.csv.gz"
        self.neural_bins = pd.read_csv(neural_bins) if os.path.exists(neural_bins) else None

    def offset(self, centre):
        """Local correction of the registration at a point (hhca.tl.registration_offset)."""
        settings = self.cfg["fields"]
        return hhca.tl.registration_offset(
            self.image,
            self.um_to_px,
            self.cytassist,
            self.cytassist_um_per_px,
            centre,
            half_um=settings["registration_tile_half_um"],
            res_um=settings["registration_res_um"],
            max_error=settings["registration_err_max"],
            max_um=settings["registration_offset_cap_um"],
        )


def extract_field(panel, field_id, centre, side, slide, outdir):
    """Write the H&E crop and the bins of one square field; returns its metadata."""
    cfg = slide.cfg
    window = hhca.tl.square_window(centre, side)
    offset = slide.offset(centre)
    field = hhca.tl.registered_field(
        slide.adata, slide.image, slide.um_to_px, window, offset["shift"], tissue_xy=slide.tissue_xy
    )
    n_tissue = field.pop("n_tissue")
    if panel == "f":
        field.update(
            hhca.tl.local_myelin(
                slide.adata,
                window,
                cfg["nerve"]["myelin_genes"],
                cfg["fields"]["f"]["display_radius_um"],
                cfg["nerve"]["neural_type"],
                slide.neural_bins,
            )
        )
    labels = field["blab"]
    meta = {
        "panel": panel,
        "id": field_id,
        "slide": slide.name,
        "centre": list(centre),
        "side_um": side,
        "offset_um": offset["offset_um"],
        "offset_err": offset["error"],
        "offset_applied": offset["applied"],
        "n_bins": int(len(labels)),
        "n_tissue_bins": n_tissue,
        "n_unassigned": int(len(field["ux"])),
        "counts": {level3_type: int((labels == level3_type).sum()) for level3_type in np.unique(labels)},
        "registration": slide.registration,
        "tif": slide.tif,
        "h5ad": slide.h5ad,
    }
    np.savez_compressed(f"{outdir}/{panel}_{field_id}.npz", **field, meta=json.dumps(meta))
    print(
        f"{panel}_{field_id}: offset {offset['offset_um']} err {offset['error']:.3f} applied {offset['applied']}; "
        f"{len(labels)} bins, {len(field['ux'])} unassigned; H&E {field['he'].shape}",
        flush=True,
    )
    return meta


def extract_window(slide, outdir):
    """Write the H&E and the bins of the Fig 5 c window (offset taken at its centre); returns its metadata."""
    cfg = slide.cfg
    x0, x1, y0, y1 = cfg["fields"]["c"]["window"]
    offset = slide.offset(((x0 + x1) / 2, (y0 + y1) / 2))
    field = hhca.tl.registered_field(
        slide.adata, slide.image, slide.um_to_px, (x0, x1, y0, y1), offset["shift"], margin_um=0.0
    )
    labels = field["blab"]
    neural = labels == cfg["nerve"]["neural_type"]
    vascular = np.isin(labels, cfg["proximity"]["vascular_types"])
    meta = {
        "panel": "c",
        "window": [x0, x1, y0, y1],
        "cmpp": slide.cytassist_um_per_px,
        "underlay": "registered microscopy H&E",
        "offset_um": offset["offset_um"],
        "offset_err": offset["error"],
        "offset_applied": offset["applied"],
        "bins": {
            "neural": int(neural.sum()),
            "vascular": int(vascular.sum()),
            "other": int((~vascular & ~neural).sum()),
        },
    }
    np.savez_compressed(f"{outdir}/c_window.npz", **field, meta=json.dumps(meta))
    print("c:", meta, flush=True)
    return meta


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="results folder of the pipeline; fields/ is created in it")
    parser.add_argument("--config", default=str(read_config.CONFIG), help="pipeline config")
    parser.add_argument("--panels", nargs="+", default=["b", "c", "f", "j"], help="panels to extract")
    return parser.parse_args()


def main():
    """Extract the fields of the requested panels."""
    args = parse_args()
    cfg = read_config.load(args.config)
    settings = cfg["fields"]
    outdir = f"{args.run_dir}/fields"
    os.makedirs(outdir, exist_ok=True)
    slide = Slide(cfg, args.run_dir)
    meta_all = {}
    if "b" in args.panels:
        meta_all["b_field"] = extract_field(
            "b", "field", settings["b"]["centre"], settings["b"]["side_um"], slide, outdir
        )
    if "f" in args.panels:
        meta_all["f_field"] = extract_field(
            "f", "field", settings["f"]["centre"], settings["f"]["side_um"], slide, outdir
        )
    if "j" in args.panels:
        for field_id, field in settings["j"]["fields"].items():
            meta_all[f"j_{field_id}"] = extract_field("j", field_id, field["centre"], field["side_um"], slide, outdir)
    if "c" in args.panels:
        meta_all["c_window"] = extract_window(slide, outdir)
    with open(f"{outdir}/fields.json", "w") as handle:
        json.dump(meta_all, handle, indent=2)


if __name__ == "__main__":
    main()
