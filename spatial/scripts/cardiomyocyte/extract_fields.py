"""Extract the small inputs of the spatial panels Fig 3 e, ED6 f and Fig 1 j from the 2 um h5ads and the H&E.

  fig3e.npz         Fig 3 e: 2 um bins in the 500 um field of H0054 endo (x, y, L3 label, NPPA, NPPB;
                    hhca.tl.extract_window)
  ed6f.npz          ED6 f: whole-section L3 raster of H0054 endo (raster_um pixels; a pixel takes the most frequent
                    non-cardiomyocyte label of its bins, else the cardiomyocyte label, so non-CM bins are drawn
                    on top; hhca.tl.label_raster) + the bins of the 340 um detail field
  fig1j_<name>.npz  Fig 1 j: H&E crop (registered microscopy image resampled at um_per_px) + bins of each 92 um
                    field (hhca.tl.he_window; only when config fig1j.centres_um is filled)

Inputs: (config/cardiomyocyte.yaml paths) h5ad_2um = CMFLAG-clean 2 um h5ads; H&E: he_tif with the CMFLAG
registration (M0, t0: um -> microscopy px).
Outputs: the npz files above in --out-dir and manifest_<panels>.json with the bin counts per output.

Usage: extract_fields.py --config config/cardiomyocyte.yaml --out-dir results/cardiomyocyte/fields
       [--only fig3e,ed6f,fig1j]
"""

import argparse
import json
import os

import anndata as ad
import numpy as np

import cm_config
import hhca


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Inputs of the spatial panels Fig 3 e, ED6 f and Fig 1 j")
    parser.add_argument("--config", required=True, help="config/cardiomyocyte.yaml")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--only", default="", help="comma-separated subset of fig3e,ed6f,fig1j (default: all)")
    return parser.parse_args()


def read_section(cfg, slide):
    """The CMFLAG-clean 2 um bins of a slide, backed."""
    return ad.read_h5ad(cm_config.path(cfg, "h5ad_2um", slide), backed="r")


def extract_fig3e(cfg, out_dir):
    """Bins and gene counts of the Fig 3 e field; returns the manifest entry."""
    params = cfg["fig3e"]
    field = hhca.tl.extract_window(
        read_section(cfg, params["slide"]), params["centre_um"], params["size_um"], params["genes"]
    )
    np.savez_compressed(f"{out_dir}/fig3e.npz", **field, n_window_all=np.int64(field["x"].size))
    return dict(
        slide=params["slide"],
        n_bins=int(field["x"].size),
        **{f"{gene}_pos": int((field[gene] > 0).sum()) for gene in params["genes"]},
    )


def extract_ed6f(cfg, out_dir):
    """Whole-section label raster and the bins of the detail field of ED6 f; returns the manifest entry."""
    params = cfg["ed6f"]
    adata = read_section(cfg, params["slide"])
    raster = hhca.tl.label_raster(adata, float(params["raster_um"]), cfg["cell_types"]["cardiomyocytes"])
    detail = hhca.tl.extract_window(adata, params["zoom_centre_um"], params["zoom_size_um"])
    np.savez_compressed(
        f"{out_dir}/ed6f.npz",
        **raster,
        zx=detail["x"],
        zy=detail["y"],
        zcode=detail["code"],
        zoom=np.array([*params["zoom_centre_um"], params["zoom_size_um"]]),
    )
    return {
        "slide": params["slide"],
        "n_bins": int(adata.n_obs),
        "n_clean": int(adata.n_obs),
        "raster": [int(n) for n in raster["img"].shape],
        "zoom_bins": int(detail["x"].size),
    }


def extract_fig1j(cfg, out_dir):
    """H&E crop and bins of every Fig 1 j field; returns the manifest entries."""
    import tifffile

    params = cfg["fig1j"]
    slide = params["slide"]
    adata = read_section(cfg, slide)
    registration = np.load(cm_config.path(cfg, "registration", slide))
    image = tifffile.memmap(cm_config.path(cfg, "he_tif", slide), mode="r")
    entries = {}
    for name, centre in params["centres_um"].items():
        field = hhca.tl.he_window(
            adata,
            image,
            (registration["M0"], registration["t0"]),
            centre,
            float(params["size_um"]),
            float(params["um_per_px"]),
        )
        np.savez_compressed(f"{out_dir}/fig1j_{name}.npz", **field)
        labels = field["cats"][field["code"]]
        entries[f"fig1j_{name}"] = {
            "centre": list(centre),
            "n_bins": int(field["x"].size),
            "n_target": int((labels == params["tiles"][name]["target"]).sum()),
        }
    return entries


def main():
    """Write the requested panel inputs and the manifest."""
    args = parse_args()
    cfg, _ = cm_config.load(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    only = {s for s in args.only.split(",") if s} or {"fig3e", "ed6f", "fig1j"}

    manifest = {}
    if "fig3e" in only:
        manifest["fig3e"] = extract_fig3e(cfg, args.out_dir)
    if "ed6f" in only:
        manifest["ed6f"] = extract_ed6f(cfg, args.out_dir)
    if "fig1j" in only and cfg["fig1j"].get("centres_um"):
        manifest.update(extract_fig1j(cfg, args.out_dir))
    with open(f"{args.out_dir}/manifest_{'_'.join(sorted(only))}.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
