"""Fig 1 j: locate the gallery H&E fields on the registered microscopy image.

The gallery pages (<name>_hires.png, one per cell type) carry one 92 um H&E tile each (config fig1j.tiles: box of
the tile in the page). This script finds each tile on the slide: the H&E is resampled to the um frame of the bins
at 1 um/px through the CMFLAG registration (M0, t0; hhca.tl.section_grey), each tile (92 um, overlay included) is
converted to grey and resized to 92 px, and normalised cross-correlation is run over the whole section for
8 orientations x 3 scales (hhca.tl.locate_tile).
The best score, the runner-up peak outside a 92 um radius and the centre in um are written to locate.json.

Inputs: the gallery pages in --gallery-dir; the slide's 2 um h5ad (paths.h5ad_2um_full), registration
(paths.registration) and H&E (paths.he_tif) of the config. Needs OpenCV (envs/substrate_cmflag.yml).
Outputs: in --out-dir, locate.json and the grey slide image slide_grey_<res>um.npy.

Usage: fig1j_locate.py --config config/cardiomyocyte.yaml --gallery-dir DIR --out-dir DIR [--um-per-px 1.0]
"""

import argparse
import json
import os

import h5py
import numpy as np
import tifffile
from PIL import Image

import cm_config
import hhca

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Locate the Fig 1 j gallery tiles on the registered H&E")
    parser.add_argument("--config", required=True, help="config/cardiomyocyte.yaml")
    parser.add_argument("--gallery-dir", required=True, help="folder with the gallery pages <name>_hires.png")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--um-per-px", type=float, default=1.0, help="resolution of the search image")
    return parser.parse_args()


def main():
    """Resample the H&E, match every gallery tile and write locate.json."""
    args = parse_args()
    cfg, _ = cm_config.load(args.config)
    gallery = cfg["fig1j"]
    slide = gallery["slide"]
    os.makedirs(args.out_dir, exist_ok=True)
    um_per_px = args.um_per_px

    with h5py.File(cm_config.path(cfg, "h5ad_2um_full", slide), "r") as h5ad:
        x = h5ad["obs"]["x_um"][:]
        y = h5ad["obs"]["y_um"][:]
    x0, x1, y0, y1 = float(x.min()) - 50, float(x.max()) + 50, float(y.min()) - 50, float(y.max()) + 50
    registration = np.load(cm_config.path(cfg, "registration", slide))
    image = tifffile.memmap(cm_config.path(cfg, "he_tif", slide), mode="r")
    section = hhca.tl.section_grey(image, (registration["M0"], registration["t0"]), (x0, x1, y0, y1), um_per_px)
    print(
        f"slide grey {section.shape} at {um_per_px} um/px, extent x {x0:.0f}-{x1:.0f} y {y0:.0f}-{y1:.0f}",
        flush=True,
    )
    np.save(f"{args.out_dir}/slide_grey_{um_per_px:g}um.npy", section.astype(np.uint8))

    size_um = float(gallery["size_um"])
    located = {}
    for name, tile in gallery["tiles"].items():
        page = Image.open(f"{args.gallery_dir}/{name}_hires.png").convert("RGB")
        rgb = np.asarray(page.crop(tuple(tile["box"]))).astype(np.float32)
        tile_grey = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        match = hhca.tl.locate_tile(section, tile_grey, size_um, um_per_px)
        located[name] = {
            "centre_um": [
                round(x0 + (match["column"] + match["n"] / 2) * um_per_px, 1),
                round(y0 + (match["row"] + match["n"] / 2) * um_per_px, 1),
            ],
            "ncc": round(match["ncc"], 4),
            "runner_up": round(match["runner_up"], 4),
            "scale": match["scale"],
            "rot90": match["rot90"],
            "flip": match["flip"],
        }
        print(name, located[name], flush=True)
    with open(f"{args.out_dir}/locate.json", "w") as handle:
        json.dump(located, handle, indent=1)


if __name__ == "__main__":
    main()
