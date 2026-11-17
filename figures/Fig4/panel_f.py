"""Fig 4f: arteriole and venule fields on H&E with the FICTURE 2 um vascular bins and hatched unassigned bins.

Fields are chosen in config/vascular.yaml (fig4f.selected, ids from fig4f.candidates).
Filled 2 um bins: the five vascular Level-3 types in the Level-3 palette. Hatched: in-tissue 2 um bins with no
cell-type assignment (no decode pixel, or removed by the CMFLAG spillover flags). Other assigned bins are left
transparent.

Inputs: the field extracts results/vascular/fig4f/fields/<id>.npz (scripts/vascular/fig4f_fields.py) and
        config/vascular.yaml.
Outputs: results/figures/Fig4/panel_f.{png,pdf}. --sheet draws every extracted candidate field to
         results/vascular/fig4f/candidates_sheet.png instead (a contact sheet for choosing the fields).

Usage: python figures/Fig4/panel_f.py [--fields ID ID] [--sheet]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel

TITLE = {"arteriole": "Arteriole", "venule": "Venule"}


def load_fields(extract_dir, ids):
    """Field extracts and their metadata, in the order of `ids`."""
    fields = [np.load(f"{extract_dir}/fields/{field_id}.npz", allow_pickle=False) for field_id in ids]
    return fields, [json.loads(str(field["meta"])) for field in fields]


def draw_sheet(fig_cfg, extract_dir):
    """Contact sheet of every extracted candidate field."""
    ids = [
        field_id
        for field_id in (fig_cfg.get("candidates") or {})
        if os.path.exists(f"{extract_dir}/fields/{field_id}.npz")
    ]
    fields, metas = load_fields(extract_dir, ids)
    titles = [
        f"{field_id}\n{meta.get('klass')}, {float(meta.get('diam_um', np.nan)):.0f} µm"
        for field_id, meta in zip(ids, metas)
    ]
    fig = hhca.pl.vessel_fields(fields, titles, fig_cfg["show_types"], n_cols=4)
    fig.savefig(f"{extract_dir}/candidates_sheet.png", dpi=150, facecolor="white")
    print(f"wrote {extract_dir}/candidates_sheet.png ({len(ids)} fields)")


def draw_panel(fig_cfg, extract_dir, ids):
    """The chosen fields side by side."""
    ids = ids or [field_id for field_id in (fig_cfg.get("selected") or {}).values() if field_id]
    if not ids:
        candidates = fig_cfg["candidates"]
        ids = [
            next(field_id for field_id, field in candidates.items() if field["klass"] == klass)
            for klass in ("arteriole", "venule")
        ]
        print("fig4f.selected is empty in config/vascular.yaml; drawing the top-ranked candidates", ids)
    fields, metas = load_fields(extract_dir, ids)
    titles = [TITLE.get(meta.get("klass"), field_id) for field_id, meta in zip(ids, metas)]
    fig = hhca.pl.vessel_fields(fields, titles, fig_cfg["show_types"])
    save_panel(fig, "Fig4", "f")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", nargs="*", default=None, help="field ids to draw (default: fig4f.selected)")
    parser.add_argument("--sheet", action="store_true", help="draw the contact sheet of all extracted candidates")
    return parser.parse_args()


def main():
    """Draw the panel, or the contact sheet with --sheet."""
    args = parse_args()
    extract_dir = hhca.results_dir("vascular", "fig4f")
    fig_cfg = load_config("vascular")["fig4f"]
    if args.sheet:
        draw_sheet(fig_cfg, extract_dir)
    else:
        draw_panel(fig_cfg, extract_dir, args.fields)


if __name__ == "__main__":
    main()
