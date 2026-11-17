#!/usr/bin/env python
"""Fig 4 f: candidate arteriole / venule fields and the per-field extracts drawn by figures/Fig4/panel_f.py.

The candidates are ranked on the vessel-structure table; an extract holds the H&E crop and the 2 um bin labels of
one field. For Fig 4 f, extract is run with --flags-dir results/substrate/cmflag/flags.

candidates (hhca.tl.field_candidates): structures on the slides with a native-resolution H&E registration
  (config fig4f.he), within the registered H&E and away from the capture edge, ranked by vascular pixel votes:
    arteriole = vessel_type Arterial(muscular) with an arterial-EC lining (ec_identity Arterial), 20-60 um
    venule    = vessel_type Venous or Venous(muscular), 15-60 um
  the top --per-class per slide and class, at least one field apart -> <run-dir>/fig4f/candidates.csv
extract (hhca.tl.extract_field): fields = config fig4f.candidates (fixed centres; --from-csv takes
  <run-dir>/fig4f/candidates.csv instead), all of them or only fig4f.selected; per field (side fig4f.field_um)
  -> <run-dir>/fig4f/fields/<id>.npz:
    he            registered H&E crop (uint8), sampled at native resolution through H_um_to_px
    ax, ay, atype 2 um bins with a FICTURE assignment: type of the highest-P1 decode pixel per 2 um key
    ux, uy        in-tissue SpaceRanger 2 um bins without an assignment (no decode pixel), plus, with
                  --flags-dir and fig4f.hatch_flagged, the bins removed by the four CMFLAG spillover flags

Inputs: the registered H&E images (config fig4f.he), the SpaceRanger tissue positions, the decode and Level-3
        factor order of pipeline `substrate` (--decode overrides the decode path) and the tables of results/vascular.
Outputs: <run-dir>/fig4f/candidates.csv (candidates), <run-dir>/fig4f/fields/<id>.npz (extract).

Usage:
  python fig4f_fields.py candidates [--run-dir results/vascular] [--config config/vascular.yaml] [--per-class 4]
  python fig4f_fields.py extract    [--run-dir results/vascular] [--flags-dir DIR] [--fields all|selected]
                                    [--from-csv] [--decode TEMPLATE_WITH_{slide}]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile

import hhca
from hhca.config import fill, load_config, resolve_roots, section_path
from hhca.grid import spillover_keys


def candidates(run_dir, per_class, cfg):
    """Rank arteriole and venule structures per slide and write the top fields to candidates.csv."""
    fig_cfg, structures = cfg["fig4f"], cfg["structures"]
    with open(section_path(cfg, "factor_order")) as handle:
        factor_order = json.load(handle)
    cal = pd.read_csv(f"{run_dir}/caliber/caliber_per_structure.csv")
    tables = []
    for slide, he in fig_cfg["he"].items():
        with tifffile.TiffFile(fill(he["tif"], resolve_roots(cfg))) as image:
            height, width = image.pages[0].shape[:2]
        vessels = hhca.io.read_vessels(
            f"{run_dir}/structures/{slide}_cells.parquet", structures["vascular_types"], factor_order
        )
        table = hhca.tl.field_candidates(
            pd.read_csv(f"{run_dir}/structures/{slide}_structs.csv"),
            cal[cal.slide == slide],
            vessels,
            he["H_um_to_px"],
            (width, height),
            float(fig_cfg["field_um"]),
            per_class,
        )
        short = slide.replace("VS_", "").replace("_henricke", "").replace("150319_34", "")
        table.insert(0, "id", [f"{short}_{klass[:3]}{rank}" for klass, rank in zip(table.klass, table["rank"])])
        table.insert(1, "slide", slide)
        tables.append(table.drop(columns="rank"))
    out = pd.concat(tables, ignore_index=True)
    os.makedirs(f"{run_dir}/fig4f", exist_ok=True)
    out.to_csv(f"{run_dir}/fig4f/candidates.csv", index=False)
    print(out.to_string(index=False))


def extract(run_dir, flags_dir, which_fields, from_csv, decode_template, cfg):
    """Write one npz per field: H&E crop, assigned 2 um bins with their type, unassigned bins."""
    fig_cfg = cfg["fig4f"]
    side = float(fig_cfg["field_um"])
    with open(section_path(cfg, "factor_order")) as handle:
        factor_order = json.load(handle)
    if from_csv:
        fields = pd.read_csv(f"{run_dir}/fig4f/candidates.csv")
    else:
        fields = pd.DataFrame(
            [dict(id=field_id, **field) for field_id, field in (fig_cfg.get("candidates") or {}).items()]
        )
    if which_fields == "selected":
        selected = [field_id for field_id in (fig_cfg.get("selected") or {}).values() if field_id]
        fields = fields[fields.id.isin(selected)]
    os.makedirs(f"{run_dir}/fig4f/fields", exist_ok=True)
    for slide, slide_fields in fields.groupby("slide"):
        he = fig_cfg["he"][slide]
        tif = fill(he["tif"], resolve_roots(cfg))
        decode = decode_template.format(slide=slide) if decode_template else section_path(cfg, "decode", slide)
        pixels = hhca.io.read_decode(decode, columns=("X", "Y", "K1", "P1"))
        tissue_xy = hhca.io.read_tissue_positions(section_path(cfg, "spaceranger", slide))
        spillover = None
        if flags_dir and fig_cfg.get("hatch_flagged", True):
            spillover = spillover_keys(f"{flags_dir}/{slide}_clean_bins.parquet")
        image = tifffile.memmap(tif)
        for row in slide_fields.itertuples():
            field = hhca.tl.extract_field(
                pixels, factor_order, tissue_xy, image, he["H_um_to_px"], (row.x_um, row.y_um), side, spillover
            )
            meta = slide_fields[slide_fields.id == row.id].iloc[0]
            np.savez_compressed(
                f"{run_dir}/fig4f/fields/{row.id}.npz",
                **field,
                meta=json.dumps(
                    dict(meta.to_dict(), side_um=side, tif=tif, flags_dir=flags_dir, decode=decode),
                    default=lambda value: value.item() if hasattr(value, "item") else str(value),
                ),
            )
            print(
                f"{row.id}: {len(field['ax'])} assigned, {len(field['ux'])} unassigned ({len(field['fx'])} CMFLAG) "
                f"2 um bins; H&E {field['he'].shape}",
                flush=True,
            )
        del image


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="cmd", required=True)
    for name in ("candidates", "extract"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", default="results/vascular", help="output directory of run_vascular.sh")
        command.add_argument("--config", default="config/vascular.yaml", help="pipeline config")
        if name == "candidates":
            command.add_argument("--per-class", type=int, default=4, help="fields kept per slide and vessel class")
        else:
            command.add_argument(
                "--flags-dir", default=None, help="directory with the CMFLAG flags; flagged bins are hatched"
            )
            command.add_argument(
                "--fields",
                choices=["all", "selected"],
                default="all",
                help="extract every candidate or only fig4f.selected",
            )
            command.add_argument(
                "--from-csv",
                action="store_true",
                help="take the fields from <run-dir>/fig4f/candidates.csv instead of the config",
            )
            command.add_argument(
                "--decode", default=None, help="decode path template with {slide} (default: config substrate.decode)"
            )
    return parser.parse_args()


def main():
    """Rank the candidate fields or write the field extracts."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.cmd == "candidates":
        candidates(args.run_dir, args.per_class, cfg)
    else:
        extract(args.run_dir, args.flags_dir, args.fields, args.from_csv, args.decode, cfg)


if __name__ == "__main__":
    main()
