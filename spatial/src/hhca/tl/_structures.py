"""Vessel structures from the FICTURE 2 um Level-3 decode of one section.

A lattice cell is vascular when at least `min_votes` of its 2 um decode pixels carry one of the vascular Level-3
types; its votes are the pixel counts per type. A vessel structure is a connected component (8-connectivity) of
vascular lattice cells.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import ndimage

from hhca.grid import bin_key, grid_index


def _vascular_codes(factor_order, vascular_types) -> list[int]:
    """K1 codes of the vascular types in the Level-3 factor order."""
    code_of = {level3_type: code for code, level3_type in enumerate(factor_order)}
    return [code_of[level3_type] for level3_type in vascular_types.values()]


def vascular_lattice(pixels, factor_order, vascular_types, lattice_um=8.0, min_votes=2) -> ad.AnnData:
    """Count the vascular decode pixels per lattice cell and vascular type.

    Parameters
    ----------
    pixels
        Decode pixels of one section with the columns ``X``, ``Y`` (um) and ``K1`` (code of the top type).
    factor_order
        Level-3 types in factor order; ``K1`` indexes this list.
    vascular_types
        Short name -> Level-3 type of the vascular types. The order breaks ties of the dominant type.
    lattice_um
        Side of a lattice cell.
    min_votes
        Vascular pixels needed for a lattice cell to be kept.

    Returns
    -------
    AnnData with one observation per vascular lattice cell and one variable per vascular type; ``X`` holds the
    votes, ``obs`` the lattice position (``gx``, ``gy``, ``key``, ``x_um``, ``y_um``), ``n_votes`` and the
    dominant type ``dom_subtype``, ``obsm["spatial"]`` the cell centres in um.
    """
    names = list(vascular_types)
    codes = _vascular_codes(factor_order, vascular_types)
    vote_columns = [f"vote_{name}" for name in names]
    vascular = pixels.loc[pixels["K1"].isin(codes).values, ["X", "Y", "K1"]]
    if vascular.empty:
        raise ValueError("the decode has no vascular pixel")
    gx, gy, key = grid_index(vascular["X"].values, vascular["Y"].values, lattice_um)
    votes = (
        pd.DataFrame({"key": key, "gx": gx, "gy": gy, "K1": vascular["K1"].values})
        .groupby(["key", "gx", "gy", "K1"])
        .size()
        .rename("n")
        .reset_index()
    )
    # lattice cell x subtype table of votes
    cells = votes.pivot_table(index=["key", "gx", "gy"], columns="K1", values="n", fill_value=0).reset_index()
    for code in codes:
        if code not in cells.columns:
            cells[code] = 0
    cells = cells.rename(columns=dict(zip(codes, vote_columns)))
    cells["n_votes"] = cells[vote_columns].sum(1)
    cells = cells[cells["n_votes"] >= min_votes].reset_index(drop=True)
    if cells.empty:
        raise ValueError("no lattice cell reaches min_votes")
    dominant = cells[vote_columns].values.argmax(1)
    cells["dom_subtype"] = pd.Categorical([names[i] for i in dominant], categories=names)
    cells["x_um"] = cells["gx"].values * lattice_um
    cells["y_um"] = cells["gy"].values * lattice_um

    obs = cells[["key", "gx", "gy", "n_votes", "dom_subtype", "x_um", "y_um"]].copy()
    obs.columns.name = None
    obs.index = cells["key"].astype(str).values
    var = pd.DataFrame({"cell_type": list(vascular_types.values()), "K1": codes}, index=names)
    return ad.AnnData(
        X=cells[vote_columns].to_numpy(np.float64),
        obs=obs,
        var=var,
        obsm={"spatial": cells[["x_um", "y_um"]].to_numpy(np.float64)},
    )


def vessel_structures(vessels) -> None:
    """Label the connected components (8-connectivity) of vascular lattice cells as vessel structures.

    Adds ``vessels.obs["struct"]``, the structure id of each lattice cell (1-based, per section).
    """
    gx, gy = vessels.obs["gx"].values, vessels.obs["gy"].values
    rows = (gy - int(gy.min())).astype(int)
    cols = (gx - int(gx.min())).astype(int)
    grid = np.zeros((rows.max() + 1, cols.max() + 1), bool)
    grid[rows, cols] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3), int))
    vessels.obs["struct"] = labels[rows, cols].astype(np.int64)


def structure_table(vessels, muscular_vsmc_fraction=0.10, lattice_um=8.0) -> pd.DataFrame:
    """Size, vote fractions, endothelial identity and vessel type of each structure.

    Vessel type: a muscular wall (VSMC vote fraction >= `muscular_vsmc_fraction`) makes an artery or vein,
    otherwise the dominant endothelial type decides (arteriole, venous, capillary/microvessel).
    """
    names = list(vessels.var_names)
    vote_columns = [f"vote_{name}" for name in names]
    cells = vessels.obs[["struct", "x_um", "y_um"]].reset_index(drop=True)
    cells[vote_columns] = np.asarray(vessels.X)
    structs = (
        cells.groupby("struct")
        .agg(
            n_bins=("struct", "size"),
            x_um=("x_um", "mean"),
            y_um=("y_um", "mean"),
            **{column: (column, "sum") for column in vote_columns},
        )
        .reset_index()
    )
    total = structs[vote_columns].sum(1).clip(lower=1)
    for name in names:
        structs[f"frac_{name}"] = structs[f"vote_{name}"] / total
    structs["area_um2"] = structs["n_bins"] * (lattice_um**2)
    structs["equiv_diam_um"] = 2.0 * np.sqrt(structs["area_um2"] / np.pi)
    # mural wall: VSMC = wall of muscular arteries and veins; pericytes = capillary mural cells
    structs["mural_VSMC_frac"] = structs["frac_VSMC"]
    structs["pericyte_frac"] = structs["frac_Pericyte"]
    # arterial / venous / capillary identity = dominant EC subtype; structures made of pericytes or VSMC
    # only have no EC lining and get "none"
    ec = structs[["frac_ArterialEC", "frac_CapillaryEC", "frac_VenousEC"]].copy()
    ec.columns = ["Arterial", "Capillary", "Venous"]
    ec_total = ec.sum(1).values
    identity = ec.idxmax(1).values.astype(object)
    identity[ec_total < 1e-9] = "none"
    structs["ec_identity"] = identity
    structs["ec_frac_total"] = ec_total
    fractions = structs[[f"frac_{name}" for name in names]]
    structs["dominant_subtype"] = fractions.idxmax(1).str.replace("frac_", "", regex=False)
    structs["vessel_type"] = np.where(
        structs["frac_VSMC"].values >= muscular_vsmc_fraction,
        np.where(identity == "Venous", "Venous(muscular)", "Arterial(muscular)"),
        np.where(
            identity == "Venous", "Venous", np.where(identity == "Arterial", "Arteriole", "Capillary/microvessel")
        ),
    )
    return structs


def clean_votes(vessels, pixels, spillover, lattice_um=8.0) -> None:
    """Recount the votes of each lattice cell without the decode pixels in CMFLAG spillover bins.

    Adds ``vessels.layers["clean"]``. The structures, their membership and their calibre stay as they are.

    Parameters
    ----------
    vessels
        Lattice cells of :func:`vascular_lattice`.
    pixels
        Decode pixels of the section (``X``, ``Y``, ``K1``).
    spillover
        Keys of the 2 um bins that carry a spillover flag (:func:`hhca.grid.spillover_keys`).
    """
    codes = vessels.var["K1"].to_numpy()
    vascular = pixels.loc[pixels["K1"].isin(codes).to_numpy()]
    x, y = vascular["X"].to_numpy(), vascular["Y"].to_numpy()
    counted = pd.DataFrame(
        {
            "key": bin_key(x, y, lattice_um),
            "K1": vascular["K1"].to_numpy(),
            "clean": ~np.isin(bin_key(x, y), spillover),
        }
    )
    votes = counted.groupby(["key", "K1"]).agg(raw=("clean", "size"), clean=("clean", "sum")).reset_index()
    wide = votes.pivot_table(index="key", columns="K1", values=["raw", "clean"], fill_value=0)
    wide = wide.reindex(vessels.obs["key"].to_numpy()).fillna(0)
    empty = np.zeros(len(wide))
    raw = np.column_stack([wide[("raw", code)].to_numpy() if ("raw", code) in wide else empty for code in codes])
    clean = np.column_stack([wide[("clean", code)].to_numpy() if ("clean", code) in wide else empty for code in codes])
    assert (raw == np.asarray(vessels.X)).all(), "recounted votes do not match the lattice"
    vessels.layers["clean"] = clean.astype(np.float64)


def structure_votes(vessels) -> pd.DataFrame:
    """Votes summed per structure: ``vote_<type>`` after the spillover mask, ``vote_<type>_raw`` before it."""
    per_cell = pd.DataFrame({"struct": vessels.obs["struct"].to_numpy()})
    for j, name in enumerate(vessels.var_names):
        per_cell[f"vote_{name}"] = np.asarray(vessels.layers["clean"])[:, j]
        per_cell[f"vote_{name}_raw"] = np.asarray(vessels.X)[:, j]
    columns = [column for column in per_cell.columns if column != "struct"]
    per_struct = (
        per_cell.groupby("struct")
        .agg(n_bins=("struct", "size"), **{column: (column, "sum") for column in columns})
        .reset_index()
    )
    for column in [*columns, "n_bins"]:
        per_struct[column] = per_struct[column].astype(np.int64)
    return per_struct
