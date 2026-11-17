"""Counts summed per lattice cell and per vessel structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.grid import bin_key


def _collapse_by_name(matrix, names):
    """Sum the columns that share a feature name (gene symbols mapped to more than one Ensembl id).

    Names are compared as they are, so symbols ending in -digit such as NKX2-5 are kept apart.
    Returns (collapsed CSR matrix, sorted unique names).
    """
    unique_names, inverse = np.unique(np.asarray(names), return_inverse=True)
    membership = sp.csr_matrix(
        (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)), shape=(len(inverse), len(unique_names))
    )
    return (matrix @ membership).tocsr(), unique_names


def lattice_counts(adata, genes, panel=None, spillover=None, lattice_um=8.0) -> pd.DataFrame:
    """Sum the library and the counts of `genes` of the 2 um bins per lattice cell.

    Parameters
    ----------
    adata
        Space Ranger 2 um bins of one section with ``obsm["spatial"]`` in um and the feature names as in the
        matrix (:func:`hhca.io.read_visium_hd`).
    genes
        Genes to sum; a gene that is not in the matrix gives zeros.
    panel
        Feature names that make up the library (default: all). Columns that share a name are summed.
    spillover
        Keys of 2 um bins to leave out (:func:`hhca.grid.spillover_keys`).

    Returns
    -------
    Table indexed by the lattice key with the library ``tot`` and one column per gene.
    """
    names = np.asarray(adata.var_names)
    matrix = adata.X.tocsr()
    if panel is not None:
        keep = np.isin(names, list(panel))
        matrix, names = matrix[:, keep], names[keep]
    matrix, symbols = _collapse_by_name(matrix, names)
    matrix = matrix.astype(np.float32)
    x_um, y_um = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    if spillover is not None:
        keep = ~np.isin(bin_key(x_um, y_um), spillover)
        matrix, x_um, y_um = matrix[keep], x_um[keep], y_um[keep]
    column_of = {symbol: i for i, symbol in enumerate(symbols)}
    per_bin = {"key": bin_key(x_um, y_um, lattice_um), "tot": np.asarray(matrix.sum(1)).ravel().astype(np.float64)}
    for gene in genes:
        per_bin[gene] = (
            np.asarray(matrix[:, column_of[gene]].todense()).ravel().astype(np.float64)
            if gene in column_of
            else np.zeros(matrix.shape[0])
        )
    return pd.DataFrame(per_bin).groupby("key", sort=True).sum()


def structure_pseudobulk(lattice, vessels, structures) -> pd.DataFrame:
    """Sum the lattice counts over the cells of each vessel structure.

    Returns
    -------
    One row per structure: its type, composition and size from `structures`, the library ``lib`` and the counts
    of every gene of `lattice`.
    """
    genes = [column for column in lattice.columns if column != "tot"]
    struct_of_key = vessels.obs.set_index("key")["struct"]
    cells = lattice.reset_index()
    cells["struct"] = cells["key"].map(struct_of_key)
    in_structure = cells.dropna(subset=["struct"])
    pseudobulk = in_structure.groupby("struct").agg({"tot": "sum", **dict.fromkeys(genes, "sum")}).reset_index()
    pseudobulk = pseudobulk.rename(columns={"tot": "lib"})
    described = ["struct", "vessel_type", "pericyte_frac", "mural_VSMC_frac", "equiv_diam_um", "n_bins", "x_um", "y_um"]
    return structures[described].merge(pseudobulk, on="struct", how="inner")


def pseudobulk_structures(
    adata, vessels, spillover, genes, library_genes, lattice_um=8.0, chunk=400_000
) -> pd.DataFrame:
    """Sum the counts of 2 um bins per vessel structure, with and without the CMFLAG spillover mask.

    Every 2 um bin is mapped to its lattice cell and from there to a structure. The matrix is read in blocks of
    `chunk` bins, so `adata` can be opened with ``backed="r"``.

    Parameters
    ----------
    adata
        2 um bins of one section (bins x genes) with ``obs["x_um"]`` and ``obs["y_um"]``.
    vessels
        Lattice cells of the section with ``obs["struct"]``.
    spillover
        Keys of the 2 um bins that carry a spillover flag.
    genes
        Genes whose counts are summed per structure.
    library_genes
        Genes that make up the library ``libc``.

    Returns
    -------
    One row per structure: total library ``lib``, library over `library_genes` ``libc``, bin count ``nb`` and the
    counts of each gene, each as ``*_raw`` (all bins) and ``*_clean`` (bins without a spillover flag).
    """
    struct_of_key = pd.Series(vessels.obs["struct"].to_numpy(), index=vessels.obs["key"].to_numpy())
    assert struct_of_key.index.is_unique
    var = np.asarray(adata.var_names).astype(str)
    x = adata.obs["x_um"].to_numpy().astype(np.float64)
    y = adata.obs["y_um"].to_numpy().astype(np.float64)
    n = adata.n_obs
    column_of = {gene: i for i, gene in enumerate(var)}
    # position of each matrix column in `genes`, -1 for the other genes
    gene_slot = np.full(len(var), -1, np.int64)
    for j, gene in enumerate(genes):
        if gene in column_of:
            gene_slot[column_of[gene]] = j
    library_genes = set(library_genes)
    in_library = np.fromiter((v in library_genes for v in var), bool, len(var))
    struct_of_bin = struct_of_key.reindex(bin_key(x, y, lattice_um)).to_numpy()
    inside = ~np.isnan(struct_of_bin)
    clean = ~np.isin(bin_key(x, y), spillover, assume_unique=False)
    structs = np.unique(struct_of_bin[inside]).astype(np.int64)
    n_structs = len(structs)
    row_of_struct = pd.Series(np.arange(n_structs), index=structs)
    sums = {
        name: np.zeros((n_structs,), np.float64)
        for name in ("lib_raw", "lib_clean", "libc_raw", "libc_clean", "nb_raw", "nb_clean")
    }
    gene_sums = {"raw": np.zeros((n_structs, len(genes)), np.float64)}
    gene_sums["clean"] = np.zeros_like(gene_sums["raw"])
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        inside_chunk = inside[start:end]
        if not inside_chunk.any():
            continue
        block = sp.csr_matrix(adata.X[start:end])
        data = block.data.astype(np.float64)
        columns = block.indices
        rows = np.repeat(np.arange(end - start), np.diff(block.indptr))
        keep = inside_chunk[rows]
        data, columns, rows = data[keep], columns[keep], rows[keep]
        struct_rows = row_of_struct.reindex(struct_of_bin[start:end][inside_chunk].astype(np.int64)).to_numpy()
        struct_row_of_bin = np.full(end - start, -1, np.int64)
        struct_row_of_bin[np.where(inside_chunk)[0]] = struct_rows
        struct_row = struct_row_of_bin[rows]
        clean_entry = clean[start:end][rows]
        for tag, mask in (("raw", np.ones_like(clean_entry)), ("clean", clean_entry)):
            sums[f"lib_{tag}"] += np.bincount(struct_row[mask], weights=data[mask], minlength=n_structs)
            sums[f"libc_{tag}"] += np.bincount(
                struct_row[mask], weights=data[mask] * in_library[columns[mask]], minlength=n_structs
            )
            target = (gene_slot[columns] >= 0) & mask
            if target.any():
                gene_sums[tag][:] += sp.coo_matrix(
                    (data[target], (struct_row[target], gene_slot[columns[target]])),
                    shape=(n_structs, len(genes)),
                ).toarray()
        bins_here = np.where(inside_chunk)[0]
        sums["nb_raw"] += np.bincount(struct_row_of_bin[bins_here], minlength=n_structs)
        sums["nb_clean"] += np.bincount(
            struct_row_of_bin[bins_here], weights=clean[start:end][bins_here], minlength=n_structs
        )
    out = pd.DataFrame({"struct": structs, **sums})
    for j, gene in enumerate(genes):
        out[f"{gene}_raw"] = gene_sums["raw"][:, j]
        out[f"{gene}_clean"] = gene_sums["clean"][:, j]
    return out
