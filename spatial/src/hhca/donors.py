"""Donor metadata of the HHCA atlas, and the files that hold the per-donor tables of the atlas pipeline."""

from __future__ import annotations

import json
import re


def parse_age(stage) -> float:
    """Age in years from a ``development_stage`` entry.

    A band such as ``"60-69"`` gives its midpoint, a plain number is returned as is and anything else
    (``"unknown"``, empty) gives NaN.
    """
    text = str(stage).strip()
    if text in ("", "unknown", "nan", "NA", "None"):
        return float("nan")
    band = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if band:
        return (int(band.group(1)) + int(band.group(2))) / 2.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_donor_map(path) -> dict[str, str]:
    """Atlas donor id -> donor (columns ``raw_donor_id``, ``donor_id``): 291 ids, 280 donors."""
    import pandas as pd

    return pd.read_csv(path).astype(str).set_index("raw_donor_id")["donor_id"].to_dict()


def _per_donor(scores, column):
    """Value of an obs column for every donor of a (donor, cell type) AnnData."""
    n_types = len(scores.obs["cell_type"].cat.categories)
    return scores.obs[column].to_numpy()[::n_types]


def write_scores(path, donors, scores, gene_set_meta) -> None:
    """Write the per-donor gene-set scores (:func:`hhca.tl.gene_set_scores`) and the donor table as one npz.

    Per level the file holds the cell types (``cats__<level>``), the score sums (``sums__<level>``, rows = donor x
    cell type) and the nuclei (``cnts__<level>``).
    """
    import numpy as np

    first = next(iter(scores.values()))
    names = list(first.var_names)
    payload = {
        "groups": np.array(names, dtype=object),
        "n_present": np.array(list(first.var["n_genes"])),
        "genes_present": np.array(list(first.var["genes"]), dtype=object),
        "gmeta_json": json.dumps({name: gene_set_meta[name] for name in names}),
        "donors": np.array(list(donors.index), dtype=object),
        "donor_sex": np.array(list(donors["sex"]), dtype=object),
        "donor_age": np.array(list(donors["age"]), dtype=float),
        "donor_tissue": np.array(list(donors["tissue"]), dtype=object),
        "donor_tissue_frac": np.array(list(donors["tissue_frac"]), dtype=float),
        "donor_assay": np.array(list(donors["assay"]), dtype=object),
        "donor_suspension": np.array(list(donors["suspension"]), dtype=object),
        "donor_disease": np.array(list(donors["disease"]), dtype=object),
        "donor_ncells": np.array(list(donors["ncells"]), dtype=np.int64),
        "levels": np.array(list(scores), dtype=object),
    }
    for level, level_scores in scores.items():
        payload[f"cats__{level}"] = np.array(list(level_scores.obs["cell_type"].cat.categories), dtype=object)
        payload[f"sums__{level}"] = np.asarray(level_scores.X)
        payload[f"cnts__{level}"] = level_scores.obs["n_nuclei"].to_numpy()
    np.savez_compressed(path, **payload)


def read_scores(path, levels=None):
    """Read a file of :func:`write_scores`.

    Returns
    -------
    The donor table, the mapping of level -> AnnData of (donor, cell type) x gene sets, and the description of
    the gene sets.
    """
    import numpy as np
    import pandas as pd

    from hhca.tl._donors import DONOR_COLUMNS, grouped

    data = np.load(path, allow_pickle=True)
    donors = pd.DataFrame(
        {column: data[f"donor_{column}"] for column in DONOR_COLUMNS},
        index=pd.Index([str(d) for d in data["donors"]], name="donor"),
    )
    var = pd.DataFrame(
        {"n_genes": data["n_present"], "genes": data["genes_present"]}, index=[str(g) for g in data["groups"]]
    )
    scores = {
        str(level): grouped(
            data[f"sums__{level}"], donors.index, data[f"cats__{level}"], data[f"cnts__{level}"], var.copy()
        )
        for level in (levels or data["levels"])
    }
    return donors, scores, json.loads(str(data["gmeta_json"]))


def write_programme_scores(path, donors, scores, extra=()) -> None:
    """Write per-donor programme scores (:func:`hhca.tl.programme_scores`) as an npz.

    ``scores`` has the shape donors x cell types x programmes and ``cnts`` donors x cell types; `extra` names
    further columns of the donor table to store (``donor_<column>``).
    """
    import numpy as np

    cell_types = list(scores.obs["cell_type"].cat.categories)
    shape = (len(donors), len(cell_types))
    payload = {
        "donors": np.array(list(donors.index), dtype=object),
        "donor_sex": np.array(list(donors["sex"]), dtype=object),
        "donor_age": np.array(list(donors["age"]), dtype=float),
        "celltypes": np.array(cell_types, dtype=object),
        "cytokines": np.array(list(scores.var_names), dtype=object),
        "scores": np.asarray(scores.X).reshape(*shape, scores.n_vars),
        "cnts": scores.obs["n_nuclei"].to_numpy().reshape(shape),
    }
    for column in extra:
        payload[f"donor_{column}"] = np.array(list(donors[column]), dtype=object)
    np.savez_compressed(path, **payload)


def read_programme_scores(path):
    """Read a file of :func:`write_programme_scores`: the donor table (the stored columns) and the scores."""
    import numpy as np
    import pandas as pd

    from hhca.tl._donors import grouped

    data = np.load(path, allow_pickle=True)
    columns = {key[len("donor_") :]: data[key] for key in data.files if key.startswith("donor_")}
    donors = pd.DataFrame(columns, index=pd.Index([str(d) for d in data["donors"]], name="donor"))
    donors["sex"] = donors["sex"].astype(str)
    donors["age"] = donors["age"].astype(float)
    cytokines = [str(c) for c in data["cytokines"]]
    scores = grouped(
        data["scores"].astype(float).reshape(-1, len(cytokines)),
        donors.index,
        data["celltypes"],
        data["cnts"].reshape(-1),
        pd.DataFrame(index=cytokines),
    )
    return donors, scores


def write_lv_expression(path, expression) -> None:
    """Write the per-donor expression of the left ventricle (:func:`hhca.tl.lv_expression`) as an npz.

    ``means`` has the shape donors x cell types x genes and ``ncells`` donors x cell types.
    """
    import numpy as np

    donors = list(expression.obs["donor"].cat.categories)
    cell_types = list(expression.obs["cell_type"].cat.categories)
    shape = (len(donors), len(cell_types))
    np.savez_compressed(
        path,
        means=np.asarray(expression.X).reshape(*shape, expression.n_vars),
        ncells=expression.obs["n_nuclei"].to_numpy().reshape(shape),
        donors=np.array(donors).astype(str),
        celltypes=np.array(cell_types).astype(str),
        genes=np.array(list(expression.var_names)),
        donor_dataset=_per_donor(expression, "dataset").astype(str),
        donor_sex=_per_donor(expression, "sex").astype(str),
    )


def read_lv_expression(path):
    """Read a file of :func:`write_lv_expression` as AnnData of (donor, cell type) x genes."""
    import numpy as np
    import pandas as pd

    from hhca.tl._donors import grouped

    data = np.load(path, allow_pickle=True)
    n_types, genes = len(data["celltypes"]), [str(g) for g in data["genes"]]
    return grouped(
        data["means"].reshape(-1, len(genes)),
        data["donors"],
        data["celltypes"],
        data["ncells"].reshape(-1),
        pd.DataFrame(index=genes),
        dataset=np.repeat(np.array([str(x) for x in data["donor_dataset"]], dtype=object), n_types),
        sex=np.repeat(np.array([str(x) for x in data["donor_sex"]], dtype=object), n_types),
    )
