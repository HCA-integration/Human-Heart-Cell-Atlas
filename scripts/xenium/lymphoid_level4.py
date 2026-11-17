#!/usr/bin/env python
"""Level-4 lymphoid states in the CellPin latent space (rejection cohort).

A 4-class lineage model {CD4 T, CD8 T, Treg, NK} applied to the whole lymphoid compartment, proliferation taken
from L3 (label8 = lineage +/- proliferating):

  * lineage reference = every atlas cell whose Final Level 4_harmonized maps to CD4 T / Treg / CD8 T / NK
    (config lymphoid_l4.lineage_from_L4; gamma-delta, MAIT, ILC and 'T / NK (Proliferating)' dropped),
    embedded through the trained CellPin panel encoder;
  * latent standardised with the reference-subsample statistics saved by transfer_labels.py (latent.npz);
  * distance-weighted kNN, K = 25 (as the L3 transfer); held-out = 20 % stratified split, seed 0;
  * query = Xenium cells whose CellPin L3 is T cells, NK cells or Lymphoid Cells (Proliferating);
    label8 = ('Proliferating ' + lineage) if L3 == 'Lymphoid Cells (Proliferating)' else lineage
    (hhca.tl.lymphoid_level4).

A second evaluation uses only the T / NK cells of the CellPin training subsample (at most 2000 per L3) as
lineage reference, to show what the subsample alone supports.

Inputs:
    --ref     reference_subset.h5ad, --extra reference_extra_lymphoid.h5ad (build_cellpin_reference.py)
    --xenium  xenium_genes.h5ad
    --model   trained CellPin model (train_cellpin.py)
    --latent  latent.npz, --pred per_cell_predictions.csv (transfer_labels.py)
    --config  config/xenium.yaml (lymphoid_l4)

Outputs (in --outdir):
    xenium_lymphoid8_cellpin.csv, xenium_lymphoid8_sensitivity.csv, extra_latent.npz, lymphoid_l4_summary.json

Usage: python lymphoid_level4.py --ref reference_subset.h5ad --extra reference_extra_lymphoid.h5ad
           --xenium xenium_genes.h5ad --model MODEL.pt --latent latent.npz --pred per_cell_predictions.csv
           --config config/xenium.yaml --outdir DIR
"""

import argparse
import json
import os

import anndata as ad
import cellpin.pp
import numpy as np
import pandas as pd
import torch
from cellpin.models import CellPin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

import hhca
from hhca.config import load_config

K = 25
SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", required=True, help="reference_subset.h5ad (CellPin training reference)")
    parser.add_argument(
        "--extra",
        required=True,
        help="all atlas T / NK / proliferating / ILC cells (build_cellpin_reference.py --extra-out)",
    )
    parser.add_argument("--xenium", required=True, help="xenium_genes.h5ad")
    parser.add_argument("--model", required=True, help="trained CellPin model (.pt)")
    parser.add_argument("--latent", required=True, help="latent.npz of transfer_labels.py (ref_emb, mu, sd)")
    parser.add_argument("--pred", required=True, help="per_cell_predictions.csv of transfer_labels.py")
    parser.add_argument("--config", required=True, help="config/xenium.yaml")
    parser.add_argument("--outdir", required=True, help="output directory")
    return parser.parse_args()


def evaluate(Z, y, classes, tag):
    """Held-out accuracy of the distance-weighted kNN on a 20 % stratified split."""
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(Z[train], y[train])
    predicted = classifier.predict(Z[test])
    report = classification_report(y[test], predicted, zero_division=0, output_dict=True)
    out = {
        "tag": tag,
        "n_reference": int(len(y)),
        "n_heldout": int(len(test)),
        "accuracy": float(accuracy_score(y[test], predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
        "per_class": {
            c: {
                "precision": report[c]["precision"],
                "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
                "support": int(report[c]["support"]),
            }
            for c in classes
            if c in report
        },
        "confusion": pd.DataFrame(
            confusion_matrix(y[test], predicted, labels=classes), index=classes, columns=classes
        ).to_dict(),
    }
    f1 = {c: round(v["f1"], 3) for c, v in out["per_class"].items()}
    print(f"[l4] {tag}: acc {out['accuracy']:.3f} balanced {out['balanced_accuracy']:.3f} F1 {f1}", flush=True)
    return out


def embed_extra_cells(model, sc_adata, extra, panel):
    """Embed the extra atlas cells like spatial queries: panel genes only, panel encoder."""
    _, extra_ds = cellpin.pp.setup_data(sc_adata=sc_adata, st_adata=extra[:, panel].copy(), layer="counts")
    assert list(extra_ds.panel_genes) == panel
    return model.get_cell_embedding(DataLoader(extra_ds, batch_size=512, shuffle=False, num_workers=0), use_mean=True)


def sensitivity(Zref, yref, classes, xenZ_query, cells, pretty):
    """Class-balanced alternatives in the same latent (not used for the labels).

    Same held-out split as `evaluate`; the test set keeps the natural class priors.
    """
    idx = np.arange(len(yref))
    train, test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=yref)
    rng = np.random.default_rng(SEED)

    def balanced_rows(rows):
        n_min = min(int((yref[rows] == c).sum()) for c in classes)
        return np.concatenate([rng.choice(rows[yref[rows] == c], n_min, replace=False) for c in classes])

    def knn_balanced(rows):
        balanced = balanced_rows(rows)
        return KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(Zref[balanced], yref[balanced])

    def logreg_weighted(rows):
        return LogisticRegression(class_weight="balanced", max_iter=2000).fit(Zref[rows], yref[rows])

    alternatives = {"knn_balanced_reference": knn_balanced, "logreg_class_weighted": logreg_weighted}
    summary, per_cell = {}, pd.DataFrame({"cell_id": cells.cell_id.values})
    for name, fit in alternatives.items():
        predicted = fit(train).predict(Zref[test])
        report = classification_report(yref[test], predicted, zero_division=0, output_dict=True)
        lineage = fit(idx).predict(xenZ_query)
        label8 = np.where(
            cells.is_prolif.values,
            "Proliferating " + pd.Series(lineage).map(pretty).values,
            pd.Series(lineage).map(pretty).values,
        )
        per_cell[f"label8_{name}"] = label8
        summary[name] = {
            "accuracy": float(accuracy_score(yref[test], predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(yref[test], predicted)),
            "f1": {c: float(report[c]["f1-score"]) for c in classes if c in report},
            "label8_counts": pd.Series(label8).value_counts().to_dict(),
        }
        f1 = {c: round(v, 3) for c, v in summary[name]["f1"].items()}
        print(
            f"[l4] sensitivity {name}: acc {summary[name]['accuracy']:.3f} "
            f"balanced {summary[name]['balanced_accuracy']:.3f} F1 {f1}",
            flush=True,
        )
    return summary, per_cell


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    settings = load_config(args.config)["lymphoid_l4"]
    lineage_of_l4 = {l4: lineage for lineage, l4_names in settings["lineage_from_L4"].items() for l4 in l4_names}
    classes = list(settings["lineage_from_L4"].keys())

    sc_adata = ad.read_h5ad(args.ref)
    st_adata = ad.read_h5ad(args.xenium)
    extra = ad.read_h5ad(args.extra)
    sc_ds, st_ds = cellpin.pp.setup_data(sc_adata=sc_adata, st_adata=st_adata, layer="counts")
    model = CellPin(sc_dataset=sc_ds, checkpoint=args.model)
    model.eval()
    extra_emb = embed_extra_cells(model, sc_adata, extra, list(sc_ds.panel_genes))

    latent = np.load(args.latent, allow_pickle=True)
    mu, sd = latent["mu"], latent["sd"]
    # extra cells that are also in the training subsample should get the embedding saved by transfer_labels.py
    ref_position = {cell: i for i, cell in enumerate(latent["ref_cells"])}
    shared = [(i, ref_position[cell]) for i, cell in enumerate(extra.obs_names.astype(str)) if cell in ref_position]
    max_diff = (
        float(np.abs(extra_emb[[i for i, _ in shared]] - latent["ref_emb"][[j for _, j in shared]]).max())
        if shared
        else float("nan")
    )
    print(
        f"[l4] {len(shared)} extra cells also in the subsample; max |Δ embedding| vs latent.npz = {max_diff:.3g}",
        flush=True,
    )

    extraZ = (extra_emb - mu) / sd
    lineage = extra.obs["cell_type_L4"].astype(str).map(lineage_of_l4)
    keep = lineage.notna().values
    Zref, yref = extraZ[keep], lineage[keep].astype(str).values
    print(
        f"[l4] lineage reference: {dict(pd.Series(yref).value_counts())}; dropped "
        f"{dict(extra.obs.loc[~keep, 'cell_type_L4'].value_counts())}",
        flush=True,
    )

    heldout_full = evaluate(Zref, yref, classes, "full atlas T/NK reference")
    in_subsample = extra.obs["in_subsample"].values[keep]
    heldout_subsample = evaluate(Zref[in_subsample], yref[in_subsample], classes, "subsample-only T/NK reference")

    predictions = pd.read_csv(args.pred)
    assert list(predictions.cell_id.astype(str)) == list(st_adata.obs_names.astype(str))
    xenZ = (latent["xen_emb"] - mu) / sd
    cells = hhca.tl.lymphoid_level4(
        xenZ,
        predictions["fine"],
        Zref,
        yref,
        settings["compartment_L3"],
        settings["proliferating_L3"],
        settings["pretty"],
        k=K,
    )
    query = predictions.index.isin(cells.index)
    cells.insert(0, "cell_id", predictions.loc[query, "cell_id"])
    if "Sample" in st_adata.obs:
        cells.insert(1, "Sample", st_adata.obs["Sample"].values[query])
    cells.to_csv(f"{args.outdir}/xenium_lymphoid8_cellpin.csv", index=False)
    print("[l4] label8 counts:\n" + cells.label8.value_counts().to_string(), flush=True)

    sensitivity_summary, sensitivity_cells = sensitivity(Zref, yref, classes, xenZ[query], cells, settings["pretty"])
    sensitivity_cells.to_csv(f"{args.outdir}/xenium_lymphoid8_sensitivity.csv", index=False)
    np.savez(f"{args.outdir}/extra_latent.npz", emb=extra_emb, cells=extra.obs_names.astype(str).values)

    summary = {
        "k": K,
        "seed": SEED,
        "max_abs_embedding_diff_vs_05": max_diff,
        "n_query": int(query.sum()),
        "heldout_full_reference": heldout_full,
        "heldout_subsample_reference": heldout_subsample,
        "label8_counts": cells.label8.value_counts().to_dict(),
        "sensitivity": sensitivity_summary,
    }
    with open(f"{args.outdir}/lymphoid_l4_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[l4] wrote", args.outdir, flush=True)


if __name__ == "__main__":
    main()
