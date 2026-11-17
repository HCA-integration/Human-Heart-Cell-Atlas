#!/usr/bin/env python
"""Transfer HHCA cell-type labels onto Xenium cells in the CellPin latent space.

Uses the model of train_cellpin.py and the reference of build_cellpin_reference.py:
  1. Reload the trained CellPin model.
  2. Embed the atlas reference and the Xenium cells through the same panel encoder, so both lie in one latent
     space (the encoder behind X_cellpin of cellpin_imputed_full.h5ad).
  3. Distance-weighted kNN (K = 25) transfer of the atlas labels in that latent, standardised with the reference
     mean / sd:
       fine   = Final Level 3_harmonized (obs['cell_type'] of the reference)
       coarse = major lineage (9 groups, summed fine probabilities)
     confidence = distance-weighted vote fraction of the winning class.
     Held-out evaluation within the reference: 20 % stratified split (seed 0); accuracy, balanced accuracy,
     per-class precision / recall / F1 / support, held-out predictions.
  4. --l4-col: Level-4 transfer within the predicted Level-3 class (hierarchical kNN) with its own held-out report.
  5. Leiden clusters of the Xenium cells in the latent; the majority transferred label of each cluster is compared
     with the cluster's marker identity from the measured panel genes.

Inputs:
    --ref      reference_subset.h5ad (build_cellpin_reference.py)
    --xenium   xenium_genes.h5ad
    --model    trained CellPin model (train_cellpin.py)
    --imputed  optional cellpin_imputed_full.h5ad; its X_cellpin is compared with the embedding

Outputs (in --outdir):
    latent.npz, per_cell_predictions.csv, heldout_L3_perclass.csv, heldout_L3_predictions.csv,
    heldout_L4_within_L3_perclass.csv (with --l4-col), cluster_crosscheck.csv, celltype_counts_fine.csv,
    celltype_counts_coarse.csv, xenium_annotated.h5ad, annotation_summary.json

Usage: python transfer_labels.py --ref reference_subset.h5ad --xenium xenium_genes.h5ad --model MODEL.pt
           --outdir DIR [--imputed cellpin_imputed_full.h5ad] [--l4-col cell_type_L4]
"""

import argparse
import json
import os

import anndata as ad
import cellpin
import cellpin.pp
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from cellpin.models import CellPin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torch.utils.data import DataLoader

FINE_COL = "cell_type"  # Final Level 3_harmonized, stored by build_cellpin_reference.py
K = 25
SEED = 0

# fine (23 harmonised Level-3 types) -> coarse lineage (9)
COARSE = {
    "Ventricular Cardiomyocytes": "Cardiomyocyte",
    "Atrial Cardiomyocytes": "Cardiomyocyte",
    "Arterial Endothelial cells": "Endothelial",
    "Capillary Endothelial cells": "Endothelial",
    "Venous Endothelial cells": "Endothelial",
    "Endocardial Endothelial Cells": "Endothelial",
    "Lymphatic Endothelial Cells": "Endothelial",
    "Fibroblasts": "Fibroblast",
    "Pericytes": "Mural",
    "Vascular Smooth Muscle cells": "Mural",
    "Macrophages": "Myeloid",
    "Monocytes": "Myeloid",
    "Dendritic cells": "Myeloid",
    "Mast cells": "Myeloid",
    "T cells": "Lymphoid",
    "NK cells": "Lymphoid",
    "ILC": "Lymphoid",
    "B cells": "Lymphoid",
    "Plasma Cells": "Lymphoid",
    "Lymphoid Cells (Proliferating)": "Lymphoid",
    "Neural Cells": "Neural",
    "Adipocytes": "Adipocyte",
    "Epicardium": "Epicardial",
}


N_NEIGHBORS_GRAPH = 15  # kNN graph of the Leiden clustering and the UMAP

# Marker sets on the measured panel. The 10x Human Multi-Tissue panel is not a cardiac panel: it has good
# endothelial, fibroblast, immune and mural markers but few cardiomyocyte genes (no TNNT2, MYH6, TTN, MYL2), so
# the cardiomyocyte set is weak (DES, S100A1, FHL2, ...) and reported separately.
MARKER_SETS = {
    "Cardiomyocyte": ["DES", "S100A1", "FHL2", "PCP4", "MYBPC1", "TIMP4", "PPP1R1A"],
    "Endothelial": ["PECAM1", "VWF", "EGFL7", "CLEC14A", "ERG", "CD34", "CD93", "RAMP2"],
    "Fibroblast": ["PDGFRA", "DPT", "COL5A2", "SFRP2", "VCAN", "FBLN1", "OGN", "PCOLCE"],
    "Myeloid": ["CD68", "CD163", "MRC1", "AIF1", "MARCO", "MS4A6A", "CD14"],
    "Lymphoid": ["CD3E", "CD3D", "TRAC", "CD2", "IL7R", "CCL5", "NKG7", "MS4A1"],
    "Mural": ["PDGFRB", "ACTA2", "MYH11", "HIGD1B", "KCNK3", "RERGL", "CNN1", "MYLK"],
    "Adipocyte": ["ADIPOQ", "LPL", "PLIN4"],
}
WEAK_MARKER_LINEAGES = {"Cardiomyocyte"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", required=True, help="reference_subset.h5ad")
    parser.add_argument("--xenium", required=True, help="xenium_genes.h5ad")
    parser.add_argument("--model", required=True, help="trained CellPin model (.pt)")
    parser.add_argument(
        "--imputed", default=None, help="cellpin_imputed_full.h5ad; its X_cellpin is compared with the embedding"
    )
    parser.add_argument("--outdir", required=True, help="annotation directory")
    parser.add_argument(
        "--l4-col", default=None, help="reference obs column with Level-4 labels (hierarchical transfer)"
    )
    parser.add_argument(
        "--min-f1", type=float, default=0.5, help="per-class held-out F1 below which a class is flagged"
    )
    parser.add_argument("--no-umap", action="store_true", help="skip the 2D embedding")
    return parser.parse_args()


def heldout_report(Z, y, k, seed, prefix):
    """20 % stratified held-out split, distance-weighted kNN."""
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    classifier = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1).fit(Z[train], y[train])
    predicted = classifier.predict(Z[test])
    accuracy = float(accuracy_score(y[test], predicted))
    balanced = float(balanced_accuracy_score(y[test], predicted))
    report = classification_report(y[test], predicted, zero_division=0, output_dict=True)
    print(f"[ann] {prefix} held-out kNN accuracy = {accuracy:.3f}  balanced = {balanced:.3f}", flush=True)
    return accuracy, balanced, report, test, predicted


def report_table(report, classes):
    """Per-class precision / recall / F1 / support, sorted by F1."""
    rows = [
        {
            "class": c,
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
            "support": int(report[c]["support"]),
        }
        for c in classes
        if c in report
    ]
    return pd.DataFrame(rows).sort_values("f1")


def embed(model, sc_ds, st_ds, imputed_path):
    """Embed the reference and the Xenium cells through the panel encoder."""
    print("[ann] embedding reference (panel encoder)", flush=True)
    ref_emb = model.get_cell_embedding(DataLoader(sc_ds, batch_size=512, shuffle=False, num_workers=0), use_mean=True)
    print("[ann] embedding xenium (panel encoder)", flush=True)
    xen_emb = model.get_cell_embedding(DataLoader(st_ds, batch_size=512, shuffle=False, num_workers=0), use_mean=True)
    print(f"[ann] ref_emb {ref_emb.shape}  xen_emb {xen_emb.shape}", flush=True)

    # the embedding should match the X_cellpin saved with the imputed matrix (same encoder)
    if imputed_path:
        saved = ad.read_h5ad(imputed_path).obsm["X_cellpin"]
        print(f"[ann] xen_emb vs saved X_cellpin mean|Δ|={np.abs(saved - xen_emb).mean():.4g}", flush=True)
    return ref_emb, xen_emb


def transfer_level3(refZ, y_fine, xenZ):
    """Transfer the Level-3 labels by kNN (whole reference as training set) and roll them up to lineages."""
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(refZ, y_fine)
    proba = classifier.predict_proba(xenZ)  # (n_xen, n_class)
    fine_classes = classifier.classes_
    fine_pred = fine_classes[proba.argmax(1)]
    fine_conf = proba.max(1)

    # coarse lineage from the summed fine probabilities
    coarse_groups = sorted(set(COARSE.values()))
    group_index = {group: i for i, group in enumerate(coarse_groups)}
    coarse_proba = np.zeros((proba.shape[0], len(coarse_groups)), dtype=np.float32)
    for j, c in enumerate(fine_classes):
        coarse_proba[:, group_index[COARSE[c]]] += proba[:, j]
    coarse_pred = np.array(coarse_groups)[coarse_proba.argmax(1)]
    coarse_conf = coarse_proba.max(1)
    print(
        f"[ann] fine conf median={np.median(fine_conf):.3f}  coarse conf median={np.median(coarse_conf):.3f}",
        flush=True,
    )
    return fine_pred, fine_conf, coarse_pred, coarse_conf


def transfer_level4(refZ, xenZ, y_fine, y_l4, classes, fine_pred, l4_col, min_f1, outdir):
    """Level-4 kNN within each predicted Level-3 class, with a held-out report per Level-4 class."""
    l4_pred = np.empty(len(xenZ), dtype=object)
    l4_conf = np.zeros(len(xenZ), np.float32)
    rows, all_true, all_pred = [], [], []
    for c in classes:
        in_ref = y_fine == c
        children, n_children = np.unique(y_l4[in_ref], return_counts=True)
        in_xenium = fine_pred == c
        if len(children) == 1:
            l4_pred[in_xenium] = children[0]
            l4_conf[in_xenium] = 1.0
            rows.append(
                {
                    "L3": c,
                    "class": children[0],
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "support": int(n_children[0] * 0.2),
                    "single_child": True,
                }
            )
            continue
        Zc, yc = refZ[in_ref], y_l4[in_ref]
        ok = np.isin(yc, children[n_children >= 5])  # stratified split needs >= 2 per class; keep classes with >= 5
        train, test = train_test_split(np.where(ok)[0], test_size=0.2, random_state=SEED, stratify=yc[ok])
        heldout_classifier = KNeighborsClassifier(n_neighbors=min(K, len(train)), weights="distance", n_jobs=-1).fit(
            Zc[train], yc[train]
        )
        predicted = heldout_classifier.predict(Zc[test])
        report = classification_report(yc[test], predicted, zero_division=0, output_dict=True)
        for child in children:
            if child in report:
                rows.append(
                    {
                        "L3": c,
                        "class": child,
                        "precision": report[child]["precision"],
                        "recall": report[child]["recall"],
                        "f1": report[child]["f1-score"],
                        "support": int(report[child]["support"]),
                        "single_child": False,
                    }
                )
        all_true.extend(yc[test])
        all_pred.extend(predicted)
        if in_xenium.any():
            classifier = KNeighborsClassifier(n_neighbors=min(K, int(in_ref.sum())), weights="distance", n_jobs=-1).fit(
                Zc, yc
            )
            proba = classifier.predict_proba(xenZ[in_xenium])
            l4_pred[in_xenium] = classifier.classes_[proba.argmax(1)]
            l4_conf[in_xenium] = proba.max(1)
    l4_df = pd.DataFrame(rows)
    l4_df["reliable"] = l4_df["f1"] >= min_f1
    l4_df.to_csv(f"{outdir}/heldout_L4_within_L3_perclass.csv", index=False)
    summary = {
        "l4_col": l4_col,
        "within_L3_heldout_accuracy_multichild": float(accuracy_score(all_true, all_pred)) if all_true else None,
        "within_L3_heldout_balanced_accuracy_multichild": float(balanced_accuracy_score(all_true, all_pred))
        if all_true
        else None,
        "reliable_L4_classes": l4_df.loc[l4_df.reliable & ~l4_df.single_child, "class"].tolist(),
        "unreliable_L4_classes": l4_df.loc[~l4_df.reliable, "class"].tolist(),
    }
    print("[ann] L4 within-L3 held-out:\n" + l4_df.to_string(index=False), flush=True)
    return l4_pred, l4_conf, summary


def leiden_clusters(xenZ):
    """Leiden clusters on the exact kNN graph of the latent (sklearn -> igraph -> leidenalg); returns the kNN too."""
    print("[ann] building the kNN graph on the latent", flush=True)
    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS_GRAPH + 1, algorithm="auto", n_jobs=-1).fit(xenZ)
    knn_dist, knn_idx = neighbors.kneighbors(xenZ)
    knn_dist, knn_idx = knn_dist[:, 1:], knn_idx[:, 1:]  # drop self
    n = xenZ.shape[0]
    edges = []
    for i in range(n):
        for j in knn_idx[i]:
            edges.append((i, int(j)))
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.simplify()
    print(f"[ann] leiden partition ({graph.vcount()} nodes, {graph.ecount()} edges)", flush=True)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0, seed=SEED
    )
    leiden = np.array([str(c) for c in partition.membership])
    print(f"[ann] leiden clusters: {len(np.unique(leiden))}", flush=True)
    return leiden, knn_idx, knn_dist


def embed_2d(xenZ, knn_idx, knn_dist):
    """UMAP on the precomputed kNN graph; the first two principal components if UMAP cannot be run."""
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=N_NEIGHBORS_GRAPH,
            random_state=SEED,
            precomputed_knn=(knn_idx.astype(np.int64), knn_dist.astype(np.float32), None),
        )
        return reducer.fit_transform(xenZ)
    except (ImportError, TypeError, ValueError) as error:
        print(f"[ann] UMAP failed ({error}); using the first two principal components", flush=True)
        return PCA(n_components=2, random_state=SEED).fit_transform(xenZ)


def cluster_marker_scores(st_adata, leiden):
    """Mean per cluster of the per-cell lineage marker scores (measured panel genes, CP10k + log1p)."""
    panel = set(st_adata.var_names)
    normalised = st_adata.copy()
    normalised.X = normalised.layers["counts"].copy()
    sc.pp.normalize_total(normalised, target_sum=1e4)
    sc.pp.log1p(normalised)
    # per-cell lineage score, z-scored across cells so that the lineages are on one scale before the argmax
    scores = {}
    for lineage, markers in MARKER_SETS.items():
        on_panel = [g for g in markers if g in panel]
        if not on_panel:
            continue
        sc.tl.score_genes(normalised, on_panel, score_name="s")
        v = normalised.obs["s"].values.astype(float)
        scores[lineage] = (v - v.mean()) / (v.std() + 1e-9)
    score_df = pd.DataFrame(scores)
    score_df["leiden"] = leiden
    return score_df.groupby("leiden").mean()


def cluster_crosscheck(leiden, fine_pred, coarse_pred, cluster_scores):
    """Compare the majority transferred lineage of each cluster with its marker identity."""
    cells = pd.DataFrame({"leiden": leiden, "fine": fine_pred, "coarse": coarse_pred})
    majority_fine = cells.groupby("leiden")["fine"].agg(lambda s: s.value_counts().index[0])
    majority_coarse = cells.groupby("leiden")["coarse"].agg(lambda s: s.value_counts().index[0])
    marker_identity = cluster_scores.idxmax(1)

    rows = []
    agree = total = 0
    agree_cells = total_cells = 0
    agree_strong = total_strong = 0  # without the weak-marker (cardiomyocyte) clusters
    for cluster in sorted(cells["leiden"].unique(), key=int):
        n_cells = int((cells.leiden == cluster).sum())
        transferred = majority_coarse[cluster]
        marker = marker_identity.get(cluster, "NA")
        ok = transferred == marker
        if marker in MARKER_SETS:
            total += 1
            agree += int(ok)
            total_cells += n_cells
            agree_cells += n_cells * int(ok)
            if transferred not in WEAK_MARKER_LINEAGES and marker not in WEAK_MARKER_LINEAGES:
                total_strong += 1
                agree_strong += int(ok)
        rows.append(
            {
                "leiden": cluster,
                "n": n_cells,
                "transfer_fine": majority_fine[cluster],
                "transfer_coarse": transferred,
                "marker_identity": marker,
                "agree": bool(ok),
                "marker_z_top": float(cluster_scores.loc[cluster].max()),
            }
        )
    rates = {
        "cluster_marker_agreement_rate": agree / total if total else float("nan"),
        "cluster_marker_agreement_cellweighted": agree_cells / total_cells if total_cells else float("nan"),
        "cluster_marker_agreement_strongmarkers": agree_strong / total_strong if total_strong else float("nan"),
    }
    print(
        f"[ann] cluster transfer-vs-marker agreement: by-cluster={rates['cluster_marker_agreement_rate']:.3f} "
        f"({agree}/{total}) | cell-weighted={rates['cluster_marker_agreement_cellweighted']:.3f} "
        f"| strong-marker-only={rates['cluster_marker_agreement_strongmarkers']:.3f}",
        flush=True,
    )
    return pd.DataFrame(rows), rates


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[ann] loading reference + xenium", flush=True)
    sc_adata = ad.read_h5ad(args.ref)
    st_adata = ad.read_h5ad(args.xenium)
    print(f"[ann] ref {sc_adata.shape}  xenium {st_adata.shape}", flush=True)

    # datasets built as in training (same gene order and panel)
    sc_ds, st_ds = cellpin.pp.setup_data(sc_adata=sc_adata, st_adata=st_adata, layer="counts")

    print("[ann] reloading trained model", flush=True)
    model = CellPin(sc_dataset=sc_ds, checkpoint=args.model)
    model.eval()

    ref_emb, xen_emb = embed(model, sc_ds, st_ds, args.imputed)

    # standardise the latent with the reference statistics, so that no dimension dominates the kNN
    mu = ref_emb.mean(0)
    sd = ref_emb.std(0)
    sd[sd < 1e-8] = 1.0
    refZ = (ref_emb - mu) / sd
    xenZ = (xen_emb - mu) / sd

    y_fine = sc_adata.obs[FINE_COL].astype(str).values
    classes = sorted(pd.unique(y_fine).tolist())
    print(f"[ann] fine classes ({len(classes)}): {classes}", flush=True)

    # held-out accuracy within the reference (snRNA -> snRNA, an upper bound)
    ref_accuracy, ref_balanced, report, heldout_rows, heldout_pred = heldout_report(
        refZ, y_fine, K, SEED, "L3 within-reference"
    )
    per_class = report_table(report, classes)
    per_class["reliable"] = per_class["f1"] >= args.min_f1
    per_class.to_csv(f"{outdir}/heldout_L3_perclass.csv", index=False)
    pd.DataFrame(
        {"ref_cell": sc_adata.obs_names[heldout_rows].astype(str), "true": y_fine[heldout_rows], "pred": heldout_pred}
    ).to_csv(f"{outdir}/heldout_L3_predictions.csv", index=False)
    print("[ann] per-class held-out F1:\n" + per_class.to_string(index=False), flush=True)

    fine_pred, fine_conf, coarse_pred, coarse_conf = transfer_level3(refZ, y_fine, xenZ)

    l4_summary = None
    if args.l4_col:
        y_l4 = sc_adata.obs[args.l4_col].astype(str).values
        l4_pred, l4_conf, l4_summary = transfer_level4(
            refZ, xenZ, y_fine, y_l4, classes, fine_pred, args.l4_col, args.min_f1, outdir
        )

    leiden, knn_idx, knn_dist = leiden_clusters(xenZ)
    X_umap = None if args.no_umap else embed_2d(xenZ, knn_idx, knn_dist)

    cluster_scores = cluster_marker_scores(st_adata, leiden)
    cluster_df, agreement = cluster_crosscheck(leiden, fine_pred, coarse_pred, cluster_scores)

    # arrays and tables
    np.savez(
        f"{outdir}/latent.npz",
        ref_emb=ref_emb,
        xen_emb=xen_emb,
        mu=mu,
        sd=sd,
        ref_cells=sc_adata.obs_names.astype(str).values,
        xen_cells=st_adata.obs_names.astype(str).values,
    )
    predictions = pd.DataFrame(
        {
            "cell_id": st_adata.obs_names.astype(str),
            "fine": fine_pred,
            "fine_confidence": fine_conf.astype(np.float32),
            "coarse": coarse_pred,
            "coarse_confidence": coarse_conf.astype(np.float32),
            "leiden": leiden,
        }
    )
    if args.l4_col:
        predictions["L4"] = l4_pred.astype(str)
        predictions["L4_confidence"] = l4_conf
    predictions.to_csv(f"{outdir}/per_cell_predictions.csv", index=False)
    cluster_df.to_csv(f"{outdir}/cluster_crosscheck.csv", index=False)

    counts_fine = pd.Series(fine_pred).value_counts()
    counts_coarse = pd.Series(coarse_pred).value_counts()
    pd.DataFrame({"n": counts_fine, "proportion": counts_fine / counts_fine.sum()}).to_csv(
        f"{outdir}/celltype_counts_fine.csv"
    )
    pd.DataFrame({"n": counts_coarse, "proportion": counts_coarse / counts_coarse.sum()}).to_csv(
        f"{outdir}/celltype_counts_coarse.csv"
    )

    # annotated AnnData (measured panel genes + embeddings)
    annotated = st_adata.copy()
    annotated.obs["cellpin_fine"] = pd.Categorical(fine_pred)
    annotated.obs["cellpin_fine_conf"] = fine_conf.astype(np.float32)
    annotated.obs["cellpin_coarse"] = pd.Categorical(coarse_pred)
    annotated.obs["cellpin_coarse_conf"] = coarse_conf.astype(np.float32)
    if args.l4_col:
        annotated.obs["cellpin_L4"] = pd.Categorical(l4_pred.astype(str))
        annotated.obs["cellpin_L4_conf"] = l4_conf
    annotated.obs["leiden"] = pd.Categorical(leiden)
    annotated.obsm["X_cellpin"] = xen_emb
    if X_umap is not None:
        annotated.obsm["X_umap"] = np.asarray(X_umap)
    annotated.write_h5ad(f"{outdir}/xenium_annotated.h5ad", compression="gzip")

    summary = {
        "n_cells": int(st_adata.n_obs),
        "k": K,
        "fine_level": "Final Level 3_harmonized (full-atlas reference)",
        "n_fine_classes": len(classes),
        "n_reference_cells": int(sc_adata.n_obs),
        "n_panel_genes_used": int(len(sc_ds.panel_genes)),
        "reference_internal_heldout_accuracy": ref_accuracy,
        "reference_internal_heldout_balanced_accuracy": ref_balanced,
        "heldout_n": int(len(heldout_rows)),
        "fine_conf_median": float(np.median(fine_conf)),
        "fine_conf_mean": float(np.mean(fine_conf)),
        "coarse_conf_median": float(np.median(coarse_conf)),
        "coarse_conf_mean": float(np.mean(coarse_conf)),
        **agreement,
        "panel_note": "Xenium Human Multi-Tissue panel (weak cardiomyocyte marker coverage)",
        "n_leiden": int(len(np.unique(leiden))),
        "fine_counts": counts_fine.to_dict(),
        "coarse_counts": counts_coarse.to_dict(),
        "per_class_ref_f1": {k: report[k]["f1-score"] for k in report if k in set(classes)},
        "per_class_ref_support": {k: int(report[k]["support"]) for k in report if k in set(classes)},
        "min_f1_reliable": args.min_f1,
        "unreliable_L3_classes": per_class.loc[~per_class.reliable, "class"].tolist(),
        "L4": l4_summary,
        "versions": {
            "cellpin": getattr(cellpin, "__version__", "?"),
            "torch": torch.__version__,
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "numpy": np.__version__,
        },
    }
    with open(f"{outdir}/annotation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[ann] wrote", outdir, flush=True)


if __name__ == "__main__":
    main()
