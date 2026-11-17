"""Certainty thresholds per cell type and the assignment of decode rows against them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hhca.celltyping._certainty import certainty_score

SAMPLE_FIELDS = [
    "margin",
    "depth_raw",
    "marker_raw",
    "spatial_raw",
    "marker_support",
    "cross_stable_bin",
    "refit_changed",
]
SCORE_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
FACTOR_COLUMNS = ["K1", "K2", "K3", "K4", "K5"]
POSTERIOR_COLUMNS = ["P1", "P2", "P3", "P4", "P5"]
SCHEMES = ["decode_argmax", "global_037", "pertype", "pertype_rare_first", "pertype_abund_first"]


def pool_type_samples(checkpoints, n_types, quantiles, weights, cap=250_000, seed=0):
    """Sample decode rows per cell type over several sections and score them with the pooled quantiles.

    Parameters
    ----------
    checkpoints
        Section -> per-row arrays (the feature checkpoints of the pipeline); the order fixes the random draw.
    n_types
        Number of factors of the model.
    cap
        Rows kept per type and section; types with fewer rows are kept whole.

    Returns
    -------
    Per type (factor index) the sampled arrays with their score ``cert_global`` (None for a type without rows),
    and the number of rows per type over all sections.
    """
    parts = {t: {field: [] for field in SAMPLE_FIELDS} for t in range(n_types)}
    n_rows = np.zeros(n_types, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for values in checkpoints.values():
        k1 = values["k1"]
        fields = {field: values[field] for field in SAMPLE_FIELDS}
        for t in range(n_types):
            rows = np.where(k1 == t)[0]
            n_rows[t] += rows.size
            if rows.size == 0:
                continue
            if rows.size > cap:
                rows = rng.choice(rows, cap, replace=False)
            for field in SAMPLE_FIELDS:
                parts[t][field].append(fields[field][rows])
    pooled = {}
    for t in range(n_types):
        if n_rows[t] == 0:
            pooled[t] = None
            continue
        sample = {field: np.concatenate(parts[t][field]) for field in SAMPLE_FIELDS}
        sample["cert_global"] = certainty_score(sample, quantiles, weights)
        pooled[t] = sample
    return pooled, n_rows


def _model_specificity(model, markers):
    """Per type: mean log2 enrichment and mean model expression of its markers, and their number."""
    expression = model.to_numpy(float)
    total = expression.sum(1)
    row_of_gene = {gene: i for i, gene in enumerate(model.index.to_numpy().astype(str))}
    specificity = {}
    for i, name in enumerate(model.columns):
        own = expression[:, i]
        other = (total - own) / max(1, expression.shape[1] - 1)
        enrichment = (own + 1) / (other + 1)
        genes = sorted(markers.get(name, []))
        rows = [row_of_gene[gene] for gene in genes if gene in row_of_gene]
        if not rows:
            specificity[i] = (np.nan, np.nan, len(genes))
        else:
            specificity[i] = (float(np.mean(np.log2(enrichment[rows]))), float(np.mean(own[rows])), len(genes))
    return specificity


def _distribution_row(name, sample, n_rows):
    """Quantiles of the score and rates / medians of its components for one type."""
    score = sample["cert_global"]
    return dict(
        cell_type=name,
        n_argmax_bins=int(n_rows),
        n_sampled=int(score.size),
        **{f"cert_q{q}": float(np.percentile(score, q)) for q in SCORE_PERCENTILES},
        marker_support_rate=float(sample["marker_support"].mean()),
        cross_stable_rate=float(np.nanmean(sample["cross_stable_bin"])),
        refit_changed_rate=float(sample["refit_changed"].mean()),
        median_marker_raw=float(np.median(sample["marker_raw"])),
        median_depth_raw=float(np.median(sample["depth_raw"])),
        median_margin=float(np.median(sample["margin"])),
        median_spatial=float(np.median(sample["spatial_raw"])),
    )


def _density_row(name, sample, n_rows, specificity):
    """Marker enrichment in the model next to the observed marker density of one type."""
    log2_enrichment, mean_expression, n_markers = specificity
    return {
        "cell_type": name,
        "n_argmax_bins": int(n_rows),
        "n_markers": n_markers,
        "marker_log2_enrichment": log2_enrichment,
        "marker_mean_pseudobulk": mean_expression,
        "empirical_median_marker_raw": float(np.median(sample["marker_raw"])),
        "empirical_median_depth_raw": float(np.median(sample["depth_raw"])),
        "frac_bins_marker_supported": float(sample["marker_support"].mean()),
    }


def _percentile_range(values):
    """5th and 95th percentile."""
    return (float(np.percentile(values, 5)), float(np.percentile(values, 95)))


def _faith(sample, keep):
    """Share of the kept rows with marker support and a stable call across decodes."""
    if keep.sum() == 0:
        return np.nan
    good = (sample["marker_support"].astype(bool)) & (sample["cross_stable_bin"].astype(bool))
    return float(good[keep].mean())


def _knee_threshold(scores):
    """Knee of the score-by-coverage curve: the score farthest from the chord between its end points."""
    s = np.sort(scores)[::-1]
    n = s.size
    if n < 20:
        return float(np.percentile(scores, 10))
    x = np.linspace(0, 1, n)
    y = (s - s.min()) / (np.ptp(s) + 1e-9)
    d = y - (1 - x)
    k = int(np.argmax(np.abs(d)))
    return float(s[k])


def _threshold_row(name, sample, n_rows, quantiles, weights, q_marker, q_depth, global_tau, rare_types):
    """Candidate thresholds of one type and the coverage and faithfulness each one gives."""
    score = sample["cert_global"]
    # density-corrected score: depth and marker scaled with the quantiles of the type, margin and spatial pooled
    corrected = certainty_score(sample, {**quantiles, "depth_raw": q_depth, "marker_raw": q_marker}, weights)
    # 1: one threshold for all types on the pooled score
    keep_global = score >= global_tau
    # 2: the same threshold on the density-corrected score
    keep_corrected = corrected >= global_tau
    # pooled score at which rule 2 starts to keep rows of this type
    if keep_corrected.sum() > 0 and (~keep_corrected).sum() > 0:
        tau_raw_equiv = float(np.percentile(score[keep_corrected], 1))
    else:
        tau_raw_equiv = float(global_tau)
    # 3: knee of the pooled score per type
    tau_knee = _knee_threshold(score)
    # 4: lowest threshold per type that reaches the faithfulness target
    taus = np.unique(np.percentile(score, np.arange(0, 100, 2)))
    achieved = [_faith(sample, score >= tau) for tau in taus]
    achieved = np.array([a if np.isfinite(a) else 0 for a in achieved])
    max_achieved = float(np.nanmax(achieved)) if achieved.size else np.nan
    target = min(0.95, max_achieved - 0.01) if np.isfinite(max_achieved) else 0.95
    reached = np.where(achieved >= target)[0]
    tau_faith = float(taus[reached[0]]) if reached.size else float(np.percentile(score, 10))
    return {
        "cell_type": name,
        "n_argmax_bins": int(n_rows),
        "is_rare": name in rare_types,
        "tau_global": global_tau,
        "tau_knee_global": round(tau_knee, 4),
        "tau_faith_calibrated": round(tau_faith, 4),
        "faith_target": round(target, 4),
        "max_achievable_faith": round(max_achieved, 4),
        "tau_raw_equiv_corrected": round(tau_raw_equiv, 4),
        "q_marker_lo": round(q_marker[0], 5),
        "q_marker_hi": round(q_marker[1], 5),
        "q_depth_lo": round(q_depth[0], 4),
        "q_depth_hi": round(q_depth[1], 4),
        # share of the type's rows kept under each rule
        "cov_global": round(float(keep_global.mean()), 4),
        "cov_corrected": round(float(keep_corrected.mean()), 4),
        "cov_knee": round(float((score >= tau_knee).mean()), 4),
        "cov_faith": round(float((score >= tau_faith).mean()), 4),
        "faith_global": round(_faith(sample, keep_global), 4) if keep_global.sum() else np.nan,
        "faith_corrected": round(_faith(sample, keep_corrected), 4) if keep_corrected.sum() else np.nan,
    }


def fit_type_thresholds(pooled, n_rows, model, markers, quantiles, weights, global_tau=0.37, rare_types=()) -> dict:
    """Describe the certainty score of each cell type and derive candidate thresholds per type.

    Rare and thin cell types have fewer marker UMI and less depth per 2 um bin than cardiomyocytes, so one
    threshold on the pooled score removes most of their rows. The density-corrected score scales depth and
    marker share with the 5-95 % range of the type itself; ``cov_corrected`` is the share of the type's rows it
    keeps at `global_tau`.

    Parameters
    ----------
    pooled, n_rows
        Sampled rows per type and rows per type (:func:`pool_type_samples`).
    model
        Genes x types model matrix, columns in factor order.
    markers
        Type -> marker genes (:func:`model_markers`).
    quantiles, weights
        Pooled quantiles and weights of the certainty score.
    global_tau
        Threshold on the score when one value is used for all types.
    rare_types
        Types marked as rare in the tables.

    Returns
    -------
    Dictionary of tables: ``distributions`` (score quantiles and component medians per type), ``density``
    (marker enrichment in the model and observed marker density), ``thresholds`` (candidate thresholds and the
    coverage each keeps) and ``normalization`` (quantiles per type, as a dictionary).
    """
    order = list(model.columns)
    specificity = _model_specificity(model, markers)
    distribution_rows, density_rows, q_marker, q_depth = [], [], {}, {}
    for t, name in enumerate(order):
        sample = pooled[t]
        if sample is None:
            continue
        distribution_rows.append(_distribution_row(name, sample, n_rows[t]))
        density_rows.append(_density_row(name, sample, n_rows[t], specificity[t]))
        q_marker[t] = _percentile_range(sample["marker_raw"])
        q_depth[t] = _percentile_range(sample["depth_raw"])
    threshold_rows = []
    for t, name in enumerate(order):
        if pooled[t] is None:
            threshold_rows.append({"cell_type": name, "n_argmax_bins": 0})
            continue
        threshold_rows.append(
            _threshold_row(
                name, pooled[t], n_rows[t], quantiles, weights, q_marker[t], q_depth[t], global_tau, rare_types
            )
        )
    return {
        "distributions": pd.DataFrame(distribution_rows),
        "density": pd.DataFrame(density_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "normalization": {
            "weights": weights,
            "global_quantiles": quantiles,
            "global_tau": global_tau,
            "q_marker_pertype": {order[t]: q_marker[t] for t in q_marker},
            "q_depth_pertype": {order[t]: q_depth[t] for t in q_depth},
        },
    }


def type_thresholds(thresholds, scores, factor_order, global_tau=0.37) -> pd.Series:
    """Threshold per cell type: the pooled score that keeps the coverage of the density-corrected score.

    Parameters
    ----------
    thresholds
        Table ``thresholds`` of :func:`fit_type_thresholds` (``cell_type``, ``cov_corrected``).
    scores
        Factor index -> pooled scores of the sampled rows of the type.
    factor_order
        Level-3 types in factor order.
    global_tau
        Threshold of a type without sampled rows.
    """
    tau = np.full(len(factor_order), global_tau, dtype=np.float64)
    coverage_corrected = dict(zip(thresholds.cell_type, thresholds.cov_corrected))
    for t, name in enumerate(factor_order):
        if t not in scores:
            continue
        score = scores[t]
        coverage = coverage_corrected.get(name, np.nan)
        if not np.isfinite(coverage) or score.size == 0:
            continue
        tau[t] = float(np.percentile(score, 100 * (1 - coverage)))  # keeps the top `coverage` share
    return pd.Series(tau, index=list(factor_order), name="tau")


def assign_labels(decode, score, tau, priority=None, p_floor=1e-3) -> np.ndarray:
    """Cell type per decode row among its top factors: the candidate of highest priority that passes its threshold.

    A factor is a candidate when its posterior is at least `p_floor` and the certainty score of the row reaches
    the threshold of that type.

    Parameters
    ----------
    decode
        Pixel decode with the top-5 factors and posteriors (``K1`` .. ``K5``, ``P1`` .. ``P5``).
    score
        Certainty score per row.
    tau
        Threshold per type, in factor order.
    priority
        Factor indices by decreasing priority (default: the rarest top factor of this decode first).

    Returns
    -------
    Factor index per row (int16), -1 where no candidate passes.
    """
    K = decode[FACTOR_COLUMNS].to_numpy(np.int16)
    P = decode[POSTERIOR_COLUMNS].to_numpy(np.float32)
    score = np.asarray(score)
    tau = np.asarray(tau, dtype=np.float64)
    n_types = tau.size
    if priority is None:
        priority = np.argsort(np.bincount(K[:, 0].astype(np.int64), minlength=n_types))
    assigned = np.full(score.shape[0], -1, dtype=np.int16)
    rank = np.full(n_types, n_types + 1, dtype=np.int32)  # lower value = higher priority
    for position, t in enumerate(priority):
        rank[t] = position
    best = np.full(score.shape[0], n_types + 2, dtype=np.int32)
    for c in range(K.shape[1]):
        factor = K[:, c]
        valid = (factor >= 0) & (P[:, c] >= p_floor) & (score >= tau[np.clip(factor, 0, n_types - 1)])
        candidate = np.where(valid, rank[np.clip(factor, 0, n_types - 1)], n_types + 2)
        take = candidate < best
        assigned[take] = factor[take]
        best[take] = candidate[take]
    return assigned


def assignment_schemes(decode, score, tau, rows_per_type, global_tau=0.37, p_floor=1e-3) -> dict[str, np.ndarray]:
    """Assign the rows of a decode in five ways, to compare the compositions they give.

    ``decode_argmax``: top factor of every row; ``global_037``: top factor if the score reaches `global_tau`;
    ``pertype``: top factor if the score reaches the threshold of its type; ``pertype_rare_first`` and
    ``pertype_abund_first``: :func:`assign_labels` with rare or abundant types (by `rows_per_type`) taking
    priority. Unassigned rows are -1.
    """
    k1 = decode["K1"].to_numpy(np.int64)
    score = np.asarray(score)
    tau = np.asarray(tau, dtype=np.float64)
    rare_first = list(np.argsort(rows_per_type))
    abundant_first = list(np.argsort(rows_per_type)[::-1])
    return {
        "decode_argmax": k1.copy(),
        "global_037": np.where(score >= global_tau, k1, -1),
        "pertype": np.where(score >= tau[k1], k1, -1),
        "pertype_rare_first": assign_labels(decode, score, tau, rare_first, p_floor).astype(np.int64),
        "pertype_abund_first": assign_labels(decode, score, tau, abundant_first, p_floor).astype(np.int64),
    }
