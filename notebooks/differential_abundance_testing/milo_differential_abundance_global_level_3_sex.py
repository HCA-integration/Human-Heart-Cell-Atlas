# Milo sex analysis — dataset and age corrected
# Global HHCA object, with neighbourhoods annotated by Level 3

import os
import sys
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc


matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sc.settings.set_figure_params(
    dpi=300,
    fontsize=12,
)

conda_prefix = os.path.dirname(os.path.dirname(sys.executable))

os.environ["PATH"] = (
    f"{conda_prefix}/bin:"
    + os.environ.get("PATH", "")
)

os.environ["R_HOME"] = f"{conda_prefix}/lib/R"

print("Python executable:", sys.executable)
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV"))
print("R path:", shutil.which("R"))
print("R_HOME:", os.environ.get("R_HOME"))

print(
    subprocess.check_output(
        ["R", "--version"],
        text=True,
    ).split("\n")[0]
)

import rpy2.robjects as ro

print(ro.r("R.version.string"))

ro.r("library(edgeR)")
ro.r("library(limma)")
ro.r("library(statmod)")

print("R bridge works")

INPUT_H5AD = (
    "/rds/general/user/snb20/projects/"
    "cardiac_single_cell_biology/live/"
    "Human_Heart_Cell_Atlas_Integration/HHCA_clean_global_object_July26.h5ad"
)

OUTPUT_DIR = Path("milo_level3_sex_analysis")
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
OBJECT_DIR = OUTPUT_DIR / "objects"

for directory in [
    OUTPUT_DIR,
    FIGURE_DIR,
    RESULTS_DIR,
    OBJECT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


LEVEL3_COL = "Final Level 3_harmonized"
SEX_COL = "sex"
DONOR_COL = "sample_id"
DATASET_COL = "dataset"
AGE_BIN_COL = "age_10yr_bins"

N_NEIGHBORS = 300
NHOOD_PROP = 0.1
SPATIAL_FDR_THRESHOLD = 0.1

adata = sc.read_h5ad(INPUT_H5AD)

print(adata)
print(
    f"Original object: "
    f"{adata.n_obs:,} nuclei and "
    f"{adata.n_vars:,} genes"
)

cluster2annotation = {

    'HsapDv:0000129': '30-40',
    'HsapDv:0000135': '40-50',
    'HsapDv:0000136': '40-50',
    'HsapDv:0000141': '40-50',
    'HsapDv:0000147': '50-60',
    'HsapDv:0000148': '50-60',
    'HsapDv:0000149': '50-60',
    'HsapDv:0000237': '20-30',
    'HsapDv:0000238': '30-40',
    'HsapDv:0000239': '40-50',
    'HsapDv:0000240': '50-60',
    'HsapDv:0000241': '60-70',
    'HsapDv:0000242': '70-80',
    'HsapDv:0000243': '80-90',
    'HsapDv:0000264': '0-20'

}

adata.obs['age_10yr_bins'] = adata.obs['development_stage_ontology_term_id'].map(cluster2annotation).astype('category')

mapping = {
    'heart left ventricle': 'Left Ventricle',
    'ventricular musculature': 'Left Ventricle',
    'wall of left ventricle': 'Left Ventricle',
    'sinoatrial node': 'SA Node',
    'atrioventricular node' : 'AV Node',              
    'right atrium auricular region': 'Right Atrium',        
    'heart right ventricle': 'Right Ventricle',                 
    'LV APEX': 'Apex',                               
    'right cardiac atrium': 'Right Atrium',
    'apex of heart': 'Apex',
    'left cardiac atrium': 'Left Atrium',
    'interventricular septum': 'IV Septum',
    'SAN': 'SA Node',
    'cardiac septum': 'IV Septum',
    'apical region of left ventricle': 'Apex',
    'cardiac muscle of left atrium': 'Left Atrium',
    'left atrium auricular region': 'Left Atrium',
    'AVN': 'AV Node',
    'cardiac muscle of right ventricle': 'Right Ventricle',
    'cardiac muscle of left ventricle': 'Left Ventricle',
    'cardiac muscle of right atrium': 'Right Atrium'
}

adata.obs['tissue_final'] = adata.obs['tissue'].map(mapping).astype('category')

adata = adata[adata.obs['tissue_final'].isin(['Left Ventricle','Right Ventricle', 'Apex'])]

required_obs_columns = [
    LEVEL3_COL,
    SEX_COL,
    DONOR_COL,
    DATASET_COL,
    AGE_BIN_COL,
]

missing_columns = [
    col
    for col in required_obs_columns
    if col not in adata.obs.columns
]

if missing_columns:
    raise KeyError(
        "The following required columns are missing from adata.obs: "
        + ", ".join(missing_columns)
    )


if "X_emb" not in adata.obsm:
    raise KeyError(
        "'X_emb' was not found in adata.obsm. "
        "Check the name of the integrated embedding."
    )


if "X_umap" not in adata.obsm:
    raise KeyError(
        "'X_umap' was not found in adata.obsm. "
        "This is required for neighbourhood graph visualisation."
    )


print("\nSex labels:")
print(adata.obs[SEX_COL].value_counts(dropna=False))

print("\nAge bins:")
print(adata.obs[AGE_BIN_COL].value_counts(dropna=False))

print("\nLevel 3 annotations:")
print(adata.obs[LEVEL3_COL].value_counts(dropna=False))

adata.obs[SEX_COL] = (
    adata.obs[SEX_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

adata = adata[
    adata.obs[SEX_COL].isin(["female", "male"])
].copy()

# Set female as the reference category.
adata.obs[SEX_COL] = pd.Categorical(
    adata.obs[SEX_COL],
    categories=["female", "male"],
    ordered=True,
)

print("\nSex distribution after filtering:")
print(adata.obs[SEX_COL].value_counts(dropna=False))

print("\nUnique donors by sex:")
print(
    adata.obs.groupby(
        SEX_COL,
        observed=True,
    )[DONOR_COL].nunique()
)

unassigned_mask = (
    adata.obs[LEVEL3_COL]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("unassigned")
)

print(
    f"\nRemoving {unassigned_mask.sum():,} "
    "Level 3 unassigned nuclei"
)

adata = adata[~unassigned_mask].copy()

print("\nLevel 3 annotations after filtering:")
print(adata.obs[LEVEL3_COL].value_counts(dropna=False))

age_mapping = {
    "0-20": 10,
    "20-30": 25,
    "30-40": 35,
    "40-50": 45,
    "50-60": 55,
    "60-70": 65,
    "70-80": 75,
    "80-90": 85,
    "90-100": 95,
}

adata.obs[AGE_BIN_COL] = (
    adata.obs[AGE_BIN_COL]
    .astype(str)
    .str.strip()
)

adata.obs["age_continuous"] = (
    adata.obs[AGE_BIN_COL]
    .map(age_mapping)
)

unmapped_age_bins = sorted(
    adata.obs.loc[
        adata.obs["age_continuous"].isna(),
        AGE_BIN_COL,
    ]
    .dropna()
    .unique()
    .tolist()
)

if unmapped_age_bins:
    print("\nThe following age labels were not mapped:")
    print(unmapped_age_bins)


# Remove donors without a usable age range.
age_missing = adata.obs["age_continuous"].isna()

print(
    f"\nRemoving {age_missing.sum():,} nuclei "
    "without a usable age range"
)

adata = adata[~age_missing].copy()

adata.obs["age_continuous"] = pd.to_numeric(
    adata.obs["age_continuous"],
    errors="raise",
)

print("\nApproximate continuous age distribution:")
print(
    adata.obs["age_continuous"]
    .value_counts()
    .sort_index()
)

print("\nUnique donors by age range and sex:")
print(
    adata.obs.groupby(
        [AGE_BIN_COL, SEX_COL],
        observed=True,
    )[DONOR_COL]
    .nunique()
    .unstack(fill_value=0)
)

adata.obs["dataset_safe"] = (
    "dataset_"
    + adata.obs[DATASET_COL]
    .astype(str)
    .str.strip()
    .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
    .str.strip("_")
)

adata.obs["dataset_safe"] = pd.Categorical(
    adata.obs["dataset_safe"]
)

print("\nDataset labels:")
print(
    adata.obs[
        [DATASET_COL, "dataset_safe"]
    ]
    .drop_duplicates()
    .sort_values(DATASET_COL)
    .to_string(index=False)
)

donor_metadata_columns = [
    SEX_COL,
    "age_continuous",
    "dataset_safe",
]

donor_metadata_nunique = (
    adata.obs.groupby(DONOR_COL)[donor_metadata_columns]
    .nunique()
)

print("\nMaximum number of metadata values per donor:")
print(donor_metadata_nunique.max())


inconsistent_donors = donor_metadata_nunique[
    (donor_metadata_nunique > 1).any(axis=1)
]

if not inconsistent_donors.empty:
    print("\nDonors with inconsistent metadata:")
    print(inconsistent_donors)

    raise ValueError(
        "Some donors have more than one value for sex, age or dataset. "
        "Resolve these donor-level metadata inconsistencies before running Milo."
    )

donor_metadata = (
    adata.obs[
        [
            DONOR_COL,
            SEX_COL,
            AGE_BIN_COL,
            "age_continuous",
            DATASET_COL,
            "dataset_safe",
        ]
    ]
    .drop_duplicates(subset=DONOR_COL)
    .copy()
)

print(
    f"\nFinal object: {adata.n_obs:,} nuclei from "
    f"{donor_metadata[DONOR_COL].nunique():,} donors"
)

print("\nDonors by sex:")
print(
    donor_metadata[SEX_COL]
    .value_counts(dropna=False)
)

print("\nDonors by dataset and sex:")
dataset_sex_table = pd.crosstab(
    donor_metadata["dataset_safe"],
    donor_metadata[SEX_COL],
    dropna=False,
)

print(dataset_sex_table.to_string())

dataset_sex_table.to_csv(
    RESULTS_DIR / "donor_counts_by_dataset_and_sex.csv"
)

donor_metadata.to_csv(
    RESULTS_DIR / "donor_metadata_used_for_milo.csv",
    index=False,
)

milo = pt.tl.Milo()
mdata = milo.load(adata)

print(mdata)

sc.pp.neighbors(
    mdata["rna"],
    use_rep="X_emb",
    n_neighbors=N_NEIGHBORS,
)

print(
    f"\nConstructed global KNN graph using "
    f"{N_NEIGHBORS} neighbours"
)

milo.make_nhoods(
    mdata["rna"],
    prop=NHOOD_PROP,
)

nhood_size = np.asarray(
    mdata["rna"].obsm["nhoods"].sum(axis=0)
).ravel()

print("\nNeighbourhood-size summary:")
print(pd.Series(nhood_size).describe())


fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(
    nhood_size,
    bins=100,
    color="#777777",
)

ax.set_xlabel("Number of cells in neighbourhood")
ax.set_ylabel("Number of neighbourhoods")
ax.set_title("Milo neighbourhood sizes")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "milo_neighbourhood_sizes.pdf",
    bbox_inches="tight",
)

plt.show()

mdata = milo.count_nhoods(
    mdata,
    sample_col=DONOR_COL,
)

print(mdata)

nhood_count_matrix = mdata["milo"].X

n_donors_per_nhood = np.asarray(
    (nhood_count_matrix > 0).sum(axis=0)
).ravel()

mdata["milo"].var["n_donors"] = n_donors_per_nhood

print("\nNumber of contributing donors per neighbourhood:")
print(
    pd.Series(n_donors_per_nhood).describe()
)


fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(
    n_donors_per_nhood,
    bins=50,
    color="#777777",
)

ax.set_xlabel("Number of donors represented")
ax.set_ylabel("Number of neighbourhoods")
ax.set_title("Donor representation across neighbourhoods")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "milo_donors_per_neighbourhood.pdf",
    bbox_inches="tight",
)

plt.show()

mdata["rna"].obs["age_continuous"] = pd.to_numeric(
    mdata["rna"].obs["age_continuous"],
    errors="raise",
)

mdata["rna"].obs[SEX_COL] = pd.Categorical(
    mdata["rna"].obs[SEX_COL],
    categories=["female", "male"],
    ordered=True,
)

mdata["rna"].obs["dataset_safe"] = pd.Categorical(
    mdata["rna"].obs["dataset_safe"]
)

print("\nModel-variable dtypes:")
print(
    mdata["rna"].obs[
        [
            "dataset_safe",
            "age_continuous",
            SEX_COL,
        ]
    ].dtypes
)

milo.da_nhoods(
    mdata,
    design="~ dataset_safe + age_continuous + tissue_final + sex",
    solver="edger",
)

print("\nMilo differential-abundance results:")
print(mdata["milo"].var.head())

results = mdata["milo"].var

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
)

axes[0].hist(
    results["PValue"].dropna(),
    bins=50,
    color="#777777",
)

axes[0].set_xlabel("P value")
axes[0].set_ylabel("Number of neighbourhoods")
axes[0].set_title("P-value distribution")


valid_results = results[
    results["SpatialFDR"].notna()
    & results["logFC"].notna()
].copy()

axes[1].scatter(
    valid_results["logFC"],
    -np.log10(
        valid_results["SpatialFDR"].clip(lower=1e-300)
    ),
    s=5,
    color="#777777",
    alpha=0.6,
    rasterized=True,
)

axes[1].axhline(
    -np.log10(SPATIAL_FDR_THRESHOLD),
    linestyle="--",
    color="black",
    linewidth=0.8,
)

axes[1].set_xlabel("log2 fold change: male versus female")
axes[1].set_ylabel("−log10 Spatial FDR")
axes[1].set_title("Sex-associated differential abundance")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "milo_sex_dataset_age_sex_corrected_model_diagnostics.pdf",
    bbox_inches="tight",
)

plt.show()

milo.build_nhood_graph(
    mdata,
    basis="X_umap",
)

milo.annotate_nhoods(
    mdata,
    anno_col=LEVEL3_COL,
)

print("\nNeighbourhood annotations:")
print(
    mdata["milo"].var[
        [
            "nhood_annotation",
            "nhood_annotation_frac",
        ]
    ].head()
)

print("\nNeighbourhood Level 3 purity:")
print(
    mdata["milo"].var[
        "nhood_annotation_frac"
    ].describe()
)

milo.plot_nhood_graph(
    mdata,
    alpha=SPATIAL_FDR_THRESHOLD,
    min_size=0.1,
)

plt.savefig(
    FIGURE_DIR / "milo_sex_dataset_age_sex_corrected_neighbourhood_graph.pdf",
    bbox_inches="tight",
)

plt.show()

milo.plot_da_beeswarm(
    mdata,
    alpha=SPATIAL_FDR_THRESHOLD,
)

plt.savefig(
    FIGURE_DIR / "milo_sex_dataset_age_sex_corrected_level3_beeswarm.pdf",
    bbox_inches="tight",
)

plt.show()

milo_results = mdata["milo"].var.copy()

milo_results["significant"] = (
    milo_results["SpatialFDR"]
    < SPATIAL_FDR_THRESHOLD
)

milo_results["direction"] = "not significant"

milo_results.loc[
    milo_results["significant"]
    & (milo_results["logFC"] > 0),
    "direction",
] = "male enriched"

milo_results.loc[
    milo_results["significant"]
    & (milo_results["logFC"] < 0),
    "direction",
] = "female enriched"


level3_summary = (
    milo_results.groupby(
        [
            "nhood_annotation",
            "direction",
        ],
        observed=True,
    )
    .size()
    .unstack(fill_value=0)
)

print("\nSignificant neighbourhoods by Level 3:")
print(level3_summary.to_string())

milo_results.to_csv(
    RESULTS_DIR
    / "milo_global_sex_dataset_age_region_corrected_neighbourhoods.csv"
)

level3_summary.to_csv(
    RESULTS_DIR
    / "milo_global_sex_dataset_age_region_corrected_level3_summary.csv"
)

#OUTPUT_OBJECT = (
#    OBJECT_DIR
#    / "mdata_milo_global_sex_dataset_age_region_corrected.h5mu"
#)

OUTPUT_OBJECT = Path(
    "/rds/general/user/snb20/projects/"
    "cardiac_single_cell_biology/live/"
    "Human_Heart_Cell_Atlas_Integration/"
    "HHCA_sex_milo_ventricular_global.h5mu"
)

OUTPUT_OBJECT.parent.mkdir(parents=True, exist_ok=True)

mdata.write_h5mu(OUTPUT_OBJECT)

print(f"\nSaved complete Milo object to:\n{OUTPUT_OBJECT}")

print("\nAnalysis complete.")