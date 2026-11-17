# hhca-analyses

Analysis code for the spatial and donor-level parts of the Human Heart Cell Atlas (HHCA).

We map the cell types of the HHCA snRNA atlas onto Visium HD sections of the human left ventricle at 2 µm
resolution, using FICTURE with a fixed atlas model and a histology-based clean-up of transcript spillover (CMFLAG).
On this map we study vessels, cardiomyocytes, nerves and the macrophages around them. The same atlas reference is
used to deconvolve standard Visium spots and to annotate Xenium sections of a healthy heart and of a
transplant-rejection cohort. On the atlas itself, per-donor scores describe how sex, age, hormones, cytokine
programmes and drug targets vary between donors.

## What is where

```
scripts/<pipeline>/   one command-line script per analysis step
src/hhca/             package shared by scripts, panels and notebooks: analysis steps (hhca.tl), panels (hhca.pl)
workflow/hpc/         SLURM jobs; run_all.sh submits everything in order
config/               one settings file per pipeline, plus the data roots (roots.yaml)
envs/                 conda environments, built automatically inside the jobs
resources/            small curated inputs: gene sets, donor map, gene panels
figures/<Fig>/        one script per figure panel, drawn from results/
notebooks/            a guided walk through each analysis
tools/set_roots.py    points the configs at your own data folders
```

| analysis | folder | figure panels |
|---|---|---|
| Visium HD cell typing: atlas model, FICTURE decode at 2 µm, certainty score, 2 µm and 8 µm AnnData files | `scripts/substrate/` | basis of all Visium HD panels |
| CMFLAG: H&E registration and removal of spillover bins | `scripts/substrate/cmflag/` | basis of all Visium HD panels |
| Vessels: structures, calibre, wall composition and markers, RAAS, drug targets by calibre | `scripts/vascular/` | Fig 4 f–k |
| Cardiomyocytes: stress score, NPPA niches, transmural gradients, agreement with proteomics, label specificity | `scripts/cardiomyocyte/` | Fig 1 j, Fig 3 d–l, ED6 d–f |
| Nerves and macrophages: myelination, nerve–vessel distance, neurotoxic drug targets, perineural macrophages | `scripts/nerve_macrophage/` | Fig 5 b–h, j–l |
| Standard Visium: deconvolution with six published methods and NNLS, simulation with known composition, chamber and pericyte panels | `scripts/deconvolution/` | Fig 3 c, ED6 c, g, h, ED8 b–c |
| Xenium: CellPin label transfer, lymphoid subtypes, rejection-cohort analyses per biopsy core | `scripts/xenium/` | Fig 6 c–l, ED9, ED6 i–j |
| Atlas donors: per-donor gene-set and cytokine scores, hormone production and reception, sex and age effects | `scripts/atlas/` | Fig 2 h–l, Fig 4 l, Fig 5 m–n |

All parameters live in `config/<pipeline>.yaml`, with a comment next to each value.

## Quick start

You need a SLURM cluster with internet access (conda-forge, PyPI) and a GPU partition with V100 or A100 cards
for three of the deconvolution methods. Nothing has to be installed beforehand: each job builds the environment
it needs from `envs/` on first use and links the helper package `hhca` (`src/hhca/`) into it.

1. Tell the code where your data are. Copy `config/roots.yaml`, fill in your paths and apply them:

   ```bash
   cp config/roots.yaml my_roots.yaml        # edit my_roots.yaml
   python tools/set_roots.py --roots my_roots.yaml
   ```

2. Set the partition and QOS names of your cluster in `workflow/hpc/cluster.sh`. Extra `sbatch` options for
   every job, such as an account or nodes to avoid, go into `HHCA_SBATCH_EXTRA`.

3. Submit everything from the repository root:

   ```bash
   bash workflow/hpc/run_all.sh                          # all pipelines
   bash workflow/hpc/run_all.sh cardiomyocyte figures    # or only some of them
   ```

The full run takes about 5.5 hours; the FICTURE decodes and the deconvolution benchmark are the longest steps.
Tables and AnnData files are written to `results/<pipeline>/`, figure panels to `results/figures/<Fig>/`, logs and
the list of job ids to `logs/`. A single panel can be redrawn with the environment of its pipeline,
`.envs/<pipeline>/bin/python figures/<Fig>/panel_<x>.py`, and a single job can be submitted by hand:

```bash
source workflow/hpc/cluster.sh
sbatch "${SBATCH_CPU[@]}" workflow/hpc/vascular.sbatch
```

Three steps need inputs beyond the standard ones and are run separately: the H&E fields of Fig 4 f
(`scripts/vascular/fig4f_fields.py`, then `figures/Fig4/panel_f.py`), the 2 µm matrices of the three AH sections
from Space Ranger `feature_slice.h5` (`workflow/hpc/substrate_regen_ah.sbatch`), and Fig 3 k, which is drawn from
the proteomics protein-group matrices.

## Notebooks

Each notebook follows one analysis from its data objects to the publication panels. It loads the objects the
pipeline starts from (the per-section Visium HD `AnnData`, the Xenium `AnnData` with their transferred labels, the
atlas), runs the analysis steps of `hhca.tl` one by one on a single section and then on all of them, and ends each
section with the publication panel drawn by the same `hhca.pl` function as the panel script. Exploratory plots use
the colours of the panels (`hhca.pl.style`). Steps that need a GPU or many hours (the atlas model and the FICTURE
decode, the H&E registration, the GPU and R deconvolution methods, CellPin training and label transfer) are run by
the pipeline; the notebooks name the command and load their outputs.

| notebook | content | panels | run time |
|---|---|---|---|
| `01_visiumhd_cell_typing` | certainty score, thresholds per cell type, 2 µm and 8 µm `AnnData`, spillover flags on the H&E and the clean-up | | 15 min |
| `02_cardiomyocytes` | label specificity, stress score across the wall, NPPA-high niches, transmural genes and their agreement with protein | Fig 3 d–i, l, ED6 d–f | 15 min |
| `03_vessels` | vessel structures and calibre, wall composition, wall markers, RAAS genes and drug targets | Fig 4 f–k | 12 min |
| `04_nerves_macrophages` | calling nerves, their distance to vessels, myelination, neurotoxic drug targets, perineural macrophages | Fig 5 b–h, j–l | 15 min |
| `05_visium_deconvolution` | reference and NNLS, spots of known composition, seven methods compared, chamber markers and atrial pericytes | Fig 3 c, ED6 c, g, h, ED8 b–c | 6 min |
| `06_xenium` | labels of the healthy section and the rejection cohort, lymphoid subtypes, immune foci, cytokine programmes by rejection type | Fig 6 c–l, ED9 a–c, ED6 i–j | 3 min |
| `07_atlas_donors` | donor table, gene-set scores, hormone production and reception, cytokine programmes, sex and age effects | Fig 2 h–l, Fig 4 l, Fig 5 m–n | 12 min |

Run times are for four CPU cores; `01`, `02` and `04` need about 20 GB of memory, the others 7 GB or less.
Create the environment, install the helper package into it and set `HHCA_ROOT` to the repository root if you
start Jupyter somewhere else:

```bash
conda env create -f envs/notebooks.yml
conda activate hhca-notebooks
pip install -e . --no-deps
```

`07_atlas_donors` runs hormone2cell and draws its panels in the environment of the hormone steps, which has its own
Jupyter kernel:

```bash
conda env create -f envs/atlas_h2c.yml
conda activate hhca-atlas-h2c
pip install -e . --no-deps
python -m ipykernel install --user --name hhca-atlas-h2c
```

The same `pip install -e . --no-deps` makes `hhca` available in any environment of your own. For development,
`pre-commit install` sets up ruff (lint and format) and shellcheck as configured in `pyproject.toml` and
`.pre-commit-config.yaml`.

## Data availability

| data | source |
|---|---|
| HHCA snRNA atlas | CELLxGENE and Cell Annotation Platform (with the publication) |
| Visium HD, 11 left-ventricle sections | with the publication |
| Standard Visium, Noseda cohort | with the publication |
| Standard Visium, Kanemaru et al. 2023 | heartcellatlas.org (`scripts/deconvolution/download_kanemaru.sh`) |
| Healthy heart Xenium section | 10x Genomics, "Human Non-diseased Heart (FFPE)" (`scripts/xenium/download_10x_healthy.sh`) |
| Transplant-rejection Xenium cohort | GEO GSE290577 (`scripts/xenium/download_gse290577.sh`) |
| huCIRA, drug2cell and hormone2cell dictionaries | public, through their Python packages (versions in `envs/`) |

The curated gene sets and tables in `resources/` are described in [`resources/README.md`](resources/README.md).

## Citation

Manuscript in preparation. Until it is out, please cite this repository (see `CITATION.cff`).

## Licence

MIT, see [`LICENSE`](LICENSE).

## Contact

Maiia Shulman
