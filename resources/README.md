# Resources

Small curated inputs that the pipelines read. Each one can be rebuilt with the script named in the last column.

| file | content | built by |
|---|---|---|
| `atlas/ALL_genesets.json` | 2,763 gene sets: 2,446 ChEMBL drug-target groups (drug2cell 0.1.2 dictionary, at least one target in the atlas), 12 cardiac signalling axes, 188 hormone producer sets and 117 hormone receptor sets (HRDB v2.0.0) | `scripts/atlas/build_genesets.py` with `atlas/cardiac_axes.yaml` |
| `atlas/cardiac_axes.yaml` | the 12 cardiac drug-mechanism axes | curated by hand |
| `atlas/cytokine_programs.json`, `atlas/cytokine_program_meta.csv` | huCIRA response programmes of 24 cytokines per atlas immune cell type: up-regulated genes (log fold change > 0.5, adjusted p < 0.05), top 50 per programme; macrophages use the monocyte programmes | `scripts/atlas/build_cytokine_programs.py` with `atlas/hucira_celltype_match.yaml` |
| `atlas/donor_map.csv` | 291 atlas donor ids → 280 donors; 11 donors are part of both the Litviňuková and the Reichart cohort | `scripts/atlas/find_duplicate_donors.py` finds the duplicate pairs in the atlas |
| `deconvolution/concordance_per_gene.csv` | gene panel of Fig 3 c: atrial and ventricular cardiomyocyte genes from the atlas (Level 2 differential expression) with their direction in snRNA and Visium | curated from the atlas |
| `deconvolution/atrial_pericyte_signature.json` | the 30-gene atrial pericyte signature of ED8 b–c | atrial vs ventricular pericytes in the atlas Level-4 profile |
| `vascular/common_panel_symbols.json` | the 18,082 genes measured in all 11 Visium HD sections (gene library of the RAAS analysis, Fig 4 i–j) | `scripts/vascular/common_panel.py` |
