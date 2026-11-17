"""Cell typing of Visium HD sections with the atlas model.

Certainty score, thresholds per type, the 2 um and 8 um AnnData objects and the CMFLAG spillover flags. The steps
are available as ``hhca.tl.<name>`` next to the other analysis steps. They are kept in this package, which needs
only numpy, pandas and scipy at import, because the scripts in ``scripts/substrate/`` run in the small
environments of the pipeline (``envs/substrate.yml``, ``certainty.yml``, ``substrate_cmflag.yml``). anndata, h5py,
pyarrow, OpenCV, scikit-image and Pillow are imported inside the functions that need them.
"""

from hhca.celltyping._bins import aggregate_bins, bins_2um, bins_8um, read_bin_positions, read_bins
from hhca.celltyping._certainty import (
    SCORE_COMPONENTS,
    call_stability,
    certainty_components,
    certainty_score,
    component_quantiles,
    decode_keys,
    match_calls,
    model_markers,
    operating_point,
    precision_coverage,
    sample_rows,
    select_threshold,
    unmatched_rows,
)
from hhca.celltyping._clean import bin_flags, clean_bins_8um
from hhca.celltyping._he import Microscopy, cytassist_scale, luminance, offset_field, register_he, sample_he
from hhca.celltyping._spillover import (
    flag_counts,
    flag_table,
    flag_thresholds,
    local_density,
    marker_depth,
    nucleus_gate,
    percent,
    spillover_flags,
    vessel_lumen,
)
from hhca.celltyping._thresholds import (
    SCHEMES,
    assign_labels,
    assignment_schemes,
    fit_type_thresholds,
    pool_type_samples,
    type_thresholds,
)

__all__ = [
    "SCHEMES",
    "SCORE_COMPONENTS",
    "Microscopy",
    "aggregate_bins",
    "assign_labels",
    "assignment_schemes",
    "bin_flags",
    "bins_2um",
    "bins_8um",
    "call_stability",
    "certainty_components",
    "certainty_score",
    "clean_bins_8um",
    "component_quantiles",
    "cytassist_scale",
    "decode_keys",
    "fit_type_thresholds",
    "flag_counts",
    "flag_table",
    "flag_thresholds",
    "local_density",
    "luminance",
    "marker_depth",
    "match_calls",
    "model_markers",
    "nucleus_gate",
    "offset_field",
    "operating_point",
    "percent",
    "pool_type_samples",
    "precision_coverage",
    "read_bin_positions",
    "read_bins",
    "register_he",
    "sample_he",
    "sample_rows",
    "select_threshold",
    "spillover_flags",
    "type_thresholds",
    "unmatched_rows",
    "vessel_lumen",
]
