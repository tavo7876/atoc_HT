"""Sea-ice comparison for CAMulator + POP2 + CICE5 checkpoint runs.

Three steps: load the two datasets, put the *reference* on the checkpoint's
native gx1v7 grid, look at them side by side.

    from ice_ens import data, regrid, metrics, plot_poles

    model = data.load_checkpoint("ck33", years=[1980])
    truth = data.load_truth()
    ref   = regrid.to_model_grid(truth, model)        # reference -> gx1v7

    plot_poles.plot_year_compare(model, ref, 1980, "north")
    metrics.compare(model, ref, "north").to_dataframe()

The checkpoint is never interpolated -- it is the field under evaluation, every
checkpoint already shares the gx1v7 grid, and POP's own TAREA stays the area
weight. Use ``method="bilinear"`` for a quick difference map and
``method="conservative"`` for any area or extent number.

Both inputs are read in place on scratch. Output goes to
``data.OUTPUT_DIR``, one subdirectory per checkpoint.
"""

from . import data, ensemble, grids, metrics, plot_poles, regrid, fluxes, stress, icefrac
from .ensemble import (
    bias_ensemble,
    climatology,
    ice_area,
    plot_area_error,
    plot_bias_grid,
    plot_month_grid,
    regrid_to_gx1v7,
)
from .fluxes import (
    FLUX_VARS,
    build_flux_cache,
    cache_path,
    flux_balance,
    flux_budget,
    flux_daily,
    flux_day,
    flux_map,
    flux_map_months,
    flux_series,
    load_flux_cache,
    load_camulator,
    load_cesm,
    monthly_mean,
    map_dir,
    plot_balance,
    plot_map_grid,
    plot_maps,
    plot_series,
    regional_mean,
)

__all__ = [
    "data", "ensemble", "fluxes", "grids", "metrics", "plot_poles", "regrid", "stress", "icefrac",
    "climatology", "regrid_to_gx1v7", "ice_area", "bias_ensemble",
    "plot_bias_grid", "plot_area_error", "plot_month_grid",
    "build_flux_cache", "load_flux_cache", "cache_path", "FLUX_VARS",
    "flux_series", "plot_series", "flux_map", "plot_maps",
    "flux_map_months", "plot_map_grid", "map_dir", "flux_day", "flux_daily",
    "flux_balance", "flux_budget", "plot_balance",
    "load_camulator", "load_cesm",
    "regional_mean", "monthly_mean",
]
