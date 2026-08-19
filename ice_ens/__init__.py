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

from . import data, grids, metrics, plot_poles, regrid

__all__ = ["data", "grids", "metrics", "plot_poles", "regrid"]
