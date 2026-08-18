"""Sea-ice comparison for CAMulator + POP2 + CICE5 checkpoint runs.

Three steps: load the two datasets, put the checkpoint on the truth's grid,
look at them side by side.

    from ice_ens import data, regrid, plot_poles

    model = data.load_checkpoint("ck33", years=[1980])
    truth = data.load_truth()
    model = regrid.to_truth_grid(model, truth)

    plot_poles.plot_year_compare(model, truth, 1980, "north")
    plot_poles.plot_month_compare(model, truth, 1980, 9, "north")

Both inputs are read in place on scratch. Output goes to
``data.OUTPUT_DIR``, one subdirectory per checkpoint.
"""

from . import data, grids, plot_poles, regrid

__all__ = ["data", "grids", "plot_poles", "regrid"]
