"""Moving the checkpoint onto the ground-truth grid so the two can be compared.

Direction: gx1v7 -> ground-truth lat/lon. Going this way means one weight file
serves every checkpoint, the truth is never altered, and the result carries 1-D
latitude/longitude that plot_poles.py already understands.

Method: conservative. For a fractional field this is the right choice -- it
preserves the area integral, so the total amount of ice survives the trip.
Verified per month by check_conservation() below.
"""

import os

import numpy as np
import xarray as xr
import xesmf as xe

from . import data, grids


def weight_file(source, target):
    """Where the weight file for this grid pair lives.

    In work, not scratch: it takes a couple of minutes to build and a second to
    read back, and scratch gets purged.
    """
    os.makedirs(data.WEIGHT_DIR, exist_ok=True)
    name = (
        f"conservative_{source.sizes['y']}x{source.sizes['x']}"
        f"_to_{target.sizes['y']}x{target.sizes['x']}.nc"
    )
    return os.path.join(data.WEIGHT_DIR, name)


def build_regridder(checkpoint, truth):
    """Conservative regridder from the checkpoint's POP grid onto the truth grid.

    Weights are written once and reused on every later call. Every checkpoint
    is on the same gx1v7 grid, so one regridder serves all of them -- build it
    once and pass it in.
    """
    source = grids.pop_grid(checkpoint)
    target = grids.f09_grid(truth)
    path = weight_file(source, target)
    return xe.Regridder(
        source, target,
        method="conservative",
        filename=path,
        reuse_weights=os.path.exists(path),
        periodic=True,
    )


def to_truth_grid(checkpoint, truth, regridder=None, var="IFRAC"):
    """Regrid a checkpoint's monthly ice fraction onto the ground-truth grid.

    Returns a Dataset with ``ICEFRAC``, directly comparable to the truth's own
    ``ICEFRAC``.

    Why the two are comparable. POP's ``IFRAC`` is ice fraction *of an ocean
    cell*, and gx1v7's land mask is binary -- a cell is ocean or it is not.
    CAM's ``ICEFRAC`` is ice area over *total* cell area, so its coastal cells
    are diluted by their land part. Filling POP land with zero before a
    conservative regrid reproduces CAM's convention exactly, which is what puts
    the two on the same footing.
    """
    if regridder is None:
        regridder = build_regridder(checkpoint, truth)

    # Land is NaN in POP output. Conservative regridding needs a number there,
    # and "no ocean means no ice" is the physically right fill.
    ice = regridder(checkpoint[var].fillna(0.0))

    # xESMF names its output dims after the destination grid (y/x). They have to
    # be renamed here: left alone, attaching latitude/longitude coordinates
    # creates *new dangling dimensions* rather than labelling the existing ones,
    # and any later multiply against a lat/lon field broadcasts into a
    # hundred-gigabyte array instead of aligning. That is not a hypothetical --
    # it silently OOM-killed this pipeline once.
    ice = ice.rename(dict(zip(ice.dims[-2:], ("latitude", "longitude"))))

    out = xr.Dataset({"ICEFRAC": ice})
    out = out.assign_coords(
        latitude=truth["latitude"].values,
        longitude=truth["longitude"].values,
    )
    for coord in ("time", "year", "month"):
        if coord in checkpoint.coords:
            out = out.assign_coords({coord: checkpoint[coord]})

    out["ICEFRAC"].attrs = {
        "long_name": "Fraction of cell area covered by sea ice",
        "units": "fraction",
        "note": f"conservatively regridded from POP {var} on gx1v7",
    }
    out.attrs.update(checkpoint.attrs)
    out.attrs["grid"] = truth.attrs.get("grid", "ground-truth lat/lon")
    out.attrs["regrid"] = "conservative (xESMF/ESMF)"
    return out


def check_conservation(checkpoint, regridded, truth, var="IFRAC"):
    """Total ice area before and after regridding, per month.

    This is the check that says the regrid behaved. Conservative regridding
    should preserve the area integral, so ``relative_error`` should sit around
    1e-4 -- that residue is POP's own ``TAREA`` disagreeing slightly with the
    area implied by its cell corners, not a regridding error. Percent-level
    values mean something is wrong.
    """
    source_area = checkpoint["TAREA"] * 1.0e-4              # cm2 -> m2
    before = (checkpoint[var].fillna(0.0) * source_area).sum(dim=("nlat", "nlon"))

    target_area = xr.DataArray(
        grids.cell_area(grids.f09_grid(truth)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": truth["latitude"].values,
            "longitude": truth["longitude"].values,
        },
    )
    after = (regridded["ICEFRAC"] * target_area).sum(dim=("latitude", "longitude"))

    return xr.Dataset(
        {
            "area_before": before,
            "area_after": after,
            "relative_error": (after - before) / before,
        }
    )
