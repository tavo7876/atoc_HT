"""Ice-area and ice-extent numbers, computed on the native gx1v7 grid.

Everything here weighs by POP's own ``TAREA``, which is the reason for keeping
the checkpoints native: the same weight applies to the checkpoint and to the
reference once the reference has been carried across, so the two totals are
built the identical way.

One caveat that matters. These numbers are only trustworthy if the reference
was regridded with ``method="conservative"``. A bilinear map does not preserve
an area integral, so a total taken off a bilinear field is wrong by an amount
nobody can bound -- it is fine for looking at a difference map and nothing else.
Both functions refuse a dataset that says it came through a non-conservative
regrid.
"""

import numpy as np
import xarray as xr

#: The conventional threshold for a cell counting as ice-covered.
EXTENT_THRESHOLD = 0.15


def _area(ds, tarea=None):
    """Cell area in m2, from POP's ``TAREA`` (which is in cm2)."""
    if tarea is None:
        if "TAREA" not in ds:
            raise KeyError(
                "no TAREA in this dataset -- pass tarea= from the checkpoint"
            )
        tarea = ds["TAREA"]
    return tarea * 1.0e-4


def _check_conservative(ds, allow_bilinear):
    regrid = ds.attrs.get("regrid", "")
    if regrid and "conservative" not in regrid and not allow_bilinear:
        raise ValueError(
            f"this field was regridded with {regrid!r}; an area integral off a "
            "non-conservative regrid is not meaningful. Rebuild it with "
            "method='conservative', or pass allow_bilinear=True if you know "
            "why you want the number anyway."
        )


def _hemisphere_mask(ds, hemisphere):
    if hemisphere is None:
        return True
    lat = ds["TLAT"]
    return lat >= 0 if hemisphere == "north" else lat < 0


def ice_area(ds, var="IFRAC", hemisphere="north", tarea=None,
             allow_bilinear=False):
    """Total sea-ice area (m2) per time step: ``sum(IFRAC * TAREA)``.

    ``hemisphere`` is ``"north"``, ``"south"``, or ``None`` for global. Cells
    outside POP's ocean mask are NaN and drop out of the sum on their own.
    """
    _check_conservative(ds, allow_bilinear)
    area = _area(ds, tarea)
    field = ds[var].where(_hemisphere_mask(ds, hemisphere))
    out = (field * area).sum(dim=("nlat", "nlon"))
    out.attrs = {"long_name": f"{hemisphere or 'global'} sea-ice area", "units": "m2"}
    return out


def ice_extent(ds, var="IFRAC", hemisphere="north", threshold=EXTENT_THRESHOLD,
               tarea=None, allow_bilinear=False):
    """Total sea-ice extent (m2): the area of every cell over ``threshold``.

    Extent counts a cell's whole area once its ice fraction clears the
    threshold, so it is always larger than the area and is the quantity the
    observational indices report.
    """
    _check_conservative(ds, allow_bilinear)
    area = _area(ds, tarea)
    covered = (ds[var] >= threshold) & _hemisphere_mask(ds, hemisphere)
    out = area.where(covered).sum(dim=("nlat", "nlon"))
    out.attrs = {
        "long_name": f"{hemisphere or 'global'} sea-ice extent",
        "units": "m2",
        "threshold": threshold,
    }
    return out


def compare(model, reference, hemisphere="north", var="IFRAC", tarea=None,
            allow_bilinear=False):
    """Checkpoint against reference, month by month, area and extent.

    Returns a Dataset in millions of km2 -- the unit these are always quoted in
    -- with the model's value, the reference's, and the difference. ``tarea``
    defaults to the checkpoint's, which is also the reference's now that it
    lives on gx1v7.
    """
    if tarea is None:
        tarea = model["TAREA"] if "TAREA" in model else reference["TAREA"]

    to_mkm2 = 1.0e-12                                 # m2 -> 10^6 km2
    fields = {}
    for name, func in (("area", ice_area), ("extent", ice_extent)):
        m = func(model, var=var, hemisphere=hemisphere, tarea=tarea,
                 allow_bilinear=allow_bilinear) * to_mkm2
        r = func(reference, var=var, hemisphere=hemisphere, tarea=tarea,
                 allow_bilinear=allow_bilinear) * to_mkm2
        fields[f"model_{name}"] = m
        fields[f"reference_{name}"] = r
        fields[f"{name}_bias"] = m - r

    out = xr.Dataset(fields)
    out.attrs["units"] = "10^6 km2"
    out.attrs["hemisphere"] = hemisphere or "global"
    return out
