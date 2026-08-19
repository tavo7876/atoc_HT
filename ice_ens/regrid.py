"""Moving the reference (ground truth) onto the checkpoint's native gx1v7 grid.

Direction: reference lat/lon -> gx1v7. The checkpoints are the thing under
evaluation, so they are never interpolated -- no regridding artifacts are
introduced into the field being judged. Every checkpoint is already on the same
gx1v7 grid, so checkpoint-vs-checkpoint and checkpoint-vs-reference are both a
plain cell-for-cell subtract, and POP's own ``TAREA`` stays the area weight for
every number that comes out.

Method. Two are offered and they are not interchangeable:

``bilinear``
    Centres only, no corners needed, cheap. Fine for a first-look difference
    map. It does *not* preserve an area integral -- never total an area or an
    extent off a bilinear field.

``conservative``
    Preserves the area integral, so ice area survives the trip. Needs cell
    corners on both grids, which :mod:`ice_ens.grids` reconstructs. This is the
    one to use for any ice-area or ice-extent number.

Convention. POP ``IFRAC`` is ice fraction of an *ocean* cell and gx1v7's land
mask is binary, so for a gx1v7 ocean cell "fraction of ocean" and "fraction of
cell" are the same thing. CAM ``ICEFRAC`` is ice area over *total* cell area,
so its coastal cells are diluted by their land part. Going this direction the
dilution cannot simply be filled away as it was in the old gx1v7 -> lat/lon
path; it is divided out using the reference's own sea fraction
``1 - LANDFRAC``, regridded the same way. See :func:`to_model_grid`.
"""

import os

import xarray as xr
import xesmf as xe

from . import data, grids

#: Below this regridded sea fraction a cell is essentially all land in the
#: reference, and dividing by it amplifies noise instead of removing dilution.
MIN_SEAFRAC = 0.05


def weight_file(source, target, method):
    """Where the weight file for this grid pair and method lives.

    In work, not scratch: conservative weights take a couple of minutes to build
    and a second to read back, and scratch gets purged.
    """
    os.makedirs(data.WEIGHT_DIR, exist_ok=True)
    name = (
        f"{method}_ref{source.sizes['y']}x{source.sizes['x']}"
        f"_to_gx1v7_{target.sizes['y']}x{target.sizes['x']}.nc"
    )
    return os.path.join(data.WEIGHT_DIR, name)


def build_regridder(truth, checkpoint, method="conservative"):
    """Regridder carrying the reference onto the checkpoint's gx1v7 grid.

    Source is the reference's regular lat/lon grid, destination is gx1v7. The
    checkpoint is only read for its grid description, so any checkpoint will do
    -- they are all on the same grid, which is the whole point of this
    direction. Build it once and pass it to every call.

    Weights are written to ``weights/`` under the repo and reused on every later
    call.
    """
    source = grids.f09_grid(truth)
    target = grids.pop_grid(checkpoint)
    path = weight_file(source, target, method)

    kwargs = {}
    if method == "conservative":
        # gx1v7's tripole seam has a couple of collapsed cells; ESMF should skip
        # them rather than refuse to build the whole map.
        kwargs["ignore_degenerate"] = True
    else:
        # Only the point methods use this; conservative warns if it is passed.
        kwargs["periodic"] = True

    return xe.Regridder(
        source, target,
        method=method,
        filename=path,
        reuse_weights=os.path.exists(path),
        **kwargs,
    )


def to_model_grid(truth, checkpoint, regridder=None, method="conservative",
                  var="ICEFRAC", undilute=True):
    """Regrid the reference's monthly ice fraction onto the native gx1v7 grid.

    Returns a Dataset on gx1v7 (``nlat``/``nlon``, with ``TLAT``/``TLONG``/
    ``TAREA`` carried over from the checkpoint) holding two fields:

    ``IFRAC``
        Ice fraction *of the sea part of the cell* -- the same convention as
        POP's ``IFRAC``, so this is what subtracts cell-for-cell against a
        checkpoint. This is the one to plot and difference.
    ``ICEFRAC``
        The raw regridded field, ice area over *total* cell area, still in CAM's
        convention. Area-conserving, so this is what
        :func:`check_conservation` weighs and what reproduces the reference's
        own total ice area.

    The un-dilution. A gx1v7 ocean cell is wholly ocean, but the reference cells
    overlapping it may be part land, and CAM puts no ice on their land part. So
    the regridded ice area is low against what the model reports for the same
    patch of sea. Dividing by the reference's sea fraction, regridded through
    the identical map, removes exactly that. Away from coasts the sea fraction
    is 1 and nothing changes. Pass ``undilute=False`` to skip it and get
    ``IFRAC`` equal to the raw field.

    Sea fraction is ``1 - LANDFRAC``, and it has to be. CAM's three fractions
    satisfy ``LANDFRAC + OCNFRAC + ICEFRAC = 1``, so ``OCNFRAC`` is the *open*
    ocean and goes to zero under the pack -- dividing by that deletes the solid
    ice instead of un-diluting it, and quietly turns the Arctic seasonal cycle
    into nonsense.
    """
    if regridder is None:
        regridder = build_regridder(truth, checkpoint, method=method)

    ice = _regrid(regridder, truth[var])

    if undilute and "LANDFRAC" in truth:
        land = truth["LANDFRAC"]
        if "time" in land.dims:       # static to 1e-14; one slice is the field
            land = land.isel(time=0, drop=True)
        sea = _regrid(regridder, 1.0 - land)
        # Mask first, divide second: dividing the whole array and masking after
        # evaluates 0/0 on every land cell and floods the log with warnings.
        ifrac = (ice / sea.where(sea >= MIN_SEAFRAC)).clip(0.0, 1.0)
    else:
        sea = None
        ifrac = ice.clip(0.0, 1.0)

    # Outside POP's ocean mask there is no ice fraction to speak of, and leaving
    # values there would let land cells into any later sum.
    mask = ocean_mask(checkpoint)
    ifrac = ifrac.where(mask)

    # Dividing by a static 2-D sea fraction pushes time to the end. Put it back
    # in front so a slice looks like every other field in this package.
    if "time" in ifrac.dims:
        ifrac = ifrac.transpose("time", "nlat", "nlon")

    out = xr.Dataset({"IFRAC": ifrac, "ICEFRAC": ice})
    if sea is not None:
        out["SEAFRAC"] = sea

    for name in data.GRID_VARS:
        if name in checkpoint:
            out[name] = checkpoint[name]
    for coord in ("time", "year", "month"):
        if coord in truth.coords:
            out = out.assign_coords({coord: truth[coord]})

    out["IFRAC"].attrs = {
        "long_name": "Fraction of sea area covered by sea ice",
        "units": "fraction",
        "note": f"reference {var} regridded ({method}) onto gx1v7"
                + (", divided by regridded 1-LANDFRAC" if undilute else ""),
    }
    out["ICEFRAC"].attrs = {
        "long_name": "Fraction of total cell area covered by sea ice",
        "units": "fraction",
        "note": f"reference {var} regridded ({method}) onto gx1v7, CAM convention",
    }
    out.attrs.update(truth.attrs)
    out.attrs["label"] = truth.attrs.get("label", "ground truth")
    out.attrs["grid"] = "gx1v7 POP T-grid"
    out.attrs["regrid"] = f"{method} (xESMF/ESMF), reference -> gx1v7"
    return out


def _regrid(regridder, field):
    """Push one field through the map and label the result with gx1v7's dims.

    xESMF names its output dims after the destination grid description (``y``/
    ``x``). They have to be renamed here: left alone, attaching ``TLAT``/
    ``TLONG`` afterwards creates *new dangling dimensions* rather than
    labelling the existing ones, and any later multiply against ``TAREA``
    broadcasts into a hundred-gigabyte array instead of aligning. That is not a
    hypothetical -- it silently OOM-killed this pipeline once.
    """
    out = regridder(field.fillna(0.0))
    return out.rename(dict(zip(out.dims[-2:], ("nlat", "nlon"))))


def ocean_mask(checkpoint, var="IFRAC"):
    """True where gx1v7 is ocean, taken from POP's own missing-value mask.

    POP writes land as fill, so any single time slice of ``IFRAC`` carries the
    mask. Returns a 2-D (``nlat``, ``nlon``) boolean.
    """
    field = checkpoint[var]
    if "time" in field.dims:
        field = field.isel(time=0)
    return field.notnull()


def check_conservation(truth, regridded, checkpoint, var="ICEFRAC"):
    """Reference ice area before and after the regrid, per month.

    This is the check that says the regrid behaved, and it only means anything
    for ``method="conservative"`` -- a bilinear field will fail it, by design.

    ``relative_error`` compares the reference's native total against the total
    over *every* gx1v7 cell, land included, so it isolates the regridding
    itself. ``area_on_land`` then reports how much of that ice landed on cells
    POP calls land: that ice is dropped from ``IFRAC`` and is a real
    land-mask mismatch between the two grids, not a regridding error.
    """
    source_area = xr.DataArray(
        grids.cell_area(grids.f09_grid(truth)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": truth["latitude"].values,
            "longitude": truth["longitude"].values,
        },
    )
    before = (truth[var].fillna(0.0) * source_area).sum(dim=("latitude", "longitude"))

    target_area = checkpoint["TAREA"] * 1.0e-4              # cm2 -> m2
    total = (regridded["ICEFRAC"] * target_area).sum(dim=("nlat", "nlon"))
    on_land = (
        regridded["ICEFRAC"].where(~ocean_mask(checkpoint), 0.0) * target_area
    ).sum(dim=("nlat", "nlon"))

    return xr.Dataset(
        {
            "area_before": before,
            "area_after": total,
            "relative_error": (total - before) / before,
            "area_on_land": on_land,
            "land_fraction_of_total": on_land / total,
        }
    )
