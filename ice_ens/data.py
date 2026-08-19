"""Finding and loading the two things being compared.

Both sides are opened in place. Nothing is copied onto /glade/work.

  checkpoint : IFRAC from the POP monthly history, gx1v7 curvilinear 384x320
  truth      : ICEFRAC from Will's zarr, regular lat/lon 192x288, year 1980

The checkpoint stays on gx1v7; the truth is what gets regridded onto it.
"""

import glob
import os
import re

import cftime
import numpy as np
import xarray as xr

# ---------------------------------------------------------------- inputs ----

#: Liyuan's case directories. Each case has a run/ full of POP history files.
CASE_ROOT = "/glade/derecho/scratch/liyuanpu"

#: Will's ground-truth run. Different grid, so this is what gets regridded --
#: onto the checkpoints' gx1v7. Covers calendar year 1980 only.
TRUTH_ZARR = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_1980_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)

# --------------------------------------------------------------- outputs ----

WORK_ROOT = "/glade/work/tvoeller/honors_thesis/code"

#: xESMF weight files. Slow to build, small to store, so they live in work.
WEIGHT_DIR = os.path.join(WORK_ROOT, "weights")

#: One subdirectory per checkpoint: output/ck33/figures, output/ck33/data.
OUTPUT_DIR = os.path.join(WORK_ROOT, "output")

#: Grid fields that have to come along -- grids.py rebuilds the cell corners
#: from ULAT/ULONG, and TAREA is what the conservation check weighs by.
GRID_VARS = ("TLAT", "TLONG", "ULAT", "ULONG", "TAREA")


def case_name(case):
    """Expand a short label into the full CESM case name.

    ``"ck33"`` -> ``"g.e21.CAMULATOR_POP_ck33"``. Anything that already looks
    like a full case name is passed straight through, so a run that does not
    follow this naming can just be given in full.
    """
    if case.startswith(("g.e21.", "g.e30.")):
        return case
    return f"g.e21.CAMULATOR_POP_{case}"


def case_label(case):
    """Collapse a full case name to the short label used for output directories."""
    return case_name(case).replace("g.e21.CAMULATOR_POP_", "")


def out_dir(case, kind="figures"):
    """Per-checkpoint output directory, created on demand."""
    path = os.path.join(OUTPUT_DIR, case_label(case), kind)
    os.makedirs(path, exist_ok=True)
    return path


#: Matches the plain monthly stream and nothing else. The run/ directories also
#: hold pop.h.nday1 daily files and pop.dv / pop.dt / restart files, which this
#: deliberately skips.
_MONTHLY = re.compile(r"\.pop\.h\.(\d{4})-(\d{2})\.nc$")


def pop_files(case, years=None):
    """Monthly POP history files for a case, sorted by date.

    ``years`` optionally restricts the result: a list of years, or two values
    read as an inclusive range.
    """
    pattern = os.path.join(
        CASE_ROOT, case_name(case), "run", f"{case_name(case)}.pop.h.[0-9]*.nc"
    )
    found = []
    for path in sorted(glob.glob(pattern)):
        match = _MONTHLY.search(path)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))

    if years is not None:
        wanted = _expand_years(years)
        found = [f for f in found if f[0] in wanted]

    return [path for _, _, path in found]


def load_checkpoint(case, years=None, var="IFRAC"):
    """Monthly coupler ice fraction for one checkpoint, on the native gx1v7 grid.

    Only ``var`` is pulled off disk. The rest of each 1.6 GB file is never
    touched, which is why a year loads in a few seconds.
    """
    files = pop_files(case, years=years)
    if not files:
        raise FileNotFoundError(
            f"no monthly POP history for {case_name(case)}"
            + (f" in years {years}" if years is not None else "")
        )

    frames = []
    for path in files:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            frames.append(ds[var].isel(time=0, drop=True).load())

    out = xr.Dataset({var: xr.concat(frames, dim="time")})

    # Grid fields are identical in every file, so read them once.
    with xr.open_dataset(files[0], decode_times=False, decode_timedelta=False) as ds:
        for name in GRID_VARS:
            if name in ds:
                out[name] = ds[name].load()

    stamps = [_month_of(path) for path in files]
    out = out.assign_coords(_month_coords(stamps))
    out.attrs["case"] = case_name(case)
    out.attrs["label"] = case_label(case)
    out.attrs["grid"] = "gx1v7 POP T-grid"
    return out


def load_truth(var="ICEFRAC", extra=("LANDFRAC",)):
    """Monthly-mean ground-truth ice fraction on its native lat/lon grid.

    The store is 6-hourly, so it is averaged to monthly means to line up with
    the POP monthly stream. A plain mean is exact here: the samples are evenly
    spaced and the calendar is noleap.

    ``LANDFRAC`` comes along by default: ``1 - LANDFRAC`` is the sea fraction
    that divides CAM's land dilution back out when the reference is carried onto
    gx1v7 -- see ``ice_ens.regrid.to_model_grid``. Pass ``extra=()`` to skip it.

    Do *not* reach for ``OCNFRAC`` for that job. In CAM,
    ``LANDFRAC + OCNFRAC + ICEFRAC = 1``: ``OCNFRAC`` is the *open* ocean and
    excludes the ice itself, so under the pack it is ~0 and dividing by it
    deletes exactly the cells the comparison is about.
    """
    ds = xr.open_zarr(TRUTH_ZARR, chunks={})

    fields = {}
    stamps = None
    for name in (var, *extra):
        if name not in ds:
            continue
        grouped = ds[name].groupby(["time.year", "time.month"]).mean("time")
        grouped = grouped.stack(ym=("year", "month")).transpose("ym", ...)
        grouped = grouped.dropna("ym", how="all")
        if stamps is None:
            stamps = [
                (int(y), int(m))
                for y, m in zip(grouped.year.values, grouped.month.values)
            ]
        fields[name] = grouped.reset_index("ym").rename(ym="time").drop_vars(
            ["year", "month"], errors="ignore"
        )

    out = xr.Dataset(fields)
    out = out.assign_coords(_month_coords(stamps))
    out = out.assign_coords(latitude=ds.latitude.values, longitude=ds.longitude.values)
    out.attrs["label"] = "ground truth"
    out.attrs["grid"] = "regular lat/lon 192x288"
    return out


def _month_of(path):
    """The (year, month) a POP monthly file *covers*, taken from its filename.

    POP writes its time coordinate at the *end* of the averaging interval, so
    the January file carries a timestamp of 1 February. Trusting that shifts
    the whole seasonal cycle by a month. The filename is the honest label.
    """
    match = _MONTHLY.search(path)
    if match is None:
        raise ValueError(f"not a monthly POP history file: {path}")
    return int(match.group(1)), int(match.group(2))


def _month_coords(stamps):
    """Mid-month time axis plus integer year/month coords, from (year, month) pairs.

    The integer coords are what everything downstream matches on -- safer than
    trusting two models' time encodings to agree.
    """
    return {
        "time": ("time", [cftime.DatetimeNoLeap(y, m, 15) for y, m in stamps]),
        "year": ("time", np.array([y for y, _ in stamps], dtype="int32")),
        "month": ("time", np.array([m for _, m in stamps], dtype="int32")),
    }


def _expand_years(years):
    years = list(years)
    if len(years) == 2 and years[1] > years[0] + 1:
        return set(range(years[0], years[1] + 1))
    return set(years)


def pick_month(ds, year, month, var=None):
    """The single (year, month) slice of a dataset.

    Returns a Dataset, or a plain 2-D array if ``var`` is given. Raises if that
    month is missing rather than silently returning nothing -- which matters
    because the ground truth only covers 1980, so asking for any other year is
    an easy mistake to make.
    """
    sel = ds.sel(time=(ds["year"] == year) & (ds["month"] == month))
    if sel.sizes.get("time", 0) != 1:
        raise KeyError(
            f"{year}-{month:02d} not in this dataset "
            f"(has {sorted(set(zip(ds.year.values.tolist(), ds.month.values.tolist())))[:3]}...)"
        )
    sel = sel.isel(time=0)
    return sel if var is None else sel[var].values
