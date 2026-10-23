"""Surface-flux time series: one CAMulator variable against one CESM variable.

Same shape as ``ensemble.py`` -- plain arguments, no config file, no paths baked
in -- but for the atmosphere side rather than the ice.

    build_flux_cache  read every daily file once, cache monthly means  <- slow
    load_flux_cache   read that cache back                             <- instant
    load_camulator    the daily emulator files for one case  -> (time, lat, lon)
    load_cesm         the 6-hourly reference zarrs           -> (time, lat, lon)
    regional_mean     cos(lat)-weighted mean over a latitude band -> 1-D in time
    monthly_mean      sub-monthly -> monthly means, or a 12-month climatology
    flux_series       drives all of the above for both sides at once
    plot_series       the two curves on one figure

The two grids are already identical -- ``latitude`` 192 x ``longitude`` 288 on
both sides -- so **nothing here regrids anything**, and no weight file is built.
That is the whole reason this module is short. What *is* expensive is reading
the emulator's 365-files-per-year daily output, which is why there is a cache.

The workflow is two steps. :func:`build_flux_cache` walks each daily file once,
pulls all six fluxes out of it at a time, averages to monthly means and writes
one small netCDF per side per year. Everything after that -- picking a variable,
changing the latitude band, widening the window, switching between a continuous
series and a climatology -- reads that cache and is instant.

The cache is keyed **per year**, not per window, so a one-year, five-year and
ten-year climatology are three selections of the same files rather than three
copies of overlapping data, and extending a window only builds the years it
adds.

Default band is the Arctic, 60-90N, because that is what the rest of this repo
is about. ``lat_range=None`` gives the global mean.

Four things about the inputs that are not obvious:

**The CAMulator files have no time dimension.** Each ``camulator.h1d.<date>.nc``
is a single ``(latitude, longitude)`` field with the date only in its
filename, so the time coordinate is parsed from the name and attached here.
It is built as ``cftime.DatetimeNoLeap`` to match CESM's noleap calendar.

**latitude is float32 on one side and float64 on the other.** The values agree
to 3.8e-6, but only 2 of the 192 are bit-identical, so an xarray binary op
between the two sides silently inner-joins down to those 2 rows and returns
nonsense. Casting the dtype does not fix that -- it keeps the wrong value at a
wider precision. ``_match_grid`` replaces the emulator's coordinate *values*
with CESM's, so both sides of the cache carry bit-identical coordinates and
``cam - cesm`` lines up. It raises if the two grids ever differ by more than
rounding. (``longitude`` already matches exactly.)

**Both models store the fluxes per 6-hour step, and this module divides that
out.** All six are written as ``J m-2`` accumulated over one step rather than as
a rate, so the raw numbers run to millions and mean nothing on a figure axis.
:data:`STEP_SECONDS` (21600, one step) converts them, and
:func:`_to_watts` applies it wherever data enters the analysis path -- not where
the cache is written, so the cached files stay a faithful copy of the source and
never need rebuilding for this. Everything the notebook sees is ``W m-2``.

The divisor is one step and not one day because a daily CAMulator file is the
*mean* of its four 6-hourly steps, not their sum. CESM says so for the four
radiative fluxes in its own attributes (``scaling: raw_W_m2 * 21600``); the
emulator side was settled numerically, and is written up in ``ice_ens/README.md``.

**The CAMulator files carry no attributes at all** -- no units, no long names.
They are unlabelled versions of the same fields CESM writes: checked value by
value on 1980-01-15. Units for the plot are therefore taken from the CESM side.
CESM labels the four radiative fluxes and leaves ``SHFLX``/``LHFLX`` bare, so
:func:`_to_watts` is also where those two pick up a unit and the sign convention
their values follow: negative on both sides, meaning positive into the surface.
"""

import datetime
import glob
import os
import re
import time as _clock

import cartopy.crs as ccrs
import cftime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr

from . import data, plot_poles

#: Where the emulator writes, under each case's run directory. One
#: subdirectory per year, one file per day.
CAMULATOR_SUBDIR = os.path.join("run", "camulator_out")

#: ``camulator.h1d.1980-01-01.nc`` -- the date is the only place the time lives.
_DAILY = re.compile(r"\.h1d\.(\d{4})-(\d{2})-(\d{2})\.nc$")

#: The six surface fluxes this comparison is about, and what
#: :func:`build_flux_cache` caches. Four radiative, two turbulent. These are the
#: variables ``output/<case>/fluxes/`` has a directory for.
FLUX_VARS = ("FLDS_J", "FSDS_J", "FSUS", "FLUS", "SHFLX", "LHFLX")

#: Seconds in one 6-hour step. Both models store all six fluxes as J m-2
#: accumulated over a step, so this is the divisor to a rate in W m-2. It is one
#: step and not one day: a daily CAMulator file is the *mean* of its four
#: 6-hourly steps, not their sum, which is checked in the module docstring.
STEP_SECONDS = 21600

#: The 2-D fields the emulator actually writes. ``U``/``V``/``T``/``Qtot`` are
#: also there but carry a ``level`` dimension and are not time series. Anything
#: here can be read with :func:`load_camulator`; only :data:`FLUX_VARS` is
#: cached.
CAMULATOR_VARS = (
    "PS", "TREFHT", "TS", "PRECT", "QFLX", "U10", "TAUX", "TAUY",
    "FSDS_J", "FLDS_J", "FSUS", "FLUS", "SHFLX", "LHFLX", "FSUTOA", "FLUT",
    "CLDHGH", "CLDLOW", "CLDMED",
)

#: Where the monthly-mean cache lives. On scratch rather than work because it
#: is the one bulky thing this package writes -- about 16 MB per side per year,
#: so a decade of both models is a few hundred megabytes. Scratch is purged on
#: a 90-day cycle; the cache is derived data and rebuilding it is one call, so
#: losing it costs time and nothing else.
CACHE_ROOT = "/glade/derecho/scratch/tvoeller/honors_thesis/flux_cache"

#: Both cached files, and every plot, key off these names.
SOURCES = ("camulator", "cesm")


# --------------------------------------------------------------- finding ----

def camulator_files(case, year):
    """The daily emulator files for one case and one year, sorted by date.

    Mirrors :func:`ice_ens.data.pop_files`, which does the same job for the POP
    monthly stream.
    """
    pattern = os.path.join(
        data.CASE_ROOT, data.case_name(case), CAMULATOR_SUBDIR, str(year),
        "camulator.h1d.*.nc",
    )
    found = sorted(path for path in glob.glob(pattern) if _DAILY.search(path))
    if not found:
        raise FileNotFoundError(
            f"no CAMulator output for {data.case_name(case)} in {year}; "
            f"looked for {pattern}"
        )
    return found


def _wanted_years(years):
    """The ``years`` shorthand -> a sorted list of integer years.

    A bare int, a list of years, or two values read as an inclusive range --
    the same shorthand :func:`ice_ens.data.load_checkpoint` takes, so every
    side of this comparison is asked for the same window in the same words.
    """
    return sorted(data._expand_years([years] if isinstance(years, int) else years))


def _years_tag(years):
    """``1980``, ``1980-1984``, or ``1980_1985`` -- for filenames and titles.

    A gap matters: a run of consecutive years collapses to a range, and
    anything else is listed out, so ``[1980, 1985]`` never gets written as if
    it covered the six years in between.
    """
    wanted = _wanted_years(years)
    if len(wanted) == 1:
        return str(wanted[0])
    if wanted == list(range(wanted[0], wanted[-1] + 1)):
        return f"{wanted[0]}-{wanted[-1]}"
    return "_".join(str(year) for year in wanted)


#: Coordinates closer than this are the same grid written at two precisions,
#: not two different grids. The observed gap is 3.8e-6 degrees.
COORD_TOL = 1e-4


def _match_grid(ds, year):
    """Put the emulator onto CESM's exact lat/lon values.

    The emulator writes ``latitude`` as float32 and CESM as float64. They agree
    to 3.8e-6 -- the same grid at two precisions -- but only **2 of the 192**
    are bit-identical, and xarray aligns indexes on exact equality. So a
    ``cam - cesm`` inner-joins down to those two rows and returns nonsense.

    Casting the dtype does *not* fix this: float32 -> float64 changes the type
    and keeps the wrong value. The coordinate values themselves have to be
    replaced, which is what this does -- CESM's are taken as canonical, so both
    sides of the cache carry bit-identical coordinates and a later subtraction
    lines up.

    Guarded by :data:`COORD_TOL`: if the two grids ever genuinely differ, this
    raises rather than quietly relabelling one as the other. Falls back to a
    plain cast when there is no CESM store for that year to align to.
    """
    if year not in set(data.truth_years()):
        return ds.assign_coords(
            latitude=ds.latitude.astype("float64"),
            longitude=ds.longitude.astype("float64"),
        )

    reference = xr.open_zarr(data.truth_zarr(year), chunks={})
    for axis in ("latitude", "longitude"):
        mine, theirs = ds[axis].values, reference[axis].values
        if mine.shape != theirs.shape:
            raise ValueError(
                f"{axis}: emulator has {mine.shape[0]} points, CESM has "
                f"{theirs.shape[0]} -- these are different grids, and this "
                f"module assumes they are not (nothing here regrids)"
            )
        gap = np.abs(mine.astype("float64") - theirs).max()
        if gap > COORD_TOL:
            raise ValueError(
                f"{axis}: emulator and CESM differ by up to {gap:g} degrees, "
                f"more than COORD_TOL={COORD_TOL:g}. That is a real grid "
                f"difference, not a precision one -- it needs regridding, "
                f"which this module deliberately does not do."
            )

    return ds.assign_coords(
        latitude=reference["latitude"].values,
        longitude=reference["longitude"].values,
    )


# ------------------------------------------------------------- the cache ----

def cache_path(case, source, year):
    """Where one side's monthly means for one year live.

    ``<CACHE_ROOT>/<case>/<source>_fluxes_monthly_<year>.nc`` -- **one file per
    year**, deliberately. Keying the cache on a whole window instead would mean
    a 1980-1984 climatology and a 1980-1989 one each stored their own copy of
    the five years they share, and widening the window would rebuild years that
    were already on disk. Per year, every window is just a different selection
    of the same files, and extending 1980-1984 to 1980-1989 builds five years,
    not ten.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    directory = os.path.join(CACHE_ROOT, data.case_label(case))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{source}_fluxes_monthly_{int(year)}.nc")


def _camulator_year(case, variables, year):
    """Every one of ``variables`` for one year, as one daily Dataset.

    One pass over the year's 365 files pulling all six fluxes, rather than six
    passes pulling one each. The file opens are what this costs, so reading six
    variables is barely more expensive than reading one -- which is the whole
    point of caching them together.
    """
    variables = list(variables)
    frames, stamps = [], []
    for path in camulator_files(case, year):
        with xr.open_dataset(path) as ds:
            missing = [name for name in variables if name not in ds]
            if missing:
                raise KeyError(
                    f"{', '.join(missing)} not in {os.path.basename(path)}; "
                    f"2-D fields are {', '.join(CAMULATOR_VARS)}"
                )
            layered = [name for name in variables if "level" in ds[name].dims]
            if layered:
                raise ValueError(
                    f"{', '.join(layered)} is 3-D (it has a `level` dimension); "
                    f"pick a level first, or use one of {', '.join(CAMULATOR_VARS)}"
                )
            frames.append(ds[variables].load())
        y, m, d = (int(g) for g in _DAILY.search(path).groups())
        stamps.append(cftime.DatetimeNoLeap(y, m, d))

    out = xr.concat(frames, dim="time").assign_coords(time=("time", stamps))
    return _match_grid(out, year)


def build_flux_cache(case, years, variables=FLUX_VARS, force=False):
    """Read both models once and cache the monthly-mean fluxes. Run per checkpoint.

    This is the slow step, and it only has to happen once per (case, year). It
    walks the emulator's daily files a single time, taking all of ``variables``
    out of each, averages both sides to monthly means on the native 192x288
    grid and writes one netCDF per side per year under :data:`CACHE_ROOT`.

    Monthly means on the *full grid* are what gets stored, not a regional
    average, so the latitude band stays a knob afterwards: a different band, a
    climatology over a different window, or a map, all come off the same cache
    without touching the raw output again.

    Roughly a minute and a half per year of emulator output; CESM is seconds.
    Years already on disk are skipped, so widening the window only pays for the
    years it adds, and a second call is instant. A cached year that is missing
    one of ``variables`` is rebuilt on its own. ``force=True`` rebuilds
    everything, which is only needed if the underlying run was rewritten.

    Returns ``{source: [paths, one per year]}``.
    """
    wanted = _wanted_years(years)
    variables = list(variables)
    paths = {}
    rebuilt = 0

    for source in SOURCES:
        built = []
        for year in wanted:
            path = cache_path(case, source, year)
            if not force and _cache_is_complete(path, variables):
                built.append(path)
                continue

            mark = _clock.time()
            if source == "camulator":
                year_ds = _camulator_year(case, variables, year)
            else:
                year_ds = _cesm_year(variables, year)

            out = year_ds.resample(time="MS").mean().load()
            out.attrs.update(
                case=data.case_name(case),
                label=data.case_label(case),
                source="CAMulator" if source == "camulator" else "CESM",
                year=int(year),
                note="monthly means on the native 192x288 grid; no regridding "
                     "(both models are already on this grid)",
            )
            _write_cache(out, path)
            built.append(path)
            rebuilt += 1
            print(f"{source:9s} {year}: built in {_clock.time() - mark:5.0f}s")

        paths[source] = built

    # Silent when there was nothing to do, because `flux_series` calls this on
    # every plot and a line of output per figure is just noise.
    if rebuilt:
        print(f"cache ready: {sum(len(v) for v in paths.values())} files for "
              f"{_years_tag(years)} in "
              f"{os.path.join(CACHE_ROOT, data.case_label(case))}")
    return paths


def _cache_is_complete(path, variables):
    """True if ``path`` exists and already holds every one of ``variables``.

    An existing file built for a shorter variable list is not good enough --
    silently reading it back would produce a ``KeyError`` several calls later,
    a long way from the cause.
    """
    if not os.path.exists(path):
        return False
    with xr.open_dataset(path) as ds:
        return all(name in ds for name in variables)


def _cesm_year(variables, year):
    """Every one of ``variables`` for one year, out of the reference zarr.

    Read one variable at a time. A whole year of one 6-hourly field is ~320 MB;
    six at once is close to two gigabytes for no benefit, since the store is
    chunked per variable and there is no per-file cost to amortise the way
    there is on the emulator side.
    """
    available = set(data.truth_years())
    if year not in available:
        raise FileNotFoundError(
            f"no CESM store for {year}; have {min(available)}-{max(available)}"
        )

    store = xr.open_zarr(data.truth_zarr(year), chunks={})
    fields = {}
    for name in variables:
        if name not in store:
            raise KeyError(
                f"{name!r} is not in {os.path.basename(data.truth_zarr(year))}"
            )
        field = store[name].load()
        field.attrs = dict(store[name].attrs)
        fields[name] = field
    return xr.Dataset(fields)


def _write_cache(ds, path):
    """Write the cache, compressed, without clobbering a good file on a failure.

    netCDF is written to a temporary name and moved into place, so an
    interrupted build leaves the previous cache intact rather than a truncated
    file that reads back as a silent hole.
    """
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    scratch = f"{path}.partial"
    ds.to_netcdf(scratch, encoding=encoding)
    os.replace(scratch, path)


def _to_watts(field, name):
    """``J m-2 per 6hr step`` -> ``W m-2``, for the six fluxes and nothing else.

    Applied wherever data enters the analysis path rather than where the cache
    is written, so the cached files stay a faithful copy of what the two models
    actually store and no rebuild is needed.

    Anything outside :data:`FLUX_VARS` comes back untouched -- ``TREFHT`` is
    kelvin and ``PS`` is pascals, and both are reachable through
    :func:`load_camulator`.

    An xarray binary op drops attributes, so they are put back by hand.
    ``scaling`` describes the stored form and would be a lie afterwards, so it
    goes. ``SHFLX``/``LHFLX`` arrive with no attributes at all on either side,
    which is why they used to plot with a bare axis label, so this is also where
    they pick up the sign convention the radiative fluxes declare for themselves
    and which their own values follow.
    """
    if name not in FLUX_VARS:
        return field

    out = field / STEP_SECONDS
    out.attrs = dict(field.attrs)
    out.attrs.pop("scaling", None)
    out.attrs["units"] = "W m-2"
    if name in ("SHFLX", "LHFLX"):
        out.attrs.setdefault("sign_convention", "positive = energy into surface")
    return out


def load_flux_cache(case, source, years):
    """Read one side's cached monthly means back, for any window. Instant.

    Concatenates the per-year files, so one year and ten years are the same
    call. Raises with the command that would build the missing years rather
    than silently falling back to a read off the raw output.

    The six fluxes come back in ``W m-2``. The files on disk hold the per-step
    energy the two models write; :func:`_to_watts` divides on the way out, so
    this and :func:`load_camulator` agree without the cache having to be
    rebuilt.
    """
    wanted = _wanted_years(years)
    missing = [y for y in wanted if not os.path.exists(cache_path(case, source, y))]
    if missing:
        raise FileNotFoundError(
            f"no {source} flux cache for {data.case_label(case)} "
            f"{', '.join(str(y) for y in missing)}; build it with "
            f"build_flux_cache({case!r}, {years!r})"
        )

    pieces = [
        xr.open_dataset(cache_path(case, source, year), use_cftime=True)
        for year in wanted
    ]
    out = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
    for name in FLUX_VARS:
        if name in out:
            out[name] = _to_watts(out[name], name)
    return out


# ------------------------------------------------- reading scratch direct ----

def load_camulator(case, var, years=1980):
    """Daily ``var`` from the emulator, stacked into one ``(time, lat, lon)`` field.

    The uncached path -- straight off scratch, about a minute and a half per
    year. Use it for a variable outside :data:`FLUX_VARS`; for the six fluxes,
    :func:`build_flux_cache` plus :func:`load_flux_cache` is the same numbers
    far faster -- including the ``W m-2`` conversion, which happens on both
    paths.
    """
    pieces = [
        _camulator_year(case, [var], year)[var] for year in _wanted_years(years)
    ]
    out = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
    out = _to_watts(out, var)
    out.attrs["case"] = data.case_name(case)
    out.attrs["label"] = data.case_label(case)
    out.attrs["source"] = "CAMulator"
    return out


def load_cesm(var, years=1980):
    """6-hourly ``var`` from the reference run, one zarr store per calendar year.

    Takes the same ``years`` shorthand as :func:`load_camulator`. Only ``var``
    is pulled out of each store, which is what keeps a multi-year window cheap.
    One of the six fluxes comes back in ``W m-2``; anything else is untouched.
    """
    wanted = _wanted_years(years)
    available = set(data.truth_years())
    missing = [y for y in wanted if y not in available]
    if missing:
        raise FileNotFoundError(
            f"no CESM store for {missing}; have "
            f"{min(available)}-{max(available)}"
        )

    pieces = []
    for year in wanted:
        ds = xr.open_zarr(data.truth_zarr(year), chunks={})
        if var not in ds:
            raise KeyError(f"{var!r} is not in {os.path.basename(data.truth_zarr(year))}")
        pieces.append(ds[var])

    out = _to_watts(
        xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0], var
    )
    out.attrs["source"] = "CESM"
    return out


# ------------------------------------------------------------ reductions ----

def regional_mean(field, lat_range=(60, 90)):
    """Area-weighted mean over a latitude band. ``(time, lat, lon)`` -> ``(time,)``.

    Weighted by ``cos(latitude)``, which is the cell-area weight on this regular
    192x288 grid -- an unweighted mean would over-count the poles badly, and the
    default band is nothing but pole.

    ``lat_range`` is ``(south, north)`` in degrees, or ``None`` for the globe.
    Latitude runs -90 to 90 ascending in both datasets, so the plain slice is
    the right way round.
    """
    if lat_range is not None:
        south, north = lat_range
        field = field.sel(latitude=slice(south, north))
        if field.sizes["latitude"] == 0:
            raise ValueError(f"no latitudes in {lat_range}; give it as (south, north)")

    weights = np.cos(np.deg2rad(field.latitude))
    out = field.weighted(weights).mean(dim=("latitude", "longitude"))
    out.attrs = dict(field.attrs)
    out.attrs["lat_range"] = "global" if lat_range is None else f"{lat_range[0]}-{lat_range[1]}N"
    return out


def monthly_mean(series, climatology=True):
    """Monthly means, or the seasonal cycle across every year in the window.

    ``climatology=True`` (the default) collapses the window onto a 12-point
    seasonal cycle indexed by ``month`` -- the five-year climatological mean.
    ``False`` gives the continuous monthly series instead: 12 points for one
    year, 60 for 1980-1984, on a real time axis.

    Safe to call on data that is already monthly, which is what comes out of
    the cache: the resample is then a no-op, and averaging monthly means into a
    climatology weights each year equally, which is what a climatology is.

    A plain mean is exact on the raw path too -- the samples are evenly spaced
    (6-hourly for CESM, daily for CAMulator) on a noleap calendar.
    """
    if climatology:
        out = series.groupby("time.month").mean(dim="time")
    else:
        out = series.resample(time="MS").mean()
    out.attrs = dict(series.attrs)
    return out


def flux_series(case, camulator_var, cesm_var, years=(1980, 1984),
                lat_range=(60, 90), climatology=True, months=None, cache=True):
    """Regional flux series for one variable from each model. The notebook's one call.

    Load each side, take the same ``lat_range`` mean of each, reduce each to
    months. Returns ``(cam, cesm)`` -- two 1-D DataArrays on the same axis.

    ``months`` narrows the series to certain calendar months, the same shorthand
    :func:`flux_map` takes: ``None`` for all of them, ``1`` for Januarys,
    ``(6, 7, 8)`` for the JJA months. Combined with ``climatology=False`` this
    is the **one point per year** view -- every January of the window on a real
    time axis, which is how a drift in one month shows up. With
    ``climatology=True`` the same subset collapses to one point per calendar
    month, so a single month gives a single point and is rarely what you want.

    ``cache=True`` reads the monthly means :func:`build_flux_cache` wrote, and
    builds them first if they are not there yet, so the first call for a
    checkpoint is slow and every one after it is immediate. Set it to ``False``
    to go straight to scratch, which is the only way to reach a variable
    outside :data:`FLUX_VARS`.

    The two variables do not have to be the same field; :func:`plot_series`
    puts them on separate axes when they are not.
    """
    if cache:
        build_flux_cache(case, years)               # a no-op once it exists
        cam_field = _from_cache(case, "camulator", years, camulator_var)
        cesm_field = _from_cache(case, "cesm", years, cesm_var)
    else:
        cam_field = load_camulator(case, camulator_var, years)
        cesm_field = load_cesm(cesm_var, years)

    # "Annual" is the right word for a map covering every month; on a series it
    # would just clutter a title that never carried a season before.
    tag = "" if months is None else _months_tag(months)

    out = []
    for field, name in ((cam_field, camulator_var), (cesm_field, cesm_var)):
        # Reduce to monthly means *first*, then drop the months not wanted.
        # The other order breaks the uncached path: `months` selects whole
        # calendar months, but the raw samples are daily or 6-hourly, and
        # resampling a January-only series onto a regular monthly axis fills
        # the eleven gaps with NaN instead of closing them.
        series = _pick_months(
            monthly_mean(regional_mean(field, lat_range), climatology=False),
            months,
        )
        if climatology:
            series = monthly_mean(series, climatology=True)
        series.attrs["variable"] = name
        series.attrs["years"] = _years_tag(years)
        series.attrs["months"] = tag
        series.attrs["climatology"] = bool(climatology)
        out.append(series)
    return tuple(out)


def flux_balance(case, down_var, up_var, years=(1980, 1984), lat_range=(60, 90),
                 climatology=True, months=None, cache=True):
    """A radiative pair and its net. Returns ``(cam, cesm)``, two Datasets.

    ``flux_balance(case, "FLDS_J", "FLUS")`` for longwave,
    ``flux_balance(case, "FSDS_J", "FSUS")`` for shortwave. Each side comes back
    holding ``down``, ``up`` and ``net``, where **net is ``down - up``** -- the
    energy the surface keeps.

    Every argument after the two variable names is passed straight to
    :func:`flux_series`, which does the banding, the monthly reduction and the
    climatology, so a balance is on exactly the same footing as the single-
    variable series beside it.

    **Net longwave is not zero and is not meant to be.** It runs near -30 to
    -50 W/m2 over the Arctic, because the surface radiates away more longwave
    than the atmosphere sends back. Net shortwave is positive and large in
    summer and near zero in the polar night. What sums to roughly zero is the
    whole surface budget, which is :func:`flux_budget`.

    Ready for :func:`plot_balance`, which reads ``summary`` off the attributes
    to know which member is the summed line.
    """
    kwargs = dict(years=years, lat_range=lat_range, climatology=climatology,
                  months=months, cache=cache)
    cam_down, cesm_down = flux_series(case, down_var, down_var, **kwargs)
    cam_up, cesm_up = flux_series(case, up_var, up_var, **kwargs)

    out = []
    for down, up in ((cam_down, cam_up), (cesm_down, cesm_up)):
        net = down - up
        net.attrs = dict(down.attrs)
        net.attrs["variable"] = f"{down_var} − {up_var}"
        combined = xr.Dataset({"down": down, "up": up, "net": net})
        combined.attrs.update(down.attrs)
        combined.attrs.update(
            summary="net",
            label=data.case_label(case),
            variable=f"{down_var} − {up_var}",
            lat_range=down.attrs.get("lat_range", ""),
        )
        out.append(combined)
    return tuple(out)


def flux_budget(case, years=(1980, 1984), lat_range=(60, 90), climatology=True,
                months=None, cache=True):
    """The whole surface energy budget. Returns ``(cam, cesm)``, two Datasets.

    Net shortwave, net longwave and the two turbulent fluxes, and the sum of all
    four -- the one combination that is *supposed* to come out near zero.

    Each side holds ``net_sw``, ``net_lw``, ``turbulent`` and ``budget``::

        net_sw    = FSDS_J - FSUS
        net_lw    = FLDS_J - FLUS
        turbulent = SHFLX + LHFLX
        budget    = net_sw + net_lw + turbulent

    The turbulent pair **adds** rather than subtracts because both are already
    on the positive-into-surface convention :func:`_to_watts` stamps on them,
    the same convention the downwelling radiative fluxes carry.

    **Where it closes.** Globally the budget lands within a few W/m2 of zero --
    about +1 for CESM over 1980-1984, and about +6 for the emulator, a gap that
    is itself a result. Over a band it does not close and is not meant to: the
    Arctic surface runs a deficit near -11 W/m2 that is made up by heat
    transported into the region, not by anything in these six fields. A residual
    over ``lat_range=(60, 90)`` is physics; a large residual over the globe
    would be a bug.

    Ready for :func:`plot_balance`.
    """
    kwargs = dict(years=years, lat_range=lat_range, climatology=climatology,
                  months=months, cache=cache)
    cam_sw, cesm_sw = flux_balance(case, "FSDS_J", "FSUS", **kwargs)
    cam_lw, cesm_lw = flux_balance(case, "FLDS_J", "FLUS", **kwargs)
    cam_sh, cesm_sh = flux_series(case, "SHFLX", "SHFLX", **kwargs)
    cam_lh, cesm_lh = flux_series(case, "LHFLX", "LHFLX", **kwargs)

    out = []
    for shortwave, longwave, sensible, latent in (
            (cam_sw, cam_lw, cam_sh, cam_lh),
            (cesm_sw, cesm_lw, cesm_sh, cesm_lh)):
        turbulent = sensible + latent
        turbulent.attrs = dict(sensible.attrs)
        turbulent.attrs["variable"] = "SHFLX + LHFLX"

        budget = shortwave["net"] + longwave["net"] + turbulent
        budget.attrs = dict(sensible.attrs)
        budget.attrs["variable"] = "surface budget"

        combined = xr.Dataset({
            "net_sw": shortwave["net"],
            "net_lw": longwave["net"],
            "turbulent": turbulent,
            "budget": budget,
        })
        combined.attrs.update(sensible.attrs)
        combined.attrs.update(
            summary="budget",
            label=data.case_label(case),
            variable="surface budget",
            lat_range=sensible.attrs.get("lat_range", ""),
        )
        out.append(combined)
    return tuple(out)


def flux_daily(case, camulator_var, cesm_var, when="1980-01",
               lat_range=(60, 90)):
    """Day-by-day through one month. Returns ``(cam, cesm)``, 1-D in time.

    The sub-monthly view. :func:`flux_series` gives one point per month, so a
    January is two dots and the weather inside it is gone. This gives 31 points
    a side.

    **Both sides come back as daily means.** CAMulator writes daily already, so
    CESM's four 6-hourly steps are averaged per day to match. Drawing CESM at
    its native cadence against a CAMulator line that is *already* a daily mean
    would make the emulator look the steadier of the two, which is a fact about
    how its output was written and not about the model. Averaged this way the
    two lines land on identical timestamps and are read off one axis.

    Reads the raw output, never the cache, since the cache is monthly means and
    this is the one question they cannot answer. That costs a few seconds for
    the month's daily files rather than being instant.

    ``when`` is ``YYYY-MM`` for a whole month or ``YYYY-MM-DD`` for one day, the
    same stamp :func:`flux_day` takes.

    ``lat_range`` is the band to average over, as in :func:`regional_mean` --
    ``None`` for the globe.
    """
    parts = str(when).split("-")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"`when` is YYYY-MM for a month or YYYY-MM-DD for a day, not {when!r}"
        )
    year = int(parts[0])
    stamp = "-".join(f"{int(part):02d}" for part in parts[1:])
    stamp = f"{year:04d}-{stamp}"

    wanted = [path for path in camulator_files(case, year)
              if f".h1d.{stamp}" in path]
    if not wanted:
        raise FileNotFoundError(
            f"no CAMulator files for {stamp} in {data.case_label(case)}"
        )

    frames, times = [], []
    for path in wanted:
        with xr.open_dataset(path) as opened:
            if camulator_var not in opened:
                raise KeyError(
                    f"{camulator_var!r} not in {os.path.basename(path)}; "
                    f"2-D fields are {', '.join(CAMULATOR_VARS)}"
                )
            frames.append(opened[[camulator_var]].load())
        y, m, d = (int(g) for g in _DAILY.search(path).groups())
        times.append(cftime.DatetimeNoLeap(y, m, d))
    cam_days = _match_grid(
        xr.concat(frames, dim="time").assign_coords(time=("time", times)), year
    )
    cam = regional_mean(_to_watts(cam_days[camulator_var], camulator_var), lat_range)

    available = set(data.truth_years())
    if year not in available:
        raise FileNotFoundError(
            f"no CESM store for {year}; have {min(available)}-{max(available)}"
        )
    store = xr.open_zarr(data.truth_zarr(year), chunks={})
    if cesm_var not in store:
        raise KeyError(
            f"{cesm_var!r} is not in {os.path.basename(data.truth_zarr(year))}"
        )
    # Partial-string selection on a CFTimeIndex: "1980-01" takes the month,
    # "1980-01-05" the four steps of that day.
    cesm_steps = store[cesm_var].sel(time=stamp)
    if cesm_steps.sizes.get("time", 0) == 0:
        raise ValueError(f"CESM has no steps stamped {stamp}")
    # Four 6-hourly steps -> one daily mean, which is what a CAMulator file
    # already is. `.resample` drops attributes, so they go back on before the
    # unit conversion, which reads `units` and `scaling` off them.
    cesm_days = cesm_steps.load().resample(time="D").mean()
    cesm_days.attrs = dict(store[cesm_var].attrs)
    cesm = regional_mean(_to_watts(cesm_days, cesm_var), lat_range)

    out = []
    for series, name in ((cam, camulator_var), (cesm, cesm_var)):
        series.attrs.update(
            variable=name,
            years=stamp,
            months="daily",
            climatology=False,
        )
        out.append(series)
    return tuple(out)


def _from_cache(case, source, years, var):
    """One variable out of a cached side, with a useful error when it is not there."""
    ds = load_flux_cache(case, source, years)
    if var not in ds:
        raise KeyError(
            f"{var!r} is not in the {source} cache, which holds "
            f"{', '.join(ds.data_vars)}. Either add it to `variables` and "
            f"rebuild with force=True, or pass cache=False to read it straight "
            f"off scratch."
        )
    return ds[var]


# -------------------------------------------------------------- plotting ----

_MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _axis_label(series):
    """``FLUS [W m-2]``, or just the name when there are no units."""
    name = series.attrs.get("variable", series.name or "")
    units = series.attrs.get("units")
    return f"{name} [{units}]" if units else name


def _time_axis(series):
    """x values and label: real dates, or 1-12 for a climatology.

    cftime objects are converted to ``datetime.datetime`` on the way out --
    matplotlib cannot plot a ``DatetimeNoLeap`` without ``nc_time_axis``
    installed, and a noleap date is always a valid real date (there is no
    29 February to fall over on), so the conversion is exact and costs nothing.
    """
    if "month" in series.dims:
        return series["month"].values, "month"

    stamps = series["time"].values
    if len(stamps) and isinstance(stamps[0], cftime.datetime):
        stamps = np.array([
            datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in stamps
        ])
    return stamps, "time"


def plot_series(cam, cesm, case=None, fig_dir=None, name=None, title=None,
                figsize=(11, 5)):
    """The two series on one figure. Returns ``(fig, ax)``.

    If the two sides are the *same* variable they share one y-axis, which is
    the comparison you actually want. If they are different variables they get
    separate axes -- CAMulator on the left, CESM on the right -- because their
    units generally differ and one shared scale would flatten one of the lines
    into the x-axis without saying so.

    A series narrowed to certain months carries that in its title and filename,
    so a Januarys-only figure never overwrites the all-months one beside it.

    ``fig_dir`` saves. When ``case`` is given and ``fig_dir`` is not, the
    figure lands in ``output/<case>/fluxes/<CAMulator variable>/`` -- one
    directory per variable, which is how ``output/`` is laid out.
    """
    cam_var = cam.attrs.get("variable", "CAMulator")
    cesm_var = cesm.attrs.get("variable", "CESM")
    same = cam_var == cesm_var

    fig, ax = plt.subplots(figsize=figsize)
    x_cam, x_label = _time_axis(cam)
    x_cesm, _ = _time_axis(cesm)
    is_clim = x_label == "month"

    ax.plot(x_cam, cam.values, color="tab:blue", marker="o", markersize=3,
            label=f"CAMulator {cam_var}")
    ax.set_xlabel(x_label)
    # The CAMulator files carry no units, so when both sides are the same field
    # the shared axis is labelled from the CESM side, which does.
    ax.set_ylabel(_axis_label(cesm if same else cam),
                  color="black" if same else "tab:blue")

    if same:
        ax.plot(x_cesm, cesm.values, color="tab:orange", marker="o",
                markersize=3, label=f"CESM {cesm_var}")
        ax.legend()
    else:
        ax.tick_params(axis="y", labelcolor="tab:blue")
        right = ax.twinx()
        right.plot(x_cesm, cesm.values, color="tab:orange", marker="o",
                   markersize=3, label=f"CESM {cesm_var}")
        right.set_ylabel(_axis_label(cesm), color="tab:orange")
        right.tick_params(axis="y", labelcolor="tab:orange")
        lines = ax.get_lines() + right.get_lines()
        ax.legend(lines, [line.get_label() for line in lines])

    band = cam.attrs.get("lat_range", "")
    years = cam.attrs.get("years", "")
    season = cam.attrs.get("months", "")
    label = case or cam.attrs.get("label", "")
    span = f"{years} climatology" if is_clim else years
    if title is None:
        pieces = [p for p in (label, f"{cam_var} vs {cesm_var}", season, band, span)
                  if p]
        title = "  |  ".join(pieces)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if is_clim:
        # A 12-point seasonal cycle reads as months, not as the integers 1-12.
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(_MONTH_LABELS)
        ax.set_xlabel("")
    else:
        fig.autofmt_xdate()
    fig.tight_layout()

    if fig_dir is None and case is not None:
        fig_dir = data.out_dir(case, os.path.join("fluxes", cam_var))
    if fig_dir is not None:
        if name is None:
            # Empty pieces are dropped, so a series with no month filter keeps
            # the name it has always had.
            name = "_".join(
                piece for piece in (f"{cam_var}_vs_{cesm_var}", season, band, years)
                if piece
            )
            if is_clim:
                name += "_clim"
            name = name.replace(" ", "")
        path = os.path.join(fig_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")

    return fig, ax


def plot_balance(cam, cesm, case=None, fig_dir=None, name=None, title=None,
                 figsize=(11, 7.5)):
    """Components above, their sum below. Returns ``(fig, (top, bottom))``.

    Draws whatever :func:`flux_balance` or :func:`flux_budget` returned. Which
    member is the summed line is read off ``attrs["summary"]``, so one plotter
    covers a two-term radiative pair and a four-term budget without caring which
    it was handed.

    **Colour is the component, line style is the model** -- CAMulator solid,
    CESM dashed. Tracking four or six lines by colour alone does not work, and
    the comparison being made is always one model against the other on the same
    quantity.

    The lower panel carries a zero reference line, because how far the sum sits
    from zero is the entire question. It gets its own axis: net longwave is tens
    of W/m2 while the downwelling and upwelling terms behind it are hundreds,
    and on one shared scale the sum would be flattened onto the axis.
    """
    summary = cam.attrs.get("summary")
    if summary is None or summary not in cam:
        raise ValueError(
            "plot_balance needs the Datasets from flux_balance or flux_budget; "
            "this one carries no `summary` attribute naming the summed member"
        )
    components = [name_ for name_ in cam.data_vars if name_ != summary]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": (2, 1), "hspace": 0.12},
    )
    x, x_label = _time_axis(cam[components[0]])
    is_clim = x_label == "month"

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for colour, member in zip(colours, components):
        for side, style, who in ((cam, "-", "CAMulator"), (cesm, "--", "CESM")):
            label = side[member].attrs.get("variable", member)
            top.plot(x, side[member].values, color=colour, linestyle=style,
                     marker="o", markersize=3, label=f"{who} {label}")

    for side, style, who in ((cam, "-", "CAMulator"), (cesm, "--", "CESM")):
        label = side[summary].attrs.get("variable", summary)
        bottom.plot(x, side[summary].values, color="tab:purple", linestyle=style,
                    marker="o", markersize=3, label=f"{who} {label}")
    bottom.axhline(0, color="black", linewidth=0.9, alpha=0.7)

    units = cam[summary].attrs.get("units", "")
    for ax in (top, bottom):
        ax.set_ylabel(f"[{units}]" if units else "")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    bottom.set_ylabel(
        f"{cam[summary].attrs.get('variable', summary)} [{units}]" if units
        else cam[summary].attrs.get("variable", summary)
    )

    band = cam.attrs.get("lat_range", "")
    years = cam.attrs.get("years", "")
    season = cam.attrs.get("months", "")
    label = case or cam.attrs.get("label", "")
    what = cam.attrs.get("variable", summary)
    span = f"{years} climatology" if is_clim else years
    if title is None:
        pieces = [p for p in (label, what, season, band, span) if p]
        title = "  |  ".join(pieces)
    top.set_title(title)

    if is_clim:
        bottom.set_xticks(range(1, 13))
        bottom.set_xticklabels(_MONTH_LABELS)
        bottom.set_xlabel("")
    else:
        bottom.set_xlabel(x_label)
        fig.autofmt_xdate()
    fig.tight_layout()

    # One directory for every balance figure rather than one per variable: a
    # balance spans two fields, so `fluxes/<var>/` has no right answer.
    if fig_dir is None and case is not None:
        fig_dir = data.out_dir(case, os.path.join("fluxes", "balance"))
    if fig_dir is not None:
        if name is None:
            name = "_".join(
                piece for piece in (what, season, band, years) if piece
            )
            if is_clim:
                name += "_clim"
            name = name.replace(" ", "").replace("−", "-")
        path = os.path.join(fig_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")

    return fig, (top, bottom)


# ------------------------------------------------------------ flux maps ----

def _pick_months(field, months):
    """Restrict a monthly field to certain calendar months. ``None`` keeps all."""
    if months is None:
        return field
    wanted = [months] if isinstance(months, int) else list(months)
    picked = field.sel(time=field["time"].dt.month.isin(wanted))
    if picked.sizes.get("time", 0) == 0:
        raise ValueError(f"no months matching {months} in this window")
    return picked


def _months_tag(months):
    """``annual``, ``Jul``, or ``JJA`` -- for filenames and titles."""
    if months is None:
        return "Annual"
    wanted = [months] if isinstance(months, int) else list(months)
    if len(wanted) == 1:
        return _MONTH_LABELS[wanted[0] - 1]
    return "".join(_MONTH_LABELS[m - 1][0] for m in wanted)


#: ``region`` -> the directory name, matching the ice output's Arctic/Antarctic.
REGION_DIRS = {"north": "Arctic", "south": "Antarctic", "global": "Global"}


def map_dir(case, var, kind, region):
    """``output/<case>/fluxes/<var>/spatial/<kind>/<Arctic|Antarctic|Global>/``.

    The same shape as the ice output's ``ice_plot/12-Month/Arctic`` — season
    above hemisphere — but nested under the variable, since fluxes are one
    directory per variable. ``kind`` is the season: ``12-Month`` for the grid,
    otherwise whatever :func:`_months_tag` produced (``JJA``, ``Jul``,
    ``Annual``). Time series stay at ``fluxes/<var>/``; only maps go under
    ``spatial/``.
    """
    if region not in REGION_DIRS:
        raise ValueError(f"region must be one of {tuple(REGION_DIRS)}, not {region!r}")
    return data.out_dir(
        case, os.path.join("fluxes", var, "spatial", kind, REGION_DIRS[region])
    )


def flux_map(case, camulator_var, cesm_var, years=(1980, 1984), months=None,
             cache=True):
    """Time-mean 2-D field from each model, for maps. Returns ``(cam, cesm)``.

    The map counterpart of :func:`flux_series`: same inputs, but the latitude
    band is *not* collapsed -- each side comes back as a full
    ``(latitude, longitude)`` field averaged over the window.

    ``months`` picks a season out of it: ``None`` for every month, ``7`` for
    Julys, ``(6, 7, 8)`` for JJA, ``(12, 1, 2)`` for DJF.

    This is a mean of monthly means, so every month counts equally regardless
    of its length -- the same convention :func:`monthly_mean` uses for a
    climatology, and the reason a JJA map here matches the JJA points on the
    series plot.
    """
    if cache:
        build_flux_cache(case, years)               # a no-op once it exists
        cam_field = _from_cache(case, "camulator", years, camulator_var)
        cesm_field = _from_cache(case, "cesm", years, cesm_var)
    else:
        cam_field = load_camulator(case, camulator_var, years).resample(time="MS").mean()
        cesm_field = load_cesm(cesm_var, years).resample(time="MS").mean()

    out = []
    for field, name in ((cam_field, camulator_var), (cesm_field, cesm_var)):
        mean = _pick_months(field, months).mean(dim="time")
        mean.attrs = dict(field.attrs)
        mean.attrs.update(
            variable=name,
            years=_years_tag(years),
            months=_months_tag(months),
            label=data.case_label(case),
        )
        out.append(mean)
    return tuple(out)


def flux_day(case, camulator_var, cesm_var, when="1980-01-01"):
    """One single day from each model. Returns ``(cam, cesm)``, both 2-D.

    The initial-conditions view. Everything else in this module goes through
    monthly means, which is the wrong instrument for asking how far apart the
    two models *start*: by the end of January a month of drift is already
    folded in. This reads one day and nothing else.

    Both sides come back as that day's **mean**, which is the only comparison
    that means anything here. A CAMulator daily file is already the mean of its
    four 6-hourly steps, so CESM's four steps for the same date are averaged to
    match. Comparing the emulator's day against a single CESM instant would be
    comparing a daily mean with a snapshot.

    ``when`` is any ``YYYY-MM-DD`` in the run. It defaults to 1980-01-01, the
    first day both models have, which is the initial condition proper.

    The output carries the same attributes :func:`flux_map` sets, so
    :func:`plot_maps` draws it with no special case and the figure lands under
    ``spatial/<date>/``.
    """
    year, month, day = (int(part) for part in str(when).split("-"))
    stamp = f"{year:04d}-{month:02d}-{day:02d}"

    wanted = [p for p in camulator_files(case, year)
              if p.endswith(f".h1d.{stamp}.nc")]
    if not wanted:
        raise FileNotFoundError(
            f"no CAMulator file for {stamp} in {data.case_label(case)}; "
            f"that year runs "
            f"{os.path.basename(camulator_files(case, year)[0])} onwards"
        )

    with xr.open_dataset(wanted[0]) as opened:
        cam_day = _match_grid(opened[[camulator_var]].load(), year)
    cam_field = _to_watts(cam_day[camulator_var], camulator_var)

    available = set(data.truth_years())
    if year not in available:
        raise FileNotFoundError(
            f"no CESM store for {year}; have {min(available)}-{max(available)}"
        )
    store = xr.open_zarr(data.truth_zarr(year), chunks={})
    if cesm_var not in store:
        raise KeyError(
            f"{cesm_var!r} is not in {os.path.basename(data.truth_zarr(year))}"
        )
    # The four steps stamped with this date. Selecting by the date rather than
    # by position means this does not care where in the year the day falls.
    same_day = store[cesm_var].sel(time=str(stamp))
    if same_day.sizes.get("time", 0) == 0:
        raise ValueError(f"CESM has no steps stamped {stamp}")
    # `.mean` drops attributes, so they go back on before the unit conversion,
    # which reads `scaling` and `units` off them.
    cesm_day = same_day.load().mean(dim="time")
    cesm_day.attrs = dict(store[cesm_var].attrs)
    cesm_field = _to_watts(cesm_day, cesm_var)

    out = []
    for field, name in ((cam_field, camulator_var), (cesm_field, cesm_var)):
        # `months` is what picks the output directory and titles the figure;
        # for one day the date is the whole selection, so there is no separate
        # `years` window to carry.
        field.attrs.update(
            variable=name,
            years="",
            months=stamp,
            label=data.case_label(case),
        )
        out.append(field)
    return tuple(out)


def _map_levels(fields, lats, region, edge, count=17, symmetric=False):
    """Contour levels sized to the region actually being drawn.

    Levels off the whole globe would put every Arctic panel in one or two
    colours -- :func:`ice_ens.plot_poles._auto_levels` exists for the same
    reason. ``symmetric`` centres them on zero, which a difference needs and a
    flux does not.
    """
    if region == "north":
        mask = lats >= edge
    elif region == "south":
        mask = lats <= -edge
    else:
        mask = np.ones_like(lats, dtype=bool)

    # `...` so this works on a 2-D (lat, lon) map and a 3-D (month, lat, lon)
    # stack alike -- latitude is the second-to-last axis either way.
    values = np.concatenate([np.ravel(np.asarray(f)[..., mask, :]) for f in fields])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-1, 1, count)

    if symmetric:
        # 99th percentile, not the max: one extreme coastal cell otherwise sets
        # the scale and flattens the pattern everywhere else to white.
        limit = np.percentile(np.abs(values), 99) or 1.0
        return _nice_levels(-limit, limit, count)

    low, high = np.percentile(values, 1), np.percentile(values, 99)
    if low == high:
        low, high = low - 1, high + 1
    return _nice_levels(low, high, count)


def _nice_levels(low, high, count):
    """Levels on a round step, widened outward to round ends.

    Raw percentiles give steps like 81722.4, and a colorbar labelled with those
    at six significant figures overlaps itself into an unreadable smear. Snap
    the step to 1, 2, 2.5 or 5 times a power of ten and push the ends out to
    multiples of it, so every tick is a number that can be read at a glance.
    The band only ever grows, so nothing that was in range falls out of it.
    """
    span = float(high) - float(low)
    if not np.isfinite(span) or span <= 0:
        return np.linspace(-1, 1, count)

    raw = span / max(count - 1, 1)
    power = 10.0 ** np.floor(np.log10(raw))
    step = next(m for m in (1, 2, 2.5, 5, 10) if raw <= m * power) * power

    start = np.floor(low / step) * step
    stop = np.ceil(high / step) * step
    return np.arange(start, stop + step / 2, step)


def _setup_global_ax(ax, resolution=plot_poles.COASTLINE):
    """Coastlines and gridlines for one whole-globe panel."""
    ax.set_global()
    ax.coastlines(resolution=resolution, linewidth=0.6, zorder=4)
    gl = ax.gridlines(linewidth=0.4, color="gray", alpha=0.6, linestyle="--",
                      draw_labels=False, zorder=5)
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-60, 61, 30))
    return gl


def plot_maps(cam, cesm, case=None, region="north", edge=60, cmap="viridis",
              diff_cmap="RdBu_r", levels=None, fig_dir=None, name=None,
              figsize=None, dpi=150):
    """The two fields side by side as maps, plus their difference.

    ``region`` is ``"north"``, ``"south"`` or ``"global"``. The polar panels
    reuse ``plot_poles``' projection, circular clip and coastlines, so a flux
    map lines up with the ice maps drawn beside it.

    The **difference panel only appears when both sides are the same
    variable** -- ``CAMulator - CESM``, on a symmetric scale centred on zero.
    Two different variables get one colorbar each and no difference, since
    subtracting unlike fields is meaningless.

    Colour limits come from the 1st-99th percentile *inside the region drawn*,
    not the globe, so an Arctic panel is not flattened to one colour by the
    tropics. Pass ``levels`` to fix them instead -- worth doing when comparing
    checkpoints by eye.

    ``fig_dir`` saves; with ``case`` and no ``fig_dir`` the figure lands in
    ``output/<case>/fluxes/<CAMulator variable>/`` like :func:`plot_series`.
    Returns ``(fig, axes)``.
    """
    cam_var = cam.attrs.get("variable", "CAMulator")
    cesm_var = cesm.attrs.get("variable", "CESM")
    same = cam_var == cesm_var

    lons = cam["longitude"].values
    lats = cam["latitude"].values
    projection = (ccrs.PlateCarree(central_longitude=180) if region == "global"
                  else plot_poles._projection(region))

    panels = [(cam.values, f"CAMulator\n{cam_var}"), (cesm.values, f"CESM\n{cesm_var}")]
    if same:
        panels.append(((cam - cesm).values, f"CAMulator − CESM\n{cam_var}"))

    if figsize is None:
        figsize = (5.0 * len(panels), 5.6) if region != "global" else (7.0, 3.4 * len(panels))
    rows, cols = (len(panels), 1) if region == "global" else (1, len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi,
                             subplot_kw={"projection": projection})
    axes = np.atleast_1d(axes).ravel()

    # One shared scale for the two model panels when they are the same field --
    # that is the whole point, the panels are then comparable by eye. Different
    # fields get a scale each: FLUS is ~390 W/m2 and SHFLX ~-20, and forcing
    # them onto one set of levels puts the second panel entirely off-scale.
    if levels is None:
        levels = _map_levels([cam.values, cesm.values], lats, region, edge)
    diff_levels = (_map_levels([panels[2][0]], lats, region, edge, symmetric=True)
                   if same else None)

    if same:
        panel_levels = [levels, levels, diff_levels]
        panel_cmaps = [cmap, cmap, diff_cmap]
    else:
        panel_levels = [_map_levels([cam.values], lats, region, edge),
                        _map_levels([cesm.values], lats, region, edge)]
        panel_cmaps = [cmap, cmap]

    meshes = []
    for ax, (values, label), panel_level, panel_cmap in zip(
            axes, panels, panel_levels, panel_cmaps):
        if region == "global":
            _setup_global_ax(ax)
        else:
            plot_poles._setup_polar_ax(ax, region, edge=edge)
        meshes.append(plot_poles._draw(
            ax, lons, lats, values, panel_level, panel_cmap,
        ))
        ax.set_title(label, fontsize=11)

    def _bar(mesh, on, label, bar_levels):
        """One colorbar, labelled at every other level, exponent in the label.

        When the ticks are large enough that they would overlap into an
        unreadable smear, a common factor is pulled out. Matplotlib's own offset
        text would sit at the right-hand end of a horizontal colorbar and
        collide with the centred label, so the factor is divided out by hand and
        written into the label instead, where it cannot collide with anything.

        The six fluxes are ``W m-2`` by the time they reach here, order 10-400,
        so this no longer fires for them; it is still right for a variable read
        with ``cache=False`` that is genuinely large, such as ``PS`` in pascals.
        """
        scale = int(np.floor(np.log10(np.max(np.abs(bar_levels)) or 1)))
        if abs(scale) >= 3:
            label = f"{label}  ×10$^{{{scale}}}$"
        else:
            scale = 0
        bar = fig.colorbar(mesh, ax=on, orientation=orientation, shrink=0.85,
                           pad=0.05, label=label, ticks=bar_levels[::2])
        axis = bar.ax.xaxis if orientation == "horizontal" else bar.ax.yaxis
        axis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v / 10 ** scale:g}")
        )
        return bar

    units = cesm.attrs.get("units", "")
    bar_label = f"{cam_var} [{units}]" if units else cam_var
    orientation = "vertical" if region == "global" else "horizontal"
    if same:
        _bar(meshes[0], axes[:2].tolist(), bar_label, levels)
        _bar(meshes[2], [axes[2]],
             f"difference [{units}]" if units else "difference", diff_levels)
    else:
        for mesh, ax, side, side_levels in zip(meshes, axes, (cam, cesm),
                                               panel_levels):
            side_units = side.attrs.get("units", "")
            side_name = side.attrs.get("variable", "")
            _bar(mesh, [ax],
                 f"{side_name} [{side_units}]" if side_units else side_name,
                 side_levels)

    span = cam.attrs.get("years", "")
    season = cam.attrs.get("months", "")
    label = case or cam.attrs.get("label", "")
    pieces = [p for p in (label, f"{cam_var} vs {cesm_var}", season, span) if p]
    fig.suptitle("  |  ".join(pieces), fontsize=13)

    if fig_dir is None and case is not None:
        fig_dir = map_dir(case, cam_var, season or "Annual", region)
    if fig_dir is not None:
        if name is None:
            # The directory already carries the region and the season, so the
            # filename only has to separate one variable pair and window from
            # another. Empty pieces are dropped rather than left as a dangling
            # underscore -- a single-day map carries its date as the season and
            # has no separate window.
            name = "_".join(
                piece for piece in (f"{cam_var}_vs_{cesm_var}", season, span)
                if piece
            ).replace(" ", "")
        path = os.path.join(fig_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")

    return fig, axes


def flux_map_months(case, camulator_var, cesm_var, years=(1980, 1984),
                    months=range(1, 13), cache=True):
    """One climatological map *per month*. Returns ``(cam, cesm)``, both 3-D.

    Twelve separate climatologies, not a twelve-month mean -- each month is
    averaged over every year in the window on its own, so the seasonal cycle
    survives instead of being averaged away. The same distinction
    ``ensemble.plot_month_grid`` draws for the ice.

    Each side comes back as ``(month, latitude, longitude)``, ready for
    :func:`plot_map_grid`. ``months`` is any subset -- ``[3, 6, 9, 12]`` for
    the four seasons.
    """
    if cache:
        build_flux_cache(case, years)               # a no-op once it exists
        cam_field = _from_cache(case, "camulator", years, camulator_var)
        cesm_field = _from_cache(case, "cesm", years, cesm_var)
    else:
        cam_field = load_camulator(case, camulator_var, years).resample(time="MS").mean()
        cesm_field = load_cesm(cesm_var, years).resample(time="MS").mean()

    wanted = list(months)
    out = []
    for field, name in ((cam_field, camulator_var), (cesm_field, cesm_var)):
        clim = _pick_months(field, wanted).groupby("time.month").mean(dim="time")
        clim = clim.sel(month=[m for m in wanted if m in set(clim.month.values)])
        clim.attrs = dict(field.attrs)
        clim.attrs.update(
            variable=name,
            years=_years_tag(years),
            months="12-Month" if len(wanted) == 12 else _months_tag(wanted),
            label=data.case_label(case),
        )
        out.append(clim)
    return tuple(out)


def plot_map_grid(cam, cesm, case=None, region="north", field="difference",
                  edge=60, cmap="viridis", diff_cmap="RdBu_r", levels=None,
                  fig_dir=None, name=None, title=None, dpi=140):
    """One panel per month, all on one shared colorbar. Returns ``(fig, axes)``.

    Takes the two 3-D fields :func:`flux_map_months` returns. ``field`` picks
    what the panels show:

    ``"difference"``
        ``CAMulator - CESM``, on a symmetric scale centred on zero -- the
        seasonal cycle of the bias, and the reason to draw this at all. Needs
        the same variable on both sides.
    ``"camulator"`` / ``"cesm"``
        that model's own field, for the seasonal cycle of the flux itself.

    **Every panel shares one colorbar**, scaled across all twelve months
    together, so the months are comparable by eye -- the same choice
    ``ensemble.plot_bias_grid`` makes. Per-panel scales would hide exactly the
    seasonal signal this figure exists to show.

    Layout follows the ice grids: three across up to six panels, four across
    beyond, and the panels shrink past six so a whole year is a figure that
    fits a page rather than a 21-inch one.
    """
    cam_var = cam.attrs.get("variable", "CAMulator")
    cesm_var = cesm.attrs.get("variable", "CESM")

    if field == "difference":
        if cam_var != cesm_var:
            raise ValueError(
                f"a difference needs the same variable on both sides, not "
                f"{cam_var} and {cesm_var}; pass field='camulator' or "
                f"field='cesm' to map one of them instead"
            )
        values = (cam - cesm)
        colormap, symmetric = diff_cmap, True
        what = f"CAMulator − CESM  {cam_var}"
    elif field == "camulator":
        values, colormap, symmetric = cam, cmap, False
        what = f"CAMulator {cam_var}"
    elif field == "cesm":
        values, colormap, symmetric = cesm, cmap, False
        what = f"CESM {cesm_var}"
    else:
        raise ValueError(
            f"field must be 'difference', 'camulator' or 'cesm', not {field!r}"
        )

    lons, lats = cam["longitude"].values, cam["latitude"].values
    present = [int(m) for m in values["month"].values]
    projection = (ccrs.PlateCarree(central_longitude=180) if region == "global"
                  else plot_poles._projection(region))

    # One scale across every month, not per panel -- see the docstring.
    if levels is None:
        levels = _map_levels([values.values], lats, region, edge,
                             symmetric=symmetric)

    n = len(present)
    ncols = n if n <= 3 else (3 if n <= 6 else 4)
    nrows = int(np.ceil(n / ncols))
    panel = 5.2 if n <= 6 else 3.4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel * ncols, panel * nrows * 0.98), dpi=dpi,
        subplot_kw={"projection": projection},
        # A circular-clipped polar panel already carries whitespace in its
        # corners; the default spacing on top of that leaves the rows floating
        # apart.
        gridspec_kw={"hspace": 0.12, "wspace": 0.02},
    )
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, month in zip(axes, present):
        if region == "global":
            _setup_global_ax(ax)
        else:
            plot_poles._setup_polar_ax(ax, region, edge=edge)
        mesh = plot_poles._draw(
            ax, lons, lats, values.sel(month=month).values, levels, colormap,
        )
        ax.set_title(_MONTH_LABELS[month - 1], fontsize=11)
    for spare in axes[n:]:
        spare.set_visible(False)

    units = cesm.attrs.get("units", "")
    label = (f"{cam_var} difference" if field == "difference" else cam_var)
    label = f"{label} [{units}]" if units else label
    scale = int(np.floor(np.log10(np.max(np.abs(levels)) or 1)))
    if abs(scale) >= 3:
        label = f"{label}  ×10$^{{{scale}}}$"
    else:
        scale = 0
    bar = fig.colorbar(mesh, ax=axes[:n].tolist(), orientation="vertical",
                       shrink=0.7, pad=0.02, label=label, ticks=levels[::2])
    bar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: f"{v / 10 ** scale:g}")
    )

    span = cam.attrs.get("years", "")
    case_label = case or cam.attrs.get("label", "")
    if title is None:
        pole = REGION_DIRS[region]
        title = (f"{case_label}  |  {what}  |  {pole}  |  "
                 f"{span} mean of each month")
    fig.suptitle(title, fontsize=13)

    if fig_dir is None and case is not None:
        fig_dir = map_dir(case, cam_var, "12-Month", region)
    if fig_dir is not None:
        if name is None:
            name = f"{cam_var}_vs_{cesm_var}_{field}_bymonth_{span}".replace(" ", "")
        path = os.path.join(fig_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")

    return fig, axes
