"""CICE ``aice`` next to POP ``IFRAC``, for one checkpoint, on the native gx1v7 grid.

  CICE  aice    ice area, aggregate over categories        fraction 0-1
  POP   IFRAC   ice fraction the coupler handed the ocean  fraction 0-1

Both sit at T-cell centres on the same gx1v7 grid, so they subtract cell for
cell -- no regridding anywhere. IFRAC is aice after a pass through the coupler,
so the two should agree closely; the difference map shows where they do not.

Workflow, like ``stress``: :func:`build_icefrac_cache` reads the CICE and the
1.6 GB POP files once and keeps the two fields (full grid, both hemispheres)
per year; everything after that reads the cache.
"""

import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import data, plot_poles

#: Same scratch tree as the stress and flux caches. Derived data -- rebuildable, purgeable.
CACHE_ROOT = "/glade/derecho/scratch/tvoeller/honors_thesis/icefrac_cache"

#: TAREA is cm2; this turns it into 10^6 km2.
CM2_TO_MKM2 = 1e-10 / 1e6

#: Shading for aice - IFRAC. The two are close, so the scale is tight.
SMALL_DIFF_LEVELS = np.arange(-0.05, 0.0501, 0.005)


# ------------------------------------------------------------------ cache ----

def cache_path(case, year):
    return os.path.join(CACHE_ROOT, data.case_label(case), f"icefrac_monthly_{year}.nc")


def build_icefrac_cache(case, years, force=False):
    """One netCDF per year: aice (CICE), IFRAC (POP), and the gx1v7 grid.

    Months are labelled from the filename, never the time stamp -- both POP and
    CICE stamp a monthly mean at the *end* of its interval (see data._month_of).
    A year with any month missing on disk is skipped and reported.
    """
    run = os.path.join(data.CASE_ROOT, data.case_name(case), "run")
    for year in sorted(data._expand_years(years)):
        out = cache_path(case, year)
        if os.path.exists(out) and not force:
            continue
        stem = os.path.join(run, f"{data.case_name(case)}.{{}}.h.{year}-{{:02d}}.nc")
        if not all(os.path.exists(stem.format(m, mo)) for m in ("pop", "cice") for mo in range(1, 13)):
            print(f"{year}: not all 12 months on disk, skipped")
            continue
        months = []
        for month in range(1, 13):
            with xr.open_dataset(stem.format("pop", month), decode_times=False,
                                 decode_timedelta=False) as pop, \
                 xr.open_dataset(stem.format("cice", month), decode_times=False,
                                 decode_timedelta=False) as ice:
                months.append(xr.Dataset({
                    "aice": (("nlat", "nlon"), ice["aice"].isel(time=0).values),
                    "IFRAC": (("nlat", "nlon"), pop["IFRAC"].isel(time=0).values),
                }))
                if month == 1:
                    grid = pop[list(data.GRID_VARS) + ["KMT"]].load()

        ds = xr.concat(months, dim="time").assign_coords(
            data._month_coords([(year, m) for m in range(1, 13)]))
        ds = ds.merge(grid.drop_vars(["time"], errors="ignore"))
        ds["aice"].attrs = {"units": "1", "long_name": "CICE aice, ice area (aggregate)"}
        ds["IFRAC"].attrs = {"units": "1", "long_name": "POP IFRAC, ice fraction from coupler"}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ds.to_netcdf(out, encoding={v: {"dtype": "float32"} for v in ("aice", "IFRAC")})
        print(f"wrote {out}")


def load_icefrac(case, years):
    """The cached years, concatenated, with ``diff = aice - IFRAC`` added."""
    files = [cache_path(case, y) for y in sorted(data._expand_years(years))]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} years not cached -- run "
                                f"build_icefrac_cache({case!r}, {years!r})")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                           data_vars="minimal", coords="minimal",
                           compat="override").load()
    ds["diff"] = ds["aice"] - ds["IFRAC"]
    ds.attrs["case"] = data.case_label(case)
    return ds


# ---------------------------------------------------------------- numbers ----

def ice_area_series(ds, hemisphere="north"):
    """Monthly hemispheric ice area of aice and IFRAC, 10^6 km2, on the native TAREA."""
    hemi = ds["TLAT"] > 0 if hemisphere == "north" else ds["TLAT"] < 0
    w = ds["TAREA"].where(hemi & (ds["KMT"] > 0), 0) * CM2_TO_MKM2
    return xr.Dataset({v: (ds[v].fillna(0) * w).sum(("nlat", "nlon")) for v in ("aice", "IFRAC")})


# ------------------------------------------------------------------ plots ----

def _season(months):
    return plot_poles.MONTH_NAMES[months[0] - 1][:3] if len(months) == 1 else \
        "".join(plot_poles.MONTH_NAMES[m - 1][0] for m in months)


def plot_icefrac_compare(ds, months, hemisphere="north", edge=60, fig_dir=None, name=None):
    """1 x 3: aice, IFRAC, aice - IFRAC, averaged over ``months`` of every year in ``ds``."""
    mean = ds.sel(time=ds["month"].isin(list(months))).mean("time")
    lons, lats, curv = plot_poles._coords(mean)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             subplot_kw={"projection": plot_poles._projection(hemisphere)})
    panels = [("aice", "CICE aice", plot_poles.ICE_LEVELS, plot_poles.ICE_CMAP),
              ("IFRAC", "POP IFRAC", plot_poles.ICE_LEVELS, plot_poles.ICE_CMAP),
              ("diff", "aice − IFRAC", SMALL_DIFF_LEVELS, plot_poles.DIFF_CMAP)]
    meshes = []
    for ax, (v, title, levels, cmap) in zip(axes, panels):
        plot_poles._setup_polar_ax(ax, hemisphere, edge=edge)
        meshes.append(plot_poles._draw(ax, lons, lats, mean[v].values, levels, cmap, curv))
        ax.set_title(title)
    fig.colorbar(meshes[0], ax=axes[:2], orientation="horizontal", fraction=0.05, pad=0.06,
                 label="sea-ice fraction")
    fig.colorbar(meshes[2], ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.06,
                 label="aice − IFRAC")

    years = f"{int(ds.year.min())}-{int(ds.year.max())}"
    season = _season(months)
    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    fig.suptitle(f"{ds.attrs.get('case', '')} {pole} sea-ice fraction, CICE aice vs POP IFRAC — "
                 f"{season} mean {years}", fontsize=14)
    if fig_dir:
        name = name or f"aice_vs_ifrac_{pole.lower()}_{season}_{years}.png"
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    return fig


def plot_aice_months(ds, hemisphere="north", var="aice", edge=60, fig_dir=None, name=None):
    """3 x 4 grid, one panel per calendar month, ``var`` averaged over every year in ``ds``."""
    clim = ds[var].groupby("month").mean("time")
    lons, lats, curv = plot_poles._coords(ds)
    levels, cmap = ((SMALL_DIFF_LEVELS, plot_poles.DIFF_CMAP) if var == "diff"
                    else (plot_poles.ICE_LEVELS, plot_poles.ICE_CMAP))

    fig, axes = plt.subplots(3, 4, figsize=(16, 13),
                             subplot_kw={"projection": plot_poles._projection(hemisphere)})
    for ax, m in zip(axes.flat, range(1, 13)):
        plot_poles._setup_polar_ax(ax, hemisphere, edge=edge)
        mesh = plot_poles._draw(ax, lons, lats, clim.sel(month=m).values, levels, cmap, curv)
        ax.set_title(plot_poles.MONTH_NAMES[m - 1])
    fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.03, pad=0.04,
                 label={"aice": "CICE aice", "IFRAC": "POP IFRAC", "diff": "aice − IFRAC"}[var])

    years = f"{int(ds.year.min())}-{int(ds.year.max())}"
    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    fig.suptitle(f"{ds.attrs.get('case', '')} {pole} {var} — monthly climatology {years}",
                 fontsize=15)
    if fig_dir:
        name = name or f"{var}_{pole.lower()}_12month_{years}.png"
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    return fig


def plot_area_series(ds, hemisphere="north", months=None, fig_dir=None, name=None):
    """Hemispheric ice area from aice and IFRAC through time, and their difference.

    ``months=None`` plots every month; ``months=[9]`` plots one point per year.
    """
    area = ice_area_series(ds, hemisphere)
    if months is not None:
        area = area.sel(time=area["month"].isin(list(months))).groupby("year").mean("time")
        x = area["year"].values
    else:
        x = area["year"].values + (area["month"].values - 0.5) / 12

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, area["aice"], "-", lw=1.2, label="CICE aice")
    ax.plot(x, area["IFRAC"], "--", lw=1.2, label="POP IFRAC")
    ax.set_ylabel("ice area (10$^6$ km$^2$)")
    ax.legend()
    ax2.plot(x, area["aice"] - area["IFRAC"], color="k", lw=1)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("aice − IFRAC")
    ax2.set_xlabel("year")

    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    when = "monthly" if months is None else _season(months)
    years = f"{int(ds.year.min())}-{int(ds.year.max())}"
    fig.suptitle(f"{ds.attrs.get('case', '')} {pole} sea-ice area, aice vs IFRAC — {when} {years}")
    fig.tight_layout()
    if fig_dir:
        name = name or f"area_aice_vs_ifrac_{pole.lower()}_{when}_{years}.png"
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    return fig
