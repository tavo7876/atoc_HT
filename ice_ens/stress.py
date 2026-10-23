"""Wind stress over the Arctic and the Barents Sea, from the POP and CICE histories.

Four vector components, two models, one grid:

  POP   TAUX,    TAUY      stress on the ocean surface   dyne/cm2 -> N/m2 (x0.1)
  CICE  strairx, strairy   atmosphere-on-ice stress      N/m2
  CICE  uatm,    vatm      lowest-level atmospheric wind  m/s   (the check)

All four sit on the gx1v7 **U-grid** (cell corners) and point along the **grid**
x/y axes, not east/north. Above ~60N the displaced pole bends those axes far
away from east/north, so every vector is rotated with the U-grid ``ANGLE``
before it is drawn or averaged. ``ANGLET`` is the same thing on the T-grid; it
is only needed for a vector that lives at cell centres, which none of these do.

Two things worth knowing before reading the figures:

* Under ice, POP's ``TAUX``/``TAUY`` is **not** the wind. The coupler hands POP
  ``(1 - ifrac) * atm/ocean stress + ifrac * ice/ocean stress``, so inside the
  pack it is mostly the ice dragging the ocean. Only in open water is it the
  wind. ``strairx``/``strairy`` is the wind's push on the ice itself.
* CICE writes ``strairx``/``strairy`` as a grid-cell mean -- it is ~0 wherever
  there is no ice. Compare its *direction* with TAUX freely, its *magnitude* only
  inside the pack.

* ``uatm``/``vatm`` is defined everywhere, ice or not. It is the monthly mean
  *vector* wind, while stress goes as ``|U| U``, so strong-wind days weigh
  more in the stress mean. Expect the two directions to agree closely, not
  exactly.

Workflow, like ``fluxes``: :func:`build_stress_cache` reads the 1.6 GB POP files
once and keeps the Arctic rows of the five fields needed; everything after that
reads the cache.
"""

import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import data, fluxes, plot_poles

#: Same scratch tree as the flux cache. Derived data -- rebuildable, purgeable.
CACHE_ROOT = "/glade/derecho/scratch/tvoeller/honors_thesis/stress_cache"

#: gx1v7 rows from here up are north of ~38N. Everything Arctic lives above it,
#: and dropping the rest makes the cache ~4x smaller.
J0 = 280

#: Barents Sea box: Norway / Kola coast to Svalbard-Franz Josef Land, Bear
#: Island to Novaya Zemlya. Land inside the box is removed by the ocean mask.
BARENTS = {"lat": (70.0, 80.0), "lon": (15.0, 60.0)}

DYNE_CM2_TO_N_M2 = 0.1

_POP_VARS = ("TAUX", "TAUY")
_CICE_VARS = ("aice", "strairx", "strairy", "uatm", "vatm")
_GRID_VARS = ("ANGLE", "ULAT", "ULON", "TLAT", "TLON", "uarea", "tarea")


# ------------------------------------------------------------------ cache ----

def cache_path(case, year):
    return os.path.join(CACHE_ROOT, data.case_label(case), f"stress_monthly_{year}.nc")


def build_stress_cache(case, years, force=False):
    """One small netCDF per year: TAUX, TAUY (N/m2), aice, strairx, strairy, uatm, vatm.

    Years cached before uatm/vatm were added need ``force=True`` once.

    Months are labelled from the filename, never the time stamp -- both POP and
    CICE stamp a monthly mean at the *end* of its interval (see data._month_of).
    """
    for year in sorted(data._expand_years(years)):
        out = cache_path(case, year)
        if os.path.exists(out) and not force:
            continue
        months = []
        for month in range(1, 13):
            stem = os.path.join(
                data.CASE_ROOT, data.case_name(case), "run",
                f"{data.case_name(case)}.{{}}.h.{year}-{month:02d}.nc",
            )
            with xr.open_dataset(stem.format("pop"), decode_times=False,
                                 decode_timedelta=False) as pop, \
                 xr.open_dataset(stem.format("cice"), decode_times=False,
                                 decode_timedelta=False) as ice:
                fields = {
                    v: (("nj", "ni"),
                        pop[v].isel(time=0, nlat=slice(J0, None)).values
                        * DYNE_CM2_TO_N_M2)
                    for v in _POP_VARS
                }
                fields |= {
                    v: (("nj", "ni"), ice[v].isel(time=0, nj=slice(J0, None)).values)
                    for v in _CICE_VARS
                }
                months.append(xr.Dataset(fields))
                if month == 1:
                    grid = ice[list(_GRID_VARS)].isel(nj=slice(J0, None)).load()
                    grid["KMU"] = (("nj", "ni"), pop["KMU"].isel(nlat=slice(J0, None)).values)
                    grid["KMT"] = (("nj", "ni"), pop["KMT"].isel(nlat=slice(J0, None)).values)

        ds = xr.concat(months, dim="time").assign_coords(
            data._month_coords([(year, m) for m in range(1, 13)])
        )
        ds = ds.merge(grid.drop_vars(["time"], errors="ignore"))
        for v in _POP_VARS:
            ds[v].attrs = {"units": "N/m2", "long_name": f"POP {v}, grid-x/y, x0.1 from dyne/cm2"}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ds.to_netcdf(out, encoding={v: {"dtype": "float32"} for v in _POP_VARS + _CICE_VARS})
        print(f"wrote {out}")


def load_stress(case, years):
    """The cached years, concatenated, with east/north components added."""
    files = [cache_path(case, y) for y in sorted(data._expand_years(years))]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} years not cached -- run "
                                f"build_stress_cache({case!r}, {years!r})")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                           data_vars="minimal", coords="minimal",
                           compat="override").load()
    ds["TAU_E"], ds["TAU_N"] = rotate(ds["TAUX"], ds["TAUY"], ds["ANGLE"])
    ds["STRAIR_E"], ds["STRAIR_N"] = rotate(ds["strairx"], ds["strairy"], ds["ANGLE"])
    if "uatm" not in ds:
        raise KeyError("cache predates uatm/vatm -- run "
                       f"build_stress_cache({case!r}, {years!r}, force=True)")
    ds["WIND_E"], ds["WIND_N"] = rotate(ds["uatm"], ds["vatm"], ds["ANGLE"])
    ds.attrs["case"] = data.case_label(case)
    return ds


def rotate(x, y, angle):
    """Grid-x/y components -> (east, north). ``angle`` in radians, from grid-x to east."""
    cos, sin = np.cos(angle), np.sin(angle)
    return x * cos - y * sin, x * sin + y * cos


# ------------------------------------------- atmosphere side: CESM vs CAMulator ----
#
# CAM's own TAUX/TAUY, from the CESM reference zarrs and the CAMulator daily
# output. Both are on the same regular 192x288 lat/lon grid, so they are already
# east/north -- no ANGLE, no regridding.
#
# CAM writes the stress *on the atmosphere*, so its sign is opposite to the
# wind (both sides correlate ~-0.87 with the lowest-level U). It is flipped
# here, so TAU_E/TAU_N mean the same as everywhere else in this module: the
# push on the surface, pointing with the wind.
#
# Cached per source per year like the flux cache, but in its own files under
# CACHE_ROOT, so the flux cache is never touched. CAMulator is the slow side:
# 365 daily files a year.

ATM_SOURCES = ("cesm", "camulator")
ATM_LAT_MIN = 40.0


def atm_cache_path(case, source, year):
    return os.path.join(CACHE_ROOT, data.case_label(case), f"atm_tau_{source}_{year}.nc")


def build_atm_stress_cache(case, years, force=False):
    """Monthly-mean surface stress from CESM and CAMulator, north of ATM_LAT_MIN."""
    for source in ATM_SOURCES:
        for year in sorted(data._expand_years(years)):
            out = atm_cache_path(case, source, year)
            if os.path.exists(out) and not force:
                continue
            if source == "cesm" and year not in data.truth_years():
                print(f"cesm {year}: no reference store, skipped")
                continue
            raw = (fluxes._cesm_year(("TAUX", "TAUY"), year) if source == "cesm"
                   else fluxes._camulator_year(case, ("TAUX", "TAUY"), year))
            raw = raw.sel(latitude=raw.latitude >= ATM_LAT_MIN)
            monthly = raw.groupby("time.month").mean("time")
            ds = xr.Dataset({"TAU_E": -monthly["TAUX"], "TAU_N": -monthly["TAUY"]})
            ds = ds.rename(month="time").assign_coords(
                data._month_coords([(year, int(m)) for m in monthly.month.values]))
            for v in ("TAU_E", "TAU_N"):
                ds[v].attrs = {"units": "N/m2",
                               "long_name": f"surface stress, -CAM TAU{v[-1].replace('E', 'X').replace('N', 'Y')}"}
            os.makedirs(os.path.dirname(out), exist_ok=True)
            ds.to_netcdf(out)
            print(f"wrote {out}")


def load_atm_stress(case, source, years):
    files = [atm_cache_path(case, source, y) for y in sorted(data._expand_years(years))]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {source} years not cached -- run "
                                f"build_atm_stress_cache({case!r}, {years!r})")
    return xr.open_mfdataset(files, combine="nested", concat_dim="time").load()


#: Shading for the stress magnitude, N/m2.
TAU_LEVELS = np.arange(0.0, 0.201, 0.02)
TAU_CMAP = "YlOrRd"


def plot_atm_stress_compare(case, months, years, zoom=False, box=BARENTS, every=None,
                            scale=None, ref=0.1, fig_dir=None, name=None):
    """1 x 2: CESM (left) and CAMulator (right) surface stress, same months and years.

    Shading is the magnitude of the mean stress vector, arrows its direction,
    Barents box in gold. ``zoom=True`` draws the Barents region only.
    """
    years = tuple(years)
    means = {src: _mean_months(load_atm_stress(case, src, years), months)
             for src in ATM_SOURCES}
    every = every or 2
    scale = scale or (0.5 if zoom else 2.5)

    proj = ccrs.NorthPolarStereo(central_longitude=40 if zoom else 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5 if zoom else 7.5),
                             subplot_kw={"projection": proj})
    for ax, src in zip(axes, ATM_SOURCES):
        m = means[src]
        if zoom:
            _setup_barents_ax(ax, box)
        else:
            plot_poles._setup_polar_ax(ax, "north", edge=60)
        lon2, lat2 = np.meshgrid(m["longitude"].values, m["latitude"].values)
        mesh = plot_poles._draw(ax, lon2, lat2, np.hypot(m["TAU_E"], m["TAU_N"]).values,
                                TAU_LEVELS, TAU_CMAP)
        # Meridians converge on a regular grid, so the longitude stride widens
        # with 1/cos(lat) to keep the arrows roughly evenly spaced on the map.
        jj, ii = np.indices(lon2.shape)
        lon_step = np.maximum(1, np.round(every / np.cos(np.radians(lat2)))).astype(int)
        thin = (jj % every == 0) & (ii % lon_step == 0)
        u, v = m["TAU_E"].values[thin], m["TAU_N"].values[thin]
        lo, la = lon2[thin], lat2[thin]
        keep = (la >= (65 if zoom else 60)) & (la < 89)
        if zoom:
            keep &= (la <= box["lat"][1]) & (la >= box["lat"][0]) & \
                    (lo % 360 >= box["lon"][0]) & (lo % 360 <= box["lon"][1])
        q = ax.quiver(lo[keep], la[keep], u[keep], v[keep], transform=ccrs.PlateCarree(),
                      scale=scale, width=0.003, color="black", zorder=6)
        ax.quiverkey(q, 0.15, 0.95, ref, f"{ref} N/m$^2$", labelpos="E",
                     coordinates="axes", fontproperties={"size": 9})
        blon, blat = _box_outline(box)
        ax.plot(blon, blat, color="gold", lw=2, transform=ccrs.PlateCarree(), zorder=7)
        ax.set_title("CESM (reference)" if src == "cesm"
                     else f"CAMulator coupled ({data.case_label(case)})")

    span = f"{years[0]}-{years[-1]}"
    season = _season(months)
    fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06,
                 label="surface wind stress magnitude (N/m$^2$), arrows = direction (−TAUX, −TAUY)")
    fig.suptitle(f"Surface wind stress, CESM vs CAMulator — {season} mean {span}", fontsize=14)
    if fig_dir:
        name = name or f"atm_tau_cesm_vs_camulator_{'barents' if zoom else 'arctic'}_{season}_{span}.png"
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    return fig


def _season(months):
    return plot_poles.MONTH_NAMES[months[0] - 1][:3] if len(months) == 1 else \
        "".join(plot_poles.MONTH_NAMES[m - 1][0] for m in months)


def _setup_barents_ax(ax, box):
    ax.set_extent([box["lon"][0] - 2, box["lon"][1] + 2,
                   box["lat"][0] - 2, box["lat"][1] + 1], ccrs.PlateCarree())
    ax.add_feature(plot_poles.cfeature.LAND, facecolor="lightgray", zorder=3)
    ax.coastlines(resolution="50m", zorder=4)
    gl = ax.gridlines(draw_labels=["bottom", "left"], x_inline=False, y_inline=False,
                      linestyle="--", color="gray", zorder=5)
    gl.xlocator = plt.FixedLocator(np.arange(0, 91, 15))
    gl.ylocator = plt.FixedLocator(np.arange(65, 86, 5))


# ------------------------------------------------------------------- mask ----

def barents_mask(ds, grid="U", box=BARENTS):
    """True on ocean cells inside the Barents box.

    ``grid="U"`` for the stress vectors, ``"T"`` for aice. Ocean is KMU/KMT > 0.
    """
    lat, lon, kmax = (("ULAT", "ULON", "KMU") if grid == "U" else ("TLAT", "TLON", "KMT"))
    lon360 = ds[lon] % 360
    return ((ds[lat] >= box["lat"][0]) & (ds[lat] <= box["lat"][1])
            & (lon360 >= box["lon"][0]) & (lon360 <= box["lon"][1])
            & (ds[kmax] > 0))


def barents_series(ds, box=BARENTS):
    """Area-weighted Barents means per month: aice, TAU_E/N, STRAIR_E/N, WIND_E/N."""
    mu, mt = barents_mask(ds, "U", box), barents_mask(ds, "T", box)
    out = {}
    for v in _VECTOR_VARS:
        out[v] = ds[v].where(mu).weighted(ds["uarea"].where(mu, 0)).mean(("nj", "ni"))
    out["aice"] = ds["aice"].where(mt).weighted(ds["tarea"].where(mt, 0)).mean(("nj", "ni"))
    return xr.Dataset(out)


def direction(east, north):
    """Compass direction the vector points *toward*, degrees: 0 = N, 90 = E, 180 = S."""
    return (np.degrees(np.arctan2(east, north)) + 360) % 360


def wind_vs_stress(series, months=None):
    """Barents-mean direction of wind, TAU and strair, and how far each stress is off the wind.

    One row per calendar month (mean seasonal cycle), or one row for ``months``
    pooled. Direction is where the vector points *toward* -- note that is the
    opposite of the meteorological "wind from" convention. ``*_minus_wind`` is
    wrapped to -180..180; positive = stress turned clockwise of the wind.
    """
    s = series if months is None else series.sel(time=series["month"].isin(list(months)))
    m = (s.groupby("month").mean("time") if months is None else s.mean("time")).astype("float64")
    out = {"wind_speed_mean_vector": np.hypot(m["WIND_E"], m["WIND_N"])}
    for name in ("WIND", "TAU", "STRAIR"):
        out[f"{name}_dir"] = direction(m[f"{name}_E"], m[f"{name}_N"])
    for name in ("TAU", "STRAIR"):
        out[f"{name}_minus_wind"] = (out[f"{name}_dir"] - out["WIND_dir"] + 180) % 360 - 180
    df = xr.Dataset(out).to_pandas() if months is None else \
        xr.Dataset(out).expand_dims(months=["".join(plot_poles.MONTH_NAMES[k - 1][0]
                                                    for k in months)]).to_pandas()
    return df.round(1)


# ------------------------------------------------------------------ plots ----

#: What each vector panel draws: east, north, title, units, arrow scale
#: (zoomed, whole Arctic) and the reference-arrow length.
PANELS = {
    "TAU":    ("TAU_E", "TAU_N", "POP TAUX/TAUY (surface ocean stress)", "N/m$^2$", 1.0, 2.5, 0.1),
    "STRAIR": ("STRAIR_E", "STRAIR_N", "CICE strairx/y (air-on-ice stress)", "N/m$^2$", 1.0, 2.5, 0.1),
    "WIND":   ("WIND_E", "WIND_N", "CICE uatm/vatm (atmospheric wind)", "m/s", 60.0, 150.0, 5),
}
_VECTOR_VARS = tuple(c for p in PANELS.values() for c in p[:2])

def _mean_months(ds, months):
    return ds.sel(time=ds["month"].isin(list(months))).mean("time")


def _box_outline(box, n=60):
    """The Barents box as a closed lon/lat path, densified so it curves on a polar map."""
    (la0, la1), (lo0, lo1) = box["lat"], box["lon"]
    lons = np.r_[np.linspace(lo0, lo1, n), np.full(n, lo1), np.linspace(lo1, lo0, n), np.full(n, lo0)]
    lats = np.r_[np.full(n, la0), np.linspace(la0, la1, n), np.full(n, la1), np.linspace(la1, la0, n)]
    return lons, lats


def _panel(ax, mean, east, north, title, every, scale, ref, units, box, zoom, shade,
           color="crimson"):
    mesh = plot_poles._draw(ax, *shade)
    sl = (slice(None, None, every), slice(None, None, every))
    u, v = mean[east].values[sl], mean[north].values[sl]
    lon_u, lat_u = mean["ULON"].values[sl], mean["ULAT"].values[sl]
    keep = np.isfinite(u) & np.isfinite(v) & (lat_u >= (65 if zoom else 60))
    # cartopy rotates east/north vectors into the map projection itself.
    q = ax.quiver(lon_u[keep], lat_u[keep], u[keep], v[keep],
                  transform=ccrs.PlateCarree(), scale=scale, width=0.003,
                  color=color, zorder=6)
    ax.quiverkey(q, 0.15, 0.95, ref, f"{ref} {units}", labelpos="E", coordinates="axes",
                 fontproperties={"size": 9})
    blon, blat = _box_outline(box)
    ax.plot(blon, blat, color="gold", lw=2, transform=ccrs.PlateCarree(), zorder=7)
    ax.set_title(title)
    return mesh


def plot_stress_map(ds, months, label="", zoom=False, box=BARENTS, panels=("TAU", "STRAIR"),
                    every=None, scale=None, bias=None, fig_dir=None, name=None):
    """aice shaded, vectors on top: one panel per entry of ``panels`` (keys of PANELS).

    ``bias`` swaps the aice shading for a checkpoint - reference ice-fraction
    bias on gx1v7, as returned by ``ensemble.bias_ensemble`` (red = too much
    ice). The vectors are then averaged over the bias's own year window, so
    both layers describe the same years. Its months must match ``months``.

    Default is POP TAU next to CICE strair; add ``"WIND"`` to see uatm/vatm
    alongside. ``scale`` overrides every panel's arrow scale at once.

    ``months`` is averaged over every year in ``ds``. ``zoom=True`` draws the
    Barents region only and masks the vectors to the box; otherwise the whole
    Arctic with the box outlined.
    """
    if bias is not None:
        if sorted(bias.attrs["months"]) != sorted(months):
            raise ValueError(f"bias is for months {bias.attrs['months']}, not {months}")
        y0, y1 = bias.attrs["window"]
        ds = ds.sel(time=(ds["year"] >= y0) & (ds["year"] <= y1))
    mean = _mean_months(ds, months)
    if bias is None:
        shade = (mean["TLON"].values, mean["TLAT"].values, mean["aice"].values,
                 plot_poles.ICE_LEVELS, plot_poles.ICE_CMAP)
        shade_label = "CICE aice (sea-ice concentration)"
    else:
        b = bias.isel(nlat=slice(J0, None))
        shade = (b["TLONG"].values, b["TLAT"].values, b.values,
                 plot_poles.DIFF_LEVELS, plot_poles.DIFF_CMAP)
        shade_label = (f"{bias.attrs['label']} IFRAC − CESM ICEFRAC "
                       "(sea-ice fraction bias, red = too much ice)")
    if zoom:
        m = barents_mask(mean, "U", box)
        for v in _VECTOR_VARS:
            mean[v] = mean[v].where(m)
    every = every or (2 if zoom else 6)

    proj = ccrs.NorthPolarStereo(central_longitude=40 if zoom else 0)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.5 if zoom else 7.5),
                             subplot_kw={"projection": proj}, squeeze=False)
    axes = axes[0]
    for ax, key in zip(axes, panels):
        e, n, what, units, s_zoom, s_arctic, ref = PANELS[key]
        if zoom:
            _setup_barents_ax(ax, box)
        else:
            plot_poles._setup_polar_ax(ax, "north", edge=60)
        mesh = _panel(ax, mean, e, n, what, every,
                      scale or (s_zoom if zoom else s_arctic), ref, units, box, zoom, shade,
                      color="crimson" if bias is None else "black")

    years = f"{int(ds.year.min())}-{int(ds.year.max())}"
    season = plot_poles.MONTH_NAMES[months[0] - 1][:3] if len(months) == 1 else \
        "".join(plot_poles.MONTH_NAMES[m - 1][0] for m in months)
    fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06,
                 label=shade_label)
    fig.suptitle(f"{label or ds.attrs.get('case', '')}  {' / '.join(panels)} over "
                 f"{'sea ice' if bias is None else 'ice bias vs CESM'} — "
                 f"{season} mean {years}", fontsize=14)
    if fig_dir:
        tag = "stress" if tuple(panels) == ("TAU", "STRAIR") else "_".join(panels).lower()
        tag += "" if bias is None else "_on_bias"
        name = name or f"{tag}_{'barents' if zoom else 'arctic'}_{season}_{years}.png"
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    return fig


def plot_barents_series(series, months=None, label="", fig_dir=None, name=None):
    """Barents-mean east and north stress (both sources), wind (right axis) and aice, per year."""
    s = series if months is None else series.sel(time=series["month"].isin(list(months)))
    annual = s.groupby("year").mean("time")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, comp, word in zip(axes[:2], ("E", "N"), ("eastward", "northward")):
        ax.plot(annual.year, annual[f"TAU_{comp}"], "o-", label="POP TAU")
        ax.plot(annual.year, annual[f"STRAIR_{comp}"], "s-", label="CICE strair")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"{word} stress (N/m$^2$)")
        # Wind on its own axis, zero lined up with the stress zero, so sign
        # agreement reads straight off the plot.
        tw = ax.twinx()
        tw.plot(annual.year, annual[f"WIND_{comp}"], "^--", color="tab:green", label="CICE uatm/vatm")
        tw.set_ylabel(f"{word} wind (m/s)", color="tab:green")
        lim = np.nanmax(np.abs(annual[f"WIND_{comp}"])) * 1.1
        slim = np.nanmax(np.abs(ax.get_ylim()))
        ax.set_ylim(-slim, slim)
        tw.set_ylim(-lim, lim)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    axes[1].text(0.01, 0.04, "negative = southward, toward Norway",
                 transform=axes[1].transAxes, fontsize=9, color="gray")
    axes[2].plot(annual.year, annual["aice"], "o-", color="tab:blue")
    axes[2].set_ylabel("Barents mean aice")
    axes[2].set_xlabel("year")
    when = "annual" if months is None else \
        "".join(plot_poles.MONTH_NAMES[m - 1][0] for m in months)
    fig.suptitle(f"{label} Barents Sea box {BARENTS['lat']}N {BARENTS['lon']}E — {when} means")
    fig.tight_layout()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, name or f"barents_series_{when}.png"),
                    dpi=150, bbox_inches="tight")
    return fig
