"""Climatological ice-fraction bias of every CAMulator checkpoint at once.

Seven functions, one per stage of the work. Everything is a plain argument --
no config file, no globals, no paths baked in.

    climatology        load -> pick months+years -> time-mean -> one 2-D field
    regrid_to_gx1v7    carry the reference onto the checkpoints' native grid
    ice_area           hemispheric ice area of one field, on its own grid
    bias_ensemble      drives the above over a dict of checkpoints
    plot_bias_grid     one bias map per checkpoint, one shared colorbar
    plot_area_error    one dot per checkpoint, zero line = reference
    plot_month_grid    one checkpoint's whole seasonal cycle, a panel a month

Sign convention everywhere: bias = checkpoint - reference, so POSITIVE means
the checkpoint has TOO MUCH ice.

The checkpoints are never regridded. They are the field under evaluation and
they all already share gx1v7, so the reference is what moves. Ice area is
always computed on a native grid, never on a regridded one.

Drawing is done with the helpers already in ice_ens.plot_poles, and the gx1v7
cell corners with ice_ens.grids -- neither is duplicated here.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xesmf as xe

from . import grids, plot_poles

#: POP grid fields that ride along on a checkpoint climatology as coordinates,
#: so the returned 2-D field knows its own grid and can be plotted or
#: area-weighted without going back to disk.
GRID_VARS = ("TLAT", "TLONG", "ULAT", "ULONG", "TAREA")


# ----------------------------------------------------------------- stage 1 ----

def climatology(path, months, years, var="IFRAC"):
    """Multi-year mean of ``var`` over the chosen months. Returns ONE 2-D field.

    ``path`` is either

    * a ``.zarr`` store, or a glob matching several of them (the reference is
      one store per calendar year, so a 5-year mean needs the glob form), or
    * a directory of POP monthly history files, or a glob matching them.

    ``months`` is a list of month numbers -- ``[8]`` is August, ``[6, 7, 8]``
    is a JJA mean. ``years`` is a ``(start, end)`` pair, inclusive, ints or
    strings.

    This is the ONE place month and year selection happens. To change what
    gets averaged, change the call, or change this function.

    A trap worth knowing about: POP stamps a monthly file at the *end* of its
    averaging interval, so ``pop.h.1980-01.nc`` -- the January mean -- carries
    a time of 1980-02-01. Selecting on that raw stamp shifts the whole
    seasonal cycle by one month. ``time_bound`` holds the true interval
    (1980-01-01 to 1980-02-01), so its midpoint is used as the time coordinate
    instead. The reference is 6-hourly instantaneous and needs no such fix.

    The returned field carries its grid as coordinates -- ``TLAT``/``TLONG``/
    ``ULAT``/``ULONG``/``TAREA`` for a checkpoint, ``latitude``/``longitude``
    for the reference -- and records what went into it in ``.attrs``:
    ``n_times``, ``years_present``, ``months``, ``window``.
    """
    y0, y1 = int(years[0]), int(years[1])
    months = list(months)
    path = str(path)

    if ".zarr" in path:
        stores = sorted(glob.glob(path)) if "*" in path else [path]
        if not stores:
            raise FileNotFoundError(f"no zarr store matches {path}")
        # Only `var` is pulled out of each store; the rest of the ~50 fields
        # are never touched, which is what keeps 35 years cheap to open.
        ds = xr.concat([xr.open_zarr(s, chunks={})[[var]] for s in stores],
                       dim="time")
    else:
        pattern = path
        if os.path.isdir(path):
            pattern = os.path.join(path, "*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no netCDF files match {pattern}")
        ds = xr.open_mfdataset(
            files, combine="nested", concat_dim="time",
            data_vars="minimal", coords="minimal", compat="override",
            decode_timedelta=False,
        )
        if "time_bound" in ds.variables:
            ds = ds.assign_coords(time=ds["time_bound"].mean("d2"))

    stamps = ds["time"].dt
    keep = stamps.month.isin(months) & (stamps.year >= y0) & (stamps.year <= y1)
    picked = ds[var].sel(time=keep)

    n_times = int(picked.sizes.get("time", 0))
    if n_times == 0:
        have = sorted(set(ds["time"].dt.year.values.tolist()))
        raise ValueError(
            f"no timesteps in {path} with month in {months} and year in "
            f"{y0}-{y1}; the file's years are {have[0]}-{have[-1]}"
        )

    field = picked.mean("time").compute()

    for name in GRID_VARS:                       # checkpoint grid, if present
        if name in ds:
            field = field.assign_coords({name: ds[name].compute()})

    field.attrs = {
        "n_times": n_times,
        "years_present": sorted(set(picked["time"].dt.year.values.tolist())),
        "months": months,
        "window": (y0, y1),
        "variable": var,
        "source": path,
    }
    return field


# ----------------------------------------------------------------- stage 2 ----

def regrid_to_gx1v7(ref_field, target_field, weights_path=None):
    """Conservatively regrid the reference onto the checkpoint's native gx1v7.

    Direction is reference -> gx1v7 and never the other way. Conservative
    (area-preserving) is the right map for a fractional field: the area
    integral survives the trip, which a bilinear map does not guarantee.

    ``target_field`` is any checkpoint climatology -- it is read only for its
    grid, and every checkpoint is on the same gx1v7, so the result is reused
    for all of them. Build it once.

    ``weights_path``, if given, is where the ESMF weight file is written. It
    takes a couple of minutes to build and a second to read back, so point it
    somewhere on /glade/work and it is reused on every later call.

    NOTE the reference is CAM ``ICEFRAC`` -- ice area over the cell's TOTAL
    area -- while POP ``IFRAC`` is ice area over the cell's OCEAN part. This
    function does not reconcile the two, so coastal cells carry a low bias
    from the reference's land dilution. See the caveat in ../README.md.
    """
    source = grids.f09_grid(ref_field)
    target = grids.pop_grid(target_field)

    kwargs = {}
    if weights_path:
        os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
        kwargs = {"filename": weights_path,
                  "reuse_weights": os.path.exists(weights_path)}

    regridder = xe.Regridder(
        source, target, method="conservative",
        # gx1v7's tripole seam has a couple of collapsed cells; skip them
        # rather than refuse to build the whole map.
        ignore_degenerate=True, **kwargs,
    )

    out = regridder(ref_field.fillna(0.0))
    # xESMF names its output dims after the destination grid ("y"/"x"). They
    # must be renamed, or attaching TLAT/TLONG later creates dangling new
    # dimensions and the next multiply broadcasts into a huge array.
    out = out.rename(dict(zip(out.dims[-2:], ("nlat", "nlon"))))

    # POP writes land as fill, so the checkpoint carries the ocean mask.
    out = out.where(target_field.notnull())
    for name in GRID_VARS:
        if name in target_field.coords:
            out = out.assign_coords({name: target_field.coords[name]})

    out.attrs = dict(ref_field.attrs)
    out.attrs["regrid"] = "conservative (xESMF/ESMF), reference -> gx1v7"
    return out


# ----------------------------------------------------------------- stage 3 ----

def ice_area(field, cell_area, lat, hemisphere):
    """Total ice area over one hemisphere, in 10^6 km^2. Returns a scalar.

    ``sum(field * cell_area)`` over the cells whose ``lat`` is in the chosen
    hemisphere. ``cell_area`` must be in m^2 -- POP's ``TAREA`` is in cm^2, so
    pass ``TAREA * 1e-4``.

    Meant to be used on NATIVE grids, one side at a time: the checkpoint with
    gx1v7's ``TAREA``/``TLAT``, the reference with its own cell area and
    ``latitude``. Nothing here is regridded, so neither total inherits a
    regridding error.

    ``hemisphere`` is ``"north"``, ``"south"``, or ``None`` for global.
    """
    if hemisphere == "north":
        mask = lat >= 0
    elif hemisphere == "south":
        mask = lat < 0
    elif hemisphere is None:
        mask = xr.ones_like(lat, dtype=bool)
    else:
        raise ValueError(f"hemisphere must be 'north', 'south' or None, got {hemisphere!r}")

    total = (field * cell_area).where(mask).sum()
    return float(total) * 1.0e-12                      # m^2 -> 10^6 km^2


# ----------------------------------------------------------------- stage 4 ----

def bias_ensemble(checkpoints, reference_path, months, years, hemisphere,
                  ref_var="ICEFRAC", weights_path=None):
    """Climatological bias of every checkpoint against one reference.

    ``checkpoints`` is ``{label: path}``. Each one is averaged over the same
    ``months``/``years`` window; the reference is averaged once over the same
    window, regridded once onto gx1v7, and reused for all of them.

    A checkpoint that does not cover the whole window is SKIPPED, with a
    message -- averaging three Augusts against everyone else's five would put
    a different climate in one panel of the same figure.

    Returns ``(biases, area_errors)``:

    ``biases[label]``
        2-D field, ``checkpoint - reference``, on gx1v7. Positive = the
        checkpoint has too much ice.
    ``area_errors[label]``
        scalar in 10^6 km^2, ``checkpoint_area - reference_area``, each side
        totalled on its OWN native grid. Positive = too much ice.

    The reference's own hemispheric area is left in
    ``biases[label].attrs["reference_area"]`` so the figures can quote it.
    """
    y0, y1 = int(years[0]), int(years[1])
    months = list(months)
    expected = (y1 - y0 + 1) * len(months)

    # -- reference: one climatology, one regrid, reused by every checkpoint --
    reference = climatology(reference_path, months, years, var=ref_var)
    covered = set(reference.attrs["years_present"])
    missing = [y for y in range(y0, y1 + 1) if y not in covered]
    if missing:
        raise ValueError(
            f"the reference at {reference_path} has no data for {missing}. "
            "If it is one store per year, pass a glob so every year is opened."
        )

    ref_cell_area = xr.DataArray(
        grids.cell_area(grids.f09_grid(reference)),
        dims=("latitude", "longitude"),
        coords={"latitude": reference["latitude"],
                "longitude": reference["longitude"]},
    )
    reference_area = ice_area(reference, ref_cell_area,
                              reference["latitude"], hemisphere)
    print(f"reference  {ref_var}  {reference.attrs['n_times']} timesteps, "
          f"{y0}-{y1} months {months}: {hemisphere} ice area "
          f"{reference_area:.3f} x10^6 km2")

    biases, area_errors, on_gx1v7 = {}, {}, None

    for label, path in checkpoints.items():
        checkpoint = climatology(path, months, years, var="IFRAC")
        n = checkpoint.attrs["n_times"]
        if n < expected:
            print(f"  SKIP {label}: only {n} of {expected} timesteps in "
                  f"{y0}-{y1} (has years {checkpoint.attrs['years_present']})")
            continue

        # Built on the first checkpoint and reused: they all share gx1v7.
        if on_gx1v7 is None:
            on_gx1v7 = regrid_to_gx1v7(reference, checkpoint,
                                       weights_path=weights_path)

        bias = checkpoint - on_gx1v7
        for name in GRID_VARS:                 # arithmetic can drop these
            if name in checkpoint.coords:
                bias = bias.assign_coords({name: checkpoint.coords[name]})
        bias.attrs = {
            "label": label, "n_times": n, "months": months, "window": (y0, y1),
            "hemisphere": hemisphere, "reference_area": reference_area,
            "long_name": "checkpoint - reference sea-ice fraction",
        }

        checkpoint_area = ice_area(checkpoint, checkpoint["TAREA"] * 1.0e-4,
                                   checkpoint["TLAT"], hemisphere)
        biases[label] = bias
        area_errors[label] = checkpoint_area - reference_area
        print(f"  {label}: {n} timesteps, {hemisphere} ice area "
              f"{checkpoint_area:.3f}, error {area_errors[label]:+.3f} x10^6 km2")

    if not biases:
        raise ValueError(f"no checkpoint covered {y0}-{y1} months {months}")
    return biases, area_errors


# ----------------------------------------------------------------- stage 5 ----

def plot_bias_grid(biases, hemisphere, fig_dir=None, months=None, years=None,
                   title=None, name=None, vmax=None, edge=60, dpi=140):
    """One bias map per panel, all sharing one diverging colorbar.

    One panel per key of ``biases``, in order. The keys are only labels -- they
    are usually checkpoints, but keying by month instead puts several months of
    one checkpoint on one shared scale. See the notebook.

    The scale is symmetric about zero (RdBu_r), so red is always too much ice
    and blue always too little, and the panels are comparable by eye. ``vmax``
    fixes the ends of that scale; left as None it is taken from the 99th
    percentile across ALL panels and rounded up, so one checkpoint's outlier
    cannot flatten everybody else.

    ``title`` replaces the auto-generated caption verbatim, and ``name``
    replaces the ``m<months>_<window>`` part of the filename. Both matter when
    the panels are NOT what the auto-title assumes: three separate months would
    otherwise be captioned as a seasonal mean and would overwrite the figure
    that really is one.

    Saved into ``fig_dir`` if given. The figure is returned either way.
    """
    labels = list(biases)
    fields = {label: biases[label].values for label in labels}

    if vmax is None:
        pooled = np.concatenate([v[np.isfinite(v)].ravel() for v in fields.values()])
        vmax = float(np.ceil(np.percentile(np.abs(pooled), 99) * 20) / 20) or 0.05
    levels = np.linspace(-vmax, vmax, 21)

    # Three across up to six panels, four across beyond that -- so a whole year
    # is 4x3 rather than 3x4, which is the shape that fits a page and a screen.
    # Panels also shrink past six: twelve months at the one-checkpoint panel
    # size is a 21x22 inch figure, which is the "blown out" look.
    n = len(labels)
    ncols = n if n <= 3 else (3 if n <= 6 else 4)
    nrows = int(np.ceil(n / ncols))
    panel = 5.2 if n <= 6 else 3.4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel * ncols, panel * 1.08 * nrows),
        subplot_kw={"projection": plot_poles._projection(hemisphere)},
        squeeze=False,
    )
    # The circular boundary fills its axes edge to edge, so a row's titles land
    # on the previous row's maps unless the rows are pushed apart.
    if nrows > 1:
        fig.subplots_adjust(wspace=0.05, hspace=0.22, top=0.93)
    else:
        fig.subplots_adjust(wspace=0.05)

    # Every checkpoint is on the same gx1v7, so the plotting coordinates are
    # built once for the whole figure. `_coords` wants a Dataset -- `in` on a
    # DataArray searches its VALUES, not its coordinate names, so a bare field
    # would silently fall through to the regular-lat/lon branch.
    lons, lats, curvilinear = plot_poles._coords(
        biases[labels[0]].to_dataset(name="bias")
    )

    mesh = None
    for ax, label in zip(axes.flat, labels):
        plot_poles._setup_polar_ax(ax, hemisphere, edge)
        mesh = plot_poles._draw(ax, lons, lats, fields[label], levels,
                                plot_poles.DIFF_CMAP, curvilinear)
        ax.set_title(label, fontsize=13 if n <= 6 else 11)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)

    # Five ticks, not one per level: a one-panel figure's colorbar is narrow
    # and 21 labels run into each other.
    ticks = np.linspace(-vmax, vmax, 5)
    bar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal",
                       pad=0.03 if nrows > 1 else 0.04,
                       shrink=0.9 if ncols == 1 else 0.5, aspect=40,
                       ticks=ticks,
                       label="sea-ice fraction bias (checkpoint − reference)")
    bar.ax.set_xticklabels([f"{t:+.2f}" for t in ticks])

    any_bias = biases[labels[0]]
    months = months or any_bias.attrs.get("months")
    years = years or any_bias.attrs.get("window")
    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    # Above the axes, not on top of them: y<1 lands the suptitle on the first
    # row's panel titles, and bbox_inches="tight" keeps it in the saved file.
    fig.suptitle(title or
                 f"{pole} sea-ice fraction bias — months {_months_text(months)}, "
                 f"{years[0]}–{years[1]} mean\npositive = too much ice",
                 y=1.02 if nrows == 1 else 0.985,
                 fontsize=15 if n <= 6 else 14)

    if fig_dir:
        path = _save(fig, fig_dir, "bias_grid", months, years, hemisphere, dpi, name)
        print(f"wrote {path}")
    return fig


# ----------------------------------------------------------------- stage 6 ----

def plot_area_error(area_errors, hemisphere, fig_dir=None, months=None,
                    years=None, title=None, name=None, dpi=140):
    """Hemispheric ice-area error per checkpoint, zero line = the reference.

    One dot per key of ``area_errors`` on a stem, in 10^6 km^2. Both sides of
    each number were totalled on their own native grid, so nothing here
    inherits a regridding error.

    ``title`` and ``name`` override the caption and the filename, exactly as in
    :func:`plot_bias_grid`.

    Saved into ``fig_dir`` if given. The figure is returned either way.
    """
    labels = list(area_errors)
    values = np.array([area_errors[label] for label in labels], dtype="float64")
    x = np.arange(len(labels))
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in values]

    # 1.5 inches a label is right for a handful of checkpoints and absurd for
    # twelve months -- it makes a 21-inch-wide strip. Past eight labels the
    # spacing tightens and the names tilt instead.
    n = len(labels)
    per_label = 1.5 if n <= 8 else 0.85
    fig, ax = plt.subplots(figsize=(per_label * n + 3.5, 5.0 if n <= 8 else 5.6))
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=2)
    ax.vlines(x, 0.0, values, color=colors, linewidth=2.5, zorder=3)
    ax.scatter(x, values, s=130, c=colors, zorder=4, edgecolor="white")

    for xi, v in zip(x, values):
        ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 13 if v >= 0 else -20), ha="center", fontsize=10)

    span = float(np.abs(values).max()) or 1.0
    ax.set_ylim(-1.45 * span, 1.45 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if n <= 8 else 45,
                       ha="center" if n <= 8 else "right")
    ax.set_ylabel("ice-area error (10$^6$ km$^2$)")
    ax.text(0.005, 0.02, "0 = reference", transform=ax.transAxes,
            fontsize=9, color="dimgray")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    if title is None:
        title = f"{pole} sea-ice area error — months {_months_text(months)}"
        if years:
            title += f", {years[0]}–{years[1]} mean"
        title += "\npositive = too much ice"
    ax.set_title(title, fontsize=13)
    fig.tight_layout()

    if fig_dir:
        path = _save(fig, fig_dir, "area_error", months, years, hemisphere, dpi, name)
        print(f"wrote {path}")
    return fig


# ----------------------------------------------------------------- stage 7 ----

def plot_month_grid(checkpoint_path, reference_path, years, hemisphere,
                    label="checkpoint", months=range(1, 13), ref_var="ICEFRAC",
                    weights_path=None, fig_dir=None, title=None, name=None,
                    vmax=None, edge=60, dpi=140):
    """The whole seasonal cycle of ONE checkpoint: one bias map per month.

    Twelve separate climatologies, not a twelve-month mean -- each month gets
    its own multi-year average on both sides, so the seasonal cycle of the bias
    is visible instead of being averaged away. Every panel shares one colorbar,
    scaled from all twelve together, so the months are comparable by eye.

    This is one checkpoint at a time on purpose. Two checkpoints would be 24
    panels; run it once per ``CASE`` instead and compare the figures.

    ``months`` is any subset -- ``[3, 6, 9, 12]`` for the four seasons. A month
    the checkpoint cannot cover over ``years`` is skipped with a message rather
    than being averaged over fewer years, so a short case still plots the
    months it has.

    Costs about ten seconds per month -- roughly two minutes for a whole year.
    The reference has to be climatologied once per month and there is no way
    around that, but the checkpoint is read through a glob pinned to the month,
    so each pass opens eleven history files rather than all 125.

    Returns ``(fig, biases, area_errors)`` -- the bias figure, and the two
    dicts keyed by month name, so either figure can be redrawn by hand with a
    different ``vmax`` without recomputing anything.
    """
    months = list(months)
    biases, area_errors = {}, {}
    for month in months:
        month_name = plot_poles.MONTH_NAMES[month - 1]

        # Hand `climatology` a glob pinned to this month rather than the whole
        # run directory. It opens 11 files instead of 125 to use 10 of them,
        # which is 3s a month instead of 17s -- the single biggest cost in the
        # loop. Same files, same timesteps, same numbers: this changes only
        # which files are OPENED, never which are averaged. (The ESMF weights
        # are already reused via `weights_path`, so there is nothing to win
        # there, and the reference's months live inside its per-year zarr
        # stores, so it cannot be narrowed the same way.)
        month_path = checkpoint_path
        if os.path.isdir(str(checkpoint_path)):
            month_path = os.path.join(
                str(checkpoint_path),
                f"*.pop.h.[0-9][0-9][0-9][0-9]-{month:02d}.nc")

        try:
            one_bias, one_area = bias_ensemble(
                {label: month_path}, reference_path=reference_path,
                months=[month], years=years, hemisphere=hemisphere,
                ref_var=ref_var, weights_path=weights_path,
            )
        except ValueError:  # raised when the checkpoint does not cover `years`
            print(f"  skipping {month_name}: {label} does not cover "
                  f"{years[0]}-{years[1]}")
            continue
        biases[month_name] = one_bias[label]
        area_errors[month_name] = one_area[label]

    if not biases:
        raise ValueError(f"{label} covered none of months {months} "
                         f"over {years[0]}-{years[1]}")

    # The auto-caption and auto-filename in stages 5 and 6 both assume the
    # panels are one averaging window, which these are not -- so both are
    # written here, and both say "each month" to keep this figure from being
    # read as, or from overwriting, a genuine multi-month mean.
    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    window = f"{years[0]}–{years[1]}"
    stem = name or f"{label}_bymonth_{years[0]}-{years[1]}"
    fig = plot_bias_grid(
        biases, hemisphere, fig_dir=fig_dir, name=stem, vmax=vmax, edge=edge,
        dpi=dpi,
        title=title or (f"{pole} sea-ice fraction bias — {label}, {window} mean "
                        f"of each month\npositive = too much ice"),
    )
    plot_area_error(
        area_errors, hemisphere, fig_dir=fig_dir, name=stem, dpi=dpi,
        title=(f"{pole} sea-ice area error — {label}, {window} mean of each "
               f"month\npositive = too much ice"),
    )
    return fig, biases, area_errors


def _months_text(months):
    return "-".join(f"{m:02d}" for m in months) if months else "?"


def _save(fig, fig_dir, kind, months, years, hemisphere, dpi, name=None):
    os.makedirs(fig_dir, exist_ok=True)
    if name is None:
        window = f"{years[0]}-{years[1]}" if years else "allyears"
        name = f"m{_months_text(months)}_{window}"
    path = os.path.join(fig_dir, f"{kind}_{name}_{hemisphere}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
