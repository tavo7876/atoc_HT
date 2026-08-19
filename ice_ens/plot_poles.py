"""Polar maps of sea-ice fraction.

The four plot_arctic_* / plot_antarctic_* functions draw one dataset. The two
*_compare functions at the bottom put a checkpoint next to the ground truth,
which is the point of the exercise -- both fields have to be on the same grid
first, so run the *reference* through ice_ens.regrid.to_model_grid before
calling them. The checkpoint itself is drawn on its native gx1v7 grid and is
never interpolated.

Two grid layouts are therefore drawn. A regular lat/lon field is contoured, as
before. A curvilinear gx1v7 field is drawn with pcolormesh instead: contouring
a tripole grid smears the seam, and pcolormesh shows the cells as they actually
are, which is the honest picture when the point is to judge the model's own
field.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from matplotlib.colors import BoundaryNorm

from . import data, grids


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ----------------------------------------------------------- knobs to turn ----

#: Ice fraction is bounded 0-1, so it always gets the same scale -- panels from
#: different checkpoints are then comparable by eye without checking colorbars.
#: Contour count is the main cost of drawing these figures: the time scales
#: roughly linearly with it, so halving the step doubles the wait. 0.1 gives ten
#: readable bands; drop to 0.05 for a smoother final figure.
ICE_LEVELS = np.arange(0.0, 1.001, 0.1)
ICE_CMAP = "Blues_r"

#: Differences are signed, so a symmetric scale centred on zero.
DIFF_LEVELS = np.arange(-0.5, 0.501, 0.05)
DIFF_CMAP = "RdBu_r"

#: Coastline detail. Measured on a 12-panel grid, 50m costs about 4 s more than
#: 110m -- not the bottleneck, so the panels may as well have the better ones.
COASTLINE = "50m"


# --------------------------------------------------------------- internals ----

def _setup_polar_ax(ax, hemisphere, edge=60, resolution=COASTLINE):
    """Land, coastlines, circular clip and gridlines for one polar panel."""
    extent = [-180, 180, edge, 90] if hemisphere == "north" else [-180, 180, -90, -edge]
    ax.set_extent(extent, ccrs.PlateCarree())
    # No OCEAN feature: the filled contours cover the whole domain, so an ocean
    # polygon underneath is never visible and costs several seconds per figure.
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)
    ax.coastlines(resolution=resolution, zorder=4)

    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + [0.5, 0.5]), transform=ax.transAxes)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=.6, color="gray", alpha=1, linestyle="--", zorder=5,
    )
    # Parallels have to follow the hemisphere, or an Arctic panel draws none.
    if hemisphere == "north":
        gl.ylocator = plt.FixedLocator(np.arange(50, 91, 10))
    else:
        gl.ylocator = plt.FixedLocator(np.arange(-90, -49, 10))
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 45))
    return gl


def _coords(ds):
    """Plotting coordinates for a dataset, whichever grid it is on.

    Returns ``(lons, lats, curvilinear)``. For gx1v7 the longitudes come from
    :func:`ice_ens.grids.pop_grid`, which has already snapped every row onto one
    branch -- raw ``TLONG`` jumps by 360 mid-row and pcolormesh would draw one
    cell straight across the globe at the seam.

    Built once per figure and passed down; rebuilding it inside a 12-panel loop
    is pure waste.
    """
    if "TLAT" in ds and "ULAT" in ds:
        grid = grids.pop_grid(ds)
        return grid["lon"].values, grid["lat"].values, True
    return ds["longitude"].values, ds["latitude"].values, False


def _draw(ax, lons, lats, values, levels, colormap, curvilinear=False):
    """One field onto one polar panel, by whichever route its grid needs."""
    if curvilinear:
        # shading="nearest" treats the given lat/lon as cell centres, so the
        # tripole seam closes on its own and nothing has to be made cyclic.
        return ax.pcolormesh(
            lons, lats, values,
            norm=BoundaryNorm(levels, ncolors=256, extend="both"),
            cmap=colormap, transform=ccrs.PlateCarree(),
            shading="nearest", zorder=2,
        )

    # Filled contours with the 0/360 seam closed so no gap shows there.
    cyclic, lons_cyclic = cutil.add_cyclic_point(values, coord=lons)
    return ax.contourf(
        lons_cyclic, lats, cyclic,
        levels=levels, transform=ccrs.PlateCarree(),
        cmap=colormap, extend="both", zorder=2,
    )


def _projection(hemisphere):
    return ccrs.NorthPolarStereo() if hemisphere == "north" else ccrs.SouthPolarStereo()


def _auto_levels(values, lats, hemisphere, edge=60, step=5):
    """Contour levels sized to the polar subset only, not the whole globe."""
    mask = lats >= edge if hemisphere == "north" else lats <= -edge
    polar = np.concatenate([np.ravel(v[mask]) for v in values])
    polar = polar[np.isfinite(polar)]
    if polar.size == 0:
        return np.arange(-50, 6, step)
    vmin = np.floor(polar.min() / step) * step
    vmax = np.ceil(polar.max() / step) * step
    return np.arange(vmin, max(vmax, vmin + step) + step, step)


# ------------------------------------------------------- single-dataset maps ----

def plot_arctic_grid(ds, year=1, var="TREFHT", colormap="YlOrRd",
                     levels=None, edge=60, title=None):
    """12-panel Jan-Dec grid of mean Arctic values. Months with no data
    in `ds` are left blank -- add more months later and they'll populate
    automatically.

    Pass `levels` for a bounded field such as ice fraction; left as None they
    are chosen automatically in steps of 5, which suits temperature.
    """
    return _grid(ds, "north", year, var, colormap, levels, edge, title)


def plot_antarctic_grid(ds, year=1, var="TREFHT", colormap="YlOrRd",
                        levels=None, edge=60, title=None):
    """12-panel Jan-Dec grid of mean Antarctic values, as `plot_arctic_grid`."""
    return _grid(ds, "south", year, var, colormap, levels, edge, title)


def plot_arctic_mean(ds, year=1, var="TREFHT", colormap="YlOrRd",
                     levels=None, edge=60, title=None):
    """Mean Arctic value for a given year, styled like the sea-ice panels."""
    return _single(ds, "north", year, var, colormap, levels, edge, title)


def plot_antarctic_mean(ds, year=1, var="TREFHT", colormap="YlOrRd",
                        levels=None, edge=60, title=None):
    """Mean Antarctic value for a given year, as `plot_arctic_mean`."""
    return _single(ds, "south", year, var, colormap, levels, edge, title)


def _grid(ds, hemisphere, year, var, colormap, levels, edge, title):
    lons, lats, curvilinear = _coords(ds)

    # pull each month's mean first so all 12 panels share one color scale
    monthly = {}
    for month in range(1, 13):
        sel = ds[var].sel(
            time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month)
        )
        if sel.sizes.get("time", 0):
            monthly[month] = sel.mean(dim="time").values

    if levels is None:
        levels = _auto_levels(list(monthly.values()), lats, hemisphere, edge)

    fig, axes = plt.subplots(
        3, 4, figsize=(16, 13), subplot_kw={"projection": _projection(hemisphere)},
    )

    mesh = None
    for month, ax in zip(range(1, 13), axes.flat):
        _setup_polar_ax(ax, hemisphere, edge)
        ax.set_title(MONTH_NAMES[month - 1])
        if month not in monthly:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        mesh = _draw(ax, lons, lats, monthly[month], levels, colormap, curvilinear)

    if mesh is not None:
        fig.colorbar(mesh, ax=axes, orientation="horizontal",
                     pad=0.02, shrink=0.6, aspect=40)

    fig.suptitle(title or f"Mean {var} by month — year {year}", y=0.93)
    return fig, axes


def _single(ds, hemisphere, year, var, colormap, levels, edge, title):
    lons, lats, curvilinear = _coords(ds)

    # average every timestep in the requested year
    values = ds[var].sel(time=ds["time"].dt.year == year).mean(dim="time").values
    if levels is None:
        levels = _auto_levels([values], lats, hemisphere, edge)

    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": _projection(hemisphere)}
    )
    _setup_polar_ax(ax, hemisphere, edge)
    mesh = _draw(ax, lons, lats, values, levels, colormap, curvilinear)
    ax.set_title(title or f"Mean {var} — year {year}")
    fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.9)
    return fig, ax


# ------------------------------------------------------- checkpoint vs truth ----

def plot_month_compare(model, truth, year, month, hemisphere="north",
                       edge=60, var="IFRAC"):
    """One month, three panels: checkpoint, ground truth, and their difference.

    Both datasets must already be on the same grid -- put the *reference*
    through ``ice_ens.regrid.to_model_grid`` first. The checkpoint stays native,
    so the coordinates are taken from it.
    """
    lons, lats, curvilinear = _coords(model)

    model_values = data.pick_month(model, year, month, var)
    truth_values = data.pick_month(truth, year, month, var)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6), subplot_kw={"projection": _projection(hemisphere)}
    )

    panels = [
        (model_values, ICE_LEVELS, ICE_CMAP, model.attrs.get("label", "checkpoint")),
        (truth_values, ICE_LEVELS, ICE_CMAP, "ground truth"),
        (model_values - truth_values, DIFF_LEVELS, DIFF_CMAP, "difference"),
    ]
    meshes = []
    for ax, (values, levels, cmap, label) in zip(axes, panels):
        _setup_polar_ax(ax, hemisphere, edge)
        meshes.append(_draw(ax, lons, lats, values, levels, cmap, curvilinear))
        ax.set_title(label)

    fig.colorbar(meshes[1], ax=axes[:2], orientation="horizontal",
                 pad=0.05, shrink=0.6, label="sea-ice fraction")
    fig.colorbar(meshes[2], ax=axes[2], orientation="horizontal",
                 pad=0.05, shrink=0.9, label="checkpoint − truth")

    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    fig.suptitle(f"{pole} sea-ice fraction — {MONTH_NAMES[month - 1]} {year}", y=0.97)
    return fig, axes


def plot_year_compare(model, truth, year, hemisphere="north", edge=60,
                      var="IFRAC", show_diff=False):
    """A whole year at a glance: the checkpoint's 12 months above the truth's.

    Set ``show_diff`` for a third row of differences. Both datasets must already
    be on the same grid -- the reference having been carried onto gx1v7.
    """
    lons, lats, curvilinear = _coords(model)

    rows = [
        (model, ICE_LEVELS, ICE_CMAP, model.attrs.get("label", "checkpoint")),
        (truth, ICE_LEVELS, ICE_CMAP, "ground truth"),
    ]
    if show_diff:
        rows.append((None, DIFF_LEVELS, DIFF_CMAP, "difference"))

    fig, axes = plt.subplots(
        len(rows), 12, figsize=(26, 2.4 * len(rows)),
        subplot_kw={"projection": _projection(hemisphere)},
    )
    axes = np.atleast_2d(axes)
    # The circular clip leaves each panel mostly empty corners, so the default
    # spacing reads as a big gap between rows.
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    ice_mesh = diff_mesh = None
    for row, (source, levels, cmap, label) in enumerate(rows):
        for month in range(1, 13):
            ax = axes[row, month - 1]
            _setup_polar_ax(ax, hemisphere, edge)

            if source is None:                       # the difference row
                values = (data.pick_month(model, year, month, var)
                          - data.pick_month(truth, year, month, var))
            else:
                values = data.pick_month(source, year, month, var)

            mesh = _draw(ax, lons, lats, values, levels, cmap, curvilinear)
            if source is None:
                diff_mesh = mesh
            else:
                ice_mesh = mesh

            if row == 0:
                ax.set_title(MONTH_NAMES[month - 1][:3], fontsize=11)

        # Row label on the left, since cartopy axes ignore set_ylabel.
        axes[row, 0].text(
            -0.12, 0.5, label, transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12,
        )

    ice_rows = axes[:2].ravel().tolist()
    fig.colorbar(ice_mesh, ax=ice_rows, orientation="vertical",
                 pad=0.01, shrink=0.75, label="sea-ice fraction")
    if diff_mesh is not None:
        fig.colorbar(diff_mesh, ax=axes[2].ravel().tolist(), orientation="vertical",
                     pad=0.01, shrink=0.9, label="checkpoint − truth")

    pole = "Arctic" if hemisphere == "north" else "Antarctic"
    fig.suptitle(
        f"{pole} sea-ice fraction, {model.attrs.get('label', 'checkpoint')} "
        f"vs ground truth — {year}",
        y=1.02, fontsize=15,
    )
    return fig, axes


def save_figures(model, truth, case, year, months=(3, 9), dpi=140, show_diff=False):
    """Write the standard set for one checkpoint into output/<label>/figures/.

    The whole-year overview for each pole, plus a closeup of each month in
    ``months`` -- March and September by default, the usual Arctic maximum and
    minimum. Pass ``months=range(1, 13)`` for all twelve.
    """
    label = data.case_label(case)
    directory = data.out_dir(case, "figures")
    written = []

    for hemisphere in ("north", "south"):
        pole = "arctic" if hemisphere == "north" else "antarctic"

        fig, _ = plot_year_compare(model, truth, year, hemisphere, show_diff=show_diff)
        path = f"{directory}/{pole}_year_{label}_{year}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

        for month in months:
            fig, _ = plot_month_compare(model, truth, year, month, hemisphere)
            path = f"{directory}/{pole}_{year}-{month:02d}_{label}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    return written
