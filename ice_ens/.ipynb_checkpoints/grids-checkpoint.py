"""Grid descriptions with cell corners, so xESMF can regrid conservatively.

Conservative regridding is the right choice for a fractional field like ice
concentration: it preserves the area-integral, so total ice area survives the
trip from the curvilinear POP grid to the regular reference grid. But it needs
cell *corners*, and neither input file ships them in the form xESMF wants, so
both grids get reconstructed here.
"""

import numpy as np
import xarray as xr


def pop_grid(ds):
    """Build an xESMF-ready description of the gx1v7 T-grid from a POP history file.

    POP stores ``ULAT``/``ULONG`` as the *northeast* corner of each T cell, so
    the full (nj+1, ni+1) corner array is that field shifted by one step in each
    direction. Two wrinkles have to be handled:

    * The periodic seam falls mid-row rather than at i=0, so longitude jumps by
      ~360 somewhere along every row. Unwrapping puts each row on one branch --
      necessary because a cell whose corners straddle the seam comes out 360
      degrees wide.
    * In the tripole rows near the north pole the grid lines curve back on
      themselves, so a row does *not* advance a net 360 degrees along i. The
      wrap-around western column therefore cannot just be shifted by a fixed
      -360; it is snapped onto whichever branch sits beside it.

    The southern edge has no U row beneath it and is extrapolated. Rows 0-1 are
    entirely land under Antarctica, so that never touches ocean data.
    """
    tlat = np.asarray(ds["TLAT"].values, dtype="float64")
    tlon = np.asarray(ds["TLONG"].values, dtype="float64")
    ulat = np.asarray(ds["ULAT"].values, dtype="float64")
    ulon = np.asarray(ds["ULONG"].values, dtype="float64")
    nj, ni = tlat.shape

    ulon = np.unwrap(ulon, period=360.0, axis=1)

    lat_b = np.empty((nj + 1, ni + 1), dtype="float64")
    lon_b = np.empty((nj + 1, ni + 1), dtype="float64")

    # [a, b] is the SW corner of T cell (a, b), i.e. the NE corner of T(a-1, b-1).
    lat_b[1:, 1:] = ulat
    lon_b[1:, 1:] = ulon

    # Southern edge: reflect the T-cell centre about the U row above it.
    lat_b[0, 1:] = 2.0 * tlat[0, :] - ulat[0, :]
    lon_b[0, 1:] = ulon[0, :]

    # Western edge wraps to the far end of the row. How many turns that is
    # depends on the row, so snap it onto the branch of the column beside it.
    lat_b[1:, 0] = ulat[:, -1]
    lat_b[0, 0] = lat_b[0, 1]
    lon_b[:, 0] = _snap_to_branch(np.r_[ulon[0, -1], ulon[:, -1]], lon_b[:, 1])

    # Cell centres must sit on the same branch as their own corners, so place
    # them relative to the SW corner rather than unwrapping them separately --
    # an independent unwrap disagrees where the tripole rows oscillate.
    lon = _snap_to_branch(tlon, lon_b[:-1, :-1])

    return xr.Dataset(
        {
            "lat": (("y", "x"), tlat),
            "lon": (("y", "x"), lon),
            "lat_b": (("y_b", "x_b"), lat_b),
            "lon_b": (("y_b", "x_b"), lon_b),
        }
    )


def _snap_to_branch(values, reference):
    """Shift ``values`` by whole turns so they land beside ``reference``."""
    return values + 360.0 * np.round((reference - values) / 360.0)


def f09_grid(ds, lat_name="latitude", lon_name="longitude"):
    """Build an xESMF-ready description of the reference regular lat/lon grid.

    The reference is a CAM finite-volume grid: uniform in longitude, uniform in
    latitude *except* that the first and last rows sit exactly on the poles and
    are therefore half-width polar caps. Taking cell edges as the midpoints
    between centres, clamped to +/-90, reproduces that correctly.
    """
    lat = np.asarray(ds[lat_name].values, dtype="float64")
    lon = np.asarray(ds[lon_name].values, dtype="float64")

    lat_b = np.empty(lat.size + 1, dtype="float64")
    lat_b[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    lat_b[0] = -90.0
    lat_b[-1] = 90.0
    lat_b = np.clip(lat_b, -90.0, 90.0)

    dlon = float(np.diff(lon).mean())
    lon_b = np.empty(lon.size + 1, dtype="float64")
    lon_b[:-1] = lon - 0.5 * dlon
    lon_b[-1] = lon[-1] + 0.5 * dlon

    return xr.Dataset(
        {
            "lat": ("y", lat),
            "lon": ("x", lon),
            "lat_b": ("y_b", lat_b),
            "lon_b": ("x_b", lon_b),
        }
    )


EARTH_RADIUS = 6.37122e6  # metres, the value CESM uses


def cell_area(grid):
    """Area (m2) of every cell in a grid built by :func:`pop_grid` / :func:`f09_grid`.

    Used to sanity-check the reconstructed corners against POP's own ``TAREA``,
    and to turn ice *fraction* into ice *area* on the reference grid.
    """
    lat_b = grid["lat_b"].values
    lon_b = grid["lon_b"].values

    if lat_b.ndim == 1:                                   # regular lat/lon
        dsin = np.diff(np.sin(np.deg2rad(lat_b)))
        dlon = np.deg2rad(np.diff(lon_b))
        return EARTH_RADIUS**2 * np.outer(dsin, dlon)

    # Curvilinear: treat each cell as a spherical quadrilateral and use the
    # spherical-excess formula on the two triangles it splits into.
    corners = [
        (lat_b[:-1, :-1], lon_b[:-1, :-1]),   # SW
        (lat_b[:-1, 1:], lon_b[:-1, 1:]),     # SE
        (lat_b[1:, 1:], lon_b[1:, 1:]),       # NE
        (lat_b[1:, :-1], lon_b[1:, :-1]),     # NW
    ]
    xyz = [_unit_vector(la, lo) for la, lo in corners]
    area = _triangle_area(xyz[0], xyz[1], xyz[2]) + _triangle_area(xyz[0], xyz[2], xyz[3])
    return EARTH_RADIUS**2 * area


def _unit_vector(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _triangle_area(a, b, c):
    """Solid angle of a spherical triangle, via l'Huilier / the vector form."""
    triple = np.abs(np.sum(a * np.cross(b, c), axis=-1))
    denom = (
        1.0
        + np.sum(a * b, axis=-1)
        + np.sum(b * c, axis=-1)
        + np.sum(c * a, axis=-1)
    )
    return 2.0 * np.arctan2(triple, denom)
