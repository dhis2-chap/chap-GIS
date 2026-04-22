"""Target-grid construction."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  registers the .odc accessor
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling


def build_grid(
    aoi: gpd.GeoDataFrame,
    resolution: float,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Build an empty template DataArray over `aoi` at the given resolution.

    Parameters
    ----------
    aoi
        AOI whose total bounds define the extent. Reprojected to `crs` if needed.
    resolution
        Pixel size in the units of `crs` (degrees for geographic, metres for
        projected).
    crs
        Target CRS written onto the DataArray via ``.rio.write_crs``.

    Returns
    -------
    xr.DataArray
        A 2D float32 array of zeros with dims ``(y, x)`` and descending ``y``
        coordinates, carrying the target CRS. Intended as a reference template
        for :meth:`xarray.DataArray.rio.reproject_match`.
    """
    if aoi.crs is None:
        raise ValueError("AOI must have a CRS")
    if str(aoi.crs) != crs:
        aoi = aoi.to_crs(crs)

    minx, miny, maxx, maxy = aoi.total_bounds
    cols = int(np.ceil((maxx - minx) / resolution))
    rows = int(np.ceil((maxy - miny) / resolution))
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    xs = minx + (np.arange(cols) + 0.5) * resolution
    ys = maxy - (np.arange(rows) + 0.5) * resolution

    da = xr.DataArray(
        np.zeros((rows, cols), dtype="float32"),
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        name="grid",
    )
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    return da


def reproject_to(
    src: xr.DataArray,
    target: xr.DataArray,
    resampling: str = "bilinear",
) -> xr.DataArray:
    """Lazy reprojection of `src` onto `target`'s grid via odc-geo.

    Preserves dask backing — the returned DataArray is a deferred node in
    the graph, computed only at terminal I/O. Pipeable as
    ``src.pipe(reproject_to, target, "bilinear")``.
    """
    out = src.odc.reproject(target.odc.geobox, resampling=resampling, dst_nodata=float('nan'))
    #out = src.rio.reproject_match(target, resampling=getattr(Resampling, resampling))
    out.attrs = {**src.attrs}
    return out.rio.write_crs(target.rio.crs)


def reproject_population_to(
    src: xr.DataArray,
    target: xr.DataArray,
    resampling: str = "bilinear",
) -> xr.DataArray:
    """Lazy reprojection of population count data, by assuming equal area pixels,
    converting to density, then back to count data after reprojection.
    """
    src_res = abs(src.rio.resolution()[0])
    tgt_res = abs(target.rio.resolution()[0])

    # 1. Convert to density
    src_density = src / src_res ** 2

    # 2. Reproject with bilinear (smooth distribution across fine pixels)
    #reprojected_density = src_density.rio.reproject_match(target, resampling=getattr(Resampling, resampling))
    reprojected_density = src_density.odc.reproject(target.odc.geobox, resampling=resampling, dst_nodata=float('nan'))

    # 3. Convert back to counts
    reprojected = reprojected_density * tgt_res ** 2

    # return
    reprojected.attrs = {**src.attrs}
    return reprojected.rio.write_crs(target.rio.crs)
