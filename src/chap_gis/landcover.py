"""Land-cover-derived masks (WorldCover class codes)."""

from __future__ import annotations

import xarray as xr
from dask_image.ndmorph import binary_dilation as _dask_binary_dilation

from .grid_check import same_grid

# WorldCover v200 class codes
WETLAND_CODES = (90, 95)
PERMANENT_WATER_CODE = 80


def land_mask(lc: xr.DataArray) -> xr.DataArray:
    return (lc > 0).rename("land_mask")


def water_mask(lc: xr.DataArray) -> xr.DataArray:
    return (lc == PERMANENT_WATER_CODE).rename("water_mask")


def wetland_mask(lc: xr.DataArray) -> xr.DataArray:
    return ((lc == WETLAND_CODES[0]) | (lc == WETLAND_CODES[1])).rename("wetland_mask")


@same_grid
def breeding_site_mask(
    lc: xr.DataArray,
    rice: xr.DataArray | None = None,
    water_edge_buffer: int = 2,
) -> xr.DataArray:
    """Identify mosquito breeding-site pixels (lazy).

    Combines WorldCover wetlands, optional rice paddies, and a buffered
    water-edge ring around permanent water bodies on land. Uses
    :func:`dask_image.ndmorph.binary_dilation` so a dask-backed ``lc``
    produces a dask-backed result.

    ``rice``, if given, must share CRS/dims/shape with ``lc``.
    """
    wet = wetland_mask(lc)
    water = water_mask(lc)
    land = land_mask(lc)

    breeding = wet
    if rice is not None:
        breeding = breeding | rice.astype(bool)

    if water_edge_buffer > 0:
        # dask-image requires a dask-backed array; chunk numpy-backed inputs
        water_dask = water if water.chunks else water.chunk()
        dilated = xr.apply_ufunc(
            _dask_binary_dilation,
            water_dask,
            kwargs={"iterations": water_edge_buffer},
            dask="allowed",
            output_dtypes=[bool],
        )
        water_edge = dilated & ~water & land
        breeding = breeding | water_edge

    breeding = breeding.rename("breeding_sites").astype(bool)
    breeding = breeding.rio.write_crs(lc.rio.crs)
    breeding.attrs.update(
        long_name="Mosquito breeding-site mask",
        description=(
            f"WorldCover wetlands {WETLAND_CODES} + optional rice + "
            f"{water_edge_buffer}-pixel buffer around water (code {PERMANENT_WATER_CODE})"
        ),
    )
    return breeding
