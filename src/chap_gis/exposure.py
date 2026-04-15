"""Distance-based mosquito exposure surface (lazy)."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

HORIZONTAL_LAMBDA_M = 651.0
VERTICAL_GAMMA_M = 22.5


@dask.delayed
def _exposure_np(
    breeding,
    elevation,
    suitability,
    land_mask,
    water_mask,
    pixel_m,
    lambda_m,
    gamma_m,
):
    """Compute the exposure array from numpy inputs (runs inside a dask task)."""
    breeding = np.asarray(breeding).astype(bool)
    elevation = np.asarray(elevation, dtype="float32")
    if not breeding.any():
        raise ValueError("no breeding sites in input mask")

    dist_px, (ny, nx) = ndimage.distance_transform_edt(
        ~breeding, return_distances=True, return_indices=True
    )
    dist_m = dist_px * pixel_m
    nearest_elev = elevation[ny, nx]
    dz = np.maximum(elevation - nearest_elev, 0.0)

    expo = np.exp(-dist_m / lambda_m) * np.exp(-dz / gamma_m)

    if suitability is not None:
        suit = np.asarray(suitability, dtype="float32")
        nearest_suit = suit[ny, nx]
        nearest_suit = np.where(np.isfinite(nearest_suit), nearest_suit, 0.0)
        expo = expo * nearest_suit
        expo[breeding] = np.where(
            np.isfinite(suit[breeding]), suit[breeding], 0.0
        )
    else:
        expo[breeding] = 1.0

    if land_mask is not None:
        expo[~np.asarray(land_mask).astype(bool)] = np.nan
    if water_mask is not None:
        expo[np.asarray(water_mask).astype(bool)] = np.nan
    return expo.astype("float32")


def exposure(
    breeding: xr.DataArray,
    elevation: xr.DataArray,
    suitability: xr.DataArray | None,
    *,
    pixel_m: float,
    land_mask: xr.DataArray | None = None,
    water_mask: xr.DataArray | None = None,
    horizontal_lambda_m: float = HORIZONTAL_LAMBDA_M,
    vertical_gamma_m: float = VERTICAL_GAMMA_M,
) -> xr.DataArray:
    """Nearest-breeding-site exposure index (lazy).

    ``exposure = exp(-d / λ) · exp(-max(Δz, 0) / γ) · S(T_nearest)``

    Distance-transform is wrapped in :func:`dask.delayed`, so the op is a
    single deferred node — the input dask arrays are materialised only when
    the terminal operation triggers compute.

    All inputs must share CRS, dims, coords and shape.
    """
    _assert_aligned(breeding, elevation)
    if suitability is not None:
        _assert_aligned(breeding, suitability)

    shape = breeding.shape
    delayed = _exposure_np(
        breeding.data,
        elevation.data,
        suitability.data if suitability is not None else None,
        land_mask.data if land_mask is not None else None,
        water_mask.data if water_mask is not None else None,
        pixel_m,
        horizontal_lambda_m,
        vertical_gamma_m,
    )
    arr = da.from_delayed(delayed, shape=shape, dtype="float32")

    out = xr.DataArray(
        arr,
        dims=breeding.dims,
        coords=breeding.coords,
        name="exposure",
        attrs={
            "long_name": "Mosquito exposure index (nearest-breeding-site model)",
            "units": "1",
            "horizontal_lambda_m": horizontal_lambda_m,
            "vertical_gamma_m": vertical_gamma_m,
            "pixel_m": pixel_m,
        },
    )
    return out.rio.write_crs(breeding.rio.crs)


def _assert_aligned(a: xr.DataArray, b: xr.DataArray) -> None:
    if a.rio.crs != b.rio.crs:
        raise ValueError(f"CRS mismatch: {a.rio.crs} vs {b.rio.crs}")
    if a.dims != b.dims:
        raise ValueError(f"dim mismatch: {a.dims} vs {b.dims}")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
