"""Distance-based mosquito exposure surface (lazy)."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

HORIZONTAL_LAMBDA_M = 651.0
VERTICAL_GAMMA_M = 22.5


def tiled_distance_transform_with_indices(
    breeding: np.ndarray,
    pixel_m: float,
    lambda_m: float,
    tile_size: int = 1500,
):
    """
    Memory-efficient tiled Euclidean Distance Transform with indices.

    Improvements:
    - Physics-informed halo (based on lambda)
    - Early tile skipping for empty regions
    - Reduced allocations and casting
    - In-place index offsetting
    - float32 distance output
    """

    breeding = np.asarray(breeding, dtype=bool, order="C")
    ny_full, nx_full = breeding.shape

    halo = int(np.ceil((4.5 * lambda_m) / pixel_m))

    dist_full = np.empty((ny_full, nx_full), dtype=np.float32, order="C")
    iy_full = np.empty((ny_full, nx_full), dtype=np.int32, order="C")
    ix_full = np.empty((ny_full, nx_full), dtype=np.int32, order="C")

    for y0 in range(0, ny_full, tile_size):
        for x0 in range(0, nx_full, tile_size):
            y1 = min(y0 + tile_size, ny_full)
            x1 = min(x0 + tile_size, nx_full)

            ys = max(y0 - halo, 0)
            ye = min(y1 + halo, ny_full)
            xs = max(x0 - halo, 0)
            xe = min(x1 + halo, nx_full)

            tile = breeding[ys:ye, xs:xe]

            if not tile.any():
                dist_full[y0:y1, x0:x1] = np.inf
                iy_full[y0:y1, x0:x1] = -1
                ix_full[y0:y1, x0:x1] = -1
                continue

            tile_uint8 = tile.view(np.uint8)

            dist_tile, (iy_tile, ix_tile) = ndimage.distance_transform_edt(
                1 - tile_uint8,
                return_distances=True,
                return_indices=True,
            )

            dist_tile = dist_tile.astype(np.float32, copy=False)

            iy_tile += ys
            ix_tile += xs

            cy0 = y0 - ys
            cx0 = x0 - xs
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)

            dist_full[y0:y1, x0:x1] = dist_tile[cy0:cy1, cx0:cx1]
            iy_full[y0:y1, x0:x1] = iy_tile[cy0:cy1, cx0:cx1]
            ix_full[y0:y1, x0:x1] = ix_tile[cy0:cy1, cx0:cx1]

    return dist_full, iy_full, ix_full


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
    """
    Compute mosquito exposure surface using optimized tiled EDT.
    """

    breeding = np.asarray(breeding, dtype=bool, order="C")
    elevation = np.asarray(elevation, dtype=np.float32, order="C")

    if not breeding.any():
        # Safe fallback for batch pipelines
        return np.full(breeding.shape, np.nan, dtype=np.float32)

    # --- Tiled EDT (optimized) ---
    dist_px, ny, nx = tiled_distance_transform_with_indices(
        breeding,
        pixel_m=pixel_m,
        lambda_m=lambda_m,
        tile_size=1500,
    )

    # --- Convert to meters ---
    dist_m = dist_px * pixel_m

    # --- Elevation coupling (exact within halo) ---
    valid = ny >= 0  # exclude skipped tiles
    nearest_elev = np.zeros_like(elevation, dtype=np.float32)
    nearest_elev[valid] = elevation[ny[valid], nx[valid]]

    dz = np.maximum(elevation - nearest_elev, 0.0)

    # --- Core exposure ---
    expo = np.exp(-dist_m / lambda_m) * np.exp(-dz / gamma_m)

    # --- Suitability weighting ---
    if suitability is not None:
        suit = np.asarray(suitability, dtype=np.float32, order="C")

        nearest_suit = np.zeros_like(expo, dtype=np.float32)
        nearest_suit[valid] = suit[ny[valid], nx[valid]]

        nearest_suit = np.where(np.isfinite(nearest_suit), nearest_suit, 0.0)

        expo *= nearest_suit

        # Correct breeding pixels
        expo[breeding] = np.where(
            np.isfinite(suit[breeding]),
            suit[breeding],
            0.0,
        )
    else:
        expo[breeding] = 1.0

    # --- Apply masks ---
    if land_mask is not None:
        expo[~np.asarray(land_mask, dtype=bool)] = np.nan

    if water_mask is not None:
        expo[np.asarray(water_mask, dtype=bool)] = np.nan

    return expo.astype(np.float32)

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
