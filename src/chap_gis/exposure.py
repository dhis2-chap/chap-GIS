"""Distance-based mosquito exposure surface (lazy)."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

from .grid_check import same_grid

HORIZONTAL_LAMBDA_M = 651.0
VERTICAL_GAMMA_M = 22.5

# Tiles beyond ~4.5 λ contribute exp(-4.5) ≈ 1.1% to exposure, so a halo of
# that size yields a near-exact EDT at the cost of clipping far-field
# distances. Increase if higher precision is needed; decrease for more speed.
EDT_HALO_LAMBDAS = 4.5
EDT_TILE_SIZE = 1500


def _tiled_distance_transform_with_indices(
    breeding: np.ndarray,
    pixel_m: float,
    lambda_m: float,
    tile_size: int = EDT_TILE_SIZE,
    halo_lambdas: float = EDT_HALO_LAMBDAS,
):
    """Tiled approximation of ``ndimage.distance_transform_edt`` with indices.

    ``scipy.ndimage.distance_transform_edt`` allocates several float64 buffers
    the size of the input — for country-scale 30 m grids that easily exceeds
    available RAM. This helper splits the array into ``tile_size`` blocks and
    runs the EDT independently on each, padded by a ``halo`` of breeding-mask
    context drawn from the surrounding raster.

    Why a halo: the EDT for any pixel inside a tile depends on the *nearest*
    breeding pixel anywhere — possibly outside the tile. We can bound the
    error by noting that the exposure kernel is ``exp(-d / lambda_m)``: any
    breeding site farther than ``halo_lambdas * lambda_m`` contributes at
    most ``exp(-halo_lambdas)`` to the final exposure (≈ 1.1% at 4.5 λ).
    Padding each tile by ``ceil(halo_lambdas * lambda_m / pixel_m)`` pixels
    therefore yields a result that matches the un-tiled EDT to within that
    far-field tolerance for every pixel in the tile interior. Tiles whose
    halo contains no breeding sites are skipped (``dist=inf``, ``i*=-1``);
    the caller must guard against the ``-1`` sentinel before indexing.
    """
    breeding = np.asarray(breeding, dtype=bool, order="C")
    ny_full, nx_full = breeding.shape

    halo = int(np.ceil((halo_lambdas * lambda_m) / pixel_m))

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
                # No breeding sites within the halo — exposure contribution
                # from anywhere reachable is below the tolerance, treat as ∞.
                dist_full[y0:y1, x0:x1] = np.inf
                iy_full[y0:y1, x0:x1] = -1
                ix_full[y0:y1, x0:x1] = -1
                continue

            dist_tile, (iy_tile, ix_tile) = ndimage.distance_transform_edt(
                ~tile,
                return_distances=True,
                return_indices=True,
            )
            dist_tile = dist_tile.astype(np.float32, copy=False)

            # indices are tile-local; shift into full-array coords
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
    """Compute the exposure array from numpy inputs (runs inside a dask task)."""
    breeding = np.asarray(breeding, dtype=bool, order="C")
    elevation = np.asarray(elevation, dtype=np.float32, order="C")
    if not breeding.any():
        raise ValueError("no breeding sites in input mask")

    dist_px, ny, nx = _tiled_distance_transform_with_indices(
        breeding, pixel_m=pixel_m, lambda_m=lambda_m
    )
    dist_m = dist_px * pixel_m

    # Tiles with no breeding sites in their halo are flagged with -1 indices.
    valid = ny >= 0
    nearest_elev = np.zeros_like(elevation, dtype=np.float32)
    nearest_elev[valid] = elevation[ny[valid], nx[valid]]
    dz = np.maximum(elevation - nearest_elev, 0.0)

    expo = np.exp(-dist_m / lambda_m) * np.exp(-dz / gamma_m)

    if suitability is not None:
        suit = np.asarray(suitability, dtype=np.float32, order="C")
        nearest_suit = np.zeros_like(expo, dtype=np.float32)
        nearest_suit[valid] = suit[ny[valid], nx[valid]]
        nearest_suit = np.where(np.isfinite(nearest_suit), nearest_suit, 0.0)
        expo *= nearest_suit
        expo[breeding] = np.where(
            np.isfinite(suit[breeding]), suit[breeding], 0.0
        )
    else:
        expo[breeding] = 1.0

    if land_mask is not None:
        expo[~np.asarray(land_mask, dtype=bool)] = np.nan
    if water_mask is not None:
        expo[np.asarray(water_mask, dtype=bool)] = np.nan
    return expo.astype(np.float32)


@same_grid
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
