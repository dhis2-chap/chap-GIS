#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rasterio",
#   "geopandas",
#   "numpy",
#   "requests",
#   "shapely",
#   "scipy",
#   "affine",
# ]
# ///
"""
Rwanda-wide Mosquito Exposure Surface (30m) — Standalone

Computes exposure x population for all of Rwanda at 30m resolution.

Pipeline:
  1. Download Rwanda ADM0 boundary from geoBoundaries
  2. Define 30m target grid over Rwanda bbox
  3. Mosaic & aggregate WorldCover 10m -> 30m (block mode)
  4. Reproject elevation (SRTM ~90m) to 30m grid
  5. Reproject rice fields (20m) to 30m grid
  5b. Load CHELSA temperature, lapse-rate downscale to 30m, compute suitability
  6. Reproject population (WorldPop 100m) to 30m grid
  7. Compute exposure using wetlands + rice + water edges + temperature
  8. Compute population x exposure
  9. Save rasters (temperature, suitability, exposure, pop_exposure)

Dispersal model:
  Horizontal: exp(-d / 651m)  -- Costantini 2013
  Vertical:   exp(-dz / 22.5m) -- Gitonga 2006
  Temperature: S(T) Mordecai/Villena TPC -- Gaussian at 25 deg C, zero below 16 deg C
"""

import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.merge
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONTAL_LAMBDA_M = 651.0   # meters, exponential decay scale (Costantini 2013)
VERTICAL_GAMMA_M = 22.5       # meters, very steep vertical decay (Gitonga 2006)
T_OPTIMAL = 25.0              # degrees C, peak for An. gambiae + P. falciparum
T_SIGMA = 5.0                 # degrees C, Gaussian width
T_MIN = 16.0                  # degrees C, minimum for P. falciparum sporogony
T_MAX = 34.0                  # degrees C, upper thermal limit (Mordecai 2013)
LAPSE_RATE = 6.5 / 1000       # degrees C per meter

# WorldCover tile names covering Rwanda (~1-3S, 28.85-30.9E)
WC_TILES = ["S03E027", "S03E030", "N00E027", "N00E030"]
WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Target resolution: 30m ~ 0.000269 deg at the equator
PIXEL_DEG = 30.0 / 111_000  # ~0.000270 deg
PIXEL_M = 30.0

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Download / caching helpers
# ---------------------------------------------------------------------------


def download_file(url: str, local: Path, label: str = "") -> Path:
    """Download a file with progress, skipping if already cached."""
    import requests

    if local.exists():
        print(f"  Cached: {local.name}")
        return local
    print(f"  Downloading {label or local.name}...")
    resp = requests.get(url, stream=True, timeout=300, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dl = 0
    with open(local, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            dl += len(chunk)
            if total:
                print(
                    f"\r    {dl // (1024 * 1024)}/{total // (1024 * 1024)} MB",
                    end="",
                    flush=True,
                )
    print()
    return local


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------


def block_mode_2d(data: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate a 2D array by taking the mode in each (factor x factor) block."""
    h = data.shape[0] // factor
    w = data.shape[1] // factor
    blocks = data[: h * factor, : w * factor].reshape(h, factor, w, factor)
    result = np.zeros((h, w), dtype=data.dtype)
    for i in range(h):
        for j in range(w):
            b = blocks[i, :, j, :].flatten()
            b = b[b > 0]
            if len(b):
                result[i, j] = np.bincount(b).argmax()
    return result


def mosaic_tiles(tile_paths, bounds):
    """Mosaic multiple raster tiles and crop to bounds."""
    datasets = [rasterio.open(p) for p in tile_paths]
    data, transform = rasterio.merge.merge(datasets, bounds=bounds)
    for ds in datasets:
        ds.close()
    return data[0], transform


def mask_to_boundary(data, transform, boundary_gdf, nodata=0):
    """Mask a 2D array to a polygon boundary."""
    tmp = CACHE_DIR / "_tmp_mask.tif"
    with rasterio.open(
        tmp, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    geoms = [g.__geo_interface__ for g in boundary_gdf.geometry]
    with rasterio.open(tmp) as src:
        masked, _ = rasterio.mask.mask(src, geoms, crop=False, filled=True, nodata=nodata)
    tmp.unlink(missing_ok=True)
    return masked[0]


# ---------------------------------------------------------------------------
# Exposure computation
# ---------------------------------------------------------------------------


def compute_suitability(temperature, t_opt=T_OPTIMAL, sigma=T_SIGMA, t_min=T_MIN):
    """Thermal suitability S(T) from Mordecai/Villena Gaussian TPC."""
    suitability = np.exp(-(((temperature - t_opt) / sigma) ** 2))
    suitability[temperature < t_min] = 0.0
    suitability[np.isnan(temperature)] = np.nan
    return suitability


def identify_breeding_sites(landcover, rice=None, include_water_edges=True):
    """Identify mosquito breeding sites from land cover classification."""
    wetland_mask = np.isin(landcover, [90, 95])
    water_mask = landcover == 80
    land_mask = landcover > 0

    breeding = wetland_mask.copy()

    if rice is not None:
        breeding = breeding | rice

    if include_water_edges:
        water_dilated = ndimage.binary_dilation(water_mask, iterations=2)
        water_edge = water_dilated & ~water_mask & land_mask
        breeding = breeding | water_edge

    return breeding, water_mask, land_mask


def compute_exposure(
    breeding, elevation, pixel_m, water_mask, land_mask,
    suitability=None, lambda_m=HORIZONTAL_LAMBDA_M, gamma_m=VERTICAL_GAMMA_M,
):
    """Compute mosquito exposure index (nearest-site model).

    exposure = exp(-d / lambda) * exp(-dz+ / gamma) * S(T_nearest)
    """
    if np.sum(breeding) == 0:
        raise ValueError("No breeding sites found")

    dist_pixels, nearest_idx = ndimage.distance_transform_edt(
        ~breeding, return_distances=True, return_indices=True,
    )
    dist_m = dist_pixels * pixel_m

    nearest_elev = elevation[nearest_idx[0], nearest_idx[1]]
    elev_diff = np.maximum(elevation - nearest_elev, 0)

    horiz_decay = np.exp(-dist_m / lambda_m)
    vert_decay = np.exp(-elev_diff / gamma_m)
    exposure = horiz_decay * vert_decay

    if suitability is not None:
        nearest_suit = suitability[nearest_idx[0], nearest_idx[1]]
        nearest_suit = np.where(np.isfinite(nearest_suit), nearest_suit, 0.0)
        exposure = exposure * nearest_suit

    exposure[~land_mask] = np.nan
    exposure[water_mask] = np.nan

    if suitability is not None:
        breeding_suit = np.where(
            np.isfinite(suitability[breeding]), suitability[breeding], 0.0,
        )
        exposure[breeding] = breeding_suit
    else:
        exposure[breeding] = 1.0

    return exposure


def identify_hotspots(pop_exposure, population, percentile=90):
    """Identify high-risk hotspot pixels based on population-weighted exposure."""
    pe_valid = pop_exposure[np.isfinite(pop_exposure)]
    pe_positive = pe_valid[pe_valid > 0]

    if len(pe_positive) == 0:
        return np.zeros_like(pop_exposure, dtype=bool), {
            "threshold": 0, "hotspot_pop": 0, "total_pop": 0, "pct": 0,
        }

    threshold = np.percentile(pe_positive, percentile)
    hotspot = (pop_exposure >= threshold) & np.isfinite(pop_exposure)
    hotspot_pop = np.nansum(population[hotspot])
    total_pop = np.nansum(population[np.isfinite(population)])

    return hotspot, {
        "threshold": threshold,
        "hotspot_pop": hotspot_pop,
        "total_pop": total_pop,
        "pct": 100 * hotspot_pop / total_pop if total_pop > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_rwanda_boundary():
    """Download Rwanda ADM0 boundary from geoBoundaries."""
    cache = CACHE_DIR / "rwanda.geojson"
    if cache.exists():
        print("  Cached: rwanda.geojson")
        return gpd.read_file(cache)

    print("  Downloading Rwanda boundary from geoBoundaries...")
    url = (
        "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
        "releaseData/gbOpen/RWA/ADM0/geoBoundaries-RWA-ADM0.geojson"
    )
    gdf = gpd.read_file(url).to_crs("EPSG:4326")
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def download_elevation():
    """Download Rwanda-wide SRTM elevation via R/geodata."""
    elev_path = CACHE_DIR / "rwanda_elevation.tif"
    if elev_path.exists():
        print("  Cached: rwanda_elevation.tif")
        return elev_path

    print("  Downloading SRTM elevation via R/geodata...")
    r_script = Path(__file__).parent / "download_rwanda_elevation.R"
    result = subprocess.run(
        ["Rscript", str(r_script)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        print(f"  R stdout: {result.stdout[-500:]}")
        print(f"  R stderr: {result.stderr[-500:]}")
        raise RuntimeError("Failed to download elevation via R")

    print(f"  {result.stdout.strip()}")
    return elev_path


def download_worldcover_tiles():
    """Download all WorldCover tiles covering Rwanda."""
    paths = []
    for tile in WC_TILES:
        filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        url = f"{WC_BASE}/{filename}"
        path = download_file(url, CACHE_DIR / filename, f"WorldCover {tile}")
        paths.append(path)
    return paths


def build_target_grid(bounds):
    """Build a 30m resolution target grid over Rwanda bbox."""
    minx, miny, maxx, maxy = bounds
    cols = int(np.ceil((maxx - minx) / PIXEL_DEG))
    rows = int(np.ceil((maxy - miny) / PIXEL_DEG))
    transform = Affine(PIXEL_DEG, 0, minx, 0, -PIXEL_DEG, maxy)
    return (rows, cols), transform


def load_worldcover_30m(tile_paths, bounds, boundary, target_shape, target_transform):
    """Mosaic WorldCover tiles, mask to boundary, aggregate 10m -> 30m."""
    print("  Mosaicking WorldCover tiles...")
    lc_10m, lc_transform = mosaic_tiles(tile_paths, bounds)
    print(f"  Mosaic shape: {lc_10m.shape}")

    # Buffer boundary by ~300m so border water bodies are included
    buffered = boundary.copy()
    buffered["geometry"] = boundary.geometry.buffer(0.0027)

    print("  Masking to buffered boundary (300m buffer for water edges)...")
    lc_10m = mask_to_boundary(lc_10m, lc_transform, buffered)

    print("  Aggregating 10m -> 30m (block mode)...")
    lc_30m = block_mode_2d(lc_10m, 3)
    print(f"  Land cover 30m: {lc_30m.shape}")

    print("  Reprojecting to target grid...")
    agg_pixel = abs(lc_transform.a) * 3
    agg_transform = Affine(agg_pixel, 0, lc_transform.c, 0, -agg_pixel, lc_transform.f)

    tmp = CACHE_DIR / "_tmp_lc30.tif"
    with rasterio.open(
        tmp, "w", driver="GTiff",
        height=lc_30m.shape[0], width=lc_30m.shape[1],
        count=1, dtype=lc_30m.dtype,
        crs="EPSG:4326", transform=agg_transform, nodata=0,
    ) as dst:
        dst.write(lc_30m, 1)

    lc_final = np.zeros(target_shape, dtype=lc_30m.dtype)
    with rasterio.open(tmp) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=lc_final,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
    tmp.unlink(missing_ok=True)

    # Re-mask to buffered boundary
    tmp2 = CACHE_DIR / "_tmp_lc30b.tif"
    with rasterio.open(
        tmp2, "w", driver="GTiff",
        height=lc_final.shape[0], width=lc_final.shape[1],
        count=1, dtype=lc_final.dtype,
        crs="EPSG:4326", transform=target_transform, nodata=0,
    ) as dst:
        dst.write(lc_final, 1)
    geoms = [g.__geo_interface__ for g in buffered.geometry]
    with rasterio.open(tmp2) as src:
        masked, _ = rasterio.mask.mask(src, geoms, crop=False, filled=True, nodata=0)
    lc_final = masked[0]
    tmp2.unlink(missing_ok=True)

    return lc_final


def load_elevation_30m(elev_path, target_shape, target_transform):
    """Reproject elevation to the target 30m grid."""
    elev = np.full(target_shape, np.nan, dtype=np.float32)
    with rasterio.open(elev_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=elev,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return elev


def load_rice_30m(rice_path, target_shape, target_transform, boundary):
    """Reproject rice field map to the target 30m grid."""
    rice = np.zeros(target_shape, dtype=np.uint8)
    geoms = [g.__geo_interface__ for g in boundary.geometry]
    with rasterio.open(rice_path) as src:
        masked, masked_transform = rasterio.mask.mask(
            src, geoms, crop=True, filled=True, nodata=0,
        )
        reproject(
            source=masked[0],
            destination=rice,
            src_transform=masked_transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    return rice > 0


def load_population_30m(pop_path, target_shape, target_transform, boundary):
    """Reproject population to the target 30m grid.

    WorldPop gives people per pixel (~100m). When reprojecting to 30m,
    we scale by the pixel area ratio to conserve total population.
    """
    pop = np.full(target_shape, np.nan, dtype=np.float32)
    geoms = [g.__geo_interface__ for g in boundary.geometry]
    with rasterio.open(pop_path) as src:
        src_pixel_area = abs(src.transform.a * src.transform.e)

        masked, masked_transform = rasterio.mask.mask(
            src, geoms, crop=True, filled=True, nodata=src.nodata,
        )
        src_data = masked[0].astype(np.float32)
        if src.nodata is not None:
            src_data[src_data == src.nodata] = np.nan
        src_data[src_data < 0] = np.nan

        reproject(
            source=src_data,
            destination=pop,
            src_transform=masked_transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    dst_pixel_area = abs(target_transform.a * target_transform.e)
    area_ratio = dst_pixel_area / src_pixel_area
    pop = pop * area_ratio
    pop[pop < 0] = np.nan
    return pop


def load_temperature_30m(chelsa_dir, target_shape, target_transform, elevation):
    """Load CHELSA monthly temperatures, compute annual mean, lapse-rate downscale to 30m."""
    monthly_30m = []
    for month in range(1, 13):
        fname = f"CHELSA_tas_{month:02d}_2010_V.2.1.tif"
        chelsa_path = chelsa_dir / fname
        if not chelsa_path.exists():
            raise FileNotFoundError(f"Missing CHELSA file: {chelsa_path}")

        temp_month = np.full(target_shape, np.nan, dtype=np.float32)
        with rasterio.open(chelsa_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=temp_month,
                dst_transform=target_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )

        sample = temp_month[np.isfinite(temp_month)]
        if len(sample) > 0:
            if np.mean(sample) > 2000:  # Kelvin * 10
                temp_month = temp_month / 10.0 - 273.15
            elif np.mean(sample) > 200:  # deg C * 10
                temp_month = temp_month / 10.0

        monthly_30m.append(temp_month)

    chelsa_annual_30m = np.nanmean(monthly_30m, axis=0).astype(np.float32)

    # Lapse-rate downscaling
    block_size = 33
    rows, cols = elevation.shape
    pad_r = (block_size - rows % block_size) % block_size
    pad_c = (block_size - cols % block_size) % block_size
    elev_padded = np.pad(elevation, ((0, pad_r), (0, pad_c)), mode="edge")
    br = elev_padded.shape[0] // block_size
    bc = elev_padded.shape[1] // block_size
    elev_blocks = elev_padded[:br * block_size, :bc * block_size].reshape(
        br, block_size, bc, block_size,
    )
    elev_coarse = np.nanmean(elev_blocks, axis=(1, 3)).astype(np.float32)
    elev_coarse_30m = np.repeat(np.repeat(elev_coarse, block_size, axis=0), block_size, axis=1)
    elev_coarse_30m = elev_coarse_30m[:rows, :cols]

    elev_anomaly = elevation - elev_coarse_30m
    temperature = chelsa_annual_30m - LAPSE_RATE * elev_anomaly
    temperature[np.isnan(elevation)] = np.nan

    return temperature


def save_raster(data, path, transform, nodata=np.nan):
    """Save a 2D float32 array as a GeoTIFF."""
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
        nodata=float(nodata) if np.isfinite(nodata) else None,
        compress="deflate",
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  Rwanda-wide Mosquito Exposure Surface (30m)")
    print("=" * 60)

    # --- 1. Boundary ---
    print("\n--- Step 1: Rwanda boundary ---")
    rwanda = download_rwanda_boundary()
    bounds = tuple(rwanda.total_bounds)
    print(f"  Bounds: {bounds}")

    # --- 2. Target grid ---
    print("\n--- Step 2: Target grid ---")
    target_shape, target_transform = build_target_grid(bounds)
    print(f"  Grid: {target_shape[0]:,} x {target_shape[1]:,} = {target_shape[0] * target_shape[1]:,} pixels")
    print(f"  Pixel: {PIXEL_DEG:.6f} deg ~ {PIXEL_M:.0f}m")

    # --- 3. WorldCover ---
    print("\n--- Step 3: WorldCover (10m -> 30m) ---")
    tile_paths = download_worldcover_tiles()
    landcover = load_worldcover_30m(tile_paths, bounds, rwanda, target_shape, target_transform)
    n_land = np.sum(landcover > 0)
    print(f"  Land pixels: {n_land:,}")

    # --- 4. Elevation ---
    print("\n--- Step 4: Elevation ---")
    elev_path = download_elevation()
    elevation = load_elevation_30m(elev_path, target_shape, target_transform)
    valid_elev = elevation[np.isfinite(elevation) & (landcover > 0)]
    print(f"  Elevation range: {np.min(valid_elev):.0f} - {np.max(valid_elev):.0f} m")

    # --- 5. Rice fields ---
    print("\n--- Step 5: Rice fields ---")
    rice_path = CACHE_DIR / "rwanda_rice_20m_2023.tif"
    if rice_path.exists():
        rice_mask = load_rice_30m(rice_path, target_shape, target_transform, rwanda)
        n_rice = np.sum(rice_mask)
        rice_km2 = n_rice * (PIXEL_M / 1000) ** 2
        print(f"  Rice pixels: {n_rice:,} ({rice_km2:.1f} km2)")
    else:
        print("  Rice map not found, proceeding without rice")
        rice_mask = None

    # --- 5b. Temperature + suitability ---
    print("\n--- Step 5b: Temperature (CHELSA -> 30m lapse-rate downscaling) ---")
    temperature = load_temperature_30m(CACHE_DIR, target_shape, target_transform, elevation)
    valid_temp = temperature[np.isfinite(temperature) & (landcover > 0)]
    print(f"  Temperature range: {np.min(valid_temp):.1f} - {np.max(valid_temp):.1f} deg C")
    print(f"  Mean: {np.mean(valid_temp):.1f} deg C")

    print("  Computing thermal suitability (Mordecai/Villena TPC)...")
    suitability = compute_suitability(temperature)
    valid_suit = suitability[np.isfinite(suitability) & (landcover > 0)]
    print(f"  Suitability range: {np.min(valid_suit):.4f} - {np.max(valid_suit):.4f}")
    print(f"  Mean: {np.mean(valid_suit):.4f}")
    pct_zero = 100 * np.sum(valid_suit == 0) / len(valid_suit)
    print(f"  Pixels below 16 deg C (zero suitability): {pct_zero:.1f}%")

    # --- 6. Population ---
    print("\n--- Step 6: Population ---")
    pop_path = CACHE_DIR / "rwa_ppp_2020_constrained.tif"
    if not pop_path.exists():
        download_file(
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
            "2020/maxar_v1/RWA/rwa_ppp_2020_constrained.tif",
            pop_path, "WorldPop 100m")
    population = load_population_30m(pop_path, target_shape, target_transform, rwanda)
    total_pop = np.nansum(population[np.isfinite(population)])
    print(f"  Total population: {total_pop:,.0f}")

    # --- 7. Breeding sites & exposure ---
    print("\n--- Step 7: Exposure computation ---")
    breeding, water_mask, land_mask = identify_breeding_sites(
        landcover, rice=rice_mask, include_water_edges=True,
    )
    n_breeding = np.sum(breeding)
    breeding_km2 = n_breeding * (PIXEL_M / 1000) ** 2
    print(f"  Breeding site pixels: {n_breeding:,} ({breeding_km2:.1f} km2)")

    print("  Computing exposure (nearest-site model)...")
    exposure = compute_exposure(
        breeding, elevation, PIXEL_M, water_mask, land_mask,
        suitability=suitability,
    )

    valid_exp = exposure[np.isfinite(exposure)]
    print(f"  Exposure range: {np.min(valid_exp):.6f} - {np.max(valid_exp):.2f}")
    print(f"  Median: {np.median(valid_exp):.4f}")
    print(f"  Mean: {np.mean(valid_exp):.4f}")

    # --- 8. Population x exposure ---
    print("\n--- Step 8: Population x exposure ---")
    pop_exposure = population * exposure

    total_exposed = np.nansum(pop_exposure[np.isfinite(pop_exposure)])
    print(f"  Exposure-weighted population: {total_exposed:,.0f}")
    print(f"  Mean per-capita exposure: {total_exposed / total_pop:.4f}")

    # Hotspots
    hotspot, stats = identify_hotspots(pop_exposure, population)
    print(f"  Hotspot threshold (top 10%): {stats['threshold']:.3f}")
    print(f"  People in hotspots: {stats['hotspot_pop']:,.0f} ({stats['pct']:.1f}%)")

    # --- 9. Save rasters ---
    print("\n--- Step 9: Save outputs ---")
    save_raster(temperature, CACHE_DIR / "rwanda_temperature.tif", target_transform)
    save_raster(suitability, CACHE_DIR / "rwanda_suitability.tif", target_transform)
    save_raster(population, CACHE_DIR / "rwanda_population.tif", target_transform)
    save_raster(exposure, CACHE_DIR / "rwanda_exposure.tif", target_transform)
    save_raster(pop_exposure, CACHE_DIR / "rwanda_pop_exposure.tif", target_transform)

    # --- 10. Summary ---
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Grid:                    {target_shape[0]:,} x {target_shape[1]:,}")
    print(f"  Resolution:              {PIXEL_M:.0f}m")
    print(f"  Land pixels:             {n_land:,}")
    print(f"  Breeding site pixels:    {n_breeding:,} ({breeding_km2:.1f} km2)")
    print(f"  Total population:        {total_pop:,.0f}")
    print(f"  Exposure-weighted pop:   {total_exposed:,.0f}")
    print(f"  Mean per-capita expos:   {total_exposed / total_pop:.4f}")
    print(f"  Hotspot population:      {stats['hotspot_pop']:,.0f} ({stats['pct']:.1f}%)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
