"""Per-sector built-up and cropland fractions (WorldCover 2021) + distance-to-water.
Clean urbanization / land-use covariates (no incidence-denominator entanglement),
to test for within-district signal beyond temperature.
"""
import numpy as np, pandas as pd
import rasterio.features as rfeat
from scipy import ndimage
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to

RES = 100.0
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027)
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
lc = np.asarray(reproject_to(land, grid, "mode").compute().values)
if lc.ndim == 3: lc = lc[0]

built = (lc == 50).astype(np.float32)      # built-up
crop = (lc == 40).astype(np.float32)       # cropland
water = (lc == 80).astype(np.float32)      # permanent water
# distance (pixels -> km) to nearest permanent-water pixel
dist_px = ndimage.distance_transform_edt(water == 0)
dist_km = dist_px * (RES / 1000.0)

sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=lc.shape, transform=grid.rio.transform(), fill=-1, dtype="int32")
ok = sect >= 0
cnt = np.bincount(sect[ok], minlength=NS).astype(float)
def per_sector(arr): return np.bincount(sect[ok], weights=arr[ok].astype(float), minlength=NS) / np.maximum(cnt, 1)

out = pd.DataFrame({
    "location_id": loc,
    "built_frac": per_sector(built),
    "cropland_frac": per_sector(crop),
    "dist_water_km": per_sector(dist_km),
}).set_index("location_id")
out.to_csv("results/rwanda_sector_landcover_extra.csv")
print("wrote results/rwanda_sector_landcover_extra.csv")
print(out.describe().round(3).to_string())
