"""Render the binned (digitized) pixel maps for each covariate of the 3D surface:
temperature, NDVI, NDVI seasonal amplitude. Same bins as build_pixel_suitability_3d.
"""
import numpy as np, xarray as xr
import rasterio.features as rfeat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")
T = tann.values.astype(np.float32)
Nd = ds["ndvi"].mean("month").values.astype(np.float32)
A = (ds["ndvi"].max("month") - ds["ndvi"].min("month")).values.astype(np.float32)
shp = T.shape

gdf = prepare_boundaries("RWA", 5)
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
inside = sect >= 0

TB = np.linspace(10, 26, 9); NBv = np.linspace(0.0, 0.9, 7); AB = np.linspace(0.0, 0.6, 6)
LAYERS = [("Temperature (C)", T, TB, "inferno"),
          ("NDVI (annual mean)", Nd, NBv, "YlGn"),
          ("NDVI seasonal amplitude", A, AB, "viridis")]

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for k, (title, arr, edges, cmap) in enumerate(LAYERS):
    nb = len(edges) - 1
    idx = np.clip(np.digitize(arr, edges) - 1, 0, nb - 1).astype(float)
    idx[~(inside & np.isfinite(arr))] = np.nan
    cm = plt.get_cmap(cmap, nb).copy(); cm.set_bad("white")
    norm = BoundaryNorm(np.arange(nb + 1) - 0.5, nb)
    im = ax[k].imshow(idx, cmap=cm, norm=norm)
    ax[k].set(title=f"{title}  ({nb} bins)"); ax[k].axis("off")
    cb = fig.colorbar(im, ax=ax[k], ticks=np.arange(nb), shrink=0.85)
    cb.ax.set_yticklabels([f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(nb)])
fig.tight_layout()
fig.savefig("results/binned_covariates.png", dpi=150, bbox_inches="tight")
print("wrote results/binned_covariates.png", flush=True)
