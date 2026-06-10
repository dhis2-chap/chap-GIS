"""Raster of predicted annual malaria CASE COUNTS per pixel, Rwanda 2021.

The 3D suitability surface S(temp,NDVI,amp) is fit (ridge, alpha=1 -- the champion
0.758 LODO config) so that a sector's population-weighted mean of S equals its
incidence per 1000. Hence S_p is a per-pixel incidence rate (cases/1000), and

    predicted_cases(p) = max(S_p, 0) * population(p) / 1000.

(Negative ridge weights in unpopulated environment cells are clipped to 0 -- they
are an artefact of the linear fit, not real negative risk.) Summing the raster
over a sector reproduces pop * incidence / 1000 = predicted cases.
Outputs GeoTIFF + NetCDF + a quick PNG.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import Ridge
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries
from chap_gis.grid import reproject_population_to

ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")
T = tann.values.astype(np.float32)
Nd = ds["ndvi"].mean("month").values.astype(np.float32)
A = (ds["ndvi"].max("month") - ds["ndvi"].min("month")).values.astype(np.float32)
shp = T.shape

gdf = prepare_boundaries("RWA", 5); NS = len(gdf); loc = gdf["location_id"].to_numpy()
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(A) & np.isfinite(pop) & (Nd > -1)
s, w = sect[ok], pop[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)

# --- fit the champion 3D surface (alpha=1) ---
NT, NN, NA = 8, 6, 5
TB = np.linspace(10, 26, NT + 1); NBv = np.linspace(0, 0.9, NN + 1); AB = np.linspace(0, 0.6, NA + 1)
ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1); ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ai = np.clip(np.digitize(A, AB) - 1, 0, NA - 1)
flat = s * (NT * NN * NA) + ti[ok] * (NN * NA) + ni[ok] * NA + ai[ok]
H = np.bincount(flat, weights=w, minlength=NS * NT * NN * NA).reshape(NS, -1)
Hn = H / np.clip(H.sum(1, keepdims=True), 1, None)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
S = Ridge(alpha=1.0).fit(Hn[keep], y).coef_.reshape(NT, NN, NA)

# --- per-pixel incidence rate (cases/1000) and predicted counts ---
rate = np.full(shp, np.nan, np.float32)               # S_p, clipped >=0
rate[ok] = np.clip(S[ti[ok], ni[ok], ai[ok]], 0, None)
counts = np.where(ok, rate * pop / 1000.0, np.nan).astype(np.float32)

ras = xr.DataArray(counts, dims=tann.dims, coords=tann.coords, name="predicted_cases")
ras = ras.rio.write_crs("EPSG:4326"); ras.rio.write_nodata(np.nan, inplace=True)
ras.to_netcdf("results/predicted_counts_2021.nc")
ras.rio.to_raster("results/predicted_counts_2021.tif")

# --- sanity check: raster sum vs sum of pop*incidence/1000 over evaluated sectors ---
tot_pred = np.nansum(counts)
sec_pred = float(np.sum([psum[i] * tgt.get(loc[i], np.nan) / 1000 for i in range(NS)
                         if keep[i] and np.isfinite(tgt.get(loc[i], np.nan))]))
print(f"raster total predicted cases = {tot_pred:,.0f}", flush=True)
print(f"sum(pop*observed_incidence/1000) over sectors = {sec_pred:,.0f}  (should be similar magnitude)", flush=True)
print(f"max per-pixel cases = {np.nanmax(counts):.2f} ; populated pixels = {ok.sum():,}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
m0 = ax[0].imshow(rate, cmap="magma"); ax[0].set_title("Predicted incidence rate (cases / 1000)")
ax[0].axis("off"); fig.colorbar(m0, ax=ax[0], shrink=.8)
pos = np.where(counts > 0, counts, np.nan)
m1 = ax[1].imshow(pos, cmap="inferno", norm=LogNorm(vmin=max(np.nanpercentile(pos, 50), 1e-3),
                                                     vmax=np.nanmax(pos)))
ax[1].set_title("Predicted case counts per pixel (log scale)"); ax[1].axis("off")
fig.colorbar(m1, ax=ax[1], shrink=.8, label="cases / pixel / year")
fig.tight_layout(); fig.savefig("results/predicted_counts_2021.png", dpi=150, bbox_inches="tight")
print("wrote results/predicted_counts_2021.{tif,nc,png}", flush=True)
