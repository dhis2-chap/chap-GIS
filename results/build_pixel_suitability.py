"""Learn a pixel-level suitability surface S(temp, NDVI) and validate at sector level.

Per sector, build the population-weighted joint histogram of pixel (annual-mean
temperature, annual-mean NDVI). Then sector_risk = sum_bins S(bin)*p(bin) is
LINEAR in the surface S, so a ridge regression of incidence on the normalised
histogram *is* a fit of the 2D suitability surface. Compare:
  - 1D: S(temp) only  (collapse NDVI axis)
  - 2D: S(temp, NDVI)
evaluated leave-one-district-out. Output the fitted surface + a pixel risk map.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries
from chap_gis.grid import reproject_population_to

ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")   # (lat,lon)
nann = ds["ndvi"].mean("month").rio.write_crs("EPSG:4326")
T = tann.values.astype(np.float32); Nd = nann.values.astype(np.float32)
shp = T.shape

gdf = prepare_boundaries("RWA", 5)
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
NS = len(gdf); loc = gdf["location_id"].to_numpy()

wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]

TB = np.linspace(10, 26, 11); NBv = np.linspace(0.0, 0.9, 9)
NT, NN = len(TB) - 1, len(NBv) - 1
ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1)
ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(pop) & (Nd > -1)
s, w = sect[ok], pop[ok].astype(np.float64)
flat2 = s * (NT * NN) + ti[ok] * NN + ni[ok]
H = np.bincount(flat2, weights=w, minlength=NS * NT * NN).reshape(NS, NT, NN)   # pop-weighted joint hist
# normalise each sector to a distribution
Hn = H / np.clip(H.sum((1, 2), keepdims=True), 1, None)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
groups = np.array([par[loc[i]] for i in range(NS) if keep[i]])
H2 = Hn[keep].reshape(keep.sum(), -1)         # 2D features
H1 = Hn[keep].sum(2)                            # 1D temp-only features (collapse NDVI)
print(f"sectors: {keep.sum()}  features: 1D={H1.shape[1]}  2D={H2.shape[1]}", flush=True)

def lodo(X, alpha=5.0):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(Ridge(alpha=alpha)).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

print("\n=== leave-one-district-out Spearman vs incidence ===")
print(f"  1D  S(temp)          : {lodo(H1):.3f}")
print(f"  2D  S(temp, NDVI)    : {lodo(H2):.3f}")
print("  (ref: sector-mean temp only ~0.55 ; sector RF all-env ~0.70)")

# fit 2D surface on all data -> S(temp,NDVI)
S2 = Ridge(alpha=5.0).fit(H2, y).coef_.reshape(NT, NN)
# pixel risk map = S at each pixel's bin
Smap = np.full(shp, np.nan, np.float32)
Smap[ok] = S2[ti[ok], ni[ok]]
xr.DataArray(Smap, dims=tann.dims, coords=tann.coords, name="pixel_suitability")\
  .rio.write_crs("EPSG:4326").to_netcdf("results/veg_temp_2021/pixel_suitability.nc")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(S2.T, origin="lower", aspect="auto", cmap="RdBu_r",
                  extent=[TB[0], TB[-1], NBv[0], NBv[-1]])
ax[0].set(xlabel="temperature (C)", ylabel="NDVI", title="Learned suitability surface S(temp, NDVI)")
fig.colorbar(im, ax=ax[0], label="ridge weight (risk)")
m = ax[1].imshow(Smap, cmap="magma"); ax[1].set(title="Pixel suitability map (Rwanda 2021)")
ax[1].axis("off"); fig.colorbar(m, ax=ax[1], shrink=.8)
fig.tight_layout(); fig.savefig("results/pixel_suitability.png", dpi=150, bbox_inches="tight")
print("\nwrote results/pixel_suitability.png and results/veg_temp_2021/pixel_suitability.nc", flush=True)
