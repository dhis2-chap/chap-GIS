"""Moisture-augmented pixel suitability surface S(temp, NDVI, ndvi_amp).

Extends results/build_pixel_suitability.py with a third axis: NDVI seasonal
amplitude (max month - min month), the near-temperature-orthogonal moisture-stress
feature that lifted the sector RF model from ~0.66 to ~0.75 LODO. Stable
year-round greenness (low amplitude) marks permanent water -> persistent breeding;
strong seasonal swing (high amplitude) marks rain-pulsed, ephemeral habitat.

Method (unchanged): sector_risk = sum_bins S(bin)*p(bin) is LINEAR in S, where
p is the per-sector population-weighted joint histogram over the binned axes; a
ridge of incidence on the normalised histogram IS a fit of S. Compare 1D/2D/3D
leave-one-district-out, render the 2021 pixel risk map and low/high-amplitude
slices of the learned surface.
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
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")
nann = ds["ndvi"].mean("month").rio.write_crs("EPSG:4326")
namp = (ds["ndvi"].max("month") - ds["ndvi"].min("month")).rio.write_crs("EPSG:4326")
T = tann.values.astype(np.float32); Nd = nann.values.astype(np.float32); A = namp.values.astype(np.float32)
shp = T.shape

gdf = prepare_boundaries("RWA", 5)
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
NS = len(gdf); loc = gdf["location_id"].to_numpy()

wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

# bins (slightly coarser than the 2D script to control the parameter count)
TB = np.linspace(10, 26, 9);  NT = len(TB) - 1     # 8 temp bins
NBv = np.linspace(0.0, 0.9, 7); NN = len(NBv) - 1   # 6 ndvi bins
AB = np.linspace(0.0, 0.6, 6);  NA = len(AB) - 1     # 5 amplitude bins
ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1)
ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ai = np.clip(np.digitize(A, AB) - 1, 0, NA - 1)

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(A) & np.isfinite(pop) & (Nd > -1)
s, w = sect[ok], pop[ok].astype(np.float64)
flat3 = s * (NT * NN * NA) + ti[ok] * (NN * NA) + ni[ok] * NA + ai[ok]
H = np.bincount(flat3, weights=w, minlength=NS * NT * NN * NA).reshape(NS, NT, NN, NA)
Hn = H / np.clip(H.sum((1, 2, 3), keepdims=True), 1, None)   # per-sector pop distribution

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
groups = np.array([par[loc[i]] for i in range(NS) if keep[i]])
H3 = Hn[keep].reshape(keep.sum(), -1)            # 3D features (temp,ndvi,amp)
H2 = Hn[keep].sum(3).reshape(keep.sum(), -1)     # collapse amp -> 2D (temp,ndvi)
H1 = Hn[keep].sum((2, 3))                          # collapse ndvi,amp -> 1D (temp)
print(f"sectors={keep.sum()} districts={pd.Series(groups).nunique()}  "
      f"feats 1D={H1.shape[1]} 2D={H2.shape[1]} 3D={H3.shape[1]}", flush=True)

def lodo(X, alpha):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(Ridge(alpha=alpha)).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

ALPHAS = [1, 5, 20, 50, 100]
print("\n=== LODO Spearman vs incidence (ridge alpha sweep) ===")
print(f"{'surface':22}" + "".join(f"{'a='+str(a):>9}" for a in ALPHAS) + f"{'best':>8}")
for name, X in (("1D S(temp)", H1), ("2D S(temp,NDVI)", H2), ("3D S(temp,NDVI,amp)", H3)):
    sc = [lodo(X, a) for a in ALPHAS]
    print(f"{name:22}" + "".join(f"{v:>9.3f}" for v in sc) + f"{max(sc):>8.3f}")
print("  (ref: 2D in report 0.669 ; sector RF temp+veg+moisture ~0.75)")

# fit the 3D surface on all data and render
BEST_A = 20
S3 = Ridge(alpha=BEST_A).fit(H3, y).coef_.reshape(NT, NN, NA)
Smap = np.full(shp, np.nan, np.float32)
Smap[ok] = S3[ti[ok], ni[ok], ai[ok]]
xr.DataArray(Smap, dims=tann.dims, coords=tann.coords, name="pixel_suitability_3d")\
  .rio.write_crs("EPSG:4326").to_netcdf("results/veg_temp_2021/pixel_suitability_3d.nc")

# population share by amplitude tercile, to label the slices
lo_a, hi_a = 0, NA - 1
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
vmin, vmax = np.nanpercentile(S3, [2, 98])
for k, (aidx, lab) in enumerate([(lo_a, f"low amp [{AB[0]:.1f}-{AB[1]:.1f}] = stable/moist"),
                                  (hi_a, f"high amp [{AB[-2]:.1f}-{AB[-1]:.1f}] = seasonal/dry")]):
    im = ax[k].imshow(S3[:, :, aidx].T, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=vmin, vmax=vmax, extent=[TB[0], TB[-1], NBv[0], NBv[-1]])
    ax[k].set(xlabel="temperature (C)", ylabel="NDVI", title=f"S(temp,NDVI | {lab})")
    fig.colorbar(im, ax=ax[k], label="risk weight")
m = ax[2].imshow(Smap, cmap="magma"); ax[2].set(title="3D pixel suitability (Rwanda 2021)")
ax[2].axis("off"); fig.colorbar(m, ax=ax[2], shrink=.8)
fig.tight_layout(); fig.savefig("results/pixel_suitability_3d.png", dpi=150, bbox_inches="tight")
print("\nwrote results/pixel_suitability_3d.png and results/veg_temp_2021/pixel_suitability_3d.nc", flush=True)
