"""3D suitability surface S(temp, NDVI, amp) with EQUAL-POPULATION (quantile) bins.

Same method as build_pixel_suitability_3d.py, but bin edges on each axis are
population-weighted quantiles so each bin holds ~the same number of people. This
spends resolution where people live (e.g. NDVI 0.45-0.60, temp 20-22) instead of
on empty equal-width cells. Reports the edges, realised pop-per-bin, and LODO vs
the equal-width version.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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

def wq_edges(arr, nb):
    """population-weighted quantile edges over valid pixels (nb bins)."""
    v = arr[ok].astype(np.float64); o = np.argsort(v); v, ww = v[o], w[o]
    cw = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    e = np.interp(np.linspace(0, 1, nb + 1), cw, v)
    e[0] = v.min() - 1e-6; e[-1] = v.max() + 1e-6
    return np.maximum.accumulate(e)   # guard monotonicity

NT, NN, NA = 8, 6, 5
TB, NBv, AB = wq_edges(T, NT), wq_edges(Nd, NN), wq_edges(A, NA)
for name, e in (("temp", TB), ("NDVI", NBv), ("amp", AB)):
    print(f"{name:5} quantile edges: " + ", ".join(f"{x:.3f}" for x in e), flush=True)

ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1)
ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ai = np.clip(np.digitize(A, AB) - 1, 0, NA - 1)

tot = w.sum()
print("\nrealised population per bin (%):")
for name, idx, nb in (("temp", ti, NT), ("NDVI", ni, NN), ("amp", ai, NA)):
    pb = np.bincount(idx[ok], weights=w, minlength=nb) / tot * 100
    print(f"  {name:5}: " + " ".join(f"{x:4.1f}" for x in pb))

flat3 = s * (NT * NN * NA) + ti[ok] * (NN * NA) + ni[ok] * NA + ai[ok]
H = np.bincount(flat3, weights=w, minlength=NS * NT * NN * NA).reshape(NS, NT, NN, NA)
Hn = H / np.clip(H.sum((1, 2, 3), keepdims=True), 1, None)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
groups = np.array([par[loc[i]] for i in range(NS) if keep[i]])
H3 = Hn[keep].reshape(keep.sum(), -1)
H2 = Hn[keep].sum(3).reshape(keep.sum(), -1)
H1 = Hn[keep].sum((2, 3))

def lodo(X, alpha):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(Ridge(alpha=alpha)).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

ALPHAS = [1, 5, 20, 50, 100]
print(f"\n=== LODO Spearman vs incidence  (EQUAL-POPULATION bins, n={keep.sum()}) ===")
print(f"{'surface':22}" + "".join(f"{'a='+str(a):>9}" for a in ALPHAS) + f"{'best':>8}")
for name, X in (("1D S(temp)", H1), ("2D S(temp,NDVI)", H2), ("3D S(temp,NDVI,amp)", H3)):
    sc = [lodo(X, a) for a in ALPHAS]
    print(f"{name:22}" + "".join(f"{v:>9.3f}" for v in sc) + f"{max(sc):>8.3f}")
print("  (equal-WIDTH bins gave: 1D 0.577 / 2D 0.696 / 3D 0.758)")

# ---- render the quantile-bin surface (fit on all data; alpha in the stable regime) ----
BEST_A = 5
S3 = Ridge(alpha=BEST_A).fit(H3, y).coef_.reshape(NT, NN, NA)
Smap = np.full(shp, np.nan, np.float32)
okm = ok  # boolean mask on the grid
Smap[okm] = S3[ti[okm], ni[okm], ai[okm]]
xr.DataArray(Smap, dims=tann.dims, coords=tann.coords, name="pixel_suitability_3d_qbins")\
  .rio.write_crs("EPSG:4326").to_netcdf("results/veg_temp_2021/pixel_suitability_3d_qbins.nc")

def slice_plot(ax, mat2d, title):
    # index-based imshow with non-uniform quantile edges shown as tick labels
    vmin, vmax = np.nanpercentile(S3, [2, 98])
    norm = BoundaryNorm(np.linspace(vmin, vmax, 11), 256)
    im = ax.imshow(mat2d.T, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(NT + 1) - 0.5); ax.set_xticklabels([f"{e:.1f}" for e in TB], fontsize=7, rotation=45)
    ax.set_yticks(np.arange(NN + 1) - 0.5); ax.set_yticklabels([f"{e:.2f}" for e in NBv], fontsize=7)
    ax.set(xlabel="temperature edges (C)", ylabel="NDVI edges", title=title)
    return im

fig, axx = plt.subplots(1, 3, figsize=(19, 5.5))
im0 = slice_plot(axx[0], S3[:, :, 0], "S(temp,NDVI | low amp = stable/moist)")
fig.colorbar(im0, ax=axx[0], label="risk weight")
im1 = slice_plot(axx[1], S3[:, :, NA - 1], "S(temp,NDVI | high amp = seasonal/dry)")
fig.colorbar(im1, ax=axx[1], label="risk weight")
m = axx[2].imshow(Smap, cmap="magma"); axx[2].set(title="3D pixel suitability — quantile bins (2021)")
axx[2].axis("off"); fig.colorbar(m, ax=axx[2], shrink=.8)
fig.suptitle("Equal-population (quantile) bins — note non-uniform edges; cells now hold ~equal population",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/pixel_suitability_3d_qbins.png", dpi=150, bbox_inches="tight")
print("\nwrote results/pixel_suitability_3d_qbins.png and .nc", flush=True)
