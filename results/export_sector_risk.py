"""Export per-sector risk (3D suitability surface) vs observed annual incidence.

risk_oof  : leave-one-district-out prediction (surface fit WITHOUT the sector's
            own district) -- the honest, generalisable risk estimate.
risk_full : surface fit on ALL sectors (matches the displayed pixel map; in-sample).
Both are the population-weighted mean suitability <p_s, S> for the sector.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
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
namecol = next((c for c in ("shapeName", "name", "ADM3_EN") if c in gdf.columns), None)
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
idx = np.where(keep)[0]
y = np.array([tgt[loc[i]] for i in idx])
groups = np.array([par[loc[i]] for i in idx])
X = Hn[keep]

ALPHA = 1.0
risk_oof = np.full(len(idx), np.nan)
for tr, te in LeaveOneGroupOut().split(X, y, groups):
    risk_oof[te] = clone(Ridge(alpha=ALPHA)).fit(X[tr], y[tr]).predict(X[te])
risk_full = Ridge(alpha=ALPHA).fit(X, y).predict(X)
print(f"sectors={len(idx)}  LODO Spearman(risk_oof, incidence)={spearmanr(risk_oof, y).statistic:.3f}", flush=True)

out = pd.DataFrame({
    "location_id": loc[idx],
    "district": groups,
    "population": np.round(psum[idx]).astype(int),
    "risk_oof": np.round(risk_oof, 3),
    "risk_full": np.round(risk_full, 3),
    "annual_incidence_per1000": np.round(y, 2),
})
if namecol:
    out.insert(1, "sector_name", gdf[namecol].to_numpy()[idx])
out["risk_oof_rank"] = out["risk_oof"].rank(ascending=False).astype(int)
out["incidence_rank"] = out["annual_incidence_per1000"].rank(ascending=False).astype(int)
out = out.sort_values("risk_oof", ascending=False)
out.to_csv("results/rwanda_sector_risk_vs_incidence.csv", index=False)
print("wrote results/rwanda_sector_risk_vs_incidence.csv", flush=True)
print(out.head(8).to_string(index=False))
