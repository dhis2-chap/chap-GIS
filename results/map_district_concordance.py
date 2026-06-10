"""Choropleth of per-district OOF concordance for the v2 gridded model
(temp + 3km focal habitat + built, within-estimator). Per district: fraction of
its same-district sector pairs ordered correctly by the LODO predictions
(0.5 = random). Second panel: gain over the temperature baseline.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

RES, KM, MIN_PAIRS = 100.0, 3.0, 3
z = np.load("results/_habitat_raw.npz", allow_pickle=True)
temp, pop, sect = z["temp"], z["pop"], z["sect"]; rice_p, wet_p, built_p = z["rice_p"], z["wet_p"], z["built_p"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
W = int(round(KM * 1000 / RES)) | 1
fhL = np.log1p(ndimage.uniform_filter(np.maximum(rice_p, wet_p), size=W, mode="nearest"))
fbL = np.log1p(ndimage.uniform_filter(built_p.astype(np.float32), size=W, mode="nearest"))
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s, hab_s, blt_s = sm(temp), sm(fhL), sm(fbL)
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
locK = loc[keep]; y = np.array([tgt[i] for i in locK]); grp = np.array([par[i] for i in locK])
logo = LeaveOneGroupOut()
def oof(X):
    p = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        gt = grp[tr]; Xt = X[tr].copy(); yt = y[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        p[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return p
pred = oof(np.column_stack([temp_s, hab_s, blt_s])[keep])
ptemp = temp_s[keep]                                  # temperature baseline (rank == temp)

def dist_conc(p, idx):
    C = Dd = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if y[i] == y[j]: continue
            dp = p[i] - p[j]; dy = y[i] - y[j]
            if dp == 0: C += .5; Dd += .5
            elif np.sign(dp) == np.sign(dy): C += 1
            else: Dd += 1
    return (C / (C + Dd), C + Dd) if (C + Dd) else (np.nan, 0)

rows = []
for d in np.unique(grp):
    idx = np.where(grp == d)[0]
    cm, n = dist_conc(pred, idx); ct, _ = dist_conc(ptemp, idx)
    rows.append((d, len(idx), n, cm, ct))
sk = pd.DataFrame(rows, columns=["district", "n_sectors", "n_pairs", "conc_model", "conc_temp"]).set_index("district")
sk["gain"] = sk.conc_model - sk.conc_temp
sk.loc[sk.n_pairs < MIN_PAIRS, ["conc_model", "conc_temp", "gain"]] = np.nan
sk.sort_values("conc_model").to_csv("results/district_concordance.csv")
v = sk.dropna(subset=["conc_model"])
print(f"districts mapped (>= {MIN_PAIRS} pairs): {len(v)}")
print(f"model per-district concordance: mean={v.conc_model.mean():.3f} median={v.conc_model.median():.3f}")
print(f"temp  per-district concordance: mean={v.conc_temp.mean():.3f}")
print(f"districts where model > temp: {(v.gain>0).sum()}/{len(v)}")

gg = cgis.io.boundaries.load("RWA", level=5).copy(); gg["location_id"] = gg["shapeID"].astype(str)
gg = gg[gg["location_id"].isin(locK)]; gg["geometry"] = gg.geometry.make_valid()
dist = gg.dissolve(by="parent").join(sk)
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
dist.plot(ax=ax[0], color="lightgrey", edgecolor="white", linewidth=.4)
dist.dropna(subset=["conc_model"]).plot(ax=ax[0], column="conc_model", cmap="RdYlGn", vmin=0.4, vmax=1.0,
        edgecolor="white", linewidth=.4, legend=True, legend_kwds={"label": "per-district OOF concordance", "shrink": .6})
for _, r in dist.iterrows():
    if pd.notna(r["conc_model"]):
        c = r.geometry.representative_point(); ax[0].annotate(f"{r['conc_model']:.2f}", (c.x, c.y), ha="center", va="center", fontsize=6.5)
ax[0].set_title(f"v2 gridded model: per-district OOF concordance\n(0.5=random; overall {0.708:.3f} vs temp {0.684:.3f})", fontsize=11); ax[0].axis("off")
dist.plot(ax=ax[1], color="lightgrey", edgecolor="white", linewidth=.4)
dist.dropna(subset=["gain"]).plot(ax=ax[1], column="gain", cmap="RdBu", vmin=-0.4, vmax=0.4,
        edgecolor="white", linewidth=.4, legend=True, legend_kwds={"label": "concordance gain over temperature", "shrink": .6})
ax[1].set_title("Gain over temperature baseline (blue = model better)", fontsize=11); ax[1].axis("off")
fig.tight_layout(); fig.savefig("results/district_concordance.png", dpi=150, bbox_inches="tight")
print("wrote results/district_concordance.{png,csv}", flush=True)
