"""Score the ORIGINAL breeding-site exposure model (previous-generation risk map)
on the headline burden-capture metric, across its parameterizations, alongside
temperature, the env 3D surface, and the gridded land-use risk map.

Exposure flavors:
  default  : lambda=651, gamma=22.5, t_opt=25 Gaussian  (coarse sweep expo_002)
  optimum  : lambda=1500, gamma=100, t_opt=29 Gaussian  (temp sweep expo_000)
  champion : logistic S(T)=1/(1+e^-3(T-23))              (per_capita_risk)
"""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw.location_id.astype(str)
hd = sw.groupby("location_id").agg(cases=("disease", "sum"), pop=("population", "mean")).reset_index()
expo_opt = sw.groupby("location_id")["mean_exposure_per_person__expo_000"].mean().rename("expo_optimum")  # 1500/100/29
swc = pd.read_csv("results/rwanda_sweep.csv"); swc["location_id"] = swc.location_id.astype(str)
expo_def = swc.groupby("location_id")["mean_exposure_per_person__expo_002"].mean().rename("expo_default")  # 651/22.5/25
champ = pd.read_csv("results/rwanda_sector_risk_vs_cases.csv")[["location_id", "per_capita_risk"]].rename(columns={"per_capita_risk": "expo_champion"})
champ["location_id"] = champ.location_id.astype(str)

z = np.load("results/_gridded_arrays.npz", allow_pickle=True)
temp, fhL, fbL, popr, sect = z["temp"], z["fhL"], z["fbL"], z["pop"], z["sect"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(popr)
s = sect[ok]; w = popr[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
feat = pd.DataFrame({"location_id": loc, "temp": sm(temp), "hab": sm(fhL), "blt": sm(fbL), "sig": sm(expit((temp - 19.) / .5))})
r3 = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv")[["location_id", "risk_oof"]]; r3["location_id"] = r3.location_id.astype(str)
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
df = (feat.merge(hd, on="location_id").merge(expo_opt, on="location_id").merge(expo_def, on="location_id")
          .merge(champ, on="location_id", how="left").merge(r3, on="location_id", how="left").merge(tgt.rename("inc"), on="location_id")
          .merge(g.assign(location_id=g.shapeID.astype(str))[["location_id", "parent"]], on="location_id"))
df = df[df["pop"].gt(0) & df["parent"].notna() & df["inc"].notna()].reset_index(drop=True)
y = df["inc"].values; grp = df["parent"].astype(str).values
logo = LeaveOneGroupOut()
def oof(cols):
    X = df[cols].values; p = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, grp): p[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return p
df["risk_temp"] = df["temp"]; df["risk_map"] = oof(["sig", "hab", "blt"])
df["risk_3d"] = df["risk_oof"].fillna(df["risk_oof"].mean()); df["oracle"] = df["cases"] / df["pop"]
pop = df["pop"].values; cas = df["cases"].values
def cap(sc, frac, P=pop, C=cas):
    o = np.argsort(-sc); cp = np.concatenate([[0], np.cumsum(P[o]) / P.sum()]); cc = np.concatenate([[0], np.cumsum(C[o]) / C.sum()])
    return np.interp(frac, cp, cc)
RANKS = [("temperature", "risk_temp"), ("exposure default (651/22.5/25)", "expo_default"),
         ("exposure optimum (1500/100/29)", "expo_optimum"), ("exposure champion (logistic)", "expo_champion"),
         ("env 3D surface", "risk_3d"), ("gridded risk map (T+land use)", "risk_map"), ("oracle", "oracle")]
FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
print(f"n={len(df)}\nburden captured by targeting top X% of population:")
print(f"{'ranking':34}" + "".join(f"{int(f*100):>6}%" for f in FRACS))
for name, col in RANKS:
    print(f"{name:34}" + "".join(f"{cap(df[col].values, f):>7.3f}" for f in FRACS))
print(f"{'random':34}" + "".join(f"{f:>7.3f}" for f in FRACS))

rng = np.random.RandomState(0); dists = df["parent"].unique(); idx_by = {d: np.where(grp == d)[0] for d in dists}
def boot_gap(col, ref="risk_temp", frac=0.5, reps=2000):
    v = []
    for _ in range(reps):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idx_by[d] for d in ds]); P = pop[idx]; C = cas[idx]
        v.append(cap(df[col].values[idx], frac, P, C) - cap(df[ref].values[idx], frac, P, C))
    return np.array(v)
print("\nburden@50% gap vs temperature (district bootstrap):")
for name, col in RANKS:
    if col == "risk_temp": continue
    b = boot_gap(col); print(f"  {name:34}{cap(df[col].values,.5):.3f}   gap {b.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]  P(>temp)={(b>0).mean():.3f}")

fig, ax = plt.subplots(figsize=(7.5, 7))
for name, col, c in [("oracle", "oracle", "k"), ("gridded risk map (T+land use)", "risk_map", "C0"),
                     ("exposure champion (logistic)", "expo_champion", "C3"), ("exposure default", "expo_default", "C5"),
                     ("temperature", "risk_temp", "C1")]:
    o = np.argsort(-df[col].values); cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cas[o]) / cas.sum()])
    ax.plot(cp, cc, color=c, label=f"{name}  (50%→{cap(df[col].values,.5):.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="random"); ax.axvline(.5, color="grey", lw=.7, ls=":")
ax.set(xlabel="population fraction targeted (ranked by risk)", ylabel="burden (cases) captured",
       title="Burden capture — original exposure model vs new risk maps")
ax.legend(loc="lower right", fontsize=8.5); ax.set_aspect("equal"); fig.tight_layout()
fig.savefig("results/burden_capture_exposure.png", dpi=150, bbox_inches="tight")
print("\nwrote results/burden_capture_exposure.png")
