"""HEADLINE METRIC: burden capture under risk-prioritised allocation.

Rank sectors by predicted risk, allocate to the top sectors until they cover a
target fraction of the population; report the share of total malaria burden
(cases) captured. Operationally: "targeting X% of people reaches Y% of cases."
random = X ; oracle (rank by actual incidence) = ceiling.

Scores the key rankings (temperature baseline, environmental 3D suitability
surface, gridded temp+land-use risk map, oracle) with district-bootstrap CIs and
a paired test vs temperature, plus a budget-fraction sweep. Honest LODO OOF for
fitted models. Burden = DHIS2 cases; population = DHIS2 population.
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
df = (feat.merge(hd, on="location_id").merge(r3, on="location_id", how="left").merge(tgt.rename("inc"), on="location_id")
          .merge(g.assign(location_id=g.shapeID.astype(str))[["location_id", "parent"]], on="location_id"))
df = df[df["pop"].gt(0) & df["parent"].notna() & df["inc"].notna()].reset_index(drop=True)
y = df["inc"].values; grp = df["parent"].astype(str).values
logo = LeaveOneGroupOut()
def oof(cols):
    X = df[cols].values; p = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, grp): p[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return p
df["risk_temp"] = df["temp"]
df["risk_map"] = oof(["sig", "hab", "blt"])
df["risk_3d"] = df["risk_oof"].fillna(df["risk_oof"].mean())
df["oracle"] = df["cases"] / df["pop"]
pop = df["pop"].values; cas = df["cases"].values
def capture(scores, frac, P=pop, C=cas):
    o = np.argsort(-scores)
    cp = np.concatenate([[0], np.cumsum(P[o]) / P.sum()]); cc = np.concatenate([[0], np.cumsum(C[o]) / C.sum()])
    return np.interp(frac, cp, cc)

RANKS = [("temperature", "risk_temp"), ("env 3D surface S(T,NDVI,amp)", "risk_3d"),
         ("gridded risk map (T+land use)", "risk_map"), ("oracle (actual incidence)", "oracle")]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
print(f"n={len(df)}  total cases={cas.sum():,.0f}\n")
print("HEADLINE: burden (% of cases) captured by targeting top X% of population")
print(f"{'ranking':32}" + "".join(f"{int(f*100):>6}%" for f in FRACS))
for name, col in RANKS:
    print(f"{name:32}" + "".join(f"{capture(df[col].values, f):>7.3f}" for f in FRACS))
print(f"{'random':32}" + "".join(f"{f:>7.3f}" for f in FRACS))

# district bootstrap: CI of burden@50% + paired gap vs temperature
rng = np.random.RandomState(0); dists = df["parent"].unique()
idx_by = {d: np.where(grp == d)[0] for d in dists}
def boot(col, paired_ref=None, frac=0.5, reps=2000):
    vals = []
    for _ in range(reps):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idx_by[d] for d in ds])
        P = pop[idx]; C = cas[idx]
        v = capture(df[col].values[idx], frac, P, C)
        vals.append(v - capture(df[paired_ref].values[idx], frac, P, C) if paired_ref else v)
    return np.array(vals)
print("\nburden@50% pop with 95% CI (district bootstrap):")
for name, col in RANKS:
    b = boot(col); print(f"  {name:32}{b.mean():.3f}  [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
g_map = boot("risk_map", "risk_temp"); g_3d = boot("risk_3d", "risk_temp")
print(f"\npaired gap vs temperature @50% pop:")
print(f"  risk map - temp : {g_map.mean():+.3f}  [{np.percentile(g_map,2.5):+.3f}, {np.percentile(g_map,97.5):+.3f}]  P(>temp)={(g_map>0).mean():.3f}")
print(f"  3D surf - temp  : {g_3d.mean():+.3f}  [{np.percentile(g_3d,2.5):+.3f}, {np.percentile(g_3d,97.5):+.3f}]  P(>temp)={(g_3d>0).mean():.3f}")
gm3 = boot("risk_map", "risk_3d")
print(f"  risk map - 3D   : {gm3.mean():+.3f}  [{np.percentile(gm3,2.5):+.3f}, {np.percentile(gm3,97.5):+.3f}]  P(>3D)={(gm3>0).mean():.3f}")

fig, ax = plt.subplots(figsize=(7.5, 7))
for name, col, c in [("oracle", "oracle", "k"), ("gridded risk map (T+land use)", "risk_map", "C0"),
                     ("env 3D surface", "risk_3d", "C2"), ("temperature", "risk_temp", "C1")]:
    o = np.argsort(-df[col].values)
    cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cas[o]) / cas.sum()])
    ax.plot(cp, cc, color=c, label=f"{name}  (50%→{capture(df[col].values,.5):.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="random"); ax.axvline(.5, color="grey", lw=.7, ls=":")
ax.set(xlabel="population fraction targeted (ranked by risk)", ylabel="burden (cases) captured",
       title="HEADLINE METRIC — burden capture by risk-prioritised allocation")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal"); fig.tight_layout()
fig.savefig("results/headline_burden_capture.png", dpi=150, bbox_inches="tight")
df[["location_id", "parent", "pop", "cases", "inc", "risk_temp", "risk_3d", "risk_map", "oracle"]].to_csv("results/headline_burden_capture.csv", index=False)
print("\nwrote results/headline_burden_capture.{png,csv}")
