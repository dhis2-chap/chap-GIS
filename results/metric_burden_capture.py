"""Targeting metric: rank sectors by predicted risk, allocate to the top sectors
until they cover 50% of the population; report the share of total malaria BURDEN
(cases) captured in that allocated half.

random = 0.50 ; oracle (rank by actual incidence) = ceiling. Compare temperature
vs the gridded risk map. Uses honest LODO out-of-fold predictions for the models.
Burden = DHIS2 cases; population = DHIS2 population (health data). Reported at
50% and across the concentration curve; sensitivity excluding >1000/1000 artifacts.
"""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

# health data: cases + population per sector
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw.location_id.astype(str)
hd = sw.groupby("location_id").agg(cases=("disease", "sum"), pop=("population", "mean")).reset_index()

# model features (gridded, 0.5 km calibrated map) + sector aggregation
z = np.load("results/_gridded_arrays.npz", allow_pickle=True)
temp, fhL, fbL, popr, sect = z["temp"], z["fhL"], z["fbL"], z["pop"], z["sect"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(popr)
s = sect[ok]; w = popr[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
feat = pd.DataFrame({"location_id": loc, "temp": sm(temp), "hab": sm(fhL), "blt": sm(fbL), "sig": sm(expit((temp - 19.0) / 0.5))})

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
df = (feat.merge(hd, on="location_id").merge(tgt.rename("inc"), on="location_id")
          .merge(g.assign(location_id=g.shapeID.astype(str))[["location_id", "parent"]], on="location_id"))
df = df[df["pop"].gt(0) & df["cases"].ge(0) & df["parent"].notna()].reset_index(drop=True)
y = df["inc"].values; grp = df["parent"].astype(str).values
logo = LeaveOneGroupOut()
def oof(cols):
    X = df[cols].values; p = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, grp): p[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return p
df["risk_temp"] = oof(["temp"])
df["risk_map"] = oof(["sig", "hab", "blt"])     # calibrated gridded risk map
df["oracle"] = df["cases"] / df["pop"]            # rank by actual incidence (ceiling)

def capture(scores, pop, cases, frac=0.5):
    o = np.argsort(-scores)
    cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()])
    cc = np.concatenate([[0], np.cumsum(cases[o]) / cases.sum()])
    return np.interp(frac, cp, cc)

def report(d, label):
    pop = d["pop"].values; cas = d["cases"].values
    print(f"\n[{label}]  n={len(d)}  total cases={cas.sum():,.0f}  total pop={pop.sum():,.0f}")
    print(f"  {'ranking':16}{'burden@50%pop':>14}{'@30%':>8}{'@70%':>8}")
    rows = {}
    for col, name in [("risk_temp", "temperature"), ("risk_map", "risk map (T+land use)"),
                      ("oracle", "oracle (actual inc)")]:
        b50 = capture(d[col].values, pop, cas, .5); b30 = capture(d[col].values, pop, cas, .3); b70 = capture(d[col].values, pop, cas, .7)
        rows[col] = (b50, b30, b70); print(f"  {name:16}{b50:>14.3f}{b30:>8.3f}{b70:>8.3f}")
    print(f"  {'random':16}{0.5:>14.3f}{0.3:>8.3f}{0.7:>8.3f}")
    o50, t50, m50 = rows['oracle'][0], rows['risk_temp'][0], rows['risk_map'][0]
    print(f"  efficiency (capture-0.5)/(oracle-0.5):  temp={ (t50-.5)/(o50-.5):.2f}  risk map={ (m50-.5)/(o50-.5):.2f}")
    return rows

full = report(df, "all sectors")
clean = report(df[df.inc <= 1000].reset_index(drop=True), "excl >1000/1000 artifacts")

# concentration curves (full)
fig, ax = plt.subplots(figsize=(7.5, 7))
pop = df["pop"].values; cas = df["cases"].values
for col, name, c in [("oracle", "oracle (actual incidence)", "k"), ("risk_map", "risk map (T+land use)", "C0"),
                     ("risk_temp", "temperature", "C1")]:
    o = np.argsort(-df[col].values)
    cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cas[o]) / cas.sum()])
    ax.plot(cp, cc, label=f"{name}  (50%pop→{capture(df[col].values,pop,cas):.2f})", color=c)
ax.plot([0, 1], [0, 1], "--", color="grey", label="random (0.50)")
ax.axvline(.5, color="grey", lw=.7, ls=":")
ax.set(xlabel="cumulative population fraction (ranked by risk)", ylabel="cumulative burden (cases) captured",
       title="Burden concentration: targeting by predicted risk")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal"); fig.tight_layout()
fig.savefig("results/burden_capture.png", dpi=150, bbox_inches="tight")
df[["location_id", "parent", "pop", "cases", "inc", "risk_temp", "risk_map", "oracle"]].to_csv("results/burden_capture.csv", index=False)
print("\nwrote results/burden_capture.{png,csv}")
