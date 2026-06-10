"""Screen covariates for WITHIN-district signal beyond temperature.

To beat temperature on within-district ranking we need a feature whose
within-district variation tracks within-district incidence AFTER removing
temperature. For each candidate: within-district-demean (subtract district mean),
residualize both incidence and the feature on demeaned temperature, then take the
Spearman of the residuals = within-district partial rank correlation | temp.
Also report the candidate's standalone within-district concordance.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv")
risk["location_id"] = risk["location_id"].astype(str)
terr = pd.read_csv("results/rwanda_sector_terrain.csv"); terr["location_id"] = terr["location_id"].astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv"); mo["location_id"] = mo["location_id"].astype(str)
df = risk.merge(terr, on="location_id", how="left").merge(
     mo.drop(columns=[c for c in mo.columns if c in risk.columns and c != "location_id"]),
     on="location_id", how="left")

# sector area -> population density
gdf = prepare_boundaries("RWA", 5)[["location_id", "geometry"]].copy()
gdf["location_id"] = gdf["location_id"].astype(str)
gdf["area_km2"] = gdf.to_crs(32735).geometry.area / 1e6
df = df.merge(gdf[["location_id", "area_km2"]], on="location_id", how="left")
df["pop_density"] = df["population"] / df["area_km2"].replace(0, np.nan)
df["log_popdens"] = np.log1p(df["pop_density"])

y = df["annual_incidence_per1000"].values
g = df["district"].astype(str).values
temp = df["temp_pop"].values

def demean(v, g):
    v = v.astype(float).copy(); out = v.copy()
    for d in np.unique(g):
        m = g == d; out[m] = v[m] - np.nanmean(v[m])
    return out

def resid_on(a_dm, t_dm):
    ok = np.isfinite(a_dm) & np.isfinite(t_dm)
    b = np.polyfit(t_dm[ok], a_dm[ok], 1)
    r = a_dm.copy(); r[ok] = a_dm[ok] - (b[0] * t_dm[ok] + b[1]); return r

def concordance(p):
    C = Dd = 0.0
    for d in np.unique(g):
        i = np.where(g == d)[0]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                if not (np.isfinite(p[i[a]]) and np.isfinite(p[i[b]])): continue
                dy = y[i[a]] - y[i[b]]
                if dy == 0: continue
                dp = p[i[a]] - p[i[b]]
                if dp == 0: C += .5; Dd += .5
                elif np.sign(dp) == np.sign(dy): C += 1
                else: Dd += 1
    return C / (C + Dd) if (C + Dd) else np.nan

y_dm, t_dm = demean(y, g), demean(temp, g)
y_res = resid_on(y_dm, t_dm)                       # incidence with temp removed (within district)

CANDS = ["temp_pop", "ndvi_ann", "ndvi_amp", "evi_ann", "ndvi_drop", "ndvi_ratio",
         "elevation_m", "rice_frac", "wetland_frac", "water_frac",
         "population", "pop_density", "log_popdens"]
print(f"sectors={len(df)}  districts={pd.Series(g).nunique()}")
print(f"\n{'feature':14}{'standalone concord':>20}{'within-r |temp':>16}{'sign':>6}")
rows = []
for c in CANDS:
    v = df[c].values.astype(float)
    f_dm = demean(v, g)
    f_res = resid_on(f_dm, t_dm)
    ok = np.isfinite(f_res) & np.isfinite(y_res)
    pr = spearmanr(f_res[ok], y_res[ok]).statistic
    con = concordance(np.where(np.isfinite(v), v, np.nanmean(v)))
    rows.append((c, con, pr))
    print(f"{c:14}{con:>20.3f}{pr:>16.3f}{('+' if pr>=0 else '-'):>6}")
pd.DataFrame(rows, columns=["feature", "standalone_concordance", "within_partial_r_given_temp"])\
  .to_csv("results/local_signal_screen.csv", index=False)
print("\n(within-r |temp = within-district partial Spearman with incidence after removing temperature)")
print("ref: temperature standalone concordance ~0.684")
