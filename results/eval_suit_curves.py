"""Score arbitrary thermal-suitability curves against disease in milliseconds,
using the cached per-sector temperature-binned weights.

risk[sector,year] = (sum_bin S(T_bin) * W[sector,year,bin]) / sector_pop
Then Spearman vs disease over region-months (raw cases and incidence),
matching the metric used throughout the sweep analysis.
"""
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline

cache = np.load("results/suit_eval_cache.npz", allow_pickle=True)
W = cache["W"]                      # (NS, NY, NB)
SPOP = cache["sector_pop"]          # (NS, NY)
C = cache["bin_centers"]            # (NB,)
LOC = cache["location_ids"]
YEARS = cache["years"]

# ---- disease (monthly), broadcast risk from sector-year ----
dis = pd.read_csv("rwanda_spray.csv")[["location_id", "time", "disease", "population"]].copy()
dis["year"] = pd.to_datetime(dis["time"]).dt.year
dis = dis[dis["population"] > 0]

def gaussian(to, sg, tmn, tmx):
    return lambda c: np.where((c >= tmn) & (c <= tmx), np.exp(-((c - to) / sg) ** 2), 0.0)

def pchip(knots, hold_high=True):
    x = np.array([k[0] for k in knots]); y = np.array([k[1] for k in knots])
    f = PchipInterpolator(x, y)
    def S(c):
        s = f(np.clip(c, x[0], x[-1]))
        s = np.where(c < x[0], 0.0, s)
        if not hold_high:
            s = np.where(c > x[-1], 0.0, s)
        return np.clip(s, 0, 1)
    return S

def cubic(knots):
    x = np.array([k[0] for k in knots]); y = np.array([k[1] for k in knots])
    f = CubicSpline(x, y, bc_type="natural")
    return lambda c: np.clip(np.where((c >= x[0]) & (c <= x[-1]), f(c), 0.0), 0, 1)

CURVES = {
    "Gaussian OPTIMUM (29,s6,[19,38])": gaussian(29, 6, 19, 38),
    "Gaussian default (25,s5,[16,34])": gaussian(25, 5, 16, 34),
    "A monotone-saturating": pchip([(12,0),(16,.05),(20,.25),(24,.6),(28,.9),(32,1),(40,1)]),
    "B monotone-steep-low":  pchip([(14,0),(18,.1),(20,.25),(22,.5),(24,.8),(26,1),(34,1)]),
    "C peaked cubic":        cubic([(14,0),(20,.25),(26,.7),(30,1),(35,.6),(40,.1),(43,0)]),
    "D gentle sigmoid":      pchip([(10,0),(18,.15),(24,.5),(30,.85),(36,1),(44,1)]),
    "E linear-in-T (pure monotone)": pchip([(12,0),(28,1.0),(42,1.0)]),
}

def score(S):
    sv = S(C)                                   # (NB,)
    pe = W @ sv                                  # (NS, NY)
    risk = np.divide(pe, SPOP, out=np.zeros_like(pe), where=SPOP > 0)
    rdf = pd.DataFrame({
        "location_id": np.repeat(LOC, len(YEARS)),
        "year": np.tile(YEARS, len(LOC)),
        "risk": risk.ravel()})
    m = dis.merge(rdf, on=["location_id", "year"], how="inner")
    raw = m["risk"].corr(m["disease"], method="spearman")
    inc = m["risk"].corr(m["disease"] / m["population"], method="spearman")
    return raw, inc, len(m)

rows = []
for name, S in CURVES.items():
    raw, inc, n = score(S)
    rows.append((name, round(raw, 4), round(inc, 4)))
res = pd.DataFrame(rows, columns=["curve", "rho_raw", "rho_inc"]).sort_values("rho_inc", ascending=False)
print(res.to_string(index=False))
print(f"\n(region-months matched: {score(CURVES['B monotone-steep-low'])[2]})")
print("Anchor target from sweep: Gaussian OPTIMUM ~ rho_raw 0.5524 / rho_inc 0.4928")
