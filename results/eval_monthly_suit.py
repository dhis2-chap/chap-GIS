"""Score monthly->yearly suitability aggregations x year-definitions vs disease.

Aggregations (per sector, per year-window), for a curve S:
  mean   : mean over the window's months of (sum_bin S(c) * W[:,month,:])
  sum    : sum over the window's months
  count  : 'suitable-month count' = mean with a step curve (handled via S itself)
  maxcal : per-pixel warmest-month over Jan-Dec  (Wmax_cal); monotone S only
  maxseas: per-pixel warmest-month over Sep-Aug  (Wmax_seas); monotone S only

Year definitions:
  calendar : Jan-Dec
  season   : Sep(y)-Aug(y+1)  (start at the August case trough)

Predictor risk = aggregated_pe / sector_pop. Scored by Spearman vs disease
incidence (cases/pop) over sector x year-window points.
"""
import sys
import numpy as np
import pandas as pd

c = np.load("results/monthly_suit_cache.npz", allow_pickle=True)
W = c["W"]                       # (NS,108,NB) pop*base binned by monthly T_nearest
SPOP = c["sector_pop"]           # (NS,9)
WMAXC = c["Wmax_cal"]            # (NS,9,NB)
WMAXS = c["Wmax_seas"]           # (NS,8,NB)
CEN = c["bin_centers"]
LOC = c["location_ids"]
MONTHS = c["months"]             # (108,2) -> (year,month)
YEARS = list(c["years"])
SEAS = list(c["season_starts"])  # start years of Sep-Aug windows
mi_of = {(int(y), int(m)): i for i, (y, m) in enumerate(MONTHS)}
yi_of = {int(y): i for i, y in enumerate(YEARS)}

dis = pd.read_csv("rwanda_spray.csv")[["location_id", "time", "disease", "population"]].copy()
dis["y"] = pd.to_datetime(dis["time"]).dt.year
dis["m"] = pd.to_datetime(dis["time"]).dt.month
dis = dis[dis.population > 0]

def gaussian(to, sg, a, b): return lambda x: np.where((x >= a) & (x <= b), np.exp(-((x - to) / sg) ** 2), 0.0)
def logistic(T0, k):        return lambda x: 1 / (1 + np.exp(-k * (x - T0)))

def windows(yeardef):
    if yeardef == "calendar":
        return [(y, [(y, m) for m in range(1, 13)], (yi_of[y], yi_of[y])) for y in YEARS]
    return [(sy, [(sy, m) for m in range(9, 13)] + [(sy + 1, m) for m in range(1, 9)],
             (yi_of[sy], yi_of[sy + 1])) for sy in SEAS]

# disease year keys
dis["key_cal"] = dis["y"]
# season key: months 9-12 belong to that year's season; months 1-8 belong to prev year's season
dis["key_seas"] = np.where(dis["m"] >= 9, dis["y"], dis["y"] - 1)

def score(S, agg, yeardef):
    Sv = S(CEN)
    rows = []
    if agg in ("maxcal", "maxseas"):
        store = WMAXC if agg == "maxcal" else WMAXS
        keys = YEARS if agg == "maxcal" else SEAS
        for j, key in enumerate(keys):
            yi = yi_of[key] if agg == "maxcal" else yi_of[key]  # normalize by start-year pop
            pe = store[:, j, :] @ Sv
            pop = SPOP[:, yi]
            risk = np.divide(pe, pop, out=np.zeros_like(pe), where=pop > 0)
            for s in range(len(LOC)):
                rows.append((LOC[s], key, risk[s]))
    else:
        for key, mlist, (yi0, yi1) in windows(yeardef):
            idx = [mi_of[(yy, mm)] for (yy, mm) in mlist]
            per_month = W[:, idx, :] @ Sv          # (NS, nmonths)
            pe = per_month.mean(1) if agg == "mean" else per_month.sum(1)
            pop = (SPOP[:, yi0] + SPOP[:, yi1]) / 2
            risk = np.divide(pe, pop, out=np.zeros_like(pe), where=pop > 0)
            for s in range(len(LOC)):
                rows.append((LOC[s], key, risk[s]))
    rdf = pd.DataFrame(rows, columns=["location_id", "key", "risk"])

    keycol = "key_cal" if (yeardef == "calendar" or agg == "maxcal") else "key_seas"
    dd = (dis.groupby(["location_id", keycol])
             .agg(disease=("disease", "sum"), pop=("population", "mean")).reset_index()
             .rename(columns={keycol: "key"}))
    m = rdf.merge(dd, on=["location_id", "key"], how="inner")
    m = m[m["pop"] > 0]
    inc = m["disease"] / m["pop"]
    return (m["risk"].corr(inc, method="spearman"),
            m["risk"].corr(m["disease"], method="spearman"), len(m))

CURVES = {
    "logistic(23,k3)": logistic(23, 3.0),
    "Gaussian(29,s6)": gaussian(29, 6, 19, 38),
}
COMBOS = [("mean", "calendar"), ("mean", "season"),
          ("sum", "calendar"), ("sum", "season"),
          ("maxcal", "calendar"), ("maxseas", "season")]

print(f"{'curve':16}{'agg':9}{'yeardef':10}{'rho_inc':>9}{'rho_raw':>9}{'n':>7}")
res = []
for cname, S in CURVES.items():
    for agg, yd in COMBOS:
        ri, rr, n = score(S, agg, yd)
        res.append((cname, agg, yd, ri, rr, n))
        print(f"{cname:16}{agg:9}{yd:10}{ri:>9.4f}{rr:>9.4f}{n:>7}")
print("\nBest by rho_inc:")
for r in sorted(res, key=lambda x: -x[3])[:5]:
    print(f"  {r[0]:16} {r[1]:8} {r[2]:9} rho_inc={r[3]:.4f}")
