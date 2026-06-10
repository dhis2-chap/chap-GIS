"""Compare within-district ranking METRICS (not models).

Old metric:   mean over districts of per-district Spearman (and Fisher-z mean).
Better A:     stratified within-district concordance (C-index over same-district
              sector pairs); 0.5=random, 1=perfect. Pools all comparable pairs.
Better B:     within-demeaned pooled rank correlation (rank within district, centre,
              pool, correlate). One high-power number.

We score a few predictors and bootstrap over DISTRICTS (resample districts with
replacement, recompute) to show each metric's sampling noise. A good metric is
both low-variance and discriminating.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

base = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv")
base["location_id"] = base["location_id"].astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv")[["location_id", "temp_pop"]]
mo["location_id"] = mo["location_id"].astype(str)
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw["location_id"].astype(str)
expo = sw.groupby("location_id")["mean_exposure_per_person__expo_000"].mean().rename("exposure")  # lam1500 g100 topt29
df = base.merge(mo, on="location_id").merge(expo, on="location_id")
y = df["annual_incidence_per1000"].values
g = df["district"].astype(str).values
rng = np.random.RandomState(0)

def mean_spearman(p, y, g, min_n=4):
    rs = [spearmanr(p[g == d], y[g == d]).statistic for d in np.unique(g)
          if (g == d).sum() >= min_n and np.ptp(p[g == d]) > 0]
    rs = np.array([r for r in rs if np.isfinite(r)])
    return rs.mean()

def fisher_spearman(p, y, g, min_n=4):
    rs = [spearmanr(p[g == d], y[g == d]).statistic for d in np.unique(g)
          if (g == d).sum() >= min_n and np.ptp(p[g == d]) > 0]
    rs = np.array([r for r in rs if np.isfinite(r)])
    return np.tanh(np.mean(np.arctanh(np.clip(rs, -.999, .999))))

def strat_concordance(p, y, g):
    C = Dd = 0.0
    for d in np.unique(g):
        i = np.where(g == d)[0]
        if len(i) < 2: continue
        pp, yy = p[i], y[i]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                dy = yy[a] - yy[b]
                if dy == 0: continue
                dp = pp[a] - pp[b]
                if dp == 0: C += 0.5; Dd += 0.5
                elif np.sign(dp) == np.sign(dy): C += 1
                else: Dd += 1
    return C / (C + Dd) if (C + Dd) else np.nan          # concordance prob in [0,1]

def within_pooled(p, y, g):
    d = pd.DataFrame({"p": p, "y": y, "g": g})
    rp = d.groupby("g")["p"].transform(lambda s: rankdata(s) - rankdata(s).mean())
    ry = d.groupby("g")["y"].transform(lambda s: rankdata(s) - rankdata(s).mean())
    m = (rp.std() > 0) & (ry.std() > 0)
    return np.corrcoef(rp, ry)[0, 1]

METRICS = {"mean-Spearman": mean_spearman, "Fisher-Spearman": fisher_spearman,
           "concordance": strat_concordance, "within-pooled-r": within_pooled}
PREDS = {"mean temperature": df["temp_pop"].values,
         "3D surface (OOF)": df["risk_oof"].values,
         "old exposure (t29)": df["exposure"].values,
         "random noise": rng.rand(len(y))}

districts = np.unique(g)
def boot_se(fn, p):
    vals = []
    for _ in range(400):
        ds = rng.choice(districts, len(districts), replace=True)
        idx = np.concatenate([np.where(g == d)[0] for d in ds])
        # relabel groups so resampled duplicate districts stay separate
        gg = np.concatenate([np.full((g == d).sum(), k) for k, d in enumerate(ds)])
        vals.append(fn(p[idx], y[idx], gg))
    return np.nanstd(vals)

print(f"{'predictor':20}" + "".join(f"{m:>18}" for m in METRICS))
for pname, p in PREDS.items():
    cells = []
    for mname, fn in METRICS.items():
        v = fn(p, y, g); se = boot_se(fn, p)
        cells.append(f"{v:.3f}+-{se:.3f}")
    print(f"{pname:20}" + "".join(f"{c:>18}" for c in cells))
print("\nconcordance/within-pooled-r: 0.5 / 0.0 = random.  '+-' is district-bootstrap SD (noise).")
