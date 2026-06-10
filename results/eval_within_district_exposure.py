"""Within-district ranking skill of the PREVIOUS-generation exposure model
(breeding-site / distance-decay / elevation / thermal-suitability), across all
swept parameterizations.

Each sweep CSV has one mean_exposure_per_person__expo_NNN column per parameter
combo (monthly long format). We aggregate to a per-sector spatial predictor (mean
over time), then score it with the SAME within-district metric and the SAME
incidence target + districts used for the new methods (so it is directly
comparable to control=0.513, 1D S(temp)=0.487, 3D surface=0.424).

The exposure index is a fixed, hand-built predictor (not fit to disease), so
within-district Spearman needs no cross-validation -- just like the temperature
control.
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

ref = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv")
ref["location_id"] = ref["location_id"].astype(str)
tgt = ref.set_index("location_id")["annual_incidence_per1000"]
par = ref.set_index("location_id")["district"]

def concordance(pred, y, grp):
    d = pd.DataFrame({"p": pred, "y": y, "g": grp}).dropna()
    C = Dd = 0.0
    for _, s in d.groupby("g"):
        pv = s.p.values; yv = s.y.values
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                dy = yv[a] - yv[b]
                if dy == 0: continue
                dp = pv[a] - pv[b]
                if dp == 0: C += 0.5; Dd += 0.5
                elif np.sign(dp) == np.sign(dy): C += 1
                else: Dd += 1
    return C / (C + Dd) if (C + Dd) else np.nan

def within(pred, y, grp, min_n=4):
    d = pd.DataFrame({"p": pred, "y": y, "g": grp}).dropna()
    rs = [spearmanr(s.p, s.y).statistic for _, s in d.groupby("g") if len(s) >= min_n]
    rs = np.array([r for r in rs if np.isfinite(r)])
    return concordance(pred, y, grp), rs.mean(), spearmanr(d.p, d.y).statistic, len(rs)

rows = []
for tag, fname in [("coarse", "rwanda_sweep"), ("refined", "rwanda_sweep_refined"), ("temp", "rwanda_sweep_temp")]:
    params = json.load(open(f"results/{fname}.params.json"))["columns"]
    df = pd.read_csv(f"results/{fname}.csv"); df["location_id"] = df["location_id"].astype(str)
    expo_cols = [c for c in df.columns if c.startswith("mean_exposure_per_person__")]
    sec = df.groupby("location_id")[expo_cols].mean()        # per-sector mean over time
    y = tgt.reindex(sec.index).values; grp = par.reindex(sec.index).values
    for col in expo_cols:
        cid = col.split("__")[1]; p = params[cid]
        c, m, pl, n = within(sec[col].values, y, grp)
        rows.append((tag, cid, p["lambda_m"], p["gamma_m"], p["t_opt"], p["sigma"],
                     p.get("water_edge_buffer_pixels"), c, m, pl, n))

res = pd.DataFrame(rows, columns=["sweep", "combo", "lambda_m", "gamma_m", "t_opt", "sigma",
                                  "water_buf", "concordance", "within_mean", "pooled", "n_distr"])
res = res.sort_values("concordance", ascending=False).reset_index(drop=True)
res.to_csv("results/within_district_exposure_sweeps.csv", index=False)
print(f"sectors scored: {sec.index.isin(tgt.index).sum()} ; parameterizations: {len(res)}\n")
with pd.option_context("display.width", 200, "display.max_rows", None):
    print(res.round(3).to_string(index=False))

print("\n=== marginal of t_opt on concordance (best lambda/gamma per t_opt) ===")
best_by_topt = res.groupby("t_opt")["concordance"].max()
for t, c in best_by_topt.items():
    edge = "  <- grid edge" if t in (res.t_opt.min(), res.t_opt.max()) else ""
    print(f"  t_opt={t:>4}: best concordance={c:.3f}{edge}")
print("\nrefs (within-district CONCORDANCE): mean-temperature control 0.684 ; "
      "old exposure(t29) 0.671 ; 3D surface 0.638 ; random 0.503")
