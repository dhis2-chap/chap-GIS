"""Choropleth of per-district out-of-fold ranking skill.

For each district (held out in LODO), compute Spearman between the model's OOF risk
(risk_oof from the 3D suitability surface) and observed incidence across that
district's sectors. Districts with <4 sectors are left grey (Spearman unreliable).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import chap_gis as cgis

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv")
risk["location_id"] = risk["location_id"].astype(str)

MIN_N = 4
rows = []
for d, sub in risk.groupby("district"):
    n = len(sub)
    rho = spearmanr(sub["risk_oof"], sub["annual_incidence_per1000"]).statistic if n >= MIN_N else np.nan
    rows.append((d, n, rho))
sk = pd.DataFrame(rows, columns=["district", "n_sectors", "rho"]).set_index("district")
pooled = spearmanr(risk["risk_oof"], risk["annual_incidence_per1000"]).statistic
valid = sk["rho"].dropna()
print(f"districts={len(sk)}  with>= {MIN_N} sectors={valid.shape[0]}", flush=True)
print(f"pooled OOF Spearman={pooled:.3f}   within-district mean={valid.mean():.3f}  median={valid.median():.3f}", flush=True)
print(f"within-district rho range: {valid.min():.2f} .. {valid.max():.2f}", flush=True)
sk.sort_values("rho").to_csv("results/district_oof_skill.csv")

# geometries: dissolve sectors -> districts
g = cgis.io.boundaries.load("RWA", level=5).copy()
g["location_id"] = g["shapeID"].astype(str)
g = g[g["location_id"].isin(risk["location_id"])]
g["geometry"] = g.geometry.make_valid()                 # fix side-location/self-intersection errors
dist = g.dissolve(by="parent")
dist = dist.join(sk, how="left")

fig, ax = plt.subplots(figsize=(10, 9))
dist.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)            # base (small/NaN districts)
dist.dropna(subset=["rho"]).plot(ax=ax, column="rho", cmap="RdYlGn", vmin=-1, vmax=1,
        edgecolor="white", linewidth=0.5, legend=True,
        legend_kwds={"label": "within-district OOF Spearman", "shrink": 0.6})
# annotate each district with its rho
for _, r in dist.iterrows():
    if pd.notna(r["rho"]):
        c = r.geometry.representative_point()
        ax.annotate(f"{r['rho']:.2f}", (c.x, c.y), ha="center", va="center", fontsize=6.5)
ax.set_title(f"Per-district out-of-fold ranking skill (3D surface)\n"
             f"pooled OOF Spearman = {pooled:.3f} ; within-district mean = {valid.mean():.3f}", fontsize=12)
ax.axis("off")
fig.tight_layout(); fig.savefig("results/district_oof_skill.png", dpi=150, bbox_inches="tight")
print("wrote results/district_oof_skill.png and results/district_oof_skill.csv", flush=True)
