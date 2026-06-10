"""Finalize the good static risk map = sigmoid-temp + habitat + built + spatial
term. Render the sector choropleth, burden curve, and report numbers.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries
z = np.load("results/_static_levers.npz")
df = pd.read_csv("results/foundation_features.csv"); df["location_id"] = df.location_id.astype(str)
pop = z["pop"]; cases = z["cases"]; sp_oof = z["spatial"]; base_oof = z["base"]; oracle = z["oracle"]
def cap(sc, frac): o = np.argsort(-sc); cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cases[o]) / cases.sum()]); return np.interp(frac, cp, cc)
FR = [.2, .3, .4, .5, .6, .7]
print("burden captured (OOF):")
print(f"{'frac':>6}" + "".join(f"{int(f*100):>7}%" for f in FR))
for nm, sc in [("baseline", base_oof), ("static map (+spatial)", sp_oof), ("oracle", oracle)]:
    print(f"{nm:22}" + "".join(f"{cap(sc, f):>8.3f}" for f in FR))

# full-data fit for the deployed map
y = df["inc_wp_w"].values; X = df[["sig_temp", "hab", "built"]].values; co = df[["lon", "lat"]].values
lin = LinearRegression().fit(X, y); res = y - lin.predict(X)
k = C(np.var(res)) * RBF(0.15, (0.03, 1.0)) + WhiteKernel(np.var(res))
gp = GaussianProcessRegressor(kernel=k, alpha=1e-6).fit(co, res)
df["risk_static"] = lin.predict(X) + gp.predict(co)
df[["location_id", "parent", "wp_pop", "cases", "risk_static"]].to_csv("results/static_risk_map.csv", index=False)

gdf = prepare_boundaries("RWA", 5)[["location_id", "geometry"]].copy(); gdf["location_id"] = gdf.location_id.astype(str)
gdf["geometry"] = gdf.geometry.make_valid()
m = gdf.merge(df[["location_id", "risk_static"]], on="location_id")
fig, ax = plt.subplots(figsize=(9, 8)); vlo, vhi = np.nanpercentile(m["risk_static"], [2, 98])
m.plot(ax=ax, column="risk_static", cmap="inferno", vmin=vlo, vmax=vhi, edgecolor="white", linewidth=.2,
       legend=True, legend_kwds={"label": "predicted risk (cleaned incidence)", "shrink": .6})
ax.set_title(f"Good static risk map: sigmoid-temp + habitat + built + spatial\nburden@50%pop = {cap(sp_oof,.5):.3f} (baseline {cap(base_oof,.5):.3f}, oracle {cap(oracle,.5):.3f})", fontsize=11)
ax.axis("off"); fig.tight_layout(); fig.savefig("results/static_risk_map.png", dpi=150, bbox_inches="tight")
print("\nwrote results/static_risk_map.{png,csv}")
