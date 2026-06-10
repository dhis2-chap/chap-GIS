# Step (lever 1) — Spatial term

**Goal.** Add a spatial random field to capture autocorrelation — neighbouring
sectors share unobserved risk drivers (vector ecology, micro-habitat, health
systems) that the covariates miss. This is the standard geostatistical upgrade for
malaria risk mapping.

**Method.** Fit the covariate model (`sigmoid-temp + habitat + built`), then a
Gaussian process (RBF + white-noise kernel on sector centroids lon/lat) on the
**training residuals**, and add the GP prediction at the held-out sectors.
Leave-one-district-out, so the spatial term must interpolate a held-out district
from its *surrounding* districts — an honest test of transferable spatial
structure. Headline metric: burden captured in the top-X%-of-population.

**Results (LODO OOF, burden captured):**

| ranking | 30% pop | 50% pop | 70% pop |
|---|---|---|---|
| baseline | 0.488 | 0.788 | 0.934 |
| **+ spatial** | **0.506** | **0.792** | 0.935 |
| oracle | 0.683 | 0.880 | 0.977 |

Paired district-bootstrap @50%: **gap +0.014, 95% CI [−0.005, +0.035], P(>baseline) = 0.92.**

**Findings.**
- **Spatial is the only lever in this phase with a positive, near-reliable signal.**
  It adds the most at the *tighter* budgets (30–40% population), where getting the
  very top sectors right matters most: +0.018 at 30%.
- The gain is **modest and just under the reliability bar** (P=0.92, CI grazes 0)
  — expected under leave-one-*district*-out, which removes a held-out district's
  own near neighbours and so handicaps the spatial term. For a **deployed** map
  (fit on all sectors) the spatial smoothing would help more than this OOF number
  suggests.

**Verdict:** *the best static-map lever* — keep it. It captures residual spatial
structure the environmental covariates don't, and helps exactly where allocation
budgets are tightest.

Artifacts: `results/eval_static_levers.py` (GP on residuals), `static_risk_map.*`.
