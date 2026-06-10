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

### Correction — the gain does not survive spatially-buffered CV

The setup is valid OOF (no target leakage: the GP is fit only on training-district
residuals and merely *predicts* at held-out coordinates). **But leave-one-district-
out leaves a held-out district's immediate neighbours in training, and a GP exploits
that proximity.** Re-running with a training buffer around the held-out district:

| training buffer | baseline | +spatial | spatial gain | P(spatial>base) |
|---|---|---|---|---|
| 0 km (standard LODO) | 0.788 | 0.794 | +0.006 | 0.94 |
| 10 km | 0.765 | 0.772 | +0.008 | 0.73 |
| 20 km | 0.762 | 0.762 | **+0.000** | 0.01 |
| 30 km | 0.743 | 0.743 | **+0.000** | 0.00 |

The gain vanishes once neighbours are buffered out (≥20 km). So the spatial signal is
**proximity-driven interpolation, not transferable spatial structure.**

**Verdict (revised):** the spatial term helps only for **interpolating among observed
sectors**; it adds nothing for **generalising to unsampled regions**, and since
Rwanda's sectors are all observed it isn't needed for a deployed map. **Do not count
it as a real static-map improvement.** The honest static map is the covariate
baseline alone.

Artifacts: `results/eval_static_levers.py` (GP on residuals),
`spatial_cv_buffer.py` (buffered-CV test), `static_risk_map.*`.
