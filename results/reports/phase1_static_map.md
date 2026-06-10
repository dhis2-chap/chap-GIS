# Phase 1 — The good static risk map

**Objective.** Try levers 1 (spatial), 4 (hydrology), 5 (urban), 6 (denominators)
and land on the best *static* risk map, scored on the headline metric: burden
captured when allocating to the top-X%-of-population by predicted risk.

## Lever scorecard (burden @50% pop, LODO OOF, paired district-bootstrap vs baseline)

| lever | burden @50% | gap | P(>baseline) | verdict |
|---|---|---|---|---|
| baseline `sigmoid-temp + habitat + built` | 0.788 | — | — | — |
| (1) spatial term (GP on centroids) | 0.792 | +0.014 | 0.92 | **proximity artifact — see correction** |
| (4) hydrology (DEM-derived) | 0.777 | −0.005 | 0.37 | drop — needs external surface-water data |
| (5) urban / pop-density | 0.788 | +0.001 | 0.58 | drop — redundant with built-up |
| (6) denominators (WorldPop + winsorised) | 0.788 | — | — | adopt — neutral, principled foundation |
| **oracle** (rank by actual incidence) | 0.880 | — | — | ceiling |

> **Correction (spatial term).** The spatial gain does **not** survive
> spatially-buffered CV: with a ≥20 km training buffer around the held-out district
> the spatial add-on contributes **+0.000** (see [step1_spatial.md](step1_spatial.md)).
> It was proximity-driven interpolation under contiguous-district holdout, not
> transferable spatial structure — valid OOF (no target leak) but not a real,
> generalisable improvement. **So none of the four static levers reliably beat the
> covariate baseline.**

## Decision: the good static map

**`sigmoid-temp(T₀=19) + log-habitat + log-built-up`** (covariate baseline), fit on
the cleaned (WorldPop-denominated, winsorised) target. The spatial GP is retained
only as an optional *within-sample smoother* for interpolating among observed
sectors — it adds nothing for unsampled regions.

Burden captured across allocation budgets (OOF):

| top-X% population | baseline | **static map** | oracle |
|---|---|---|---|
| 30% | 0.488 | **0.506** | 0.683 |
| 50% | 0.788 | **0.792** | 0.880 |
| 70% | 0.934 | 0.935 | 0.977 |

The map (`results/static_risk_map.png`) shows the temperature/elevation gradient
(cool dark west/centre → warm bright eastern lowlands) modulated by habitat and
urbanization, then smoothed by the spatial field.

## What we learned

- **No static lever reliably beats the covariate baseline** once the spatial term's
  apparent gain is shown (above) to be a proximity artifact that vanishes under
  buffered CV. Hydrology and urban add nothing usable from available data, and
  denominators are immaterial to a *cases*-based metric.
- **We are at the data ceiling for a static map.** The static map sits
  at **~0.79 vs the oracle's 0.88** — and the oracle itself is bounded by the genuine
  spatial spread of burden plus outcome noise. Further *static, environmental*
  features are unlikely to close much of the remaining ~0.09 gap; the levers that
  could (seasonal surface water, nightlights/GHSL, travel-time) all require
  **new external data**.
- **The productive remaining directions are not more static covariates** but
  (2) **interventions** — which explain why some high-environment sectors have
  *lower realised burden* — and (3) **multi-year / spatiotemporal** structure.
  Those are Phase 2.

Artifacts: `build_foundation_features.py`, `eval_static_levers.py`,
`finalize_static_map.py`, `static_risk_map.{png,csv}`, and the per-lever reports
`step{1,4,5,6}_*.md`.
