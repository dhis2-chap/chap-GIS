# Rwanda risk-map work — handoff / state report

*Written 2026-06-11. Self-contained summary so the work can resume after a context
reset. Branch `mean-exposure-per-person`. All analysis lives in `results/`; per-step
detail in `results/reports/`. Read `results/INVESTIGATION_REPORT.md` §0 for the
headline metric and `results/reports/SUMMARY.md` for the improvement program.*

---

## 1. What this project is

Building and evaluating a malaria **risk map for Rwanda** at admin **sector** level
(~404 sectors, DHIS2 org-units), from environmental rasters + routine case data
(2013–2021). Started from a hand-built "mosquito-exposure" index; evolved into a
covariate regression with a decision-relevant evaluation metric.

## 2. THE HEADLINE METRIC — burden capture

**"Rank sectors by predicted risk; allocate to the top sectors until they cover
X% of the population; report the share of total malaria burden (cases) captured."**
Operationally: *target X% of people → reach Y% of cases.* Random = X%; oracle (rank
by actual incidence) = ceiling. Reported at 50% population, LODO out-of-fold, with
**paired district-bootstrap** CIs. Code: `results/evaluate_headline.py`,
`metric_burden_capture.py`. Rank-correlation / within-district **concordance**
(stratified C-index over same-district sector pairs) is the secondary metric.

**Burden @50% population (the key scoreboard):**

| ranking | burden@50% | note |
|---|---|---|
| random | 0.50 | — |
| temperature only | ~0.65 | the real baseline |
| **old breeding-site exposure model (any params)** | **0.62–0.66** | = temperature; adds nothing (P 0.25–0.58) |
| env 3D surface `S(temp,NDVI,amp)` | 0.77 | beats temp, P≈0.99 |
| **gridded temp+land-use risk map** | **0.79** | beats temp +0.11, P≈0.99 |
| oracle (actual incidence) | 0.88 | ceiling |

Key result: the **environmental risk maps beat temperature by ~+0.11 burden@50%
(P≈0.99)**, roughly *doubling* targeting efficiency; the **old exposure model does
not beat temperature** in any parameterization (it is a temperature index).

## 3. CURRENT BEST DELIVERABLE — per-capita risk CSV

`results/sector_percapita_risk.csv` (404 sectors, full-dataset fit).
**WITHIN-DISTRICT formulation** (mechanistically-correct coefficients), built by:
`results/export_percapita_risk.py` (UNCOMMITTED as of writing — commit it).

- within-district slopes `[sigmoid_temp +3861, habitat +6670, built −3679]`
  (habitat **+**, urban **−** — correct), estimated district-demeaned;
- **+ empirical-Bayes district intercept** (observed district burden, mean
  shrinkage 0.84) to supply the between-district level.

Columns: `risk_per1000_yr` (PRIMARY: within slopes + EB district intercept —
best estimate for Rwanda-as-observed), `risk_env_score` (pure transferable
environmental contribution, correct signs), `risk_pooled_per1000_yr` (reference,
habitat sign flips), `observed_incidence_per1000_yr`, `risk_normalized`, `risk_rank`.

**Caveat:** the absolute level comes partly from observed district burden (EB
intercept) → not a pure environmental prediction, does NOT transfer to unobserved
districts. Use `risk_env_score` for the pure/transferable object.

## 4. Model components (definitions)

- **Temperature term:** `sigmoid((T−19)/k)` per pixel. A sigmoid does NOT improve
  *ranking* (rank-invariant to monotone transforms) but reliably improves
  *calibration* (LODO R² 0.355→0.41, P=0.95); T₀≈19 °C is a clean interior optimum.
  Lapse-rate downscaling the temperature changed nothing (washes out under
  pop-weighted sector aggregation; CHELSA already orographically downscaled).
- **Land use:** `log1p(focal rice∪wetland)` (habitat) and `log1p(focal built-up)`
  (urban), focal radius ~0.5–3 km. The only covariates beyond temperature that add
  signal; built-up ≈ urbanization (pop-density redundant with it).
- **Gridded version:** 100 m grid (`build_gridded_*`, `gridded_*` rasters); nominal
  100 m, effective ~1 km (CHELSA temp) / 0.5–3 km (land-use focal); validated at
  sector level. Linear + per-pixel transform → re-aggregates exactly to sectors.
- **Foundation feature table:** `results/foundation_features.csv` (built by
  `build_foundation_features.py`) — all sector covariates + cleaned target
  `inc_wp_w` (WorldPop-denominated, winsorised) + `parent`(district) + centroids.

## 5. Findings ledger — what helped vs what didn't

**Helps:** sigmoid-temp (calibration only); habitat + built-up land use
(within-district & burden-capture); the within-district formulation (correct signs).

**Saturated / no value:**
- Pop-weighting rasters; temperature bands; elevation bands (redundant w/ temp).
- Lapse-rate temperature downscaling (washes out).
- DEM-derived hydrology (needs external seasonal-surface-water data).
- Urban/pop-density beyond built-up (redundant).
- Denominator cleaning (neutral on cases-based metric; adopted as foundation).
- Feature interactions incl. temp×habitat (hurts — Rwanda mostly above threshold).
- Nonlinear learners (RF/GBM) — help pooled, hurt within-district (overfit).
- Old breeding-site exposure machinery (= temperature).
- **Spatial GP term — apparent +0.014 gain is a PROXIMITY ARTIFACT**: valid OOF
  (no target leak) but vanishes (+0.000) under ≥20 km spatially-buffered CV
  (`spatial_cv_buffer.py`). Keep only as a within-sample smoother.

**Two genuinely productive directions (Phase 2):**
- **Interventions (bednets/IRS):** all positively correlate with burden (targeting
  confound). LLIN per-capita +0.019 (P=0.92); IRS hurts OOF. **Circular for an
  allocation map — excluded.** (`eval_lever2_interventions.py`)
- **Spatiotemporal / seasonality:** the static map is seasonally blind (temperature
  aseasonal). Lagged **rainfall** recovers +0.30 within-year seasonal skill OOF
  (~74% of climatology ceiling) vs ~0 for temperature. A *separate* product
  (early-warning). HIGHEST-VALUE next direction. (`eval_lever3_temporal.py`)

## 6. Methodology notes / caveats

- **Within vs between sign flips:** habitat is +within-district, −pooled (collinear
  with temperature; the pooled regression attributes shared signal to temperature).
  Use within-district formulation for interpretable coefficients.
- **CV discipline:** LODO-by-district is fair for covariate models but TOO OPTIMISTIC
  for any spatial model — use spatially-buffered/blocked CV there.
- **Denominators:** DHIS2 incidence has facility-catchment artifacts (8 sectors
  >1000/1000). Use WorldPop denominator + winsorise. Burden metric uses cases
  (robust). Anomalous low sectors (e.g. Karembo obs 20) are likely artifacts; EB
  district intercept stabilises them.
- **vs Malaria Atlas Project:** MAP = point-level Bayesian model-based geostatistics
  (INLA-SPDE GP) on prevalence surveys, ML-stacked covariates, mechanistic
  temperature-suitability index, transmission models for interventions, full
  posterior uncertainty, ~5 km surfaces. Ours = areal sector regression on routine
  cases, hand-engineered covariates, bootstrap CIs. To "do it properly": point/
  disaggregation geostatistics + spatial-block CV, mechanistic TSI, Bayesian
  uncertainty, transmission model for interventions.

## 7. Recommendations / next steps

1. **Ship the static map** (`sigmoid-temp + habitat + built`, within-district
   formulation) — burden@50% ~0.79 vs old exposure model's ~0.65.
2. **Don't bolt interventions on** (confound) — keep as separate effect-modifier.
3. **Build the spatiotemporal product** (static spatial risk + lagged rainfall) for
   seasonal early-warning — the main remaining headroom (and the whole temporal axis).
4. **For more static gain, acquire external data**: JRC/Sentinel seasonal surface
   water; VIIRS/GHSL urbanization; travel-time-to-facility. DEM-only proxies failed.
5. **If resourced**, move to MAP-style point/disaggregation geostatistics with
   spatial-block validation and Bayesian uncertainty.

## 8. Git state

Committed through **`e1c1917`** on `mean-exposure-per-person` (pushed).
**Uncommitted:** `results/export_percapita_risk.py` (the within-district CSV
generator) — should be committed. Large outputs (`*.csv/.nc/.tif/.png/.npz/.log`)
are gitignored as regenerable; regenerate from the scripts. Untracked & unrelated:
`.claude/`, root `malaria-modeling-weekend-read.*`, `rwanda_spray.csv`.

## 9. Key file index

- Metric: `evaluate_headline.py`, `metric_burden_capture.py`, `score_original_exposure_burden.py`
- Static map: `build_foundation_features.py`, `eval_static_levers.py`, `finalize_static_map.py`, `spatial_cv_buffer.py`
- Deliverable: `export_percapita_risk.py` → `sector_percapita_risk.csv`
- Phase 2: `eval_lever2_interventions.py`, `eval_lever3_temporal.py`
- Gridded/earlier: `build_gridded_*.py`, `sigmoid_*.py`, `final_*.py`, `eval_within_*.py`
- Reports: `results/reports/SUMMARY.md`, `step{1,4,5,6}_*.md`, `step2_interventions.md`, `step3_temporal.md`, `phase1_static_map.md`, this file
- Original investigation: `results/INVESTIGATION_REPORT.md` (§0 = headline metric)
