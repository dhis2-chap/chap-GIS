# Rwanda malaria exposure-model investigation — report

**Scope:** how well the `dynamic-periods` mosquito-exposure ("risk map") model predicts
malaria, whether its parameters/structure can be improved, and what actually drives
predictive skill. Country: **Rwanda**, admin **sectors** (≈404 with disease data, DHIS2
org-units), monthly **2013–2021**. Branch: `mean-exposure-per-person`.

---

## TL;DR

1. The hand-built temperature exposure model (breeding sites → distance-decay kernel →
   thermal suitability, pop-weighted) has a hard skill ceiling and **a plain
   "sector mean temperature" baseline matches or beats it** at every level
   (national, within-district). The breeding/distance/elevation/curve/aggregation
   machinery adds **no validated value** for spatial burden ranking here.
2. Parameter tuning **converged** (best: λ=1500 m, γ=100 m, t_opt≈29 °C) and a sharp
   **logistic ~23 °C threshold** curve beat the Gaussian — but all of this moved skill
   by ≤0.02 (noise). Temperature is aseasonal in Rwanda (~1.2 °C annual range), so
   monthly resolution, aggregation method, and season-aligned years made **no
   difference**.
3. The real lever is **other covariates + a nonlinear model**. Vegetation (NDVI/EVI)
   supplies independent signal via a **temperature×vegetation interaction**
   (warm + low-NDVI lowlands = high risk). A RandomForest on all environmental
   covariates reaches **ρ ≈ 0.70 (leave-one-district-out)** vs ~0.55 for temperature
   alone; adding elevation/terrain → **~0.74**.
4. A **pixel-level `S(temp, NDVI)` suitability surface** built from just two satellite
   rasters reaches **0.669 (LODO)** — nearly matching the full 10-covariate model and
   far above temperature alone — confirming the mechanism.

**Recommendation:** replace the hand-built temperature exposure index with a
RandomForest/GBM on `{mean/max/min/dewpoint temperature, NDVI, EVI, humidity, rainfall,
elevation}`; treat interventions (bednets/IRS) separately to avoid the targeting
confound. The productive signal is temperature×vegetation, not breeding-habitat geometry.

---

## 0. Headline metric: burden capture under risk-prioritised allocation

**This is now the primary metric** — it expresses model value in directly actionable
terms and discriminates models far more sharply than rank correlation.

**Definition.** Rank sectors by predicted risk; allocate interventions to the
top-ranked sectors until they cover a target share of the population; report the
share of total malaria **burden (cases)** falling in the allocated part —
*"targeting X% of people reaches Y% of cases."* Random targeting captures X%;
ranking by **actual** incidence (the oracle) is the ceiling.

**Burden captured by targeting the top 50% of population** (DHIS2 cases, LODO
out-of-fold predictions, district-bootstrap 95% CI):

| ranking | burden @50% pop | 95% CI |
|---|---|---|
| random | 0.50 | — |
| temperature (baseline) | 0.66 | [0.54, 0.81] |
| env 3D surface `S(T,NDVI,amp)` | 0.77 | [0.69, 0.89] |
| gridded risk map (T + land use) | 0.77 | [0.67, 0.85] |
| **oracle** (rank by actual incidence) | 0.88 | [0.83, 0.93] |

- **Both environmental risk maps reliably beat temperature** — paired gap **+0.11,
  P≈0.99** (bootstrap CI excludes 0): ~77% of burden in the top-half allocation vs
  temperature's ~66%, closing ~65% of the gap to the oracle. Temperature alone
  closes only ~36%; adding the environmental signal roughly **doubles targeting
  efficiency**.
- The vegetation/moisture 3D surface and the temp+land-use gridded map are
  **statistically indistinguishable** here (+0.004, P=0.57). On a *global*
  allocation metric the between-district gradient dominates and both encode it;
  they diverge only on *within-district* ranking (a separate question).
- Budget sweep (top X% pop → % burden captured): **30%→~50%, 50%→~77%, 70%→~94%**
  for the risk maps, vs 44 / 65 / 92% for temperature.

Artifacts: `results/evaluate_headline.py`, `headline_burden_capture.{png,csv}`.
Rank-correlation (Spearman / within-district concordance) is retained below as a
secondary diagnostic.

---

## 1. The model under test

Per 30 m pixel, exposure ("risk") =
`exp(−d/λ) · exp(−max(Δz,0)/γ) · S(T_nearest)`, where `d` = distance to nearest mosquito
breeding site (WorldCover wetlands + rice + water-edge buffer), `Δz` = elevation rise
above it, and `S` = thermal suitability (Gaussian TPC) at that site's temperature.
Pop-weighted (`pop·expo`) and summed to sectors; `mean_exposure_per_person = Σ(pop·expo)/Σpop`.
Defaults: λ=651 m, γ=22.5 m, t_opt=25 °C, σ=5.

Evaluation metric throughout: **Spearman correlation vs disease** (raw cases and, to remove
the population confound, incidence = cases/pop). Computed at several levels — region-month,
sector-year, and pure-spatial sector means — and, for the covariate models,
**leave-one-district-out (LODO) cross-validation** (the honest "generalise to a new
district" number).

---

## 2. Parameter sweep (exposure-model tuning)

Implemented a sweep capability (`dynamic-periods --grid-config`, see §7) and an efficient
cached evaluator (the distance transform is temperature-independent, so it is computed
once and the kernel/suitability swept cheaply).

| sweep | best params | metric (region-month) |
|---|---|---|
| coarse (12 combos) | λ=900, γ=50, t_opt=27 | ρ_raw 0.477 |
| refined (18 combos) | **λ=1500, γ=100, t_opt=29** | ρ_raw 0.552 / ρ_inc 0.493 |
| thermal-focused | t_opt **peaks at 29** (31, 33 worse) | converged |

Marginal effects: **t_opt** was the only strong lever (and peaks at 29); **γ** plateaus by
~100; **λ** is essentially saturated (and even weaker under incidence — it partly rode the
population channel). Conclusion: the search **converged**; further tuning is noise.

**Caveat:** t_opt≈29 °C sits well above the ~25 °C malaria literature optimum. Rwanda is
cool highland (sector temps 13–24 °C, median ~20), so the curve only ever operates on its
rising flank; "t_opt=29" is a data-driven fit that steepens the response across 14–24 °C,
not a mechanistic transmission optimum.

---

## 3. Suitability-curve shape

Replaced the Gaussian with splines/logistic curves (scored instantly via a cached
per-sector temperature histogram). A **sharp logistic threshold** won:

| curve | ρ_inc | ρ_raw (region-month) |
|---|---|---|
| **logistic, T0≈23 °C, k≈3** | **0.515** | 0.571 |
| Gaussian optimum (t_opt 29, σ6) | 0.496 | 0.552 |
| gentle/monotone splines | 0.30–0.41 | — |

Mechanism: the population×kernel-weighted breeding-site temperatures cluster tightly at
19–23 °C, so a **selective ~23 °C switch** (≈1 °C transition) discriminates best. Smoother
or gentler curves do worse. This is the only place curve choice mattered — and it's still
a ≈0.02 gain.

---

## 4. The control: does the machinery beat plain temperature?

Control = `S(sector mean temperature)` — no breeding sites, no distance kernel, no
population weighting. (Ranking by it ≡ ranking by mean temperature, since `S` is monotone.)

| level | CONTROL (sector mean temp) | FULL champion exposure model |
|---|---|---|
| spatial, vs incidence | **0.627** | 0.581 |
| spatial, vs cases | **0.682** | 0.636 |
| region-month, vs incidence | **0.523** | 0.515 |
| within-district (Fisher-avg) | **0.617** | 0.577 |

**The control matches or beats the full model everywhere.** The two predictors agree at
Spearman 0.934; where they differ, the full model damps the signal. The within-district
result also shows the temperature signal is **local**, not just the gross national gradient
(within-district ρ ≈ national ρ). The breeding/distance machinery is not justified for
spatial burden ranking.

---

## 5. Temporal sophistication — no benefit (temperature is aseasonal)

- **Disease is strongly bimodal** (peaks Nov–Jan and May–Jun, trough August; rainfall-driven).
- **Temperature is nearly aseasonal** (monthly mean 19.1–20.3 °C) and *anti*-correlates with
  the seasonal case cycle. So temperature suitability cannot explain seasonality.
- Built a monthly-resolved cache and tested monthly→yearly aggregations and a season-aligned
  (Sep→Aug) disease year:
  - mean ≈ annual; **mean, sum, warmest-month, softmax (any intensity)** all land in a
    **0.515–0.519** band (within noise).
  - **Season alignment slightly *hurt*** (predictor has no temporal signal to align to).

Temporal modelling of temperature is a dead end here; seasonality needs a moisture term.

---

## 6. Covariates — the real improvement

Rich covariates available in `data/inputs/chap_data_level5_clean_2013-2021.csv` (join 404/404):
mean/max/min/dewpoint temperature, **EVI, NDVI**, relative humidity, rainfall (CHIRPS/ERA5/IRI),
plus interventions (bednets `llins_*`, IRS `irs_*`). Also aggregated from rasters: elevation,
rice fraction, wetland/marshland fraction, water fraction.

**Single-covariate spatial Spearman vs incidence:** relative_humidity −0.65, rainfall_era5
−0.62, mean_temp +0.59, max_temp +0.55, dewpoint +0.47; **EVI +0.11, NDVI −0.00** (≈ zero).
Humidity/rainfall are negative because they track altitude (cool wet highlands = low
transmission) — i.e. more temperature proxies, not an independent moisture axis.

**Cross-validated multivariate models (leave-one-district-out Spearman vs incidence):**

| feature set | Linear | RandomForest | GradBoost |
|---|---|---|---|
| temp only | 0.55–0.57 | 0.53 | 0.52 |
| all temperature (max/min/dewpoint) | 0.61 | 0.65 | 0.62 |
| temp + vegetation (EVI/NDVI) | 0.63 | 0.67 | 0.64 |
| ALL env (10) | 0.67 | **0.70** | 0.70 |
| ALL env + terrain (elev/rice/wetland/water) | 0.68 | **0.74** | 0.73 |
| ALL env + interventions | 0.70 | 0.74 | **0.75** ⚠ |

Key points:
- **Vegetation is a suppressor:** EVI/NDVI have ≈0 marginal correlation but add the biggest
  multivariate lift — alone meaningless, but *conditional on temperature* they separate
  warm-vegetated valleys/wetlands from warm-dry areas. This is the signal the breeding-site
  machinery tried and failed to encode.
- **Nonlinearity matters:** RF/GBM beat linear (interactions). Best validated non-intervention
  model: **RF on ENV+terrain ≈ 0.74 LODO** vs 0.55 temp-only.
- **Terrain habitat layers (rice/wetland/water) are the *least* important features**
  (importances 0.02–0.03); the terrain gain is essentially **elevation** sharpening the
  temperature axis. Top RF importances: relative_humidity 0.41, elevation 0.11,
  min_temp 0.11, evi 0.06, ndvi 0.045.
- ⚠ **Interventions inflate skill via targeting:** IRS is deployed where malaria is high, so
  `irs_coverage` predicts cases partly through reverse causality, not protection. The 0.75 is
  not pure exposure skill; treat interventions as effect modifiers, not predictors.

LODO and 8-fold GroupKFold gave essentially the same numbers (≈0.70 env), so the result is
robust to CV granularity.

---

## 7. Pixel-level temperature×NDVI suitability (synthesis)

Sourced **MODIS 13Q1 (250 m, 16-day) NDVI+EVI** for 2021 via Microsoft Planetary Computer,
composited to monthly, co-registered with CHELSA monthly temperature
(`results/veg_temp_2021/stack.nc`).

**Cell-by-cell NDVI×temperature** (~7.0M pixel-months):
- Pooled correlation weakly negative (NDVI −0.11), but the relationship is **non-monotonic**:
  mean NDVI peaks ~0.59 at 19–20 °C then **drops to ~0.49 at 21–23 °C** (the warm, populous,
  high-malaria band).
- **Strongly seasonal — sign flips:** wet/transition months show positive NDVI–temp
  coupling, the dry season (Jun–Sep) strongly negative (to −0.43). Warm lowland vegetation
  collapses in the dry season — a **moisture-stress signal temperature can't see**.

**Learned suitability surface `S(temp, NDVI)`** (ridge on per-sector pop-weighted joint
histogram = a fit of the 2D surface), leave-one-district-out:

| surface | LODO Spearman |
|---|---|
| 1D `S(temp)` | 0.490 |
| **2D `S(temp, NDVI)`** | **0.669** |

The 2-raster pixel suitability **nearly matches the full 10-covariate sector model (0.70)**.
The surface puts highest risk at **warm temperature (~20–24 °C) + low/moderate NDVI**
(warm, dry, settled lowlands) — confirming the temperature×vegetation interaction.

---

## 8. Conclusions & recommendations

- The mosquito-exposure raster model is, for Rwanda spatial burden, **equivalent to
  sector mean temperature** — the breeding-site/distance/elevation/curve machinery adds
  nothing validated. Don't invest further in tuning it.
- **Improve by going multivariate + nonlinear.** Recommended model: RandomForest/GBM on
  `{mean/max/min/dewpoint temperature, NDVI, EVI, relative humidity, rainfall, elevation}`
  → ρ ≈ 0.70–0.74 LODO. Vegetation (interacting with temperature) is the key addition.
- For a **physical risk surface**, the learned pixel `S(temp, NDVI)` (≈0.67 from two
  rasters) is a strong, interpretable replacement for the current exposure index.
- Likely further gains: a **dry-season NDVI / moisture-stress** axis (the seasonal sign-flip
  suggests it), and proper handling of **interventions** as effect modifiers.
- Hard limits: temperature alone ≈ 0.55; rainfall-driven **seasonality** is unreachable
  without a moisture term; sector incidence has **denominator outliers** (8 sectors with
  >1000/1000 are facility-catchment artifacts — flag/exclude).

---

## 9. Artifacts

**Code (committed on `mean-exposure-per-person`):**
- `src/chap_gis/exposure.py` — split into `compute_distance_field` + `exposure_from_field`
  (EDT reused across sweeps); `exposure()` unchanged.
- `src/chap_gis/pipelines/malaria_exposure.py` — `MalariaExposureParams` exposes
  λ/γ/thermal; `ExposureSweepSpec`; `reproject_layers`.
- `src/chap_gis/cli/dynamic.py` — `dynamic-periods --grid-config` sweep path (wide CSV +
  manifest); `get_health_data` keeps only the disease column.
- Tag `pre-exposure-sweep` marks the pre-feature baseline.

**Analysis scripts (`results/`):** `build_suit_eval_cache.py`, `eval_suit_curves.py`,
`build_monthly_suit_cache.py`, `eval_monthly_suit.py`, `softmax_agg.py`,
`gen_optimum_rasters_2021.py`, `gen_champion_rasters_2021.py`, `agg_terrain_covariates.py`,
`build_veg_temp_stack.py`, `build_pixel_suitability.py`. Sweep grids:
`rwanda_sweep_grid.json`, `rwanda_sweep_refined_grid.json`, `rwanda_sweep_temp_grid.json`.

**Outputs (gitignored — large/regenerable):**
- Per-sector CSVs: `rwanda_sector_incidence_vs_risk.csv`, `rwanda_sector_risk_vs_cases.csv`
  (per-capita risk = champion logistic curve), `rwanda_sector_control_temp.csv`,
  `rwanda_sector_terrain.csv`.
- Raster stacks: `optimum_rasters_2021/` (Gaussian optimum, 7 layers),
  `champion_rasters_2021/` (logistic curve, 7 layers), `veg_temp_2021/stack.nc`
  (NDVI/EVI/temp monthly) and `veg_temp_2021/pixel_suitability.nc`.
- Caches: `suit_eval_cache.npz` (annual), `monthly_suit_cache.npz`, `softmax_pe.npz`.
- Plots: `suitability_curves.png`, `suitability_splines.png`, `suitability_best.png`,
  `control_vs_full.png`, `veg_temp_pixel_analysis.png`, `pixel_suitability.png`.

**New deps used:** `scikit-learn`, `planetary-computer`, `odc-stac` (installed into the venv
via `uv`; not yet added to `pyproject.toml`).

**Optimum config:** λ=1500 m, γ=100 m, water-edge buffer=2, 30 m; Gaussian t_opt=29/σ6/[19,38]
or **champion** `S(T)=1/(1+e^(−3(T−23)))`.

---

## 10. Caveats

- Most skill numbers are **spatial**; the model is a burden classifier, not a temporal/early-
  warning model (seasonality needs moisture).
- The "districts" used for LODO are the 46 DHIS2 `parent` groups, not exactly the 30 ADM2
  districts (close enough; a spatial join would make it exact).
- Vegetation/raster work uses **2021 only**; covariate models use the 2013–2021 mean.
- The warm+low-NDVI signal may partly encode **land use/settlement**, not pure mosquito
  ecology.
- Population denominators: a few sectors are facility catchments → implausible incidence;
  exclude >1000/1000 for clean analysis.
