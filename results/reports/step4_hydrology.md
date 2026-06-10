# Step (lever 4) — Hydrology / breeding-water axis

**Goal.** Add water-availability features beyond static land cover — where water
pools and persists drives mosquito breeding. Targeted a moisture-dynamics axis the
NDVI-amplitude work only partly captured.

**Method.** DEM-derived hydrology on the 30 m elevation: **slope**, **valley index**
(depth below the ~1 km local mean elevation), a **topographic-wetness proxy** (TWI ≈
flatness × focal upslope flatness), plus **focal permanent-water** and **focal
wetland** fractions. Added to the baseline; LODO OOF; burden-capture.

**Results (burden @50% pop):**

| feature set | burden @50% | gap vs baseline | P |
|---|---|---|---|
| baseline | 0.788 | — | — |
| **+ hydrology (5 features)** | **0.777** | **−0.005** | 0.37 |

**Findings.**
- **Hydrology does not help — it slightly hurts** (−0.005, P=0.37, i.e. more often
  worse than better). The five DEM-derived features add parameters and noise
  without new signal that survives cross-validation.
- **Why:** the genuinely useful hydrology signal is *seasonal surface water and
  flood dynamics*, which **DEM-derived proxies cannot represent**. Slope/valley/TWI
  describe where water *could* accumulate topographically, but not where it
  *actually* stands seasonally; and the static permanent-water/wetland fractions
  are already partly captured by the habitat term. Rwanda also has little permanent
  open water at sector scale.

**Verdict:** *no value from DEM-derived hydrology — drop it.* The real upgrade
requires **external seasonal-surface-water data** not in the current pipeline:
JRC Global Surface Water (occurrence/seasonality) or Sentinel-1 flood-frequency
composites. Recommend adding one of those as a future data source; the DEM alone
is not enough.

Artifacts: `results/build_foundation_features.py` (hydrology features), `eval_static_levers.py`.
