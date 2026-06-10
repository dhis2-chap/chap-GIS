# Step (lever 3) — Multi-year / spatiotemporal (seasonality)

**Goal.** The static map predicts a constant per sector and is **blind to the
seasonal cycle**. Test whether a rainfall-driven spatiotemporal model on the
2013–2021 monthly panel recovers the within-year timing — a genuinely new
capability (early warning), not just a better spatial ranking.

**The seasonal cycle is real and rainfall-driven.** National mean cases by month:

| month | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cases | 990 | 722 | 610 | 581 | 720 | 802 | 504 | 329 | 455 | 716 | 1009 | 1087 |

Bimodal (peaks Nov–Jan and May–Jun, trough Aug). **Cases track lagged rainfall**
(ρ = +0.67 with rain at 2-month lag) — and **temperature is aseasonal**, so the
static temperature map cannot represent any of this.

**Method.** Remove each (sector, year) annual mean to isolate the within-year
seasonal **anomaly**; predict it from climate (temperature, rainfall at lags 0–3,
NDVI, EVI, humidity); leave-one-**year**-out; pooled Spearman of predicted vs
actual anomaly.

| model | seasonal-anomaly skill |
|---|---|
| temperature only | **+0.045** (≈ 0 — confirms aseasonality) |
| **climate incl. lagged rainfall** | **+0.300** |
| month-of-year climatology | +0.406 (data-driven seasonal ceiling) |

**Findings.**
- **Rainfall recovers the seasonality temperature can't.** A spatiotemporal model
  with lagged rainfall reaches **+0.30 within-year seasonal skill** out-of-sample,
  vs ~0 for temperature — i.e. ~**74% of the climatological ceiling** (0.30/0.41).
- This is a **new product capability**: a *when-and-where* model that predicts the
  monthly rise and fall, enabling seasonal early-warning, which the static
  allocation map fundamentally cannot do.
- Lever 3 is therefore **not** about improving the spatial burden-capture metric
  (where it adds little) — its value is the **temporal axis**, addressing the
  report's standing gap that "seasonality needs a moisture term."

**Verdict:** *the highest-value structural direction* — but it's a different
product. Recommend building a proper spatiotemporal model (lagged rainfall + the
static spatial risk as the level term) if a seasonal/early-warning capability is
wanted; the static map remains the right tool for one-off spatial allocation.

Artifacts: `results/eval_lever3_temporal.py`.
