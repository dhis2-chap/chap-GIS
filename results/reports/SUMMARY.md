# Risk-map improvement program — summary

Headline metric throughout: **burden captured by allocating to the top-X%-of-
population by predicted risk** (LODO out-of-fold; paired district-bootstrap).
Baseline static map = `sigmoid-temp(T₀=19) + log-habitat + log-built-up`;
burden@50% ≈ 0.79; oracle (rank by actual incidence) = 0.88.

## Phase 1 — levers 1, 4, 5, 6 → the good static map

| lever | burden@50% | gain vs baseline | P | verdict |
|---|---|---|---|---|
| (6) denominators (WorldPop + winsorised) | 0.788 | ~0 | — | adopt (neutral foundation) — [report](step6_denominators.md) |
| (1) spatial term (GP on centroids) | 0.792 | +0.014 | 0.92 | **proximity artifact — vanishes under buffered CV** — [report](step1_spatial.md) |
| (4) hydrology (DEM-derived) | 0.777 | −0.005 | 0.37 | drop; needs external surface-water data — [report](step4_hydrology.md) |
| (5) urban / pop-density | 0.788 | +0.001 | 0.58 | drop; redundant w/ built-up — [report](step5_urban.md) |

**Good static map** = covariate baseline `sigmoid-temp + habitat + built`
(burden@50% **0.788**). The spatial term's apparent gain is **proximity-driven** and
contributes +0.000 under ≥20 km spatially-buffered CV (`spatial_cv_buffer.py`) — it
is a valid-OOF *within-sample smoother* for interpolating among observed sectors,
not a transferable improvement. See [phase1_static_map.md](phase1_static_map.md);
render `static_risk_map.png`.

**Conclusion:** we are at the static-data ceiling (0.79 vs oracle 0.88). **No static
lever reliably beats the covariate baseline**; further static gains need *new
external data* (seasonal surface water; VIIRS/GHSL; travel-time).

## Phase 2 — levers 2, 3

- **(2) Interventions** — [report](step2_interventions.md). All intervention
  features correlate *positively* with burden (targeting confound). LLIN per-capita
  gives a small gain (+0.019, P=0.92); IRS hurts out-of-fold (−0.019). **Do not use
  for an allocation map — it's circular** (past allocation predicting burden to
  guide future allocation). Valid only for surveillance of realised burden.
- **(3) Multi-year / spatiotemporal** — [report](step3_temporal.md). The static map
  is blind to the bimodal seasonal cycle (temperature is aseasonal). A rainfall-
  driven model recovers **+0.30 within-year seasonal skill** out-of-sample (~74% of
  the climatology ceiling) vs ~0 for temperature. **Highest-value direction**, but
  it's a *different product* (a when-and-where / early-warning model), not an
  improvement to the static spatial map.

## Overall recommendation

1. **Ship the good static map** (`sigmoid-temp + habitat + built`) for one-off
   spatial allocation — it reaches ~0.79 burden@50% vs the old exposure model's
   ~0.65 (= temperature). (Optional spatial smoothing only where interpolating among
   observed sectors; it doesn't generalise to unsampled areas.)
2. **Don't bolt interventions onto it** (confound); keep them as a separate
   effect-modifier analysis.
3. **The real next product is spatiotemporal** — add lagged rainfall on top of the
   static spatial risk to get seasonal early-warning. That, plus external
   surface-water / nightlights / travel-time data, is where the remaining headroom
   (static 0.79 → oracle 0.88, and the entire temporal axis) actually lives.
