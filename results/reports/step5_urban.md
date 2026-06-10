# Step (lever 5) — Sharper urbanization / accessibility

**Goal.** Strengthen the urbanization signal (urban areas have lower malaria) with a
finer measure than WorldCover built-up, and add accessibility (care-seeking /
detection) signal.

**Method.** Added **focal log population density** (a finer settlement-intensity
proxy than the 100 m WorldCover built-up class) to the baseline; LODO OOF;
burden-capture.

**Results (burden @50% pop):**

| feature set | burden @50% | gap vs baseline | P |
|---|---|---|---|
| baseline (incl. built-up) | 0.788 | — | — |
| **+ focal pop-density** | **0.788** | **+0.001** | 0.58 |

**Findings.**
- **No improvement** — population density is redundant with the built-up term
  already in the baseline (both encode urbanization), confirming the earlier
  within-district finding. Adding it changes nothing.
- **The real upgrade is data the pipeline doesn't have.** A sharper urbanization
  signal needs **VIIRS night-time lights** or **GHSL built-up volume**, and the
  *detection/accessibility* axis (low reported cases ≠ low true burden) needs a
  **travel-time-to-facility friction surface**. None of these are in `chap_gis.io`,
  so this lever is **data-limited**, not idea-limited.

**Verdict:** *no gain from available data.* The urbanization signal is already
captured by built-up. Flag VIIRS/GHSL + travel-time as future data acquisitions —
they are the only way to push this lever further.

Artifacts: `results/build_foundation_features.py` (logpopdens), `eval_static_levers.py`.
