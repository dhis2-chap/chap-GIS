# Step (lever 2) — Interventions (bednets / IRS)

**Goal.** Add intervention coverage (LLIN bednet distribution, IRS spraying) as
predictors of *realised* burden — they explain why some high-environment sectors
have lower-than-expected cases.

**Features (per sector, 2013–2021).** LLIN per-capita dispensed (mass + EPI), IRS
spraying coverage, IRS population-protected per-capita, IRS months-covered.

**Raw spatial correlation with incidence — all positive:**

| feature | ρ vs incidence |
|---|---|
| llins_pc | +0.31 |
| irs_coverage | +0.55 |
| irs_protected_pc | +0.55 |
| irs_months | +0.54 |

**The positive sign is the targeting confound:** interventions — especially IRS —
are deployed *where malaria is already high*, so they correlate positively with
burden through reverse causality, not protection.

**Effect on the headline metric (burden @50% pop, LODO OOF, paired bootstrap):**

| model | burden @50% | gap vs static | P |
|---|---|---|---|
| static covariates (baseline) | 0.788 | — | — |
| + LLIN per-capita | **0.803** | +0.019 | 0.92 |
| + IRS (3 features) | 0.748 | −0.019 | 0.13 |
| + all interventions | 0.780 | +0.009 | 0.64 |

**Findings.**
- **LLIN per-capita gives a small, near-reliable gain** (+0.019, P=0.92) — bednet
  distribution is routine/universal (ANC, EPI, mass campaigns), so its per-capita
  intensity carries transferable signal.
- **IRS *hurts* out-of-fold** (−0.019). IRS is allocated by administrative decision
  to chosen districts; holding out a whole district, that decision can't be
  predicted from other districts' IRS–incidence relationship, so it adds noise.
- Combined, the two roughly cancel.

**Critical caveat — circularity.** The intervention "gain" is largely the targeting
confound: you'd be **using past allocation to predict burden to guide future
allocation**. For a risk map meant to *direct* interventions this is circular and
inappropriate. Interventions are valid only for predicting *current realised
burden given known coverage* (e.g. surveillance), not for prioritising where to
deploy.

**Verdict:** *do not add interventions to an allocation risk map.* They are a
reverse-causal confound; the apparent LLIN gain is not exposure skill. Keep them
out of the prioritisation model (treat as effect modifiers in a separate
evaluation, as the original report recommended).

Artifacts: `results/eval_lever2_interventions.py`, `sector_interventions.csv`.
