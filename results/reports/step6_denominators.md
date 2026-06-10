# Step (lever 6) — Fix the denominators

**Goal.** Test whether cleaning the case-rate denominators (use WorldPop instead of
DHIS2 facility-catchment population; train on a winsorised target) improves the
risk map, and lock in a clean evaluation setup.

**Setup.** Headline metric = burden captured in the top-50%-of-population. Three
denominator choices, baseline model `sigmoid-temp + habitat + built`, LODO OOF:

| population for the metric | target the model is trained on | burden @50% | oracle |
|---|---|---|---|
| DHIS2 (facility catchment) | DHIS2 incidence | 0.786 | 0.880 |
| WorldPop | DHIS2 incidence | 0.785 | 0.880 |
| WorldPop | **WorldPop-denominated, winsorised** | **0.788** | 0.880 |

**Findings.**
- **The denominator choice barely moves the burden metric.** Swapping DHIS2 for
  WorldPop population changes burden@50% by <0.001 and leaves the oracle ceiling
  unchanged at 0.880. The metric is intrinsically robust here because *burden =
  actual cases* (a count, independent of the denominator); only the population
  used to define "the top 50%" changes, and the two population surfaces rank
  sectors similarly.
- **Training on the cleaned (WorldPop-denominated, 99th-pct winsorised) target is
  marginally better** (0.788 vs 0.785) — de-noising the 8 facility-catchment
  outliers (>1000/1000) helps the fit slightly, without hurting.

**Decision.** Adopt **population = WorldPop, target = cleaned winsorised incidence**
as the going-forward evaluation setup. It is the more defensible denominator and
is at worst neutral, at best a small gain.

**Verdict:** *low impact* for the headline metric, but free and principled —
keep it as the foundation. The achievable ceiling (oracle 0.88) is set by genuine
spatial spread of burden plus residual outcome noise, not by the denominator.

Artifacts: `results/build_foundation_features.py`, `eval_static_levers.py`.
