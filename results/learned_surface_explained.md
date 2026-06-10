---
title: "The learned suitability surface: how it is built and evaluated"
subtitle: "Rwanda malaria exposure model — pixel surface $S(\\text{temp}, \\text{NDVI}, \\text{amp})$"
date: "June 2026"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
---

# What we are trying to build

We want a **suitability surface** $S$: a function that takes an environmental
state — here a triple of *(temperature, annual-mean NDVI, NDVI seasonal
amplitude)* — and returns a number we interpret as **malaria risk** for living
under that state. Once we have $S$, two things follow:

1. a **pixel risk map** — evaluate $S$ at every 250 m pixel's environment; and
2. a **sector prediction** — combine the surface with where people live to get a
   predicted disease level per administrative sector, which we can check against
   observed incidence.

The difficulty: **we never observe risk at the pixel level.** We only observe
**disease counts aggregated to sectors** ($\approx 404$ of them). So we cannot fit $S$
pixel-by-pixel. The trick described below recovers a pixel-level $S$ *purely from
sector-level outcomes*, by exploiting one structural assumption that makes the
problem linear.

---

# The ingredients

| symbol | meaning | source |
|---|---|---|
| $p$ | a pixel (250 m) | MODIS / CHELSA grid |
| $T_p, N_p, A_p$ | temperature, NDVI, NDVI amplitude at pixel $p$ | `stack.nc` (2021) |
| $w_p$ | population in pixel $p$ | WorldPop 2021 |
| $\text{sec}(p)$ | the sector pixel $p$ belongs to | rasterized boundaries |
| $y_s$ | observed malaria **incidence** (cases/1000) in sector $s$ | DHIS2 |
| $d(s)$ | the **district** sector $s$ belongs to | DHIS2 parent |

NDVI amplitude is $A_p = \max_{\text{month}} \text{NDVI} - \min_{\text{month}}
\text{NDVI}$ over 2021 — a measure of how much the vegetation dries down
seasonally (the moisture-stress axis).

---

# The one assumption that makes it work

We assume a sector's risk is the **population-weighted average of a per-environment
suitability**. In words: a sector is risky to the extent that *its people live in
risky environments*. Formally, if pixel $p$ has suitability $S(T_p, N_p, A_p)$,
the sector's predicted risk is

$$
\hat y_s \;=\; \frac{\sum_{p \in s} w_p \, S(T_p, N_p, A_p)}{\sum_{p \in s} w_p}.
\tag{1}
$$

This is the *entire* model. Everything else is bookkeeping to turn (1) into
something we can fit.

The key observation: **equation (1) is linear in the unknown surface $S$.** $S$
appears only multiplied by known weights and summed. Linear models are easy to
fit. We just have to write (1) in matrix form.

---

# Step 1 — Discretise the environment into cells

We cannot estimate $S$ at every possible real-valued triple, so we **bin** each
axis and assume $S$ is constant within a cell. With the equal-width bins of the
3D surface:

- temperature: 8 bins, edges $10, 12, \dots, 26\,^\circ\mathrm C$
- NDVI: 6 bins, edges $0.00, 0.15, \dots, 0.90$
- amplitude: 5 bins, edges $0.00, 0.12, \dots, 0.60$

That is a grid of $8 \times 6 \times 5 = 240$ **cells**. Write $c(p)$ for the cell
pixel $p$ falls into, and let $S_c$ be the (unknown) suitability of cell $c$.
The surface is now just a **vector of 240 numbers** $\mathbf S = (S_1, \dots,
S_{240})$.

---

# Step 2 — Summarise each sector by a population histogram

For each sector $s$ we count **how much population sits in each cell**:

$$
H_s(c) \;=\; \sum_{\substack{p \in s \\ c(p) = c}} w_p,
\qquad
p_s(c) \;=\; \frac{H_s(c)}{\sum_{c'} H_s(c')}.
\tag{2}
$$

$p_s$ is the **population distribution of sector $s$ over the 240 cells** — the
fraction of the sector's people living in each environment. By construction
$\sum_c p_s(c) = 1$. In code this is one `np.bincount` over the flattened cell
index, then a normalisation (`Hn` in the script).

Each sector is now a single row of 240 numbers. Stack all sectors:
$\mathbf X \in \mathbb R^{404 \times 240}$, row $s$ equal to $p_s$.

---

# Step 3 — The model is a matrix product

Substitute the binned surface into (1). Because $S$ is constant within a cell,
the population-weighted average becomes a weighted sum over cells:

$$
\hat y_s \;=\; \sum_{c} S_c \, p_s(c) \;=\; \langle \mathbf p_s, \mathbf S\rangle,
\qquad\text{i.e.}\qquad
\hat{\mathbf y} = \mathbf X\,\mathbf S .
\tag{3}
$$

So the predicted-incidence vector is **just $\mathbf X$ times the surface**. This
is an ordinary linear regression with the unknown surface $\mathbf S$ playing the
role of the regression coefficients, and the per-sector histograms playing the
role of the features.

> **This is the crux.** We did not fit a pixel surface and then average it. We
> defined each sector by *where its population sits in environment-space*, and
> asked linear regression: *what risk value must each environment cell carry so
> that the population-weighted cell mix predicts each sector's incidence?* The
> answer **is** the surface.

## A 2-cell worked example

Suppose only two cells, "cool" and "warm". Two sectors:

- Sector A: 10% of people cool, 90% warm $\Rightarrow \mathbf p_A = (0.1, 0.9)$
- Sector B: 70% cool, 30% warm $\Rightarrow \mathbf p_B = (0.7, 0.3)$

Then $\hat y_A = 0.1\,S_\text{cool} + 0.9\,S_\text{warm}$ and
$\hat y_B = 0.7\,S_\text{cool} + 0.3\,S_\text{warm}$. Given the observed
incidences $y_A, y_B$, regression solves these for $(S_\text{cool},
S_\text{warm})$ — the two cells' risk. With 404 sectors and 240 cells it is the
same thing at scale.

---

# Step 4 — Fit with ridge regression (and why)

We have 240 unknowns and ~404 sectors, but the columns of $\mathbf X$ are highly
**collinear** (neighbouring cells co-occur in the same sectors) and many cells are
nearly **empty** (almost nobody lives at 12 °C or NDVI 0.85). Plain least squares
would be unstable — it would fit wild, oscillating risk values to empty cells.

We therefore use **ridge regression**, which penalises large surface values:

$$
\hat{\mathbf S} \;=\; \arg\min_{\mathbf S}\;
\underbrace{\sum_s \big(y_s - \langle \mathbf p_s, \mathbf S\rangle\big)^2}_{\text{fit}}
\;+\;
\alpha \underbrace{\textstyle\sum_c S_c^2}_{\text{smoothness / shrinkage}} .
\tag{4}
$$

The penalty weight $\alpha$ trades fit against stability. Large $\alpha$ shrinks
the whole surface toward a flat constant (under-fit); small $\alpha$ lets it chase
the data (risk of over-fit). We report a small sweep $\alpha \in \{1,5,20,50,100\}$
rather than fixing one value, so the comparison between surfaces is not an
artefact of a lucky penalty.

*(Note: because every row of $\mathbf X$ sums to 1, adding a constant to all $S_c$
shifts every prediction equally — that constant is absorbed by the regression
intercept, and the risk **ranking** we ultimately score is unaffected by it.)*

---

# Step 5 — Evaluate honestly: leave-one-district-out

If we fit $\mathbf S$ on all sectors and then scored it on those same sectors, we
would be grading the model on data it has seen. Instead we use **leave-one-
district-out (LODO)** cross-validation. The ~404 sectors group into 46 districts.
For each district $d$:

1. fit the surface $\hat{\mathbf S}^{(-d)}$ using only sectors **not** in $d$
   (equation 4 on the training sectors);
2. predict the held-out district's sectors: $\hat y_s = \langle \mathbf p_s,
   \hat{\mathbf S}^{(-d)}\rangle$ for $s \in d$.

Collect the out-of-fold predictions for all sectors and compute one number:

$$
\rho \;=\; \text{Spearman}\big(\hat{\mathbf y}_{\text{oof}},\; \mathbf y\big).
$$

Grouping by **district** (not random sectors) is deliberate: it forces the model
to generalise to a *spatially new region*, the honest "would this work somewhere
we haven't calibrated" test.

We score with **Spearman rank correlation** (not $R^2$) because we care about
getting the **ordering** of sectors right — which are higher-burden than which —
and Spearman is invariant to any monotone distortion of the scale and robust to
the handful of sectors with implausible incidence (facility-catchment
denominators).

---

# Step 6 — Read out the surface and the map

Two distinct objects come out, and it is worth keeping them straight:

- **The skill numbers ($\rho$)** come from the *out-of-fold* predictions (Step 5).
  This is what we trust as "how good is the model".
- **The displayed surface and pixel map** are fit *once on all sectors* (good for
  visualisation, not a skill claim). The surface is $\hat{\mathbf S}$ reshaped
  back to the $8\times6\times5$ grid; the pixel map assigns each pixel its cell's
  value, $\text{map}(p) = \hat S_{c(p)}$.

---

# Results

Leave-one-district-out Spearman vs incidence, equal-width bins, ridge $\alpha$
sweep (best in **bold**):

| surface | $\alpha{=}1$ | 5 | 20 | 50 | 100 | best |
|---|---|---|---|---|---|---|
| 1D $S(\text{temp})$ | 0.577 | 0.571 | 0.541 | 0.484 | 0.459 | **0.577** |
| 2D $S(\text{temp},\text{NDVI})$ | 0.696 | 0.688 | 0.673 | 0.652 | 0.627 | **0.696** |
| 3D $S(\text{temp},\text{NDVI},\text{amp})$ | 0.758 | 0.742 | 0.709 | 0.677 | 0.628 | **0.758** |

The 1D/2D rows are the same machinery with the histogram **marginalised** over the
unused axes (sum $p_s$ over NDVI and/or amplitude before fitting). Adding the
seasonal-amplitude axis lifts the surface from 0.696 to **0.758** — matching the
full sector random-forest model (~0.75) but using only two satellite raster
stacks and remaining fully interpretable.

## Binning choice changes the answer

The bin edges are themselves a modelling choice. Equal-width bins waste resolution
on near-empty environments; **equal-population (quantile) bins** put a bin boundary
where people are, so each bin holds the same number of people:

| surface | equal-width (best) | equal-population (best) |
|---|---|---|
| 1D $S(\text{temp})$ | 0.577 | **0.646** |
| 2D $S(\text{temp},\text{NDVI})$ | 0.696 | **0.714** |
| 3D $S(\text{temp},\text{NDVI},\text{amp})$ | **0.758** | 0.752 |

Equal-population binning sharply helps the low-dimensional surfaces (the 1D gain
is almost entirely better bin *placement*, not new signal) but does nothing for
the 3D surface and makes its ridge fit more sensitive to $\alpha$ — denser, more
collinear histograms destabilise under heavy shrinkage.

---

# Things to keep in mind

- **The surface only "sees" inhabited cells.** ~83% of people live at 18–24 °C and
  ~74% in NDVI 0.45–0.60, so most of the 240 cells are empty and carry no fitted
  weight (the pale regions of the surface plot). The amplitude axis is the most
  *population-dispersed* of the three, which is why it adds discriminating power.
- **In-sample surface $\ne$ out-of-fold skill.** The picture is fit on everything; the
  $\rho$ values are cross-validated. Do not read a skill claim off the map.
- **Linearity is an assumption.** Equation (1) says a sector's risk is a *linear*
  population-weighted mean of cell suitabilities. The surface itself can be
  arbitrarily nonlinear in (temp, NDVI, amp) — that nonlinearity is captured by
  letting each cell have its own value — but interactions *between sectors* or
  saturation effects are not modelled.
- **Single year (2021).** The whole vegetation/amplitude analysis uses one year;
  confirming the amplitude gain across multiple years is the main outstanding
  robustness check.
- **A data seam.** The MODIS NDVI mosaic has an ~8-pixel all-year gap (a tile
  seam) that shows as a thin NaN line on the maps; it is excluded from every
  sector histogram and does not affect the statistics.
