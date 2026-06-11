"""Champion static risk map: within-district covariate regression.

The "champion" map ranks sectors by malaria risk from three per-sector
covariates — a sigmoid temperature suitability, focal mosquito *habitat*
(rice ∪ wetland) and focal *built-up* land — fit in a **within-district**
formulation (district-demeaned slopes, mechanistically-correct signs: habitat
``+``, built ``-``) plus an **empirical-Bayes district intercept** that supplies
the between-district level from observed burden.

This module holds the pure transforms and a pydantic params model; all raster /
CSV I/O lives in :mod:`chap_gis.cli.champion`.

Pipeline:
    rasters ──compute_sector_features──▶ per-sector covariates + wp_pop
    (+ observed cases) ──fit_champion_risk──▶ per-sector risk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio.features as rfeat
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LinearRegression

from ..grid import build_grid, reproject_to, reproject_population_to

#: Covariates entering the regression, in coefficient order.
FEATURE_COLS: tuple[str, ...] = ("sig_temp", "hab", "built")

#: WorldCover class codes used to build the focal land-use features.
_WETLAND_CLASSES = (90, 95)  # herbaceous wetland + mangroves
_BUILT_CLASS = 50


class ChampionRiskParams(BaseModel):
    """Parameters for the champion static risk map.

    Grid/focal geometry plus the sigmoid-temperature transform. Defaults
    reproduce ``results/build_foundation_features.py`` /
    ``results/export_percapita_risk.py``.
    """

    model_config = ConfigDict(frozen=True)

    resolution_m: float = Field(default=100.0, gt=0)
    focal_km: float = Field(default=1.0, gt=0, description="Focal radius for land-use features")
    aoi_buffer_deg: float = Field(default=0.0027, ge=0)
    meters_per_degree: int = Field(default=111_000, gt=0)

    # sigmoid temperature suitability: expit((T - t0) / k)
    t0: float = Field(default=19.0, description="Sigmoid-temp midpoint (°C)")
    k: float = Field(default=0.5, gt=0, description="Sigmoid-temp slope (°C)")

    winsor_quantile: float = Field(default=0.99, gt=0, le=1.0)
    raster_year: int = Field(default=2021, description="Environmental snapshot year")


@dataclass
class ChampionFit:
    """Diagnostics from :func:`fit_champion_risk` (for logging)."""

    beta_within: dict[str, float]
    beta_pooled: dict[str, float]
    shrink_mean: float
    n_sectors: int
    n_districts: int


def _focal_mean(arr: np.ndarray, km: float, res_m: float) -> np.ndarray:
    """Square uniform focal mean of `arr` with an odd window of radius ~`km`."""
    size = int(round(km * 1000.0 / res_m)) | 1  # force odd
    return ndimage.uniform_filter(arr.astype(np.float32), size=size, mode="nearest")


def compute_sector_features(
    gdf,
    *,
    land,
    rice,
    tas,
    pop,
    params: ChampionRiskParams,
) -> pd.DataFrame:
    """Population-weighted per-sector covariates from loaded rasters.

    Reprojects every raster onto a common analysis grid, derives the per-pixel
    champion covariates (``sig_temp``, ``hab``, ``built``), then pop-weights
    them to one row per sector.

    Parameters
    ----------
    gdf
        Sector boundaries with a ``location_id`` column (EPSG:4326).
    land, rice, tas, pop
        WorldCover landcover, rice fraction, mean temperature (°C), and
        WorldPop count rasters — already loaded (lazy is fine).
    params
        :class:`ChampionRiskParams`.

    Returns
    -------
    pandas.DataFrame
        Columns ``location_id, wp_pop, sig_temp, hab, built, lon, lat``.
    """
    res = params.resolution_m
    grid = build_grid(gdf, resolution=res / params.meters_per_degree, crs="EPSG:4326")

    def _to_grid(src, how: str) -> np.ndarray:
        a = np.asarray(reproject_to(src, grid, how).compute().values, dtype=np.float32)
        return a[0] if a.ndim == 3 else a

    lc = _to_grid(land, "mode")
    if "band" in getattr(rice, "dims", ()):
        rice = rice.squeeze("band", drop=True)
    rg = _to_grid(rice.astype("float32"), "average")
    temp = _to_grid(tas, "bilinear")

    popg = np.asarray(
        reproject_population_to(pop, grid, "bilinear").compute().values, dtype=np.float32
    )
    popg = popg[0] if popg.ndim == 3 else popg
    popg = np.clip(np.nan_to_num(popg, nan=0.0), 0, None)

    # per-pixel champion covariates
    sig = expit((temp - params.t0) / params.k)
    habitat = ((rg > 0) | np.isin(lc, _WETLAND_CLASSES)).astype(np.float32)
    built = (lc == _BUILT_CLASS).astype(np.float32)
    hab_pix = np.log1p(_focal_mean(habitat, params.focal_km, res))
    blt_pix = np.log1p(_focal_mean(built, params.focal_km, res))

    # rasterize sectors and pop-weight each covariate to one row per sector
    ns = len(gdf)
    sect = rfeat.rasterize(
        ((geom, i) for i, geom in enumerate(gdf.geometry)),
        out_shape=temp.shape,
        transform=grid.rio.transform(),
        fill=-1,
        dtype="int32",
    )
    ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(popg)
    s = sect[ok]
    w = popg[ok].astype(np.float64)
    psum = np.bincount(s, weights=w, minlength=ns)

    def _pop_mean(a: np.ndarray) -> np.ndarray:
        num = np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=ns)
        return num / np.maximum(psum, 1e-9)

    cent = gdf.to_crs(4326).geometry.representative_point()
    return pd.DataFrame(
        {
            "location_id": gdf["location_id"].to_numpy(),
            "wp_pop": psum,
            "sig_temp": _pop_mean(sig),
            "hab": _pop_mean(hab_pix),
            "built": _pop_mean(blt_pix),
            "lon": cent.x.to_numpy(),
            "lat": cent.y.to_numpy(),
        }
    )


def fit_champion_risk(df: pd.DataFrame) -> tuple[pd.DataFrame, ChampionFit]:
    """Fit the within-district + empirical-Bayes risk model.

    Expects one row per sector with the :data:`FEATURE_COLS`, a district key
    ``parent``, and the (winsorised, WorldPop-denominated) target ``inc_wp_w``.

    Adds three columns to a copy of `df`:

    ``risk_within``
        within-district slopes + EB district intercept (primary best estimate
        for the region as observed; clipped at 0).
    ``risk_env``
        the pure environmental contribution ``X · beta_within`` (transferable,
        correct signs).
    ``risk_pooled``
        ordinary pooled regression prediction (reference; habitat sign flips
        under collinearity with temperature).
    """
    X = df[list(FEATURE_COLS)].to_numpy(dtype=float)
    y = df["inc_wp_w"].to_numpy(dtype=float)
    g = df["parent"].astype(str).to_numpy()

    # within-district (district-demeaned) slopes -> correct-sign covariate effects
    Xd, yd = X.copy(), y.copy()
    for d in np.unique(g):
        m = g == d
        Xd[m] -= Xd[m].mean(0)
        yd[m] -= yd[m].mean()
    beta_w = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
    env = X @ beta_w

    # empirical-Bayes district intercept (observed district burden) for the level
    e = y - env
    edf = pd.DataFrame({"g": g, "e": e})
    alpha = e.mean()
    rj = edf.groupby("g")["e"].mean() - alpha
    nj = edf.groupby("g")["e"].size()
    sigma2 = ((edf.e - edf.groupby("g")["e"].transform("mean")) ** 2).sum() / (
        len(e) - nj.size
    )
    tau2 = max(0.0, rj.var(ddof=1) - sigma2 * (1.0 / nj).mean())
    shrink = tau2 / (tau2 + sigma2 / nj)
    u = shrink * rj
    risk_within = np.clip(alpha + env + pd.Series(g).map(u).to_numpy(), 0, None)

    # pooled covariate model (reference)
    lin = LinearRegression().fit(X, y)
    risk_pooled = np.clip(lin.predict(X), 0, None)

    out = df.copy()
    out["risk_within"] = risk_within
    out["risk_env"] = env
    out["risk_pooled"] = risk_pooled

    info = ChampionFit(
        beta_within=dict(zip(FEATURE_COLS, np.round(beta_w, 1))),
        beta_pooled=dict(zip(FEATURE_COLS, np.round(lin.coef_, 1))),
        shrink_mean=float(shrink.mean()),
        n_sectors=len(out),
        n_districts=int(nj.size),
    )
    return out, info
