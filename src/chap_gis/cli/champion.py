"""``champion-map`` — build the champion static malaria risk map.

End-to-end parallel to :func:`chap_gis.cli.dynamic.dynamic_periods`: from a
country + a per-sector case CSV, load the environmental rasters, build the
per-sector covariates, fit the within-district + empirical-Bayes risk model and
write one risk row per sector.

I/O only — the feature/model transforms live in
:mod:`chap_gis.pipelines.champion_risk`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import chap_gis as cgis
from chap_gis.cli.dynamic import chunk, prepare_boundaries
from chap_gis.pipelines.champion_risk import (
    ChampionRiskParams,
    compute_sector_features,
    fit_champion_risk,
)

logger = logging.getLogger(__name__)


def get_case_data(input_csv: str, gdf) -> tuple[pd.DataFrame, int, "pd.Series | None"]:
    """Per-sector observed cases from a (sector, time) case CSV.

    Accepts the same loose column conventions as ``dynamic-periods``
    (``time``/``date``/``time_period``, ``location_id``/``location``,
    ``disease``/``disease_cases``/``cases``). Sums cases per sector over the
    full period.

    Returns ``(cases_df[location_id, cases], n_years, names_or_None)`` where
    ``n_years`` is the number of distinct calendar years (for annualising risk)
    and ``names`` maps ``location_id -> sector_name`` if the CSV carries one.
    """
    if not (input_csv and Path(input_csv).exists()):
        raise FileNotFoundError(
            f"case CSV not found: {input_csv!r}. The champion map is fit from "
            "observed cases — provide a per-sector(-time) case CSV."
        )

    df = pd.read_csv(input_csv)
    tcol = next(c for c in ("time", "date", "time_period") if c in df)
    lcol = next(c for c in ("location_id", "location") if c in df)
    dcol = next(c for c in ("disease", "disease_cases", "cases") if c in df)
    df = df.rename(columns={tcol: "time", lcol: "location_id", dcol: "disease"})
    df["location_id"] = df["location_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["location_id", "time", "disease"])

    n_years = int(df["time"].dt.year.nunique()) or 1

    names = None
    for ncol in ("location_name", "sector_name", "name"):
        if ncol in df:
            names = df.drop_duplicates("location_id").set_index("location_id")[ncol]
            names.name = "sector_name"
            break

    cases = (
        df.groupby("location_id")["disease"].sum().rename("cases").reset_index()
    )
    return cases, n_years, names


def _district_and_names(country: str, level: int):
    """District (``parent``) and sector-name lookups from the boundary table."""
    b = cgis.io.boundaries.load(country, level=level)
    id_col = next((c for c in ("shapeID", "shapeName") if c in b.columns), None)
    b["location_id"] = (b[id_col] if id_col else b.index).astype(str)
    parent = (
        b.set_index("location_id")["parent"] if "parent" in b.columns else None
    )
    names = (
        b.set_index("location_id")["shapeName"] if "shapeName" in b.columns else None
    )
    return parent, names


def champion_map(
    country: str,
    level: int = 5,
    input_csv: str = "./data/inputs/disease-data.csv",
    out_path: Path = Path("champion_risk.csv"),
    raster_year: int = 2021,
    resolution_m: float = 100.0,
):
    """Build the champion static risk map for `country` at admin `level`.

    Loads the environmental rasters for ``raster_year``, builds the per-sector
    covariates (sigmoid-temp, focal habitat, focal built-up), aggregates the
    observed cases from `input_csv`, fits the within-district + empirical-Bayes
    model, and writes one risk row per sector to `out_path`.

    Output columns: ``location_id, sector_name, district, population,
    risk_per1000_yr`` (PRIMARY), ``risk_normalized, risk_env_score``
    (transferable environmental part), ``risk_pooled_per1000_yr`` (reference),
    ``observed_incidence_per1000_yr, risk_rank``.
    """
    params = ChampionRiskParams(raster_year=raster_year, resolution_m=resolution_m)
    logger.info("Building champion risk map for %s (level %d, year %d)",
                country, level, raster_year)

    gdf = prepare_boundaries(country, level)
    parent, bnames = _district_and_names(country, level)
    if parent is None:
        raise ValueError(
            f"boundary level {level} for {country} has no 'parent' (district) "
            "column; the within-district model needs a district grouping."
        )

    cases, n_years, csv_names = get_case_data(input_csv, gdf)
    names = csv_names if csv_names is not None else bnames
    logger.info("Loaded cases for %d sectors over %d year(s)", len(cases), n_years)

    # --- raster I/O (environmental snapshot for `raster_year`) ----------------
    yr = params.raster_year
    aoi = cgis.aoi.buffered(gdf, params.aoi_buffer_deg)
    wc_year = max(2020, min(yr, 2021))  # WorldCover coverage is 2020–2021
    logger.info("Loading rasters (WorldCover %d, CHELSA/WorldPop %d)...", wc_year, yr)

    land = chunk(cgis.io.worldcover.load(aoi=aoi, start=wc_year, end=wc_year, country_code=country))
    rice = cgis.io.rice.load(country_code=country)
    gdf0 = cgis.io.boundaries.load(country, level=0)
    tas = (
        cgis.io.chelsa.load(gdf0, start=f"{yr}-01", end=f"{yr}-12", country_code=country)
        .mean("time")
        .rio.write_crs("EPSG:4326")
    )
    pop = cgis.io.worldpop.load(country_code=country, start=yr, end=yr).squeeze(drop=True)
    pop.rio.write_crs("EPSG:4326", inplace=True)

    # --- transforms (library) -------------------------------------------------
    logger.info("Computing per-sector covariates...")
    feat = compute_sector_features(gdf, land=land, rice=rice, tas=tas, pop=pop, params=params)

    feat["parent"] = feat["location_id"].map(parent)
    # Inner-join: a sector absent from the case CSV has no observed burden, so it
    # cannot be grounded in the burden-based fit (the EB district intercept would
    # otherwise float it to a fabricated risk). Genuine in-CSV zeros are kept.
    feat = feat.merge(cases, on="location_id", how="inner")

    # WorldPop-denominated incidence target, winsorised
    inc_wp = feat["cases"] / feat["wp_pop"].replace(0, np.nan) * 1000.0
    cap = inc_wp.quantile(params.winsor_quantile)
    feat["inc_wp"] = inc_wp
    feat["inc_wp_w"] = inc_wp.clip(upper=cap)

    fit_df = feat[(feat.wp_pop > 0) & feat.parent.notna() & feat.inc_wp_w.notna()].reset_index(drop=True)
    if fit_df.empty:
        raise ValueError("no sectors with population, district and cases — nothing to fit.")

    logger.info("Fitting within-district + empirical-Bayes model on %d sectors...", len(fit_df))
    scored, info = fit_champion_risk(fit_df)

    # --- assemble output (mirror results/export_percapita_risk.py) -----------
    rw = scored["risk_within"].to_numpy()
    env = scored["risk_env"].to_numpy()
    rng = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else np.zeros_like(a)

    out = pd.DataFrame(
        {
            "location_id": scored.location_id,
            "sector_name": scored.location_id.map(names) if names is not None else pd.NA,
            "district": scored.parent,
            "population": scored.wp_pop.round().astype(int),
            "risk_per1000_yr": np.round(rw / n_years, 2),
            "risk_normalized": np.round(rng(rw), 4),
            "risk_env_score": np.round(rng(env), 4),
            "risk_pooled_per1000_yr": np.round(scored.risk_pooled.to_numpy() / n_years, 2),
            "observed_incidence_per1000_yr": np.round(scored.inc_wp.to_numpy() / n_years, 2),
        }
    )
    out["risk_rank"] = out["risk_per1000_yr"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("risk_rank").reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info(
        "within-district slopes [sig_temp, habitat, built] = %s  (habitat +, built −: mechanistic)",
        info.beta_within,
    )
    logger.info("pooled slopes (reference)                          = %s", info.beta_pooled)
    logger.info("EB district-intercept mean shrinkage = %.2f", info.shrink_mean)
    logger.info("Done. Wrote %s (%d sectors, %d districts).",
                out_path, info.n_sectors, info.n_districts)
    return out
