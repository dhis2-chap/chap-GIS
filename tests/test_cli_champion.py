"""Tests for the champion static risk map.

Covers the library transforms in ``chap_gis.pipelines.champion_risk`` (the
within-district + empirical-Bayes fit, which is the scientific core) and an
end-to-end ``champion-map`` CLI smoke test with monkeypatched loaders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chap_gis as cgis
from chap_gis.cli import champion as cli_champion
from chap_gis.cli.champion import champion_map
from chap_gis.pipelines.champion_risk import (
    FEATURE_COLS,
    ChampionRiskParams,
    compute_sector_features,
    fit_champion_risk,
)


# ---------------------------------------------------------------------------
# Unit tests — fit_champion_risk (the scientific core, no rasters)
# ---------------------------------------------------------------------------


def _synthetic_sectors():
    """Sectors in two districts with a known within-district relationship.

    Within each district, burden rises with habitat and sigmoid-temp and falls
    with built-up; the two districts sit at different baseline levels. The
    within-district fit should recover those signs regardless of the baselines.
    """
    rng = np.random.default_rng(0)
    rows = []
    for district, base in (("D_hi", 400.0), ("D_lo", 80.0)):
        for _ in range(25):
            hab = rng.uniform(0, 1)
            built = rng.uniform(0, 1)
            sig = rng.uniform(0, 1)
            inc = base + 300 * hab - 200 * built + 150 * sig + rng.normal(0, 5)
            rows.append((district, sig, hab, built, max(inc, 0.0)))
    return pd.DataFrame(rows, columns=["parent", "sig_temp", "hab", "built", "inc_wp_w"])


def test_fit_recovers_within_district_signs():
    df = _synthetic_sectors()
    scored, info = fit_champion_risk(df)

    # habitat + and sigmoid-temp + ; built − (the mechanistic signs)
    assert info.beta_within["hab"] > 0
    assert info.beta_within["sig_temp"] > 0
    assert info.beta_within["built"] < 0
    assert info.n_districts == 2
    assert info.n_sectors == len(df)
    assert 0.0 <= info.shrink_mean <= 1.0


def test_fit_adds_expected_columns_and_is_nonnegative():
    df = _synthetic_sectors()
    scored, _ = fit_champion_risk(df)

    for col in ("risk_within", "risk_env", "risk_pooled"):
        assert col in scored.columns
    assert (scored["risk_within"] >= 0).all()
    assert (scored["risk_pooled"] >= 0).all()
    # the EB intercept keeps the two districts at distinct levels
    means = scored.groupby("parent")["risk_within"].mean()
    assert means["D_hi"] > means["D_lo"]


def test_fit_is_deterministic():
    df = _synthetic_sectors()
    a, _ = fit_champion_risk(df)
    b, _ = fit_champion_risk(df)
    pd.testing.assert_series_equal(a["risk_within"], b["risk_within"])


# ---------------------------------------------------------------------------
# Unit test — compute_sector_features (raster -> per-sector covariates)
# ---------------------------------------------------------------------------


def _gdf_with_location_id(xxx_adm2):
    gdf = xxx_adm2.copy()
    gdf["location_id"] = gdf["shapeID"].astype(str)
    return gdf


def test_compute_sector_features_one_row_per_sector(
    xxx_adm2, xxx_landcover_native, xxx_rice_native, xxx_tas_monthly, xxx_pop_yearly
):
    gdf = _gdf_with_location_id(xxx_adm2)
    # Real WorldCover arrives float; mirror that (uint8 can't mode-reproject with NaN fill).
    land = xxx_landcover_native.astype("float32").rio.write_crs("EPSG:4326")
    tas = xxx_tas_monthly.mean("time").rio.write_crs("EPSG:4326")
    pop = xxx_pop_yearly.isel(time=0, drop=True)
    params = ChampionRiskParams(resolution_m=10_000.0)

    feat = compute_sector_features(gdf, land=land, rice=xxx_rice_native, tas=tas, pop=pop, params=params)

    assert set(feat["location_id"]) == set(gdf["location_id"])
    assert len(feat) == len(gdf)
    assert set(FEATURE_COLS) <= set(feat.columns)
    assert (feat["wp_pop"] > 0).all()
    assert feat[list(FEATURE_COLS)].notna().all().all()
    # sigmoid-temp at a uniform 25 °C (>> t0=19) saturates near 1
    assert (feat["sig_temp"] > 0.9).all()


# ---------------------------------------------------------------------------
# End-to-end CLI smoke test (monkeypatched loaders)
# ---------------------------------------------------------------------------


@pytest.fixture
def adm2_with_districts(xxx_adm2):
    """ADM2 fixture with a ``parent`` (district) column — two sectors each."""
    gdf = xxx_adm2.copy()
    gdf["parent"] = gdf["shapeID"].map(
        {
            "XXX-ADM2-SW": "XXX-D-S", "XXX-ADM2-SE": "XXX-D-S",
            "XXX-ADM2-NW": "XXX-D-N", "XXX-ADM2-NE": "XXX-D-N",
        }
    )
    return gdf


@pytest.fixture
def patched_champion_loaders(
    monkeypatch, adm2_with_districts, xxx_landcover_native,
    xxx_rice_native, xxx_tas_monthly, xxx_pop_yearly,
):
    land = xxx_landcover_native.astype("float32").rio.write_crs("EPSG:4326")
    monkeypatch.setattr(cgis.io.boundaries, "load", lambda country, level: adm2_with_districts.copy())
    monkeypatch.setattr(cgis.io.worldcover, "load", lambda **kw: land)
    monkeypatch.setattr(cgis.io.rice, "load", lambda **kw: xxx_rice_native)
    monkeypatch.setattr(cgis.io.chelsa, "load", lambda *a, **kw: xxx_tas_monthly)
    monkeypatch.setattr(cgis.io.worldpop, "load", lambda **kw: xxx_pop_yearly)


def test_champion_map_writes_well_formed_risk_csv(
    patched_champion_loaders, tmp_path, xxx_disease_csv, adm2_with_districts
):
    out_path = tmp_path / "risk.csv"
    champion_map(
        country="XXX",
        level=2,
        input_csv=str(xxx_disease_csv),
        out_path=out_path,
        resolution_m=10_000.0,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)

    expected_cols = {
        "location_id", "sector_name", "district", "population",
        "risk_per1000_yr", "risk_normalized", "risk_env_score",
        "risk_pooled_per1000_yr", "observed_incidence_per1000_yr", "risk_rank",
    }
    assert expected_cols <= set(df.columns)

    # one row per sector present in the case CSV (all four), districts attached
    assert set(df["location_id"]) == set(adm2_with_districts["shapeID"])
    assert set(df["district"]) == {"XXX-D-S", "XXX-D-N"}
    assert df["sector_name"].notna().all()

    # risk_rank is a 1..n permutation; risk is non-negative and never NaN
    assert sorted(df["risk_rank"]) == list(range(1, len(df) + 1))
    assert (df["risk_per1000_yr"] >= 0).all()
    assert df["risk_per1000_yr"].notna().all()
    assert (df["population"] > 0).all()


def test_champion_map_drops_sectors_absent_from_case_csv(
    patched_champion_loaders, tmp_path, adm2_with_districts
):
    """A sector with no rows in the case CSV is dropped, not zero-filled — it
    has no observed burden to ground the EB-intercept fit."""
    # Keep SW + SE (district D-S) and NW (D-N); drop NE entirely. Two sectors in
    # one district keep the within-district / EB fit non-degenerate.
    csv = tmp_path / "partial.csv"
    pd.DataFrame(
        {
            "time": ["2018-01", "2018-02"] * 3,
            "location_id": [
                "XXX-ADM2-SW", "XXX-ADM2-SW",
                "XXX-ADM2-SE", "XXX-ADM2-SE",
                "XXX-ADM2-NW", "XXX-ADM2-NW",
            ],
            "disease": [10.0, 12.0, 4.0, 6.0, 8.0, 9.0],
        }
    ).to_csv(csv, index=False)

    out_path = tmp_path / "risk.csv"
    champion_map(
        country="XXX", level=2, input_csv=str(csv),
        out_path=out_path, resolution_m=10_000.0,
    )
    df = pd.read_csv(out_path)
    assert set(df["location_id"]) == {"XXX-ADM2-SW", "XXX-ADM2-SE", "XXX-ADM2-NW"}
    assert "XXX-ADM2-NE" not in set(df["location_id"])
