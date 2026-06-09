"""Tests for ``chap_gis.cli.dynamic`` — the multi-month / multi-year pipeline.

These tests deliberately target the correctness concerns documented in
``src/chap_gis/cli/dynamic.py`` (see module docstring there). Several of them
are expected to **fail on the current branch** and should turn green when the
listed bugs are fixed.

The orchestration is hard to test today because every external loader is
called from inside the function body and ``MalariaExposureParams`` is
constructed inline. Until the refactor described in ``dynamic.py`` lands,
these tests rely on heavy monkeypatching — once the refactor is done, the
end-to-end test can call the pure inner function directly with the same
fixtures and drop most of the patches.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import chap_gis as cgis
from chap_gis.aggregate import (
    aggregate_population_by_year,
    aggregate_temperature_by_month,
)
from chap_gis.cli import dynamic as cli_dynamic
from chap_gis.cli.dynamic import (
    _simulate_monthly_disease_data,
    dynamic_periods,
    get_health_data,
    prepare_boundaries,
)


# ---------------------------------------------------------------------------
# Unit tests — aggregators
# ---------------------------------------------------------------------------


def _gdf_with_location_id(xxx_adm2):
    """XXX ADM2 fixture, plus a ``location_id`` column matching shapeID."""
    gdf = xxx_adm2.copy()
    gdf["location_id"] = gdf["shapeID"].astype(str)
    return gdf


def test_aggregate_population_returns_expected_shape(xxx_adm2, xxx_pop_yearly):
    gdf = _gdf_with_location_id(xxx_adm2)
    agg = aggregate_population_by_year(xxx_pop_yearly, gdf)

    assert set(agg.dims) == {"location_id", "time"}
    assert agg.sizes["location_id"] == len(gdf)
    assert agg.sizes["time"] == xxx_pop_yearly.sizes["time"]
    assert "population" in agg.data_vars
    assert set(agg.location_id.values) == set(gdf["location_id"].values)
    # population uses the raster's native time dtype — should be datetime64
    assert np.issubdtype(agg.time.dtype, np.datetime64)


def test_aggregate_temperature_time_is_datetime(xxx_adm2, xxx_tas_monthly):
    """Regression: the final inner merge in ``dynamic_periods`` joins on time.

    ``health_xr`` and ``pop_agg`` carry ``datetime64`` time. This aggregator
    currently casts time to ``YYYY-MM`` strings, which makes the inner join
    drop every row (the e2e test below pins the same bug from the other end).
    """
    gdf = _gdf_with_location_id(xxx_adm2)
    agg = aggregate_temperature_by_month(xxx_tas_monthly, gdf)

    assert "tas" in agg.data_vars
    assert np.issubdtype(agg.time.dtype, np.datetime64), (
        f"expected datetime64 time, got {agg.time.dtype}. "
        "String-typed time breaks the final inner merge in dynamic_periods."
    )


# ---------------------------------------------------------------------------
# Unit tests — small helpers
# ---------------------------------------------------------------------------


def test_simulate_monthly_disease_data_shape(xxx_adm2):
    gdf = _gdf_with_location_id(xxx_adm2)
    df = _simulate_monthly_disease_data(gdf, "location_id")
    assert set(df.columns) == {"location_id", "time", "disease"}
    assert len(df) == len(gdf) * 36
    assert df["time"].nunique() == 36
    assert set(df["location_id"]) == set(gdf["location_id"])


def test_prepare_boundaries_standardizes_columns(monkeypatch, xxx_adm2):
    monkeypatch.setattr(
        cgis.io.boundaries, "load", lambda country, level: xxx_adm2.copy()
    )
    gdf = prepare_boundaries("XXX", level=2)
    assert list(gdf.columns) == ["geometry", "location_id"]
    assert set(gdf["location_id"]) == {"XXX-ADM2-SW", "XXX-ADM2-SE", "XXX-ADM2-NW", "XXX-ADM2-NE"}
    assert gdf.crs.to_string() == "EPSG:4326"


def test_get_health_data_csv_renames_chap_columns(xxx_disease_csv, xxx_adm2):
    gdf = _gdf_with_location_id(xxx_adm2)
    ds = get_health_data(str(xxx_disease_csv), gdf)
    assert set(ds.dims) == {"location_id", "time"}
    assert "disease" in ds.data_vars
    assert np.issubdtype(ds.time.dtype, np.datetime64)
    assert ds.sizes["time"] == 36
    assert set(ds.location_id.values) == {"XXX-ADM2-SW", "XXX-ADM2-SE", "XXX-ADM2-NW", "XXX-ADM2-NE"}


def test_get_health_data_drops_extra_columns(tmp_path, xxx_adm2):
    """Only the disease signal survives — extra CSV columns (e.g. a prior run's
    tas/population) must not ride into the final merge and collide with the
    computed regional aggregates."""
    gdf = _gdf_with_location_id(xxx_adm2)
    df = _simulate_monthly_disease_data(gdf, "location_id")
    df["tas"] = 25.0
    df["population"] = 1000.0
    df["pop_exposure"] = 1.0
    csv = tmp_path / "rich.csv"
    df.to_csv(csv, index=False)

    ds = get_health_data(str(csv), gdf)
    assert set(ds.data_vars) == {"disease"}


def test_get_health_data_falls_back_to_simulation(tmp_path, xxx_adm2):
    gdf = _gdf_with_location_id(xxx_adm2)
    ds = get_health_data(str(tmp_path / "missing.csv"), gdf)
    assert "disease" in ds.data_vars
    assert ds.sizes["time"] == 36


# ---------------------------------------------------------------------------
# End-to-end CLI test (heavy monkeypatching — see module docstring)
# ---------------------------------------------------------------------------


def _stub_run_exposure_pipeline(
    *,
    aoi,
    landcover_native,
    elev_native,
    tas_monthly,
    population_native,
    rice_native,
    params,
):
    """Stand in for ``pipelines.malaria_exposure.run``.

    Returns a properly georeferenced pixel-level Dataset with ``pop_exposure``.
    We do not stub ``_calculate_monthly_exposure_from_vars`` itself because
    that function contains the string-time bug we want the e2e assertion to
    catch.
    """
    times = tas_monthly.time.values
    xs = (np.arange(8) + 0.5) / 8.0
    ys = (np.arange(8)[::-1] + 0.5) / 8.0
    data = np.ones((len(times), 8, 8), dtype="float32") * 10.0
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
        name="pop_exposure",
    ).rio.write_crs("EPSG:4326")
    return xr.Dataset({"pop_exposure": da})


@pytest.fixture
def patched_loaders(
    monkeypatch,
    xxx_adm2,
    xxx_pop_yearly,
    xxx_tas_monthly,
    xxx_elev_native,
    xxx_landcover_native,
    xxx_rice_native,
):
    """Patch every external loader the CLI touches.

    Also patches ``run_exposure_pipeline`` so the heavyweight pixel pipeline
    is bypassed — we are testing CLI orchestration, not the exposure model.
    Returns a dict of call counters so tests can assert how often each loader
    fired (e.g. to catch the per-year reload of static layers).
    """
    counts = {"worldcover": 0, "elevation": 0, "rice": 0,
              "worldpop": 0, "chelsa": 0, "boundaries": 0}

    def _count(name, value):
        def _inner(*a, **kw):
            counts[name] += 1
            return value
        return _inner

    monkeypatch.setattr(cgis.io.boundaries, "load", _count("boundaries", xxx_adm2))
    monkeypatch.setattr(cgis.io.worldpop, "load", _count("worldpop", xxx_pop_yearly))
    monkeypatch.setattr(cgis.io.chelsa, "load", _count("chelsa", xxx_tas_monthly))
    monkeypatch.setattr(cgis.io.worldcover, "load", _count("worldcover", xxx_landcover_native))
    monkeypatch.setattr(cgis.io.elevation, "load", _count("elevation", xxx_elev_native))
    monkeypatch.setattr(cgis.io.rice, "load", _count("rice", xxx_rice_native))
    monkeypatch.setattr(cli_dynamic, "run_exposure_pipeline", _stub_run_exposure_pipeline)

    return counts


def test_dynamic_periods_writes_full_csv(
    patched_loaders, tmp_path, xxx_disease_csv, xxx_adm2
):
    """End-to-end smoke test. Expected to fail today due to the time-dtype bug.

    The closing ``xr.merge(..., join="inner")`` aligns datetime64 ``health_xr``
    against string-typed ``tas_agg`` / ``exposure_ds`` and produces an empty
    intersection, so the output CSV has 0 rows (or all-NaN env columns).
    """
    out_path = tmp_path / "out.csv"
    dynamic_periods(
        country="XXX",
        level=2,
        inter=True,
        input_csv=str(xxx_disease_csv),
        out_path=out_path,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)

    assert {"time", "location_id", "disease", "population", "tas", "pop_exposure"} <= set(df.columns)

    expected_times = (
        pd.date_range("2017-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist()
    )
    expected_locations = set(xxx_adm2["shapeID"].astype(str))
    expected_pairs = {(loc, t) for loc in expected_locations for t in expected_times}
    actual_pairs = set(zip(df["location_id"].astype(str), df["time"].astype(str)))

    assert actual_pairs == expected_pairs, (
        f"output is missing {len(expected_pairs - actual_pairs)} (location, time) pairs "
        f"and has {len(actual_pairs - expected_pairs)} unexpected ones. "
        "Today this fails because the closing xr.merge(..., join='inner') aligns "
        "datetime64 health_xr against string-typed tas_agg / exposure_ds and "
        "produces an empty intersection."
    )
    assert not df.duplicated(subset=["location_id", "time"]).any(), (
        "duplicate (location_id, time) rows in output"
    )
    assert df["population"].notna().all(), "population NaNs imply the inner merge collapsed"
    assert df["tas"].notna().all(), "tas NaNs imply the inner merge collapsed"
    assert df["pop_exposure"].notna().all()


def test_dynamic_periods_loads_static_layers_once(
    patched_loaders, tmp_path, xxx_disease_csv
):
    """Static layers must not be reloaded per year.

    The disease CSV covers 3 years; worldcover/elevation/rice are static across
    years. They are currently loaded inside the yearly loop, so this test will
    fail until the loads are hoisted out (and worldcover's year clamp is
    addressed — see concern (3) in dynamic.py).
    """
    out_path = tmp_path / "out.csv"
    dynamic_periods(
        country="XXX",
        level=2,
        inter=True,
        input_csv=str(xxx_disease_csv),
        out_path=out_path,
    )

    assert patched_loaders["worldcover"] == 1, (
        f"worldcover loaded {patched_loaders['worldcover']} times — "
        "static layers should be hoisted out of the yearly loop"
    )
    assert patched_loaders["elevation"] == 1
    assert patched_loaders["rice"] == 1
    # CHELSA + WorldPop are loaded once each by get_environmental_data
    assert patched_loaders["chelsa"] == 1
    assert patched_loaders["worldpop"] == 1


# ---------------------------------------------------------------------------
# Parameter-sweep path (grid_config) — exercises the real exposure machinery
# ---------------------------------------------------------------------------


def test_dynamic_periods_sweep_writes_one_column_set_per_combo(
    patched_loaders, monkeypatch, tmp_path, xxx_disease_csv, xxx_adm2,
    xxx_landcover_native, xxx_rice_native,
):
    """With a grid_config, the CSV gains a pop_exposure/mean column per combo
    and a sidecar params manifest.

    Uses a coarse base resolution so the analysis grid is tiny — this path runs
    the *real* reproject/breeding/distance-field code (run_exposure_pipeline is
    bypassed), unlike the stubbed e2e test above.
    """
    # Real WorldCover arrives float (open_mfdataset masks to NaN nodata); the
    # uint8 fixture can't be mode-reprojected with a NaN fill, so mirror reality.
    lc_float = xxx_landcover_native.astype("float32").rio.write_crs("EPSG:4326")
    monkeypatch.setattr(cgis.io.worldcover, "load", lambda **kw: lc_float)
    rice_float = (
        xxx_rice_native.squeeze("band", drop=True)
        if "band" in xxx_rice_native.dims
        else xxx_rice_native
    ).astype("float32").rio.write_crs("EPSG:4326")
    monkeypatch.setattr(cgis.io.rice, "load", lambda **kw: rice_float)

    grid_cfg = tmp_path / "grid.json"
    grid_cfg.write_text(
        json.dumps(
            {
                "lambda_m": [400, 651],
                "gamma_m": [22.5],
                "water_edge_buffer_pixels": [1, 2],
                "thermal": [{"t_opt": 25, "sigma": 5, "t_min": 16, "t_max": 34}],
                "base": {"resolution_m": 3000.0},
            }
        )
    )
    out_path = tmp_path / "sweep.csv"
    dynamic_periods(
        country="XXX",
        level=2,
        inter=True,
        input_csv=str(xxx_disease_csv),
        out_path=out_path,
        grid_config=grid_cfg,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)

    # 2 buffers * 1 thermal * 2 lambda * 1 gamma = 4 combos
    tags = [f"expo_{i:03d}" for i in range(4)]
    expected_cols = {"location_id", "time", "disease", "tas", "population"}
    for t in tags:
        expected_cols |= {f"pop_exposure__{t}", f"mean_exposure_per_person__{t}"}
    assert expected_cols <= set(df.columns)

    # Full (location, time) coverage, same as the single-combo e2e test.
    expected_times = (
        pd.date_range("2017-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist()
    )
    expected_locations = set(xxx_adm2["shapeID"].astype(str))
    expected_pairs = {(loc, t) for loc in expected_locations for t in expected_times}
    actual_pairs = set(zip(df["location_id"].astype(str), df["time"].astype(str)))
    assert actual_pairs == expected_pairs

    # Exposure columns are populated (not all-NaN) for at least one combo.
    assert df["pop_exposure__expo_000"].notna().any()

    # Manifest sidecar maps every tag to its parameters.
    manifest_path = out_path.with_suffix(".params.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["columns"]) == set(tags)
    assert manifest["columns"]["expo_000"]["lambda_m"] == 400
    assert manifest["columns"]["expo_001"]["lambda_m"] == 651
    assert manifest["columns"]["expo_000"]["water_edge_buffer_pixels"] == 1
    assert manifest["columns"]["expo_002"]["water_edge_buffer_pixels"] == 2
