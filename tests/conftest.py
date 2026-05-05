"""Shared test fixtures: tiny synthetic DataArrays."""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pathlib import Path

from chap_gis.io import boundaries


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require network access or external credentials",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def outputs_folder():
    return SCRIPT_DIR.parent / 'data' / 'outputs'


def _grid(shape=(20, 20), crs="EPSG:32636", res=30.0):
    ny, nx = shape
    xs = np.arange(nx) * res
    ys = np.arange(ny)[::-1] * res
    da = xr.DataArray(
        np.zeros(shape, dtype="float32"),
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
    )
    return da.rio.write_crs(crs)


@pytest.fixture
def grid():
    return _grid()


@pytest.fixture
def rwanda_adm0():
    gdf = boundaries.load('RWA', level=0)
    return gdf


@pytest.fixture
def rwanda_adm2():
    gdf = boundaries.load('RWA', level=2)
    return gdf


@pytest.fixture
def xxx_adm0():
    import geopandas as gpd
    return gpd.read_file(DATA_DIR / "geoBoundaries-XXX-ADM0.geojson")


@pytest.fixture
def xxx_adm2():
    import geopandas as gpd
    return gpd.read_file(DATA_DIR / "geoBoundaries-XXX-ADM2.geojson")


@pytest.fixture
def temperature(grid):
    ny, nx = grid.shape
    temp = np.full((ny, nx), 25.0, dtype="float32")
    temp[:2, :] = 10.0  # below t_min
    temp[-2:, :] = np.nan
    return xr.DataArray(
        temp, dims=grid.dims, coords=grid.coords
    ).rio.write_crs(grid.rio.crs)


@pytest.fixture
def landcover(grid):
    ny, nx = grid.shape
    lc = np.full((ny, nx), 10, dtype="uint8")  # trees
    lc[5, 5] = 80  # water
    lc[10:12, 10:12] = 95  # wetland
    return xr.DataArray(
        lc, dims=grid.dims, coords=grid.coords
    ).rio.write_crs(grid.rio.crs)


@pytest.fixture
def elevation(grid):
    ny, nx = grid.shape
    elev = np.full((ny, nx), 1500.0, dtype="float32")
    return xr.DataArray(
        elev, dims=grid.dims, coords=grid.coords
    ).rio.write_crs(grid.rio.crs)
