import numpy as np
import pytest
import xarray as xr

from chap_gis.grid_check import same_grid


def _da(shape=(20, 20), crs="EPSG:32636", res=30.0, origin=(0.0, 0.0)):
    ny, nx = shape
    x0, y0 = origin
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny)[::-1] + 0.5) * res
    return xr.DataArray(
        np.zeros(shape, dtype="float32"),
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
    ).rio.write_crs(crs)


def _da_3d(time_len=3, **kwargs):
    spatial = _da(**kwargs)
    arr = np.broadcast_to(spatial.values, (time_len, *spatial.shape)).copy()
    return xr.DataArray(
        arr,
        dims=("time", "y", "x"),
        coords={"time": np.arange(time_len), "y": spatial["y"], "x": spatial["x"]},
    ).rio.write_crs(spatial.rio.crs)


def test_matching_grids_pass():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray):
        return a + b

    out = f(_da(), _da())
    assert out.shape == (20, 20)


def test_crs_mismatch_raises():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray):
        return a

    with pytest.raises(ValueError, match="CRS mismatch"):
        f(_da(crs="EPSG:32636"), _da(crs="EPSG:4326"))


def test_shape_mismatch_raises():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray):
        return a

    with pytest.raises(ValueError, match="shape mismatch"):
        f(_da(shape=(20, 20)), _da(shape=(20, 30)))


def test_transform_mismatch_raises():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray):
        return a

    with pytest.raises(ValueError, match="transform mismatch"):
        f(_da(origin=(0.0, 0.0)), _da(origin=(100.0, 0.0)))


def test_optional_none_argument_skipped():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray | None = None):
        return a

    f(_da(), None)


def test_optional_value_still_checked():
    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray | None = None):
        return a

    with pytest.raises(ValueError, match="CRS mismatch"):
        f(_da(crs="EPSG:32636"), _da(crs="EPSG:4326"))


def test_rank_mismatched_but_aligned_pass():
    """2D + 3D (time, y, x) sharing the spatial grid should pass."""

    @same_grid
    def f(a: xr.DataArray, b: xr.DataArray):
        return a

    f(_da(), _da_3d())


def test_explicit_param_subset():
    @same_grid("a", "b")
    def f(a, b, c):
        return a

    f(_da(), _da(), _da(crs="EPSG:4326"))  # c not checked

    with pytest.raises(ValueError, match="CRS mismatch"):
        f(_da(crs="EPSG:32636"), _da(crs="EPSG:4326"), _da())


def test_unknown_param_name_raises():
    with pytest.raises(TypeError, match="unknown parameter"):

        @same_grid("nope")
        def f(a: xr.DataArray):
            return a


def test_non_dataarray_value_raises():
    @same_grid("a", "b")
    def f(a, b):
        return a

    with pytest.raises(TypeError, match="expected xr.DataArray"):
        f(_da(), "not-an-array")


def test_single_dataarray_argument_no_check():
    @same_grid
    def f(a: xr.DataArray):
        return a

    f(_da())
