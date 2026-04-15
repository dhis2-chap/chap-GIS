import numpy as np
import pytest
import xarray as xr

from chap_gis.exposure import exposure


def _mk(arr, template):
    return xr.DataArray(arr, dims=template.dims, coords=template.coords).rio.write_crs(
        template.rio.crs
    )


def test_exponential_decay_with_distance(elevation):
    breeding_np = np.zeros(elevation.shape, dtype=bool)
    breeding_np[0, 0] = True
    breeding = _mk(breeding_np, elevation)

    expo = exposure(breeding, elevation, None, pixel_m=30.0)
    # At breeding pixel, exposure = 1
    assert expo.values[0, 0] == pytest.approx(1.0)
    # Decays exponentially: at (0, 1), d = 30m, λ = 651m
    assert expo.values[0, 1] == pytest.approx(np.exp(-30.0 / 651.0), rel=1e-5)


def test_raises_on_crs_mismatch(elevation):
    breeding = xr.DataArray(
        np.ones(elevation.shape, dtype=bool),
        dims=elevation.dims,
        coords=elevation.coords,
    ).rio.write_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        exposure(breeding, elevation, None, pixel_m=30.0)


def test_raises_on_empty_breeding(elevation):
    breeding = _mk(np.zeros(elevation.shape, dtype=bool), elevation)
    # exposure is lazy — the error surfaces when the dask graph is computed
    with pytest.raises(ValueError, match="no breeding sites"):
        exposure(breeding, elevation, None, pixel_m=30.0).compute()
