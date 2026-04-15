import numpy as np

from chap_gis.suitability import thermal_suitability


def test_peaks_at_t_opt(temperature):
    s = thermal_suitability(temperature)
    assert np.nanmax(s.values) == 1.0
    assert np.nanargmax(s.values.ravel()) is not None


def test_zero_below_t_min(temperature):
    s = thermal_suitability(temperature)
    # rows 0-1 are 10°C (below t_min=16)
    assert np.all(s.values[:2, :] == 0.0)


def test_preserves_nan(temperature):
    s = thermal_suitability(temperature)
    assert np.all(np.isnan(s.values[-2:, :]))


def test_crs_propagated(temperature):
    s = thermal_suitability(temperature)
    assert s.rio.crs == temperature.rio.crs
