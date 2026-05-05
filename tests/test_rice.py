import pytest
from pathlib import Path

import xarray as xr

from chap_gis.io.rice import load
from chap_gis.io.cache import cache_dir

def test_load_rice(xxx_adm0):
    # test that data downloads and returns correctly
    da = load(country_code='RWA')
    assert isinstance(da, xr.DataArray)
