import pytest
from pathlib import Path

import xarray as xr

from chap_gis.io.worldpop import load
from chap_gis.io.cache import cache_dir

def test_load_population():
    country = 'RWA'
    year = 2026

    # test that data downloads and returns correctly
    da = load(country, year)
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)