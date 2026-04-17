import pytest
from pathlib import Path

import xarray as xr

from chap_gis.io.chelsa import load_monthly_tas
from chap_gis.io.cache import cache_dir

def test_load_chelsa(rwanda_adm0):
    # test that data downloads and returns correctly
    year = 2021
    da = load_monthly_tas(rwanda_adm0, year)
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)
