import pytest
from pathlib import Path

import xarray as xr

from chap_gis.io.elevation import load
from chap_gis.io.cache import cache_dir

def test_load_elevation():
    bbox = [10, 59, 11+1, 60+1]  # example: Oslo-ish

    # test that data downloads and returns correctly
    da = load(bbox)
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)
