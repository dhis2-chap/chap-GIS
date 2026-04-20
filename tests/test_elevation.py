import pytest
from pathlib import Path
import logging

import xarray as xr

from chap_gis.io.elevation import load
from chap_gis.io.cache import cache_dir

def test_load_elevation(rwanda_adm0):
    # test that data downloads and returns correctly
    da = load(rwanda_adm0)
    assert isinstance(da, xr.DataArray)

    logging.info(da)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)
