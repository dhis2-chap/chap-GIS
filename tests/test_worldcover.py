import pytest
from pathlib import Path
import logging

import xarray as xr

from chap_gis.io.worldcover import load
from chap_gis.io.cache import cache_dir

@pytest.mark.integration
def test_load_worldcover(rwanda_adm0):
    # test that data downloads and returns correctly
    year = 2021
    da = load(rwanda_adm0, year)
    logging.info(da)
    
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)
