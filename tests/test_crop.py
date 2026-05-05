import pytest
from pathlib import Path

import xarray as xr

from chap_gis.io.crop import load
from chap_gis.io.cache import cache_dir

@pytest.mark.integration
def test_load_rice(rwanda_adm0):
    # test that data downloads and returns correctly
    da = load(rwanda_adm0)
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)
