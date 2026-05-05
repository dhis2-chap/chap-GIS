from pathlib import Path

import pytest
import xarray as xr

from chap_gis.io.rice import _ISO3_TO_ZENODO_PREFIX, download, load


FIXTURES = Path(__file__).parent / "data"


def test_load_rice(monkeypatch):
    # Use the synthetic XXX raster so the test stays offline regardless of
    # whether anyone has pre-staged data/inputs/.
    monkeypatch.setattr("chap_gis.io.rice._inputs_dir", lambda: FIXTURES)
    da = load(country_code="XXX")
    assert isinstance(da, xr.DataArray)
    assert da.name == "rice"


def test_download_rejects_unknown_country(tmp_path, monkeypatch):
    # Isolate inputs dir so the test doesn't accidentally find a pre-staged file.
    monkeypatch.setattr("chap_gis.io.rice._inputs_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match="not in the Jiang"):
        download(country_code="USA")


def test_country_map_covers_real_countries():
    # Spot-check a few; the full list is in rice._ISO3_TO_ZENODO_PREFIX.
    for iso3 in ("RWA", "MWI", "GMB", "KEN"):
        assert iso3 in _ISO3_TO_ZENODO_PREFIX


@pytest.mark.integration
def test_download_zenodo(tmp_path, monkeypatch):
    # Redirect the inputs dir to tmp_path so the test never touches data/inputs.
    monkeypatch.setattr("chap_gis.io.rice._inputs_dir", lambda: tmp_path)
    # The smallest single-file country in the dataset (~824 KB) keeps the test fast.
    files = download(country_code="GMB")
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert files[0].exists()
    assert files[0].stat().st_size > 0
