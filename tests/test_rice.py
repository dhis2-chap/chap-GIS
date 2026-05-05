import pytest
import xarray as xr

from chap_gis.io.rice import _ISO3_TO_ZENODO_PREFIX, download, load


def test_load_rice(xxx_adm0):
    # test that data downloads and returns correctly (uses pre-staged file)
    da = load(country_code='RWA')
    assert isinstance(da, xr.DataArray)


def test_download_rejects_unknown_country():
    with pytest.raises(ValueError, match="not in the Jiang"):
        download(country_code="USA")


def test_country_map_covers_test_inputs():
    # Pre-staged inputs (rwa, mwi) must be in the map so download() is idempotent.
    assert "RWA" in _ISO3_TO_ZENODO_PREFIX
    assert "MWI" in _ISO3_TO_ZENODO_PREFIX


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
