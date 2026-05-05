"""Every loader module satisfies the ``DataSource`` protocol structurally."""

import pytest

from chap_gis.io import SOURCES, DataSource


@pytest.mark.parametrize("dataset_id", sorted(SOURCES))
def test_module_satisfies_datasource(dataset_id):
    module = SOURCES[dataset_id]
    assert isinstance(module, DataSource), (
        f"{module.__name__} does not satisfy DataSource protocol "
        f"(missing dataset_id, load, or download)"
    )
    assert module.dataset_id == dataset_id


def test_dataset_ids_are_climate_api_shaped():
    for dataset_id in SOURCES:
        assert dataset_id.islower(), dataset_id
        assert " " not in dataset_id, dataset_id
        assert "_" in dataset_id, dataset_id
