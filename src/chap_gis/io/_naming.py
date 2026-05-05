"""Cache prefix construction shared by all loaders.

The canonical prefix is ``{iso3_lower}_{dataset_id}`` when a country is given,
or just ``{dataset_id}`` for global / country-agnostic loads. ``dataset_id``
follows the climate-api ``{provider}_{var}_{period}`` style so caches are
trivially relatable to dhis2/climate-api outputs.
"""

from __future__ import annotations


def dataset_prefix(country_code: str | None, dataset_id: str) -> str:
    """Build the on-disk cache prefix for a (country, dataset) pair."""
    if country_code and country_code.strip():
        return f"{country_code.strip().lower()}_{dataset_id}"
    return dataset_id
