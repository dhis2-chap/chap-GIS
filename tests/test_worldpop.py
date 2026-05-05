import pytest
from pathlib import Path
import logging

import xarray as xr

import chap_gis
from chap_gis.io.worldpop import load
from chap_gis.io.cache import cache_dir
from chap_gis.grid import reproject_population_to

@pytest.mark.integration
def test_load_population():
    country = 'RWA'
    year = 2021

    # test that data downloads and returns correctly
    da = load(country, year)
    assert isinstance(da, xr.DataArray)
    
    # test that source file is located in cachedir
    pth = Path(da.encoding['source'])
    assert str(cache_dir()) in str(pth)

@pytest.mark.integration
def test_population_reprojects():
    # load population
    country = 'RWA'
    year = 2021
    population = load(country, year)

    # load country
    aoi = chap_gis.io.boundaries.load(country, level=0)

    # create grid
    resolution_m = 30
    grid = chap_gis.grid.build_grid(
        aoi, resolution=resolution_m / 111_000, crs="EPSG:4326"
    )
    logging.info(population)
    logging.info(f'Population sum {population.sum().compute().item()}, mean {population.mean().compute().item()}')

    # reproject
    population_reproj = population.pipe(reproject_population_to, grid, 'nearest').compute()
    logging.info(population_reproj)
    logging.info(f'Population reprojected sum {population_reproj.sum().compute().item()}, mean {population_reproj.mean().compute().item()}')

    # assert approx totals preserved
    poptot_orig = population.sum().compute().item()
    poptot_reproj = population_reproj.sum().compute().item()
    poptot_rel_change = abs(poptot_orig - poptot_reproj) / poptot_orig
    logging.info(f'Population relative change after reprojection {poptot_rel_change}')
    assert (poptot_rel_change < 0.02)  # <2 percent change
