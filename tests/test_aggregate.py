import logging

import pytest
import rioxarray
import xarray as xr

from chap_gis.aggregate import aggregate_to_regions

@pytest.mark.integration
def test_aggregate_real(rwanda_adm2, outputs_folder):
    regions = rwanda_adm2
    logging.info(regions)

    # load grid data from outputs
    grid = rioxarray.open_rasterio(outputs_folder / 'population.nc')
    logging.info(grid)

    # run the aggregate
    ds_agg = aggregate_to_regions(grid, regions, 'sum', 'shapeName')

    # inspect ds
    logging.info(ds_agg)

    # convert to pandas
    df_agg = ds_agg.to_dataframe()

    # inspect df
    logging.info(df_agg)
