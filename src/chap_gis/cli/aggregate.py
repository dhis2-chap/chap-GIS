"""``aggregate`` subcommand: aggregate gridded outputs to admin regions and emit a CHAP CSV."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pandas as pd
import rioxarray
import xarray as xr
import numpy as np

from dhis2eo.integrations.chap import dataframe_to_chap_csv

import chap_gis as cgis
from chap_gis.aggregate import aggregate_to_regions


logger = logging.getLogger(__name__)


def _simulate_monthly_disease_data(regions_df, location_id_field):
    # set location id field
    regions_df['location_id'] = regions_df[location_id_field]
    regions_df = regions_df[['location_id']]
    regions_df = regions_df.drop_duplicates(subset=["location_id"])

    #set population to random number for now
    pop_df = regions_df.copy()
    pop_df["population"] = np.random.randint(1000, 500000, size=len(pop_df))

    # create df for timeframe (hardcoded for now)
    time_df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=12*3, freq="MS")})

    # crossjoin regions with time
    final = pop_df.merge(time_df, how='cross')

    # add random disease data
    from random import uniform
    final['disease'] = [uniform(0, 30) for _ in range(len(final))]

    return final


def aggregate(
    out_dir: str,
    country: str,
    level: int,
) -> None:
    """Aggregate the various dataset outputs to country administrative boundaries and output to chap CSV.

    Administrative boundaries are fetched dynamically from GeoBoundaries (should be easy later to switch out with custom geojson file).
    Disease data needs to exist as a CSV file in the inputs folder, with the name "disease_data_<countrycode>".
    """
    out_dir = Path(out_dir).resolve()
    logger.info(f'Aggregating nc files in folder {out_dir} to country boundaries {country}-ADM{level}')

    # load admin boundaries for country and level
    logger.info('Loading boundary regions')
    gdf = cgis.io.boundaries.load(country, level=level)
    logger.info(gdf)

    # load disease data based on required location and naming convention
    logger.info('Loading disease data')
    filename = f'disease_data_{country.lower()}.csv'
    disease_path = out_dir.parent / 'inputs' / filename
    if disease_path.exists():
        logger.info(f'--> {disease_path}')
        df = pd.read_csv(disease_path, parse_dates=["time"])
    else:
        logger.info('--> Generating dummy data')
        df = _simulate_monthly_disease_data(regions_df=gdf, location_id_field='shapeName')

    assert 'location_id' in df.columns
    assert 'time' in df.columns
    assert 'disease' in df.columns

    # convert to xarray
    logger.info(df)
    output = df.set_index(["location_id", "time"]).to_xarray()
    logger.info(output)

    # aggregate for each nc file and join to a single output dataset
    for path in out_dir.glob('*.nc'):
        logger.info('')
        logger.info('----------------------------------------------------------')
        logger.info(f'File: {path}')

        # open nc file
        ds = rioxarray.open_rasterio(path)
        ds_name = path.stem

        # hacky fixes
        # TODO: we need to standardize this earlier in the pipelines and not cleanup here
        # squeeze away unneeded dimensions
        ds = ds.squeeze(drop=True)
        # add time dim
        ds = ds.expand_dims(time=[pd.Timestamp("2021-01-01")])

        # inspect input
        logger.info('Input xarray:')
        logger.info(ds)

        # aggregate to geojson
        # TODO: right now we have no way to map each nc file to specific statistic
        agg = aggregate_to_regions(ds, gdf, statistic='mean', id_field='location_id')

        # prefix "mean" variable with variable name before merging
        agg = agg.rename({'mean': f'{ds_name}_mean'})
        logger.info(agg)

        # cleanup for memory
        del ds
        gc.collect()

        # left join to main disease dataset
        output = xr.merge([output, agg], join='left')

    # prepare and output to chap csv
    # example of preparing data for chap, see: https://climate-tools.dhis2.org/guides/import-chap/harmonize-to-chap/
    logger.info(f'Merged dataset of all aggregates:')
    logger.info(output)
    out_path = out_dir / 'chap-output.csv'
    # map columns to chap names
    # these are the required columns, all others will be included with their original variables names
    column_map = {
        "time_period": "time",
        "location": "location_id",
        "disease_cases": "disease",
        "population": "population",
    }

    
    start_date = pd.to_datetime(output['time'].min().values).strftime('%Y-%m')
    end_date = pd.to_datetime(output['time'].max().values).strftime('%Y-%m')
    logger.info(f'Start date: {start_date}, End date: {end_date}')

    # output chap-csv
    dataframe_to_chap_csv(
        output.to_dataframe().reset_index(),
        column_map=column_map,
        freq='monthly',
        start=start_date,  # these dont quite work yet, need to look up the expected format :P
        end=end_date,  # these dont quite work yet, need to look up the expected format :P
        output_path=out_path,
    )
