
import xarray as xr
import geopandas as gpd

import pandas as pd
from exactextract import Writer
from exactextract.feature import JSONFeature
import exactextract

import logging

logger = logging.getLogger(__name__)


# custom xarray output writer class
# TODO: remove after we get it merged into exactextract
# see: https://github.com/isciences/exactextract/pull/192

class XArrayWriter(Writer):
    """
    Writer that returns an :py:class:`xarray.Dataset` with dimensions
    ``(feature, <dim_name>)``.

    Unlike the other writers, the non-spatial dimension (e.g. time) cannot
    be inferred from exactextract's internal data model, which only tracks
    band indices. When the input raster is an :py:class:`xarray.DataArray`
    or :py:class:`xarray.Dataset`, ``dim_coords`` are resolved automatically
    by :py:func:`exact_extract` if ``dim_name`` matches a coordinate on the
    input. For all other raster types, integer band indices are used unless
    ``dim_coords`` is provided explicitly via ``output_options``.

    Args:
        dim_name: Name of the non-spatial output dimension. Defaults to
            ``"band"``. Common values: ``"time"``, ``"level"``.
        dim_coords: Coordinate values for the non-spatial dimension, one per
            band in the source raster (e.g. ``ds["time"].values``). If not
            provided, 0-based integer indices are used.
    """

    def __init__(self, *, dim_name="band", dim_coords=None):
        super().__init__()
        self._dim_name = dim_name
        self._dim_coords = dim_coords
        self._ops = []
        self._id_cols = []
        self._rows = []

    def add_operation(self, op):
        self._ops.append(op)

    def add_column(self, col_name):
        self._id_cols.append(col_name)

    def write(self, feature):
        f = JSONFeature()
        feature.copy_to(f)

        props = f.feature["properties"]
        row = {col: props.get(col) for col in self._id_cols}
        for op in self._ops:
            row[op.name] = props.get(op.name)
        self._rows.append(row)

    def features(self):
        import numpy as np
        import xarray as xr
        from collections import defaultdict
        import re

        if not self._rows:
            return xr.Dataset()

        feature_ids = (
            [r[self._id_cols[0]] for r in self._rows]
            if self._id_cols
            else list(range(len(self._rows)))
        )

        # Group ops by (var_name, stat) — each group spans one full set of bands
        groups = defaultdict(list)
        for op in self._ops:
            stat = op.stat
            suffix = f"_{stat}"
            prefix = op.name[:-len(suffix)] if op.name.endswith(suffix) else op.name
            # prefix is e.g. "t2m_band_1", "band_1", "t2m", ""
            # strip band index to get var name
            var_name = re.sub(r"_?band_\d+$", "", prefix) or "values"
            groups[(var_name, stat)].append(op)

        # All groups must have the same number of bands
        group_lengths = {len(ops) for ops in groups.values()}
        if len(group_lengths) != 1:
            raise ValueError(
                f"Unequal number of bands across stat/variable groups: {group_lengths}"
            )
        n_bands = group_lengths.pop()

        dim_coords = (
            np.asarray(self._dim_coords)
            if self._dim_coords is not None
            else np.arange(n_bands)
        )

        if len(dim_coords) != n_bands:
            raise ValueError(
                f"Length of dim_coords ({len(dim_coords)}) does not match "
                f"number of bands per group ({n_bands})."
            )

        # Use plain var_name when there is only one stat, var_name_stat otherwise
        multi_stat = len({stat for _, stat in groups}) > 1

        data_vars = {}
        for (var_name, stat), ops in groups.items():
            da_name = f"{var_name}_{stat}" if multi_stat else var_name
            data = np.array(
                [[r[op.name] for op in ops] for r in self._rows],
                dtype=float,
            )
            data_vars[da_name] = (["feature", self._dim_name], data)

        return xr.Dataset(
            data_vars,
            coords={
                self._dim_name: dim_coords,
                "feature": feature_ids,
            },
        )

################
# main function

def aggregate_to_regions(grid: xr.DataArray, regions: gpd.GeoDataFrame, statistic: str, id_field: str):
    # until we get native xarray support, we need to init our custom writer class with options
    writer = XArrayWriter(
        #dim_name="time",
        #dim_coords=grid.coords["time"].values,
    )
    
    # get the raw aggregate
    agg_ds = exactextract.exact_extract(grid, regions, [statistic], 
                                        strategy='raster-sequential', 
                                        include_cols=[id_field],
                                        output=writer)

    # rename default aggregation xarray dims
    agg_ds = agg_ds.rename({'feature': id_field, 'band': 'time'})

    # return
    return agg_ds

def aggregate_population_by_year(pop_da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Aggregates population raster data to administrative regions.
    
    This function handles:
    1. CRS verification for spatial alignment.
    2. Zonal statistics (sum) across all time steps.
    3. Restoring location names from the GeoDataFrame.
    4. Restoring actual year/time labels from the input raster.
    5. Standardizing variable names for downstream merging.
    """
    logger.info("Starting regional population aggregation...")
    
    # 1. Coordinate Reference System (CRS) check
    # exact_extract requires the raster and gdf to have defined CRSs.
    if pop_da.rio.crs is None:
        logger.warning("Raster CRS missing. Defaulting to EPSG:4326.")
        pop_da = pop_da.rio.write_crs("EPSG:4326")
    
    # 2. Execute Zonal Statistics
    # aggregate_to_regions returns dims ['location_id', 'time'] 
    # but initially uses integer indices (0, 1, 2...) for coordinates.
    agg_ds = aggregate_to_regions(
        pop_da, 
        gdf, 
        statistic="sum", 
        id_field='location_id'
    )

    # 3. Restore Coordinate Labels
    # We replace the integer indices with the actual metadata from our inputs.
    agg_ds = agg_ds.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": pop_da.time.values
    })

    # 4. Standardize Variable Names
    # exact_extract outputs the name of the statistic ('sum'). 
    # We rename it to 'population' to match health data expectations.
    if "sum" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"sum": "population"})
    elif "values" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"values": "population"})

    # 5. Clean up Metadata
    # Remove any residual 'band' coordinates if they exist
    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds

def aggregate_temperature_by_month(tas_da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Aggregates CHELSA monthly temperature rasters to administrative regions.
    
    This function handles:
    1. CRS verification.
    2. Zonal statistics (mean) across all monthly time steps.
    3. Restoring location and time metadata (formatted as YYYY-MM).
    4. Renaming the output variable to 'tas'.
    """
    logger.info("Starting regional temperature aggregation...")
    
    # 1. Coordinate Reference System (CRS) check
    if tas_da.rio.crs is None:
        logger.warning("Temperature raster CRS missing. Defaulting to EPSG:4326.")
        tas_da = tas_da.rio.write_crs("EPSG:4326")
    
    # 2. Execute Zonal Statistics
    agg_ds = aggregate_to_regions(
        tas_da, 
        gdf, 
        statistic="mean", 
        id_field='location_id'
    )

    if isinstance(agg_ds, xr.DataArray):
        agg_ds = agg_ds.to_dataset(name="tas")

    
    agg_ds = agg_ds.assign_coords({
        "location_id": gdf["location_id"].values,
        "time": pd.to_datetime(tas_da.time.values)
    })

    # 4. Standardize Variable Names
    if "mean" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"mean": "tas"})
    elif "values" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"values": "tas"})

    # 5. Clean up Metadata
    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds
