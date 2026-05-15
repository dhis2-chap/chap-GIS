
import xarray as xr
import geopandas as gpd

import pandas as pd
from exactextract import Writer
from exactextract.feature import JSONFeature
import exactextract
from dask.diagnostics import ProgressBar

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
        feature.copy_to(f);

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
    # 1. TRIGGER COMPUTE HERE
    # If the grid is a Dask array (lazy), exact_extract will crash or crawl.
    # Computing it once here turns it into a NumPy array that the C++ engine 
    # can scan through very quickly.
    if hasattr(grid.data, "dask"):
        logger.info(f"Computing exposure raster chunks for regional aggregation...")
        # This context manager will show progress for the .compute() call
        with ProgressBar():
            grid = grid.compute()

    # 2. Setup the writer
    writer = XArrayWriter()
    
    # 3. Execute - this will now be MUCH faster because it's working on a NumPy array
    agg_ds = exactextract.exact_extract(
        grid, 
        regions, 
        [statistic], 
        strategy='raster-sequential', 
        include_cols=[id_field],
        output=writer,
        progress=True
    )

    # 4. Rename default aggregation xarray dims
    # Note: Check if 'feature' and 'band' exist in agg_ds before renaming
    rename_dict = {}
    if 'feature' in agg_ds.dims: rename_dict['feature'] = id_field
    if 'band' in agg_ds.dims: rename_dict['band'] = 'time'
    
    if rename_dict:
        agg_ds = agg_ds.rename(rename_dict)

    return agg_ds

def aggregate_population_by_year(pop_da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Aggregates population raster data to administrative regions.
    Internalizes the .compute() to avoid exactextract 'bad variant access'.
    """
    logger.info("Starting regional population aggregation...")
    
    # CRS check
    if pop_da.rio.crs is None:
        logger.warning("Raster CRS missing. Defaulting to EPSG:4326.")
        pop_da = pop_da.rio.write_crs("EPSG:4326")
    
    # CRITICAL: exact_extract cannot handle Dask arrays. 
    # We compute here to provide a NumPy-backed array to the C++ engine.
    if hasattr(pop_da.data, "dask"):
        logger.info("Computing population chunks for regional aggregation...")
        pop_da = pop_da.compute()

    # Execute Zonal Statistics
    agg_ds = aggregate_to_regions(
        pop_da, 
        gdf, 
        statistic="sum", 
        id_field='location_id'
    )

    # Restore Coordinate Labels
    agg_ds = agg_ds.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": pop_da.time.values
    })

    # Standardize Variable Names
    if "sum" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"sum": "population"})
    elif "values" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"values": "population"})

    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds


def aggregate_temperature_by_month(tas_da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Aggregates CHELSA monthly temperature rasters to administrative regions.
    Internalizes the .compute() to avoid exactextract 'bad variant access'.
    """
    logger.info("Starting regional temperature aggregation...")
    
    # CRS check
    if tas_da.rio.crs is None:
        logger.warning("Temperature raster CRS missing. Defaulting to EPSG:4326.")
        tas_da = tas_da.rio.write_crs("EPSG:4326")
    
    # CRITICAL: Compute chunks before passing to exact_extract engine
    if hasattr(tas_da.data, "dask"):
        logger.info("Computing temperature chunks for regional aggregation...")
        tas_da = tas_da.compute()

    # Execute Zonal Statistics
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

    # Standardize Variable Names
    if "mean" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"mean": "tas"})
    elif "values" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"values": "tas"})

    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds
