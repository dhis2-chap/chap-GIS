
import xarray as xr
import geopandas as gpd

from exactextract import Writer
from exactextract.feature import JSONFeature
import exactextract

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
