"""WorldPop population loader + density-preserving reproject helper."""

from __future__ import annotations

import xarray as xr
from dhis2eo.data.worldpop import pop_total

from .cache import cache_dir

def load(
    iso3: str,
    year: int,
) -> xr.DataArray:
    """Load a WorldPop people-per-pixel raster as a lazy DataArray."""
    
    # find files from cache or download
    files = pop_total.yearly.download(start=str(year),
                                  end=str(year),
                                  country_code=iso3,
                                  dirname=cache_dir(),
                                  prefix=f'{iso3}_worldpop_population')
    
    # open as xarray
    ds = xr.open_mfdataset(files)
    da = ds['pop_total']
    encoding = da.encoding

    # set nans to 0s
    da = da.fillna(0)

    # convert to float (otherwise breaks things later)
    da = da.astype('float32')

    # make it spatial
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    
    # add metadata
    da.name = "population"
    da.attrs.update(
        long_name="Population count per pixel",
        units="people",
        source=f"WorldPop {year}",
    )
    da.encoding = encoding

    return da
