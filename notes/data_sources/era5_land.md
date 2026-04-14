# ERA5-Land

Reanalysis climate data from ECMWF / Copernicus.

- **Variables:** 2m temperature, 2m dewpoint temperature (monthly means)
- **Resolution:** ~9 km
- **Coverage:** Global
- **Access:** Copernicus Climate Data Store (CDS) API via the `cdsapi` Python package
- **Used in malaria-research:** `malaria_research/data/era5.py`, `scripts/download_era5.py`; humidity and suitability models
