# chap-GIS

## Repository layout

- **`src/chap_gis/`** — package code. Functions that fulfil the conventions of this repo.
- **`data/inputs/`** - manually downloaded data (which we won't download dynamically via functions)
- **`data/cache/`** - dynamically downloaded and cached data (ignored by .gitignore to avoid tracking very large data)
- **`data/outputs/`** - final data outputs produced by the scripts (ignored by .gitignore to avoid tracking very large data)
- **`scripts/`** — concrete scripts that use the functionality of the package.
- **`raw_scripts/`** — inspiration scripts; perform a wanted operation but are not adapted to the GIS conventions used here.
- **`notes/data_sources/`** — markdown files, each describing a data source.

## Conventions

- Command-line interfaces use [**cyclopts**](https://cyclopts.readthedocs.io/)
  (Typer-style decorator-based CLIs, native type-annotation parsing).

## Malaria Exposure Analyses

Runs a mosquito suitability and exposure analysis for a specified country and year.

Data is dynamically downloaded on-demand, except for the rice dataset (see requirements). 

Outputs a single `<countrycode>_exposure.nc` file with all the variables in the output folder (see how to run). 

### Requirements

1. Register and get Copernicus CDSE OpenEO client credentials (required for elevation and worldcover data):
    - Register a CDSE account at https://dataspace.copernicus.eu/
    - Create and get OAuth client credentials
        - Go to https://shapps.dataspace.copernicus.eu/dashboard/#/
        - Under User Settings, and OAuth Clients, create a new Client.
        - Copy the Client ID and Client Secret.
        - Set these as environment variables `CDSE_OAUTH_CLIENT_ID`, and `CDSE_OAUTH_CLIENT_SECRET`.
    - CDSE allows X free processing "credits" each month. There is no easy way to track credit usage with client credentials. This has to be done programmatically via openeo:
        - `jobs = conn.list_jobs()` and `conn.job(job_id).describe()`
        - For more info about monthly free credits, see: https://documentation.dataspace.copernicus.eu/APIs/openEO/credit_usage.html

2. Manually download rice fields data for 2023:
    - Go to: https://zenodo.org/records/13729353
    - Download the tiff file for a country (Africa only)
    - Save the file to `data/inputs` as `<countrycode>_rice_fields.tif` (all lowercased)

3. Setup the required environment variables:
    - Create an `.env` file at the project root folder, with the following parameters:
        - `CDSE_OAUTH_CLIENT_ID`
            CDSE OAuth client ID needed for some datasets (see requirements).
        - `CDSE_OAUTH_CLIENT_SECRET`
            CDSE OAuth client secret needed for some datasets (see requirements).
        - `CHAP_GIS_CACHE` (optional, default is `data/cache`)
            The target path for downloading data. After first download, files will be reused. 
            Remember to clear the cache if changing the country used for analysis. 

### Running the analysis

Script location: `scripts/exposure_analysis.py`

Simplest possible example to run the analysis for Rwanda 2021 (the default year when not specified, see [the section on adjusting parameters](#adjusting-analysis-parameters)):

```
  uv run scripts/exposure_analysis.py --country=RWA
```

The result of the analysis should output a series of .nc files in the output folder, as specified by the `CHAP_GIS_CACHE` environment variables (the default is `data/outputs`). 

### Visualizing the results

To visualize the results of the analysis you can run the `vizualize` subcommand pointing to the output folder, e.g.:

```
  uv run scripts/exposure_analysis.py visualize data/outputs
```

### Aggregate to regions and produce Chap-compatible CSV files

To analyze the gridded output data in Chap, it's first necessary to aggregate the gridded outputs to region boundaries and produce Chap-compatible CSV files. Assuming you have downloaded the below boundary file to :

```
  uv run scripts/exposure_analysis.py aggregate data/outputs data/cache/geoBoundaries-RWA-ADM2.geojson shapeISO
```

This will:

1. Create a CSV file for every .nc file in the specified outputs folder, aggregated to the specified geojson file, using the specified region id property. 
2. Merge all the CSV files and produce a Chap-compatible CSV file containined all the variables. 

### Adjusting analysis parameters

#### Analysis parameters

The following parameters can be used to modify the analysis:

- country:
    ISO3 3-letter country code. Used to download administrative boundaries, worldpop data,
    and access the correct rice fields dataset (see requirements). 
- year_worldcover: 
    Year to use for worldcover landuse data. Valid years: 2020 and 2021.
- year_chelsa:
    Year to use for CHELSA monthly temperature data. Valid years: 1979 and 2021.
- year_worldpop:
    Year to use for WorldPop population data. Valid years: 2015 and 2030.
- resolution_m:
    Spatial resolution of analysis. Default is 30m. 
- out_dir:
    Target folder for final analysis output dataset. Default is `data/outputs`. 

The latest possible year where we have data from all sources is 2021. This is the default if none
of the years are specified. 

#### Visualizing parameters

...


#### Aggregate parameters

...
