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

- Register and get Copernicus S3 credentials
  - ... 
- Manually download rice fields data for 2023:
  - Go to: https://zenodo.org/records/13729353
  - Download the tiff file for a country (Africa only)
  - Save the file to `data/inputs` as `<countrycode>_rice_fields.tif` (all lowercased)

### How to run

Script location: `scripts/exposure_analysis.py`

To run the analysis call this script from commandline with the following parameters:

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

Simplest possible example to run the analysis for Rwanda 2021:

```
  python -m scripts/exposure_analysis.py --country=RWA
```

### Environment variables

Environment variables must be specified in .env file of the project root:

- CHAP_GIS_CACHE
    The target path for downloading data. After first download, files will be reused. 
    Remember to clear the cache if changing the country used for analysis. 
    Default is `data/cache`. 

- COPERNICUS_S3_ACCESS_KEY
    Copernicus S3 access key needed to download Copernicus 30m elevation data (see requirements).

- COPERNICUS_S3_SECRET_KEY
    Copernicus S3 secret key needed to download Copernicus 30m elevation data (see requirements).
