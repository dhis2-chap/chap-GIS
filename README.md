# chap-GIS

## Repository layout

- **`src/chap_gis/`** — package code. Functions that fulfil the conventions of this repo.
- **`data/inputs/`** - manually downloaded data (which we won't download dynamically via functions)
- **`data/zarr/`** - zarr archives produced by dynamic download data functions which has to be recreated by each user (ignored by .gitignore to avoid tracking very large data)
- **`scripts/`** — concrete scripts that use the functionality of the package.
- **`raw_scripts/`** — inspiration scripts; perform a wanted operation but are not adapted to the GIS conventions used here.
- **`notes/data_sources/`** — markdown files, each describing a data source.

## Conventions

- Command-line interfaces use [**cyclopts**](https://cyclopts.readthedocs.io/)
  (Typer-style decorator-based CLIs, native type-annotation parsing).
