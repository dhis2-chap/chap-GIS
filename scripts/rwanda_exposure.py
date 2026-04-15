"""Rwanda-wide mosquito exposure surface (30 m), lazy xarray pipeline.

Driver script that composes chap_gis functions. Everything stays dask-lazy
until the terminal ``to_netcdf`` / ``to_raster`` calls, which trigger a
single combined compute.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from cyclopts import App

import chap_gis as cgis
from chap_gis.grid import reproject_to

app = App(name="rwanda-exposure", help=__doc__)


@app.default
def run(
    *,
    elevation: Path,
    chelsa_dir: Path,
    country: str = "RWA",
    year_worldcover: int = 2021,
    year_chelsa: int = 2010,
    year_worldpop: int = 2020,
    resolution_m: float = 30.0,
    rice: Path | None = None,
    out_dir: Path = Path("data/cache"),
) -> None:
    """Compute and write the country-wide exposure rasters."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi = cgis.io.boundaries.load_adm(country, level=0)
    buffered = cgis.aoi.buffered(aoi, distance=0.0027)  # ~300 m buffer

    grid = cgis.grid.build_grid(
        aoi, resolution=resolution_m / 111_000, crs="EPSG:4326"
    )

    # --- unary steps pipe, multi-arg steps are plain calls; both stay lazy ---

    landcover = (
        cgis.io.worldcover.load(buffered, year=year_worldcover)
        .pipe(reproject_to, grid, "mode")
        .astype("uint8")
    )

    elev_native = cgis.io.elevation.load(elevation)
    elev = elev_native.pipe(reproject_to, grid, "bilinear")

    tas_annual = (
        cgis.io.chelsa.load_monthly_tas(chelsa_dir, year=year_chelsa)
        .pipe(cgis.climate.annual_mean)
    )
    tas_on_grid = tas_annual.pipe(reproject_to, grid, "bilinear")
    coarse_elev_on_grid = (
        elev_native
        .pipe(reproject_to, tas_annual, "average")
        .pipe(reproject_to, grid, "bilinear")
    )
    temperature = cgis.climate.lapse_rate_downscale(
        tas_on_grid, coarse_elev_on_grid, elev
    )

    suitability = temperature.pipe(cgis.suitability.thermal_suitability)

    population = (
        cgis.io.worldpop.load(country, year=year_worldpop)
        .pipe(cgis.io.worldpop.reproject_density, grid)
    )

    rice_mask = None
    if rice is not None and rice.exists():
        rice_mask = (
            cgis.io.elevation.load(rice)
            .pipe(reproject_to, grid, "nearest")
            .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))
        )

    breeding = cgis.landcover.breeding_site_mask(
        landcover, rice=rice_mask, water_edge_buffer=2
    )

    expo = cgis.exposure.exposure(
        breeding,
        elev,
        suitability,
        pixel_m=resolution_m,
        land_mask=cgis.landcover.land_mask(landcover),
        water_mask=cgis.landcover.water_mask(landcover),
    )

    pop_exposure = (population * expo).rename("pop_exposure")
    pop_exposure.attrs.update(long_name="Population-weighted exposure", units="people")
    pop_exposure = pop_exposure.rio.write_crs(grid.rio.crs)

    out_ds = xr.Dataset(
        {
            "temperature": temperature,
            "suitability": suitability,
            "population": population,
            "exposure": expo,
            "pop_exposure": pop_exposure,
        }
    )

    # Terminal compute — single dask graph executes here.
    nc_path = out_dir / f"{country.lower()}_exposure.nc"
    out_ds.to_netcdf(nc_path)
    print(f"  Wrote {nc_path}")

    prefix = country.lower()
    for name, data in out_ds.data_vars.items():
        path = out_dir / f"{prefix}_{name}.tif"
        data.rio.to_raster(path, compress="deflate")
        print(f"  Wrote {path}")

    _, stats = cgis.hotspots.identify_hotspots(pop_exposure, population)
    print(f"  Hotspot threshold (top 10%): {stats['threshold']:.3f}")
    if "pct" in stats:
        print(f"  People in hotspots: {stats['hotspot_pop']:,.0f} ({stats['pct']:.1f}%)")


if __name__ == "__main__":
    app()
