"""One-shot generator for synthetic test fixtures under tests/data/.

All produced files are committed to the repo; pytest does not invoke this
script. Re-run manually after editing to regenerate:

    uv run python tests/data/_build.py

The boundaries use a deliberately fake ISO3 (XXX) and a unit-square AOI in
EPSG:4326 so it is obvious the data is synthetic and unrelated to any real
country.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from shapely.geometry import Polygon


HERE = Path(__file__).parent


def _adm0() -> gpd.GeoDataFrame:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    return gpd.GeoDataFrame(
        {
            "shapeName": ["XXX"],
            "shapeISO": ["XXX"],
            "shapeID": ["XXX-ADM0-1"],
            "shapeGroup": ["XXX"],
            "shapeType": ["ADM0"],
            "geometry": [poly],
        },
        crs="EPSG:4326",
    )


def _adm2() -> gpd.GeoDataFrame:
    quadrants = [
        ("SW", Polygon([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])),
        ("SE", Polygon([(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)])),
        ("NW", Polygon([(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)])),
        ("NE", Polygon([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])),
    ]
    return gpd.GeoDataFrame(
        {
            "shapeName": [name for name, _ in quadrants],
            "shapeISO": ["XXX"] * 4,
            "shapeID": [f"XXX-ADM2-{name}" for name, _ in quadrants],
            "shapeGroup": ["XXX"] * 4,
            "shapeType": ["ADM2"] * 4,
            "geometry": [geom for _, geom in quadrants],
        },
        crs="EPSG:4326",
    )


def _disease_csv() -> pd.DataFrame:
    """36-month dummy disease data for the XXX ADM2 quadrants.

    Uses CHAP-style column names (``time_period``, ``location``, ``disease_cases``)
    so tests exercise the column-standardization branch of ``get_health_data``.
    """
    times = pd.date_range("2017-01-01", periods=36, freq="MS")
    locations = ["SW", "SE", "NW", "NE"]
    rng = np.random.default_rng(seed=0)
    rows = []
    for loc in locations:
        for t in times:
            rows.append({
                "time_period": t.strftime("%Y-%m"),
                "location": loc,
                "disease_cases": float(rng.integers(0, 50)),
            })
    return pd.DataFrame(rows)


def _rice_raster() -> xr.DataArray:
    """Synthetic 16x16 rice raster aligned with the XXX unit-square AOI."""
    data = np.zeros((1, 16, 16), dtype="uint8")
    data[0, 6:10, 6:10] = 1  # a small "rice patch" near the center
    xs = (np.arange(16) + 0.5) / 16.0
    ys = (np.arange(16)[::-1] + 0.5) / 16.0
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1], "y": ys, "x": xs},
    )
    return da.rio.write_crs("EPSG:4326")


def main() -> None:
    for level, gdf in [(0, _adm0()), (2, _adm2())]:
        out = HERE / f"geoBoundaries-XXX-ADM{level}.geojson"
        gdf.to_file(out, driver="GeoJSON")
        print(f"wrote {out} ({out.stat().st_size} bytes)")
    rice_path = HERE / "xxx_jiang_rice_fields.tif"
    _rice_raster().rio.to_raster(rice_path)
    print(f"wrote {rice_path} ({rice_path.stat().st_size} bytes)")

    csv_path = HERE / "xxx_disease.csv"
    _disease_csv().to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({csv_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
