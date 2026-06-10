"""Co-registered monthly NDVI / EVI / temperature rasters for Rwanda, 2021.

NDVI+EVI from MODIS 13Q1 (250 m, 16-day) via Microsoft Planetary Computer,
composited to monthly means; CHELSA monthly temperature reprojected onto the
same 250 m EPSG:4326 grid. Output: results/veg_temp_2021/stack.nc with
dims (month, y, x).
"""
import numpy as np, xarray as xr
from pathlib import Path
import planetary_computer as pc, pystac_client
import odc.stac
import rioxarray  # noqa

import chap_gis as cgis
from chap_gis.io import chelsa
from chap_gis.grid import reproject_to

RES = 0.0025  # ~275 m in EPSG:4326
gdf0 = cgis.io.boundaries.load("RWA", level=0)
bbox = [float(v) for v in gdf0.total_bounds]
outdir = Path("results/veg_temp_2021"); outdir.mkdir(parents=True, exist_ok=True)

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
items = list(cat.search(collections=["modis-13Q1-061"], bbox=bbox,
                        datetime="2021-01-01/2021-12-31").items())
print(f"MODIS 13Q1 items: {len(items)}", flush=True)

veg_ck, temp_ck = outdir / "_veg.nc", outdir / "_temp.nc"
if veg_ck.exists() and temp_ck.exists():
    print("loading checkpoints (skipping MODIS download) ...", flush=True)
    veg = xr.open_dataset(veg_ck); temp = xr.open_dataarray(temp_ck)
else:
    ds = odc.stac.load(
        items, bands=["250m_16_days_NDVI", "250m_16_days_EVI"],
        crs="EPSG:4326", resolution=RES, bbox=bbox,
        groupby="solar_day", chunks={"x": 2048, "y": 2048},
    ).rename({"250m_16_days_NDVI": "ndvi", "250m_16_days_EVI": "evi"})
    for k in ("ndvi", "evi"):                      # MODIS VI: fill -3000, valid>=-2000, scale 1e-4
        ds[k] = ds[k].where(ds[k] > -2000) * 1e-4
    print(f"scenes loaded: {ds.sizes.get('time')} -> monthly composite", flush=True)
    veg = ds.groupby("time.month").mean().compute()
    veg.to_netcdf(veg_ck)
    ydim, xdim = veg["ndvi"].odc.spatial_dims      # odc may name dims latitude/longitude
    tas = chelsa.load(gdf0, start="2021-01", end="2021-12", country_code="RWA")
    for d in ("x", "y"):
        if d in tas.coords:
            tas[d] = np.round(tas[d].astype("float64"), 10)
    tas = tas.rio.write_crs("EPSG:4326")
    tas_m = reproject_to(tas, veg["ndvi"].isel(month=0), "bilinear")
    temp = (tas_m.assign_coords(month=("time", np.arange(1, 13)))
                 .swap_dims({"time": "month"}).drop_vars("time").compute())
    temp.to_netcdf(temp_ck)

ndvi = veg["ndvi"].drop_vars(["spatial_ref"], errors="ignore")
evi = veg["evi"].drop_vars(["spatial_ref"], errors="ignore")
t = temp.drop_vars(["spatial_ref"], errors="ignore").rename("temperature")
out = xr.merge([ndvi.rename("ndvi"), evi.rename("evi"), t],
               compat="override", join="override").rio.write_crs("EPSG:4326")
out.to_netcdf(outdir / "stack.nc")
print(f"WROTE {outdir/'stack.nc'}  dims={dict(out.sizes)}", flush=True)
print("ndvi range:", float(veg.ndvi.min()), float(veg.ndvi.max()),
      "| evi:", float(veg.evi.min()), float(veg.evi.max()),
      "| temp:", float(temp.min()), float(temp.max()), flush=True)
