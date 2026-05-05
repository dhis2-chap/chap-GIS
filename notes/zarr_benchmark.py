"""Benchmark NetCDF / GeoTIFF / Zarr for a representative chap_gis output.

Run from the repo root:

    uv pip install 'zarr>=3' numcodecs   # one-shot, not in pyproject
    uv run python notes/zarr_benchmark.py

The synthetic dataset is a 12-month float32 stack of shape (12, 1024, 1024) —
roughly the size of one CHELSA-tas year for a small country at native 1 km.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr


SHAPE = (12, 1024, 1024)
RNG_SEED = 0
OUT_DIR = Path(__file__).parent / "_bench"
RESULTS: list[tuple[str, str, float, int]] = []  # (format, op, seconds, bytes)


def make_dataset() -> xr.DataArray:
    rng = np.random.default_rng(RNG_SEED)
    data = rng.standard_normal(SHAPE, dtype=np.float32)
    times = np.array(
        [np.datetime64(f"2021-{m:02d}-01") for m in range(1, 13)],
        dtype="datetime64[ns]",
    )
    ys = np.linspace(2.0, 1.0, SHAPE[1], dtype=np.float64)
    xs = np.linspace(28.0, 31.0, SHAPE[2], dtype=np.float64)
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
        name="tas",
    )
    return da.rio.write_crs("EPSG:4326")


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@contextmanager
def timed(label: str, fmt: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    RESULTS.append((fmt, label, dt, 0))
    print(f"  {fmt:24} {label:32} {dt:7.3f}s")


def reset_outdir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)


def bench_per_month_netcdf(da: xr.DataArray) -> None:
    fmt = "netcdf-per-month"
    print(f"\n[{fmt}]")
    files = []
    with timed("write (12 files)", fmt):
        for t in range(da.sizes["time"]):
            slc = da.isel(time=[t])
            path = OUT_DIR / fmt / f"tas_{t:02d}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            slc.to_netcdf(path)
            files.append(path)

    size = dir_size(OUT_DIR / fmt)
    print(f"  {fmt:24} {'on-disk size':32} {size / 1024 / 1024:6.1f} MB")
    RESULTS.append((fmt, "size_bytes", 0.0, size))

    with timed("open + load full", fmt):
        ds = xr.open_mfdataset(files, combine="by_coords")
        _ = ds["tas"].load()
        ds.close()

    with timed("open + load 1 month", fmt):
        ds = xr.open_mfdataset(files, combine="by_coords")
        _ = ds["tas"].isel(time=6).load()
        ds.close()

    with timed("open + load 200x200 box", fmt):
        ds = xr.open_mfdataset(files, combine="by_coords")
        _ = ds["tas"].isel(y=slice(200, 400), x=slice(200, 400)).load()
        ds.close()


def bench_single_netcdf(da: xr.DataArray) -> None:
    fmt = "netcdf-single"
    print(f"\n[{fmt}]")
    path = OUT_DIR / "tas.nc"
    with timed("write", fmt):
        da.to_netcdf(path)

    size = dir_size(path)
    print(f"  {fmt:24} {'on-disk size':32} {size / 1024 / 1024:6.1f} MB")
    RESULTS.append((fmt, "size_bytes", 0.0, size))

    with timed("open + load full", fmt):
        d = xr.open_dataset(path)["tas"]
        _ = d.load()
        d.close()

    with timed("open + load 1 month", fmt):
        d = xr.open_dataset(path)["tas"]
        _ = d.isel(time=6).load()
        d.close()

    with timed("open + load 200x200 box", fmt):
        d = xr.open_dataset(path)["tas"]
        _ = d.isel(y=slice(200, 400), x=slice(200, 400)).load()
        d.close()


def bench_zarr(da: xr.DataArray, label: str, chunks: dict) -> None:
    fmt = label
    print(f"\n[{fmt}] chunks={chunks}")
    path = OUT_DIR / f"{label}.zarr"
    chunked = da.chunk(chunks)

    with timed("write", fmt):
        chunked.to_zarr(path, mode="w")

    size = dir_size(path)
    print(f"  {fmt:24} {'on-disk size':32} {size / 1024 / 1024:6.1f} MB")
    RESULTS.append((fmt, "size_bytes", 0.0, size))

    with timed("open + load full", fmt):
        d = xr.open_zarr(path)["tas"]
        _ = d.load()

    with timed("open + load 1 month", fmt):
        d = xr.open_zarr(path)["tas"]
        _ = d.isel(time=6).load()

    with timed("open + load 200x200 box", fmt):
        d = xr.open_zarr(path)["tas"]
        _ = d.isel(y=slice(200, 400), x=slice(200, 400)).load()


def main() -> None:
    print(f"shape={SHAPE} dtype=float32 nominal_size_mb={(np.prod(SHAPE) * 4) / 1024 / 1024:.1f}")
    reset_outdir()
    da = make_dataset()

    bench_per_month_netcdf(da)
    bench_single_netcdf(da)
    bench_zarr(da, "zarr-time1-y512-x512", {"time": 1, "y": 512, "x": 512})
    bench_zarr(da, "zarr-time12-y512-x512", {"time": 12, "y": 512, "x": 512})
    bench_zarr(da, "zarr-time1-y256-x256", {"time": 1, "y": 256, "x": 256})

    print("\n=== summary (markdown table) ===")
    print("| format | write s | full read s | 1-month read s | box read s | size MB |")
    print("|---|---:|---:|---:|---:|---:|")
    fmts = sorted({f for f, _, _, _ in RESULTS})
    for f in fmts:
        rows = {op: (s, b) for fmt, op, s, b in RESULTS if fmt == f}
        size = next((b for op, (_, b) in rows.items() if op == "size_bytes"), 0)
        write = rows.get("write (12 files)", rows.get("write", (0, 0)))[0]
        full = rows.get("open + load full", (0, 0))[0]
        one = rows.get("open + load 1 month", (0, 0))[0]
        box = rows.get("open + load 200x200 box", (0, 0))[0]
        print(f"| `{f}` | {write:.2f} | {full:.2f} | {one:.2f} | {box:.2f} | {size / 1024 / 1024:.1f} |")


if __name__ == "__main__":
    main()
