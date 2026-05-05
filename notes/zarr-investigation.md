# Zarr as the chap_gis persistence format — investigation

> Ticket: [CLIM-676](https://dhis2.atlassian.net/browse/CLIM-676)
> Related: CLIM-672 (caching consolidation), CLIM-673 (CLI thinning), CLIM-677 (lazy serialize/deserialize).

## TL;DR — recommendation

**Adopt Zarr selectively, not universally.**

| Layer | Today | Recommended |
|---|---|---|
| `cli/analyze.py` outputs (7 NetCDFs read back by `aggregate`) | Per-variable NetCDF | **Single multi-variable Zarr store** |
| `io/chelsa.py` per-month NetCDF cache | 12 small `.nc` files | **Single multi-month NetCDF** (bundle, no Zarr) |
| `io/{worldcover,elevation,crop,rice}.py` GeoTIFF cache | GeoTIFF | Status quo |
| External-source caches in their native formats | TIF/NetCDF | Status quo |

The two structural wins worth chasing now are:

1. **Bundle the analyze outputs into one Zarr store**, so `aggregate` opens a single store instead of stitching seven NetCDFs back together. The Zarr-vs-NetCDF question is a tie at our current scale; the *bundling* is the win.
2. **Bundle CHELSA per-month files into a single file** (NetCDF is fine). Per-month + `open_mfdataset` is the slowest read pattern in our pipeline today.

Defer a full Zarr migration of caches until either (a) cloud storage is on the roadmap or (b) cache footprint becomes a problem.

## Benchmark

Synthetic dataset: 12 months × 1024 × 1024 float32 (~48 MB nominal), local SSD, single Python process. Source: [`notes/zarr_benchmark.py`](./zarr_benchmark.py).

| format | write s | full read s | 1-month read s | 200×200 box read s | size MB |
|---|---:|---:|---:|---:|---:|
| `netcdf-per-month` (12 files, `open_mfdataset`) | 0.07 | 0.20 | 0.05 | 0.06 | 48.3 |
| `netcdf-single` (multi-month, no chunks) | 0.01 | 0.01 | 0.01 | 0.00 | 48.0 |
| `zarr-time1-y512-x512` | 0.23 | 0.04 | 0.01 | 0.02 | 44.4 |
| `zarr-time12-y512-x512` | 0.06 | 0.03 | 0.03 | 0.02 | 44.4 |
| `zarr-time1-y256-x256` | 0.16 | 0.10 | 0.02 | 0.04 | 44.4 |

What the numbers say:

- **Format barely matters at this scale.** All five variants finish in well under a second; cold open is dominated by Python imports, not I/O.
- **Per-month NetCDF read is 20× slower than single-file NetCDF read.** This is the strongest signal in the table — it pinpoints `chelsa.py`'s 12-files-per-year pattern as the real friction, not the choice of format.
- **Zarr is 7 % smaller on disk** because xarray's default Zarr encoder applies zlib; NetCDF was uncompressed. Apples-to-apples (compressed NetCDF) would close most of that gap.
- **Zarr write is slower** at this size — compression overhead — but doesn't matter for a once-per-run output.
- **Chunking matters more than format.** `time=12, y=512, x=512` (one chunk) gives near-NetCDF write speed; `time=1, y=512, x=512` is 4× slower because it pays per-chunk overhead 12 times. For real workloads with dask, smaller chunks regain the lead via parallelism.

These numbers don't extrapolate to 100s of GB. At cache-scale (entire-country CHELSA stacks, year-stacks of WorldCover) the picture would shift; benchmark again before committing to a cache-format change.

## Question-by-question

### Read/write performance

Tied at our scale. Zarr only pulls ahead when there's a dask scheduler doing parallel writes (multi-process / multi-node) or when reads happen against object storage. Neither applies to chap_gis today.

### Cloud story

Zarr wins cleanly on S3/GCS — each chunk is one object, partial reads via byte-range, no metadata-blob bottleneck. NetCDF on cloud storage stalls on header reads and sliced access patterns. **But** the cloud move isn't on the roadmap. Treat this as a tiebreaker, not a forcing function.

### Tooling compatibility

- **CHAP**: doesn't read rasters from chap_gis — only the aggregated CSV (`cli/aggregate.py`). Format choice is invisible to CHAP.
- **xarray / rioxarray / dask**: first-class Zarr support; `to_zarr` / `open_zarr` are stable.
- **GDAL CLI**: Zarr driver since GDAL 3.6 (Oct 2022). `gdalinfo Zarr:store.zarr` works. Older GDAL installs (pre-3.6) won't.
- **QGIS**: GDAL-bound, so Zarr works on QGIS 3.30+ (which ships GDAL ≥ 3.6). Older long-term-release QGIS users won't see the store.
- **Notebooks**: zero impact — xarray hides the format.

No blockers; one watch-out is people on old QGIS.

### Single-file vs. directory

Zarr is a directory tree by default. For chap_gis-internal use this is fine — outputs are written once at the end of `analyze`, then read once by `aggregate` on the same machine. No shipping happens.

If we ever need to hand a single artifact to a teammate or CI, `ZipStore` (Zarr v2) / `FsspecStore("zip://...")` (Zarr v3) gives a one-file representation — read-only, write-once. That's a non-blocker rather than a feature.

### Inputs vs. outputs

- **Outputs**: under our full control, format is internal-only, the only readers are inside `chap_gis`. **Easy Zarr win.**
- **External-source caches**: come from CHELSA / WorldCover / WorldPop / NASA STAC in their native formats. Re-encoding to Zarr on each download means doubling I/O on cache-write for no read-side win until we move to cloud. **Skip for now.**

### Chunking strategy

Rule of thumb for monthly stacks at small-country resolution:

- ~1 MB chunks (e.g. `time=1, y=512, x=512` for float32) is the sweet spot for partial reads.
- Don't go below 256×256 — chunk-bookkeeping overhead eats the gain.
- For static rasters (elevation, landcover, crop, rice) chunk on `(y=2048, x=2048)` ≈ 16 MB chunks — they're read-whole more often than sliced.
- For multi-month stacks, chunk `time=1` so a single-month read materialises one chunk per spatial tile, not 12.

### Migration cost

Output bundling: **small**. One PR.

- `cli/analyze.py` last block (line ~189–210 today): replace the seven `ds[var].to_netcdf(out_dir / fn)` with one `ds.to_zarr(out_dir / "exposure.zarr", mode="w")`.
- `cli/aggregate.py`: replace its `xr.open_mfdataset(out_dir / "*.nc")` reshape pass with `xr.open_zarr(out_dir / "exposure.zarr")` — same Dataset shape, no reshape needed.
- Docs (README "Running the analysis" + "Aggregate") need a one-line update on the output path.
- Add `zarr>=3` to `[project.dependencies]`.

CHELSA bundling (NetCDF, no Zarr): **smaller still**. Replace the per-month write loop in `io/chelsa.py::download` with a single multi-month write at the end of the year window. Touches one file; tests already exist.

A full cache → Zarr migration is **bigger** — six loaders × native-format readers + cache key changes — and is *not* what this ticket recommends.

## Follow-up tickets if this lands

1. **Bundle `analyze` outputs into one Zarr store** (this recommendation, scoped). Touches `cli/analyze.py`, `cli/aggregate.py`, README, deps. ~½ day.
2. **Bundle CHELSA per-month NetCDFs into a single multi-month NetCDF.** Independent of the Zarr decision. ~½ day.
3. **Re-evaluate cache → Zarr** *if* cloud storage gets prioritised, *or* a 100 GB+ cache benchmark shows real wins.

## Appendix — what was *not* benchmarked

- Compressed NetCDF (zlib level 4). Would close the size gap with Zarr.
- Multi-process / dask-distributed writes. Where Zarr should pull ahead.
- Cache-scale (10–100 GB) reads. Where Zarr's chunked random access matters most.
- Object-storage reads (S3 latency).

If anyone disagrees with the recommendation, those four are the most likely places to find counter-evidence — re-run there.
