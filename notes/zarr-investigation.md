# Zarr as the chap_gis persistence format — investigation

> Ticket: [CLIM-676](https://dhis2.atlassian.net/browse/CLIM-676)
> Related: CLIM-672 (caching consolidation), CLIM-673 (CLI thinning), CLIM-677 (lazy serialize/deserialize).

## TL;DR — recommendation

**One serialization format, and that format is NetCDF.** Zarr stays on the table as a future migration when (and only when) cloud storage lands.

| Layer | Today | Recommended |
|---|---|---|
| `cli/analyze.py` outputs (7 separate NetCDFs read back by `aggregate`) | Per-variable NetCDF | **Single multi-variable NetCDF** (`exposure.nc`) |
| `io/chelsa.py` per-month NetCDF cache | 12 small `.nc` files | **Single multi-month NetCDF** |
| External-source caches (`worldcover`, `elevation`, `crop`, `rice`) | TIF as downloaded | **Status quo** — these are the source format, not our serialization |

The two structural wins worth chasing now:

1. **Bundle `analyze` outputs into one multi-variable NetCDF**, so `aggregate` opens a single file instead of stitching seven NetCDFs back together. The format-vs-format question is a tie at our current scale; the *bundling* is the win.
2. **Bundle CHELSA per-month files into one multi-month NetCDF.** Per-month + `open_mfdataset` was the slowest read pattern in the benchmark, ~20× slower than a single-file read.

Why NetCDF over Zarr when forced to pick one:

- External sources arrive as NetCDF/GeoTIFF; choosing Zarr forces re-encoding on every cache write.
- NetCDF was as fast or faster than Zarr in the benchmark at our scale.
- A `.nc` file is a single file; a Zarr store is a directory tree. Simpler artifact for shipping, version control, and CI.
- Universal tool support: every QGIS install reads NetCDF; Zarr requires GDAL ≥ 3.6.
- The ecosystem is decades-old and stable. Zarr 3 is still settling (the consolidated-metadata warning fires in our benchmark today).

The tradeoff we're trading away:

- Parallel-write throughput via dask (Zarr's strength on multi-process workloads).
- Cloud-native object-store reads (Zarr's strength on S3/GCS).

Both are dormant in chap_gis today. The migration to Zarr stays cheap because writing one multi-variable NetCDF per output maps 1:1 to one multi-variable Zarr store — a one-line `to_netcdf` → `to_zarr` swap when we cross that bridge. See [Future migration path](#future-migration-path-to-zarr) below.

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
- **Single-file NetCDF beats every Zarr variant on every operation** at this scale. Zarr's wins (parallel writes, partial-read efficiency) need a workload we don't have today.
- **Zarr is 7 % smaller on disk** because xarray's default Zarr encoder applies zlib; NetCDF was uncompressed. Apples-to-apples (compressed NetCDF) would close most of that gap.
- **Chunking matters more than format.** `time=12, y=512, x=512` (one chunk) gives near-NetCDF write speed; `time=1, y=512, x=512` is 4× slower because it pays per-chunk overhead 12 times. For real workloads with dask, smaller chunks regain the lead via parallelism — but we're not there yet.

These numbers don't extrapolate to 100s of GB. At cache-scale (entire-country CHELSA stacks, year-stacks of WorldCover) the picture would shift; benchmark again before committing to a cache-format change.

## Question-by-question

### Read/write performance

Tied at our scale, with a slight edge to NetCDF. Zarr only pulls ahead when there's a dask scheduler doing parallel writes (multi-process / multi-node) or when reads happen against object storage. Neither applies to chap_gis today.

### Cloud story

Zarr wins cleanly on S3/GCS — each chunk is one object, partial reads via byte-range, no metadata-blob bottleneck. NetCDF on cloud storage stalls on header reads and sliced access patterns. **But** the cloud move isn't on the roadmap. Treat this as the *trigger* for a future Zarr migration, not a reason to switch now.

### Tooling compatibility

- **CHAP**: doesn't read rasters from chap_gis — only the aggregated CSV (`cli/aggregate.py`). Format choice is invisible to CHAP.
- **xarray / rioxarray / dask**: first-class support for both formats; `to_netcdf` / `open_dataset` and `to_zarr` / `open_zarr` are equally stable.
- **GDAL CLI**: NetCDF driver since forever; Zarr driver since GDAL 3.6 (Oct 2022).
- **QGIS**: NetCDF supported on every install; Zarr requires QGIS 3.30+ (which ships GDAL ≥ 3.6). Old long-term-release users would lose access.
- **R / `ncdf4` / matlab**: NetCDF is the lingua franca; Zarr support is sparser outside the Python ecosystem.

NetCDF wins on the breadth axis.

### Single-file vs. directory

NetCDF is a single `.nc` file. Easy to email, easy to ship as a CI artifact, easy to git-LFS, easy to inspect with `ncdump`.

Zarr is a directory tree by default. For chap_gis-internal use this would be fine, but for any artifact handed across a boundary (teammate, CI, archive) we'd reach for `ZipStore` — read-only, write-once, an extra concept.

### Inputs vs. outputs

- **Outputs we generate**: NetCDF, single multi-variable file per pipeline run. Trivially Zarr-able later.
- **External-source caches**: stay in their native arrival format (NetCDF for CHELSA, GeoTIFF for WorldCover/elevation/crop/rice). Re-encoding on cache write means doubling I/O for no read-side win — and worse, breaks the "cache the bytes the source gave us" property that makes the cache reproducible.

GeoTIFF caches are *not* a serialization-format choice on our side — they're preserving what the source delivered. That's true under either format-consistency rule.

### Chunking strategy

For NetCDF (no chunking by default) the benchmark says: don't bother. Single contiguous storage is fastest at our scale.

For Zarr (when we eventually adopt it), the rules of thumb:

- ~1 MB chunks (e.g. `time=1, y=512, x=512` for float32) is the sweet spot for partial reads.
- Don't go below 256×256 — chunk-bookkeeping overhead eats the gain.
- For multi-month stacks, chunk `time=1` so a single-month read materialises one chunk per spatial tile, not 12.

### Migration cost

NetCDF-bundling outputs: **small**. One PR.

- `cli/analyze.py` final block (lines ~189–210): replace the seven `ds[var].to_netcdf(out_dir / fn)` calls with `ds.to_netcdf(out_dir / "exposure.nc")`.
- `cli/aggregate.py`: replace its `xr.open_mfdataset(out_dir / "*.nc")` + reshape pass with `xr.open_dataset(out_dir / "exposure.nc")` — same Dataset shape, no reshape needed.
- README "Running the analysis" + "Aggregate": one-line update on the output path.
- No new dependency.

CHELSA per-month bundling: **smaller still**. Replace the per-month write loop in `io/chelsa.py::download` with a single multi-month write at the end of the window. Touches one file; tests already exist.

A future Zarr migration is documented below; it stays cheap precisely because we're picking the same logical layout (one store / file per pipeline output) under both formats.

## Future migration path to Zarr

When cloud storage becomes a priority (object-store cache, S3-backed outputs) — or a benchmark at cache-scale flips the speed picture — the upgrade is mechanical:

1. Add `zarr>=3` to `[project.dependencies]`.
2. `ds.to_netcdf(out_dir / "exposure.nc")` → `ds.to_zarr(out_dir / "exposure.zarr", mode="w")`.
3. `xr.open_dataset(out_dir / "exposure.nc")` → `xr.open_zarr(out_dir / "exposure.zarr")`.
4. Pick chunks (see above).
5. Decide whether to migrate cached external-source files at the same time, or leave them as TIF/NetCDF (still likely the right call — they're external input).

Because we're committing now to **one logical artifact per pipeline output** (multi-variable, single-file/store), the Zarr swap will be a few lines and a chunk-size choice — not a re-architecture.

## Follow-up tickets if this lands

1. **Bundle `analyze` outputs into one multi-variable NetCDF.** Touches `cli/analyze.py`, `cli/aggregate.py`, README. ~½ day. Replaces the seven-file output with `exposure.nc`.
2. **Bundle CHELSA per-month NetCDFs into a single multi-month NetCDF.** Touches `io/chelsa.py`. ~½ day. Independent of #1.
3. **Re-evaluate Zarr migration** *if and when* cloud storage gets prioritised, *or* a 100 GB+ cache benchmark shows real Zarr wins.

## Appendix — what was *not* benchmarked

- Compressed NetCDF (zlib level 4). Would close the size gap with Zarr.
- Multi-process / dask-distributed writes. Where Zarr should pull ahead.
- Cache-scale (10–100 GB) reads. Where Zarr's chunked random access matters most.
- Object-storage reads (S3 latency).

If anyone disagrees with the recommendation, those four are the most likely places to find counter-evidence — re-run there before flipping the format choice.
