# chap_gis — coding conventions

This package follows a "data-in / data-out" style on the
**xarray + rioxarray + dask + STAC** stack. Code here must obey these
conventions; experimental or unconventional code belongs in `raw_scripts/`.

## 1. General principles

- CF-compliant metadata + xarray-in / xarray-out + `.pipe()` composition +
  accessors for domain verbs + lazy by default.
- Functions take and return `DataArray` / `Dataset` — never raw numpy arrays.
- Preserve dims, coords, CRS, and attrs; don't silently drop them.
- Stay lazy; call `.compute()` only at the end (or never, inside library code).

## 2. Metadata

Follow CF + STAC conventions — they are the data contract.

- CF for netCDF/Zarr: `standard_name`, `units`, `long_name`, proper `time`
  encoding, `grid_mapping` for CRS.
- STAC for cataloging scenes/collections.
- Spatial dims: `x`, `y` (projected) or `lon`, `lat` (geographic) — don't mix.
- Time dim is always `time` (`datetime64[ns]` or `CFTimeIndex`).
- Bands/variables: lowercase, matching STAC common names (`red`, `nir`, …).
- CRS lives on the object via `rioxarray` (`.rio.write_crs(...)`); never pass
  CRS as a side-channel argument. Reproject explicitly at boundaries.

## 3. Single-input functions

Signature: `f(da, *args, **kwargs) -> da`.

- Use `keep_attrs=True` in reductions.
- Propagate CRS via `.rio.write_crs()` when building new arrays.
- Accept already-chunked inputs; document chunking assumptions in the docstring.
- Prefer `apply_ufunc` / `map_blocks` over manual loops.

## 4. Composition with `.pipe()`

Pipelines read top-to-bottom and each step is independently testable:

```python
result = (
    ds.pipe(mask_clouds)
      .pipe(ndvi)
      .pipe(lambda da: da.resample(time="1ME").median())
)
```

## 5. Accessors for domain verbs

Register `xr.register_dataarray_accessor` / `register_dataset_accessor` for
domain namespaces (`da.veg.ndvi()`), mirroring `.rio`, `.cf`, `xvec`.

## 6. Binary functions (two grids in, one out)

Two grids rarely share CRS, resolution, extent, or time axis.

- **Align explicitly** — don't rely on broadcasting:
  `a, b = xr.align(a, b, join="inner")` (use `join="exact"` in libraries to
  fail loudly).
- **Regridding is the caller's job** — provide it as a separate step
  (`b.rio.reproject_match(a)`, `xesmf.Regridder`). The choice of method is a
  scientific decision.
- **Name inputs by role**: `(reference, target)`, `(observed, predicted)`, or
  `(a, b)` with a docstring — pick one and stick to it. The first argument is
  the one whose metadata/grid is preserved.
- **Preserve metadata from one parent deliberately** — `keep_attrs=True` only
  propagates from the left operand; copy attrs and CRS explicitly when needed
  and append to `attrs["history"]`.
- **Time vs space**: align in time for paired series; broadcast a static grid
  against a time cube; resample to a common cadence *before* the function.
- **Validate the contract up front** (CRS, dims, shape, coord values), or
  apply the same `xarray-schema` to both inputs.
- **Return type matches inputs**: `DataArray + DataArray → DataArray`,
  `Dataset + Dataset → Dataset`. Return a `Dataset` (not a tuple) when the
  operation produces multiple named outputs.

> **Contract:** inputs must share CRS, dims, and coords. Caller aligns;
> function asserts. Output inherits metadata from the first ("primary")
> argument. Return the same type as inputs.

## 7. Typing and validation

- Static: `jaxtyping` + `beartype` for shape/dtype on signatures.
- Runtime schemas: `xarray-schema`, `xarray-dataclasses`, `pandera`.
- Metadata contracts: `cf-xarray`, `rioxarray` CRS checks, `pint-xarray`.
- Catalog models: `pystac`, `stac-pydantic`.

Insert validators between pipeline steps — they inspect metadata only and run
instantly on lazy arrays. Catches dims, dtypes, CRS, resolution, units,
chunks, CF metadata pre-compute.

## 8. Testing

- `xarray.testing.assert_allclose` / `assert_identical` for array equality
  including coords and attrs.
- Build tiny synthetic cubes in fixtures; avoid real rasters in unit tests.
- `hypothesis` + `hypothesis-geometry` for property-based tests on geometry.

## References

- Source notes: `~/Notes/geospatial-xarray-conventions.md`
- CF Conventions: https://cfconventions.org
- xarray user guide — "Working with custom accessors", `apply_ufunc`
- `xclim` — a strong real-world example of CF-aware xarray library design
