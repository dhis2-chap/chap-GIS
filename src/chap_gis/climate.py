"""Climate-derived variables."""

from __future__ import annotations

import xarray as xr

DEFAULT_LAPSE_RATE = 6.5e-3  # K per metre


def annual_mean(tas: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Reduce a time-varying temperature DataArray to its annual mean."""
    out = tas.mean(dim=dim, keep_attrs=True)
    out.attrs["cell_methods"] = (
        out.attrs.get("cell_methods", "") + f" {dim}: mean"
    ).strip()
    return out


def lapse_rate_downscale(
    coarse_tas: xr.DataArray,
    coarse_elev: xr.DataArray,
    fine_elev: xr.DataArray,
    lapse_rate: float = DEFAULT_LAPSE_RATE,
) -> xr.DataArray:
    """Downscale `coarse_tas` from coarse to fine elevation.

    All three inputs must share CRS, dims, coords, and shape — i.e. both the
    coarse temperature and the coarse elevation must already have been
    resampled onto the fine target grid before this call. The output is
    ``coarse_tas - lapse_rate * (fine_elev - coarse_elev)``.
    """
    _assert_aligned(coarse_tas, coarse_elev)
    _assert_aligned(coarse_tas, fine_elev)

    anomaly = fine_elev - coarse_elev
    out = coarse_tas - lapse_rate * anomaly
    out = out.where(fine_elev.notnull())
    out.name = coarse_tas.name
    out.attrs = {
        **coarse_tas.attrs,
        "history": f"lapse_rate_downscale(lapse_rate={lapse_rate})",
    }
    out = out.rio.write_crs(coarse_tas.rio.crs)
    return out


def _assert_aligned(a: xr.DataArray, b: xr.DataArray) -> None:
    if a.rio.crs != b.rio.crs:
        raise ValueError(f"CRS mismatch: {a.rio.crs} vs {b.rio.crs}")
    if a.dims != b.dims:
        raise ValueError(f"dim mismatch: {a.dims} vs {b.dims}")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
