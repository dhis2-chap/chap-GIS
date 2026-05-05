"""Runtime grid-alignment check for functions taking multiple DataArrays.

The :func:`same_grid` decorator declares that the ``xr.DataArray`` arguments
of a function must share a spatial grid (CRS, affine transform, spatial
shape). Comparison is by ``odc.geo.GeoBox`` equality, which collapses those
three things into a single value and ignores non-spatial dims — a 2D mask
and a 3D ``(time, y, x)`` array are aligned if their spatial extents match.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable, get_type_hints

import odc.geo.xr  # noqa: F401  registers the .odc accessor
import xarray as xr


def same_grid(*params):
    """Assert that selected ``xr.DataArray`` arguments share the same GeoBox.

    Two call shapes::

        @same_grid                           # all xr.DataArray-annotated params
        def f(a: xr.DataArray, b: xr.DataArray): ...

        @same_grid("a", "b")                 # explicit subset
        def g(a, b, scale): ...

    ``None``-valued arguments are skipped, so ``xr.DataArray | None`` optional
    inputs work without ceremony. On mismatch raises :class:`ValueError` with
    a message naming the disagreeing parameters and which of CRS, shape, or
    transform differs (the legacy ``"CRS mismatch"`` substring is preserved).
    """
    if len(params) == 1 and callable(params[0]) and not isinstance(params[0], str):
        return _build(params[0], names=None)

    if not all(isinstance(p, str) for p in params):
        raise TypeError("same_grid arguments must be parameter name strings")

    def deco(func):
        return _build(func, names=params or None)

    return deco


def _build(func: Callable, *, names: tuple[str, ...] | None) -> Callable:
    sig = inspect.signature(func)

    if names is None:
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        names = tuple(p for p in sig.parameters if _is_dataarray_annotation(hints.get(p)))

    unknown = [n for n in names if n not in sig.parameters]
    if unknown:
        raise TypeError(
            f"same_grid: unknown parameter(s) {unknown} on {func.__name__}"
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        _check(func.__name__, names, bound.arguments)
        return func(*args, **kwargs)

    return wrapper


def _is_dataarray_annotation(ann) -> bool:
    if ann is xr.DataArray:
        return True
    args = getattr(ann, "__args__", None)
    if args:
        return any(a is xr.DataArray for a in args)
    return False


def _check(func_name: str, names: tuple[str, ...], arguments: dict) -> None:
    pairs = []
    for n in names:
        v = arguments.get(n)
        if v is None:
            continue
        if not isinstance(v, xr.DataArray):
            raise TypeError(
                f"{func_name}: argument {n!r} expected xr.DataArray, "
                f"got {type(v).__name__}"
            )
        pairs.append((n, v))
    if len(pairs) < 2:
        return

    ref_name, ref = pairs[0]
    ref_box = ref.odc.geobox
    for name, da in pairs[1:]:
        box = da.odc.geobox
        if box == ref_box:
            continue
        diffs = []
        if box.crs != ref_box.crs:
            diffs.append(
                f"CRS mismatch ({ref_name}={ref_box.crs} vs {name}={box.crs})"
            )
        if tuple(box.shape) != tuple(ref_box.shape):
            diffs.append(
                f"shape mismatch ({ref_name}={tuple(ref_box.shape)} vs "
                f"{name}={tuple(box.shape)})"
            )
        if box.affine != ref_box.affine:
            diffs.append(
                f"transform mismatch ({ref_name}={tuple(ref_box.affine)[:6]} vs "
                f"{name}={tuple(box.affine)[:6]})"
            )
        detail = "; ".join(diffs) or f"GeoBox mismatch ({ref_name} vs {name})"
        raise ValueError(f"{func_name}: {detail}")
