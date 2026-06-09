"""Tests for the exposure parameter-sweep machinery.

Covers the EDT/kernel split in ``chap_gis.exposure`` and the
``ExposureSweepSpec`` grid enumeration / config loading used by the
``dynamic-periods`` sweep path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from chap_gis.exposure import (
    compute_distance_field,
    exposure,
    exposure_from_field,
)
from chap_gis.pipelines.malaria_exposure import (
    ExposureSweepSpec,
    ThermalParams,
    combo_tag,
)
from chap_gis.cli.dynamic import _load_sweep_spec
from chap_gis.suitability import thermal_suitability


def _mk(arr, template):
    return xr.DataArray(arr, dims=template.dims, coords=template.coords).rio.write_crs(
        template.rio.crs
    )


def test_field_reuse_matches_direct_exposure(elevation):
    """A field built with the sweep's max lambda, reused for a smaller lambda,
    must reproduce a direct ``exposure()`` call.

    The 20x20 fixture is smaller than either halo, so the EDT is exact (single
    tile) and the two paths should agree to float precision.
    """
    breeding_np = np.zeros(elevation.shape, dtype=bool)
    breeding_np[0, 0] = True
    breeding_np[15, 17] = True
    breeding = _mk(breeding_np, elevation)

    # Vary elevation so the Δz term is exercised, not a no-op.
    elev_np = elevation.values.copy()
    elev_np[10:, :] += 40.0
    elev = _mk(elev_np, elevation)

    suit = _mk(np.full(elevation.shape, 0.7, dtype="float32"), elevation)

    ref = exposure(
        breeding, elev, suit, pixel_m=30.0, horizontal_lambda_m=400.0, vertical_gamma_m=22.5
    ).compute()

    # Field built with the larger sweep lambda (bigger halo), reused for 400.
    field = compute_distance_field(
        breeding.values, elev.values, pixel_m=30.0, lambda_m=900.0
    )
    got = exposure_from_field(field, suit.values, lambda_m=400.0, gamma_m=22.5)

    np.testing.assert_allclose(ref.values, got, rtol=1e-6, equal_nan=True)


def test_exposure_from_field_no_suitability_sets_breeding_to_one(elevation):
    breeding_np = np.zeros(elevation.shape, dtype=bool)
    breeding_np[0, 0] = True
    field = compute_distance_field(
        breeding_np, elevation.values, pixel_m=30.0, lambda_m=651.0
    )
    expo = exposure_from_field(field, None, lambda_m=651.0, gamma_m=22.5)
    assert expo[0, 0] == pytest.approx(1.0)
    assert expo[0, 1] == pytest.approx(np.exp(-30.0 / 651.0), rel=1e-5)


def test_compute_distance_field_raises_on_empty_breeding(elevation):
    with pytest.raises(ValueError, match="no breeding sites"):
        compute_distance_field(
            np.zeros(elevation.shape, dtype=bool),
            elevation.values,
            pixel_m=30.0,
            lambda_m=651.0,
        )


def test_combos_count_order_and_tags():
    spec = ExposureSweepSpec(
        lambda_m=[400.0, 651.0],
        gamma_m=[22.5],
        water_edge_buffer_pixels=[1, 2],
        thermal=[ThermalParams(), ThermalParams(t_opt=27.0)],
    )
    combos = spec.combos()
    # 2 buffers * 2 thermal * 2 lambda * 1 gamma
    assert len(combos) == 8
    assert [c.tag for c in combos] == [combo_tag(i) for i in range(8)]
    # nested order is water_buffer -> thermal -> lambda -> gamma
    first = combos[0]
    assert (first.water_edge_buffer_pixels, first.thermal.t_opt, first.lambda_m) == (
        1, 25.0, 400.0,
    )
    # tags are unique
    assert len({c.tag for c in combos}) == 8


def test_default_spec_is_single_default_combo():
    combos = ExposureSweepSpec().combos()
    assert len(combos) == 1
    c = combos[0]
    assert c.lambda_m == 651.0 and c.gamma_m == 22.5
    assert c.water_edge_buffer_pixels == 2
    assert c.thermal.t_opt == 25.0


def test_load_sweep_spec_json(tmp_path):
    cfg = tmp_path / "grid.json"
    cfg.write_text(
        json.dumps(
            {
                "lambda_m": [400, 900],
                "gamma_m": [22.5, 50],
                "water_edge_buffer_pixels": [2],
                "thermal": [{"t_opt": 25, "sigma": 5, "t_min": 16, "t_max": 34}],
            }
        )
    )
    spec = _load_sweep_spec(cfg)
    assert spec.lambda_m == [400.0, 900.0]
    assert len(spec.combos()) == 4


def test_combo_tag_consistency_with_thermal_t_max_none():
    spec = ExposureSweepSpec(thermal=[ThermalParams(t_max=None)])
    (c,) = spec.combos()
    assert c.thermal.t_max is None
