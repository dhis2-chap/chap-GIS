import numpy as np

from chap_gis.landcover import breeding_site_mask, land_mask, water_mask


def test_water_mask(landcover):
    m = water_mask(landcover)
    assert bool(m.values[5, 5])
    assert not bool(m.values[0, 0])


def test_breeding_includes_wetlands(landcover):
    b = breeding_site_mask(landcover, water_edge_buffer=0)
    assert bool(b.values[10, 10])
    assert not bool(b.values[0, 0])


def test_breeding_water_edge_buffer(landcover):
    b0 = breeding_site_mask(landcover, water_edge_buffer=0)
    b2 = breeding_site_mask(landcover, water_edge_buffer=2)
    # edge ring only appears with buffer > 0
    assert b2.values.sum() > b0.values.sum()
    # the water pixel itself is not a breeding site (masked off)
    assert not bool(b2.values[5, 5])


def test_land_mask_shape(landcover):
    assert land_mask(landcover).shape == landcover.shape
