#!/usr/bin/env python
"""int_image() on floating point data outside the range 0 to 1."""

import numpy as np
import pytest

from machinevisiontoolbox.base.types import int_image


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_out_of_range_floats_are_clipped(dtype):
    x = np.array([-0.5, 0.0, 0.5, 1.0, 1.2], dtype=dtype)
    out = int_image(x, "uint8")
    assert out.dtype == np.uint8
    assert list(out) == [0, 0, 128, 255, 255]


def test_clipping_applies_to_uint16_and_to_color_arrays():
    x = np.full((2, 2, 3), 2.0)
    assert (int_image(x, "uint16") == 65535).all()
    assert (int_image(-x, "uint16") == 0).all()


def test_values_in_range_are_unchanged():
    x = np.linspace(0, 1, 11)
    assert list(int_image(x, "uint8")) == [round(v * 255) for v in x]
