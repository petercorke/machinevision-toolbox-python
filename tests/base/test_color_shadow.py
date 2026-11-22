#!/usr/bin/env python
"""Tests of shadow_invariant() and esttheta().

A synthetic image is built so that the right answer is known.  Pixels of one material
that are lit or in shadow differ only in their position along a line in the
log-chromaticity plane; the line perpendicular to it, at angle ``theta``, is the
invariant direction.  So projecting onto ``theta`` gives the same grey value for every
pixel, and sweeping the angle finds ``theta`` again.  The one thing replaced in
esttheta() is the user's mouse clicks on the region.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest
from matplotlib.figure import Figure

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import esttheta, shadow_invariant

THETA = 0.7  # the invariant angle of the synthetic image, radians


def synthetic(theta=THETA, shape=(6, 40), d0=(0.2, -0.1)):
    """Linear RGB image whose pixels differ only along the shadow direction.

    Returns the image and the grey value that the invariant should have everywhere.
    """
    c = np.array([np.cos(theta), np.sin(theta)])  # the invariant direction
    u = np.array([-np.sin(theta), np.cos(theta)])  # the direction shadows move pixels
    t = np.linspace(-1, 1, shape[1])
    d = np.array(d0)[:, None] + u[:, None] * t  # log chromaticities, (2, W)
    r, b = np.exp(d)
    g = 1 / (r * b)  # makes the geometric mean of R, G, B equal to 1
    row = np.stack([r, g, b], axis=1)  # (W, 3)
    im = np.broadcast_to(row, (shape[0], shape[1], 3)).copy()
    return im, float(np.dot(d0, c))


class TestShadowInvariant:
    def test_constant_for_lit_and_shadowed_pixels(self):
        im, expected = synthetic()
        gs = shadow_invariant(im, THETA)
        assert gs.shape == im.shape[:2]
        nt.assert_allclose(gs, expected, atol=1e-9)

    def test_wrong_angle_is_not_constant(self):
        im, _ = synthetic()
        gs = shadow_invariant(im, THETA + 0.5)
        assert np.ptp(gs) > 0.1

    def test_exp_exponentiates(self):
        im, expected = synthetic()
        nt.assert_allclose(shadow_invariant(im, THETA, exp=True), np.exp(expected))

    def test_green_channel_as_denominator(self):
        # without the geometric mean, the chromaticities are R/G and B/G
        im, _ = synthetic()
        rgb = im.reshape(-1, 3)
        d = np.log(np.stack([rgb[:, 0] / rgb[:, 1], rgb[:, 2] / rgb[:, 1]]))
        c = np.array([np.cos(THETA), np.sin(THETA)])
        gs = shadow_invariant(im, THETA, geometricmean=False)
        nt.assert_allclose(gs.ravel(), d.T @ c)

    def test_identity_sharpening_changes_nothing(self):
        im, _ = synthetic()
        nt.assert_allclose(
            shadow_invariant(im, THETA, sharpen=np.eye(3)), shadow_invariant(im, THETA)
        )

    def test_sharpening_is_applied(self):
        im, _ = synthetic()
        sharpen = np.array([[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [-0.1, -0.1, 1.2]])
        assert not np.allclose(
            shadow_invariant(im, THETA, sharpen=sharpen), shadow_invariant(im, THETA)
        )

    def test_default_angle_comes_from_the_camera_filters(self):
        # with no angle, the direction is worked out from the peak responses of the
        # Sony ICX204 sensor: 610, 538 and 460 nm
        im, _ = synthetic()
        c2 = 1.4388e-2
        e_r, e_b = -c2 / 610e-9, -c2 / 460e-9
        c = np.array([e_b, -e_r])
        c /= np.linalg.norm(c)
        rgb = im.reshape(-1, 3)
        denom = np.prod(rgb, axis=1) ** (1 / 3)
        d = np.log(np.stack([rgb[:, 0] / denom, rgb[:, 2] / denom]))
        nt.assert_allclose(shadow_invariant(im).ravel(), d.T @ c)

    def test_primaries_without_an_angle(self):
        # the direction is worked out from the filters' peak wavelengths, in
        # any units, since only their ratio matters
        im, _ = synthetic()
        primaries = np.array([600, 540, 450]) * 1e-9
        out = shadow_invariant(im, primaries=primaries)
        c = np.array([-1 / primaries[2], 1 / primaries[0]])
        c /= np.linalg.norm(c)
        d = shadow_invariant(im, np.arctan2(c[1], c[0]))
        nt.assert_allclose(out, d)
        # a different set of peaks gives a different direction
        other = shadow_invariant(im, primaries=np.array([650, 540, 420]) * 1e-9)
        assert not np.allclose(out, other)
        # nanometres give the same answer as metres
        nt.assert_allclose(shadow_invariant(im, primaries=primaries * 1e9), out)

    def test_default_primaries_are_the_sony_sensor(self):
        im, _ = synthetic()
        nt.assert_allclose(
            shadow_invariant(im),
            shadow_invariant(im, primaries=np.array([610, 538, 460]) * 1e-9),
        )

    def test_not_a_color_image_raises(self):
        with pytest.raises(ValueError):
            shadow_invariant(np.ones((4, 4)), THETA)
        with pytest.raises(ValueError):
            shadow_invariant(np.ones((4, 4, 4)), THETA)

    def test_zero_pixels_give_nan_not_an_exception(self):
        im, _ = synthetic(shape=(2, 6))
        im[0, 0] = 0
        gs = shadow_invariant(im, THETA)
        assert not np.isfinite(gs[0, 0])
        assert np.all(np.isfinite(gs[1]))


@pytest.fixture
def agg_backend():
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    plt.switch_backend(previous)


class TestEstTheta:
    """The region is picked with the mouse, so the clicks are supplied."""

    def click_whole_image(self, monkeypatch, shape):
        h, w = shape
        corners = [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]
        monkeypatch.setattr(Figure, "ginput", lambda self, n=-1, **kw: corners)

    def test_recovers_the_invariant_angle(self, agg_backend, monkeypatch, capsys):
        array, _ = synthetic()
        self.click_whole_image(monkeypatch, array.shape[:2])
        theta = esttheta(Image(array, colororder="RGB"))
        # the angle is searched in steps of 0.02 rad
        assert theta == pytest.approx(THETA, abs=0.021)
        out = capsys.readouterr().out
        assert f"{array.shape[0] * array.shape[1]} pixels selected" in out
        assert "best angle" in out

    def test_a_different_angle_is_found(self, agg_backend, monkeypatch):
        array, _ = synthetic(theta=1.9)
        self.click_whole_image(monkeypatch, array.shape[:2])
        assert esttheta(Image(array, colororder="RGB")) == pytest.approx(1.9, abs=0.021)

    def test_sharpening_is_accepted(self, agg_backend, monkeypatch):
        array, _ = synthetic()
        self.click_whole_image(monkeypatch, array.shape[:2])
        theta = esttheta(Image(array, colororder="RGB"), sharpen=np.eye(3))
        assert theta == pytest.approx(THETA, abs=0.021)
