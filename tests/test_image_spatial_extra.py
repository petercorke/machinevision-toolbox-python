#!/usr/bin/env python
"""More tests for ``ImageSpatial``: filtering, rank filters, pyramids, labelling
and the similarity measures.

Test images are small synthetic arrays, and expected values are worked out
directly with NumPy, so nothing is mocked.
"""


import cv2
import numpy as np
import numpy.testing as nt
import pytest
import scipy.ndimage as ndi

from machinevisiontoolbox import Image, Kernel


@pytest.fixture
def square():
    """40 x 40 uint8 image, a bright square at rows and columns 10..29."""
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 200
    return Image(arr)


@pytest.fixture
def binary():
    """The same square as a 0 / 255 image."""
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    return Image(arr)


# --------------------------------------------------------------------------- #
# convolution and gradients


class TestConvolve:
    def test_full_valid_and_same_sizes(self, square):
        box = np.ones((3, 3)) / 9
        assert square.convolve(box, mode="same").shape == (40, 40)
        assert square.convolve(box, mode="valid").shape == (38, 38)
        assert square.convolve(box, mode="full").shape == (42, 42)

    def test_bad_mode(self, square):
        with pytest.raises(ValueError):
            square.convolve(np.ones((3, 3)), mode="outside")

    def test_pad_border_value(self):
        im = Image(np.full((6, 6), 100, dtype=np.uint8))
        box = np.ones((3, 3)) / 9
        # padding with the image value leaves a constant image constant
        assert (im.convolve(box, border="pad", bordervalue=100).array == 100).all()
        # padding with zero darkens the corner: 4 of its 9 neighbours are real
        corner = im.convolve(box, border="pad").array[0, 0]
        assert corner == pytest.approx(100 * 4 / 9, abs=1)

    def test_kernel_may_be_an_image_or_a_kernel(self, square):
        box = np.ones((3, 3), dtype=np.float32) / 9
        a = square.convolve(box)
        b = square.convolve(Image(box))
        c = square.convolve(Kernel.Box(1))
        nt.assert_array_equal(a.array, b.array)
        assert c.shape == a.shape

    def test_smooth_equals_convolution_with_a_gaussian(self, square):
        K = Kernel.Gauss(2)
        nt.assert_array_equal(square.smooth(2).array, square.convolve(K).array)


class TestGradients:
    @pytest.fixture
    def ramp(self):
        # intensity equal to the column number
        return Image(np.tile(np.arange(20, dtype=np.float32), (20, 1)))

    def test_sobel_of_a_ramp(self, ramp):
        Iu, Iv = ramp.gradients()
        # the toolbox's Sobel kernel is normalised, so it has unit gain on a
        # ramp of slope one
        assert abs(Iu.array[10, 10]) == pytest.approx(1)
        assert Iv.array[10, 10] == pytest.approx(0)

    def test_custom_kernel_and_mode(self, ramp):
        Iu, Iv = ramp.gradients(Kernel.Sobel(), mode="valid")
        assert Iu.shape == (18, 18) and Iv.shape == (18, 18)

    def test_direction(self):
        horizontal = Image(np.ones((3, 3), dtype=np.float32))
        vertical = Image(np.ones((3, 3), dtype=np.float32))
        d = Image.direction(horizontal, vertical)
        nt.assert_allclose(d.array, np.pi / 4, rtol=1e-6)

    def test_direction_of_a_vertical_gradient(self):
        zero = Image(np.zeros((2, 2), dtype=np.float32))
        one = Image(np.ones((2, 2), dtype=np.float32))
        nt.assert_allclose(Image.direction(zero, one).array, np.pi / 2, rtol=1e-6)

    def test_direction_shapes_must_match(self):
        with pytest.raises(ValueError, match="same shape"):
            Image.direction(
                Image(np.ones((2, 2), dtype=np.float32)),
                Image(np.ones((3, 3), dtype=np.float32)),
            )


class TestHarris:
    def test_strongest_response_is_at_a_corner(self, square):
        strength = square.Harris_corner_strength().array
        v, u = np.unravel_index(np.argmax(strength), strength.shape)
        corners = [(10, 10), (10, 29), (29, 10), (29, 29)]
        assert min(np.hypot(u - cu, v - cv) for cu, cv in corners) <= 2

    def test_flat_image_has_no_response(self):
        flat = Image(np.full((20, 20), 50, dtype=np.uint8))
        assert np.abs(flat.Harris_corner_strength().array).max() == 0


# --------------------------------------------------------------------------- #
# window, zero crossings, scale space, pyramid


class TestWindow:
    def test_maximum_over_a_window(self, square):
        out = square.window(np.max, h=1)
        # the bright square grows by one pixel on every side
        assert out.array[9, 9] == 200 and out.array[8, 8] == 0
        assert out.array[30, 30] == 200 and out.array[31, 31] == 0

    def test_structuring_element(self):
        arr = np.zeros((7, 7), dtype=np.uint8)
        arr[3, 3] = 9
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        out = Image(arr).window(np.max, se=cross)
        assert out.array[3, 4] == 9 and out.array[4, 3] == 9
        assert out.array[4, 4] == 0  # a diagonal neighbour is outside the cross

    def test_color_image_is_refused(self):
        with pytest.raises(ValueError, match="single channel"):
            Image(np.zeros((4, 4, 3), dtype=np.uint8)).window(np.max, h=1)

    def test_function_must_be_callable(self, square):
        with pytest.raises(TypeError, match="not callable"):
            square.window(5, h=1)

    @pytest.mark.parametrize(
        "border", ["constant", "replicate", "reflect", "mirror", "wrap"]
    )
    def test_border_options(self, square, border):
        assert square.window(np.mean, h=1, border=border).shape == (40, 40)

    def test_bad_border(self, square):
        with pytest.raises(ValueError, match="not a valid option"):
            square.window(np.max, h=1, border="outside")


def test_zerocross():
    # negative for columns up to 19, positive from 20: crossing between them
    arr = np.tile(np.arange(40, dtype=np.float32) - 19.5, (10, 1))
    out = Image(arr).zerocross().array
    assert out.dtype == bool
    assert set(np.nonzero(out[5])[0]) == {19, 20}


class TestScaleSpace:
    def test_sizes_and_scales(self, square):
        g, lap, scales = square.scalespace(3, sigma=1)
        assert len(g) == 3 and len(lap) == 2 and len(scales) == 3
        assert all(im.shape == (40, 40) for im in g + lap)
        nt.assert_allclose(scales, [0.5, np.sqrt(0.25 + 1), np.sqrt(0.25 + 2)])

    def test_each_level_is_smoother(self, square):
        g, _, _ = square.scalespace(4, sigma=2)
        # total variation falls as the image is blurred
        tv = [np.abs(np.diff(im.array.astype(float), axis=1)).sum() for im in g]
        assert all(a >= b for a, b in zip(tv, tv[1:]))


class TestPyramid:
    def test_halves_each_level(self, square):
        pyr = square.pyramid(N=3)
        assert [p.shape for p in pyr] == [(40, 40), (20, 20), (10, 10), (5, 5)]

    def test_default_stops_at_one_pixel(self):
        pyr = Image(np.zeros((8, 8), dtype=np.uint8)).pyramid()
        assert pyr[-1].shape[0] == 1 or pyr[-1].shape[1] == 1

    def test_color_image_becomes_mono(self):
        pyr = Image(np.zeros((16, 16, 3), dtype=np.uint8), colororder="RGB").pyramid(
            N=1
        )
        assert pyr[1].shape == (8, 8)

    def test_constant_border_is_not_supported(self, square):
        with pytest.raises(ValueError, match="not supported"):
            square.pyramid(border="constant")

    def test_sigma_must_be_a_scalar(self, square):
        with pytest.raises(ValueError, match="scalar"):
            square.pyramid(sigma=[1, 2])


# --------------------------------------------------------------------------- #
# canny


class TestCanny:
    def test_finds_the_edges_of_a_square(self, square):
        edges = square.canny().array
        assert edges.dtype == np.uint8
        assert set(np.unique(edges)) <= {0, 255}
        v, u = np.nonzero(edges)
        assert len(u) > 40
        # every edge pixel is within two pixels of the square's boundary
        inside = (u >= 8) & (u <= 31) & (v >= 8) & (v <= 31)
        core = (u >= 12) & (u <= 27) & (v >= 12) & (v <= 27)
        assert inside.all()
        assert not core.any()

    def test_flat_image_has_no_edges(self):
        assert Image(np.full((20, 20), 80, dtype=np.uint8)).canny().array.max() == 0

    @pytest.mark.xfail(
        strict=True,
        reason="canny() computes its thresholds from the mean of the image and "
        "ignores sigma, th0 and th1",
    )
    def test_thresholds_have_an_effect(self, square):
        default = (square.canny().array > 0).sum()
        strict = (square.canny(th0=250, th1=255).array > 0).sum()
        assert strict < default


# --------------------------------------------------------------------------- #
# rank and median filters


class TestRankFilter:
    @pytest.fixture
    def noisy(self):
        arr = np.full((9, 9), 50, dtype=np.uint8)
        arr[4, 4] = 255  # one bright pixel
        arr[2, 6] = 0  # one dark pixel
        return Image(arr)

    def test_median_removes_isolated_noise(self, noisy):
        out = noisy.medianfilter(h=1)
        assert (out.array == 50).all()

    @pytest.mark.parametrize(
        "rank, reference",
        [
            ("min", ndi.minimum_filter),
            ("minimum", ndi.minimum_filter),
            ("max", ndi.maximum_filter),
            ("maximum", ndi.maximum_filter),
            ("median", ndi.median_filter),
            ("med", ndi.median_filter),
        ],
    )
    def test_named_ranks(self, noisy, rank, reference):
        out = noisy.rankfilter(h=2, rank=rank)
        expected = reference(noisy.array, size=5, mode="nearest")
        nt.assert_array_equal(out.array, expected)

    def test_integer_rank_zero_is_the_maximum(self, noisy):
        # rank 0 selects the largest value of the window
        out = noisy.rankfilter(h=1, rank=0)
        expected = ndi.maximum_filter(noisy.array, size=3, mode="nearest")
        nt.assert_array_equal(out.array, expected)

    def test_footprint(self, noisy):
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        out = noisy.rankfilter(footprint=cross, rank=0)
        expected = ndi.maximum_filter(noisy.array, footprint=cross, mode="nearest")
        nt.assert_array_equal(out.array, expected)

    def test_color_image_is_filtered_by_plane(self):
        arr = np.zeros((7, 7, 3), dtype=np.uint8)
        arr[3, 3] = (255, 128, 64)
        out = Image(arr, colororder="RGB").rankfilter(h=1, rank=0)
        assert out.shape == (7, 7, 3)
        assert tuple(out.array[2, 2]) == (255, 128, 64)

    def test_needs_a_window(self, noisy):
        with pytest.raises(ValueError, match="footprint or h"):
            noisy.rankfilter(rank=0)

    def test_unknown_rank_name(self, noisy):
        with pytest.raises(ValueError, match="unknown rank"):
            noisy.rankfilter(h=1, rank="middle")

    def test_rank_must_be_int_or_str(self, noisy):
        with pytest.raises(TypeError):
            noisy.rankfilter(h=1, rank=1.5)

    def test_rank_must_not_be_negative(self, noisy):
        with pytest.raises(ValueError, match=">= 0"):
            noisy.rankfilter(h=1, rank=-1)

    def test_rank_is_deprecated(self, noisy):
        with pytest.warns(DeprecationWarning, match="rankfilter"):
            out = noisy.rank(h=1, rank=0)
        nt.assert_array_equal(out.array, noisy.rankfilter(h=1, rank=0).array)


# --------------------------------------------------------------------------- #
# distance transform and labelling


class TestDistanceTransform:
    def test_distance_to_the_nearest_set_pixel(self):
        arr = np.zeros((41, 41), dtype=np.uint8)
        arr[20, 20] = 255
        d = Image(arr).distance_transform()
        assert d.array[20, 20] == 0
        # OpenCV approximates the Euclidean distance with a small mask
        assert d.array[20, 25] == pytest.approx(5, abs=0.3)
        assert d.array[10, 20] == pytest.approx(10, abs=0.6)

    def test_distance_to_the_nearest_zero_pixel(self, binary):
        d = binary.distance_transform(invert=True)
        assert d.array[0, 0] == 0
        # the middle of a 20 pixel wide square is 10 pixels from the edge
        assert d.array[19, 19] == pytest.approx(10, abs=1)

    def test_l1_norm(self):
        arr = np.zeros((21, 21), dtype=np.uint8)
        arr[10, 10] = 255
        d = Image(arr).distance_transform(norm="L1")
        assert d.array[10, 15] == pytest.approx(5)
        assert d.array[13, 14] == pytest.approx(7)  # city-block: 3 + 4

    def test_needs_an_8_bit_image(self):
        with pytest.raises((TypeError, ValueError)):
            Image(np.zeros((5, 5), dtype=np.float32)).distance_transform()


class TestLabels:
    def test_binary_connectivity(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 0] = arr[1, 1] = 255  # touch at a corner only
        im = Image(arr)
        labels4, n4 = im.labels_binary(connectivity=4)
        labels8, n8 = im.labels_binary(connectivity=8)
        # the count includes the background label
        assert n4 == 3 and n8 == 2
        assert labels4.array[0, 0] != labels4.array[1, 1]
        assert labels8.array[0, 0] == labels8.array[1, 1]

    def test_two_separate_blobs(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[1:3, 1:3] = 255
        arr[6:9, 6:9] = 255
        labels, n = Image(arr).labels_binary()
        assert n == 3
        assert labels.array[1, 1] != labels.array[6, 6]
        assert labels.array[0, 0] == 0

    def test_uint16_labels(self, binary):
        labels, n = binary.labels_binary(ltype="uint16")
        assert n == 2
        assert set(np.unique(labels.array)) == {0, 1}

    def test_bad_connectivity(self, binary):
        with pytest.raises(ValueError, match="connectivity"):
            binary.labels_binary(connectivity=6)

    def test_bad_label_type(self, binary):
        with pytest.raises(TypeError, match="ltype"):
            binary.labels_binary(ltype="float")

    def test_mser_labels(self):
        im = Image.Read("castle.png", mono=True)
        labels, n = im.labels_MSER()
        assert labels.shape == im.shape
        if n == 0:
            pytest.skip("OpenCV 5 finds no MSERs on this image, see Issue 54")
        assert labels.array.max() <= n - 1

    @pytest.mark.skipif(
        not hasattr(cv2, "ximgproc"), reason="needs opencv-contrib-python"
    )
    def test_graph_segmentation(self, square):
        labels, n = square.labels_graphseg()
        assert labels.shape == square.shape
        # the square and its background are two segments
        assert n == 2
        assert labels.array[20, 20] != labels.array[0, 0]


# --------------------------------------------------------------------------- #
# similarity measures

A = np.array([[10, 20], [30, 40]], dtype=float)
B = np.array([[20, 10], [30, 50]], dtype=float)


def za(x):
    return x - x.mean()


class TestSimilarityMeasures:
    @pytest.fixture
    def pair(self):
        return Image(A), Image(B)

    def test_sad(self, pair):
        assert pair[0].sad(pair[1]) == pytest.approx(np.abs(A - B).sum())

    def test_ssd(self, pair):
        assert pair[0].ssd(pair[1]) == pytest.approx(((A - B) ** 2).sum())

    def test_ncc(self, pair):
        expected = (A * B).sum() / np.sqrt((A**2).sum() * (B**2).sum())
        assert pair[0].ncc(pair[1]) == pytest.approx(expected, rel=1e-5)

    def test_zsad(self, pair):
        assert pair[0].zsad(pair[1]) == pytest.approx(np.abs(za(A) - za(B)).sum())

    def test_zssd(self, pair):
        assert pair[0].zssd(pair[1]) == pytest.approx(((za(A) - za(B)) ** 2).sum())

    def test_zncc(self, pair):
        a, b = za(A), za(B)
        expected = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
        assert pair[0].zncc(pair[1]) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("name", ["sad", "ssd", "ncc", "zsad", "zssd", "zncc"])
    def test_identical_images(self, name):
        im = Image(A)
        value = getattr(im, name)(im)
        assert value == pytest.approx({"ncc": 1, "zncc": 1}.get(name, 0), abs=1e-6)

    @pytest.mark.parametrize("name", ["sad", "ssd", "ncc", "zsad", "zssd", "zncc"])
    def test_shapes_must_match(self, name):
        with pytest.raises(ValueError, match="not equal"):
            getattr(Image(A), name)(Image(np.zeros((3, 3))))

    def test_correlation_of_flat_images_is_zero(self):
        zero = Image(np.zeros((2, 2)))
        one = Image(np.ones((2, 2)))
        assert zero.ncc(zero) == 0
        assert one.zncc(one) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="sad, ssd and ncc subtract and square uint8 arrays without "
        "converting them, so the arithmetic wraps around at 256",
    )
    def test_uint8_images(self):
        a = Image(A.astype(np.uint8))
        b = Image(B.astype(np.uint8))
        assert a.sad(b) == pytest.approx(np.abs(A - B).sum())
        assert a.ncc(b) <= 1


class TestSimilarity:
    @pytest.fixture
    def scene(self):
        rng = np.random.default_rng(0)
        image = Image(rng.integers(0, 255, (20, 20)).astype(np.float32))
        template = Image(image.array[5:8, 5:8].copy())
        return image, template

    @pytest.mark.parametrize("metric", ["ncc", "zncc"])
    def test_correlation_peaks_at_the_template(self, scene, metric):
        image, template = scene
        out = image.similarity(template, metric)
        assert out.shape == (18, 18)
        v, u = np.unravel_index(np.argmax(out.array), out.shape)
        assert (u, v) == (5, 5)
        assert out.array[v, u] == pytest.approx(1.0, abs=1e-4)

    def test_ssd_is_zero_at_the_template(self, scene):
        image, template = scene
        out = image.similarity(template, "ssd")
        v, u = np.unravel_index(np.argmin(out.array), out.shape)
        assert (u, v) == (5, 5)
        assert out.array[v, u] == pytest.approx(0, abs=1e-3)

    def test_zssd_is_zero_at_the_template(self, scene):
        image, template = scene
        out = image.similarity(template, "zssd")
        v, u = np.unravel_index(np.argmin(out.array), out.shape)
        assert (u, v) == (5, 5)

    def test_template_must_have_odd_size(self, scene):
        image, _ = scene
        with pytest.raises(ValueError, match="odd"):
            image.similarity(Image(np.zeros((2, 3), dtype=np.float32)))

    def test_unknown_metric(self, scene):
        image, template = scene
        with pytest.raises(ValueError, match="bad metric"):
            image.similarity(template, "nonsense")

    def test_template_is_unchanged_by_ncc(self, scene):
        image, template = scene
        before = template.array.copy()
        image.similarity(template, "ncc")
        nt.assert_array_equal(template.array, before)

    @pytest.mark.xfail(
        strict=True,
        reason="the z metrics subtract the template mean in place, which "
        "changes the caller's template",
    )
    def test_template_is_unchanged_by_zncc(self, scene):
        image, template = scene
        before = template.array.copy()
        image.similarity(template, "zncc")
        nt.assert_array_equal(template.array, before)

    @pytest.mark.xfail(
        strict=True,
        raises=Exception,
        reason="the in-place mean subtraction fails on a uint8 image",
    )
    def test_zncc_on_uint8_images(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 255, (20, 20), dtype=np.uint8)
        image = Image(arr)
        out = image.similarity(Image(arr[5:8, 5:8].copy()), "zncc")
        assert out.shape == (18, 18)
