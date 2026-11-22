#!/usr/bin/env python
"""More tests for ``ImagePointFeatures``.

Most tests build feature objects by hand from ``cv2.KeyPoint`` values with known
coordinates, scale, angle, strength and descriptors, so the expected answers are
exact and do not depend on which detector, or which OpenCV version, found them.
The detector tests use a blurred checkerboard whose corners are at known places.
Nothing is mocked.
"""

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

from machinevisiontoolbox import Image
from machinevisiontoolbox.ImagePointFeatures import (
    BaseFeature2D,
    FeatureMatch,
    HarrisFeature,
    SIFTFeature,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_features(n=5, descriptors=True, image=None, cls=SIFTFeature):
    """Feature i is at (10(i+1), 20+5i), size 2+i, angle 30i degrees,
    strength 0.1(i+1), id i, with descriptor [4i, 4i+1, 4i+2, 4i+3]."""
    kps = [
        cv2.KeyPoint(
            10.0 * (i + 1), 20.0 + 5 * i, 2.0 + i, 30.0 * i, 0.1 * (i + 1), 0, i
        )
        for i in range(n)
    ]
    des = np.arange(4 * n, dtype=np.float32).reshape(n, 4) if descriptors else None
    f = cls(kps, des, scale=True, orient=True)
    f._feature_type = "SIFT"
    f._image = image
    return f


@pytest.fixture
def checkerboard():
    """240 x 240, 40 pixel squares: interior corners at multiples of 40."""
    board = ((np.indices((240, 240)) // 40).sum(axis=0) % 2 * 200 + 30).astype(np.uint8)
    return Image(cv2.GaussianBlur(board, (0, 0), 1.5))


# --------------------------------------------------------------------------- #
# BaseFeature2D: indexing, properties, printing


class TestIndexing:
    def test_len(self):
        assert len(make_features(5)) == 5
        assert len(BaseFeature2D()) == 0

    def test_int(self):
        f = make_features()[2]
        assert len(f) == 1
        assert f.u == 30 and f.v == 30
        nt.assert_array_equal(f.descriptor, [8, 9, 10, 11])

    def test_negative_int(self):
        assert make_features()[-1].u == 50

    def test_slice(self):
        f = make_features()[1:4]
        assert len(f) == 3
        nt.assert_array_equal(f.u, [20, 30, 40])
        nt.assert_array_equal(f.descriptor[:, 0], [4, 8, 12])

    @pytest.mark.parametrize(
        "index",
        [[0, 2, 4], (0, 2, 4), np.array([0, 2, 4])],
        ids=["list", "tuple", "int array"],
    )
    def test_sequence(self, index):
        f = make_features()[index]
        nt.assert_array_equal(f.u, [10, 30, 50])

    def test_bool_array(self):
        f = make_features()[np.array([True, False, True, False, True])]
        nt.assert_array_equal(f.u, [10, 30, 50])
        nt.assert_array_equal(f.descriptor[:, 0], [0, 8, 16])

    def test_without_descriptors(self):
        f = make_features(descriptors=False)[1:3]
        assert len(f) == 2
        assert f._descriptor is None

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="descriptor is typed as ndarray | None but array_result calls "
        "len() on it, so it raises when there are no descriptors",
    )
    def test_descriptor_property_without_descriptors(self):
        assert make_features(descriptors=False).descriptor is None

    def test_one_dimensional_descriptor_is_shared(self):
        f = make_features(descriptors=False)
        f._descriptor = np.arange(4, dtype=np.float32)
        nt.assert_array_equal(f[1:3]._descriptor, np.arange(4))

    def test_scale_and_orientation_flags_are_kept(self):
        sub = make_features()[1:3]
        assert sub._has_scale and sub._has_orient

    def test_iteration(self):
        parts = list(make_features(3))
        assert [len(p) for p in parts] == [1, 1, 1]
        assert [p.u for p in parts] == [10, 20, 30]


class TestProperties:
    def test_arrays_for_several_features(self):
        f = make_features(4)
        nt.assert_array_equal(f.u, [10, 20, 30, 40])
        nt.assert_array_equal(f.v, [20, 25, 30, 35])
        nt.assert_array_equal(f.id, [0, 1, 2, 3])
        nt.assert_allclose(f.strength, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
        nt.assert_allclose(f.scale, [2, 3, 4, 5])
        nt.assert_array_equal(f.octave, [0, 0, 0, 0])
        nt.assert_allclose(f.orientation, np.radians([0, 30, 60, 90]), rtol=1e-6)
        nt.assert_array_equal(f.p, [[10, 20, 30, 40], [20, 25, 30, 35]])
        assert f.descriptor.shape == (4, 4)

    def test_scalars_for_one_feature(self):
        f = make_features(4)[2]
        assert f.u == 30 and f.v == 30 and f.id == 2
        assert f.scale == pytest.approx(4)
        assert f.orientation == pytest.approx(np.pi / 3, rel=1e-6)
        assert f.p.shape == (2, 1)


class TestPrinting:
    def test_several(self):
        assert str(make_features(3)) == "SIFTFeature features, 3 points"
        assert repr(make_features(3)) == str(make_features(3))

    def test_one_with_scale_and_orientation(self):
        s = str(make_features(3)[1])
        assert s.startswith("SIFTFeature: (20.0, 25.0), strength=0.20")
        assert "scale=3.0" in s and "orient=0.5°" in s and s.endswith("id=1")

    def test_one_without_scale_and_orientation(self):
        f = make_features(3)[1]
        f._has_scale = f._has_orient = False
        s = str(f)
        assert "scale" not in s and "orient" not in s

    def test_list(self, capsys):
        make_features(3).list()
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("SIFT feature 0: (10.0, 20.0), strength=0.10")

    def test_table(self, capsys):
        make_features(3).table()
        out = capsys.readouterr().out
        assert "strength" in out and "scale" in out and "orient" in out
        assert "20.0, 25.0" in out

    def test_table_without_scale_and_orientation(self, capsys):
        f = make_features(3)
        f._has_scale = f._has_orient = False
        f.table()
        out = capsys.readouterr().out
        assert "strength" in out and "scale" not in out


# --------------------------------------------------------------------------- #
# adding, gridify, subset, sort, filter


class TestAdd:
    def test_adds_keypoints_and_descriptors(self):
        a, b = make_features(3), make_features(2)
        c = a + b
        assert len(c) == 5
        assert c.descriptor.shape == (5, 4)
        assert c._feature_type == "SIFT"

    def test_none_and_empty_list_are_ignored(self):
        a = make_features(3)
        assert (a + None) is a
        assert (a + []) is a
        assert ([] + a) is a

    def test_different_types_are_an_error(self):
        a = make_features(3)
        b = make_features(3)
        b._feature_type = "ORB"
        with pytest.raises(TypeError, match="different feature types"):
            a + b

    def test_adding_other_things_is_an_error(self):
        with pytest.raises(AssertionError):
            make_features(3) + 5
        with pytest.raises(ValueError):
            5 + make_features(3)


class TestGridify:
    def test_limits_features_per_cell(self):
        # ten features in one corner cell of a 2 x 2 grid of a 100 x 100 image
        image = Image(np.zeros((100, 100), dtype=np.uint8))
        kps = [cv2.KeyPoint(5.0 + i, 5.0, 1, 0, 0.1, 0, i) for i in range(10)]
        kps.append(cv2.KeyPoint(80.0, 80.0, 1, 0, 0.1, 0, 10))
        f = SIFTFeature(kps)
        f._image = image
        g = f.gridify(2, 3)
        # three from the crowded cell and the one in the far corner
        assert len(g) == 4

    def test_needs_an_image(self):
        with pytest.raises(ValueError, match="no image"):
            make_features().gridify(2, 3)


class TestSubset:
    def test_every_nth(self):
        f = make_features(10).subset(N=5)
        assert len(f) == 5
        nt.assert_array_equal(f.id, [0, 2, 4, 6, 8])
        assert f._feature_type == "SIFT"

    def test_small_set_is_kept(self):
        assert len(make_features(3).subset(N=100)) == 3


class TestSort:
    def test_by_strength_descending(self):
        f = make_features(4).sort()
        nt.assert_array_equal(f.id, [3, 2, 1, 0])
        assert f.descriptor[0, 0] == 12  # descriptors move with the keypoints
        assert f._feature_type == "SIFT"

    def test_by_scale_ascending(self):
        f = make_features(4).sort(by="scale", descending=False)
        nt.assert_array_equal(f.id, [0, 1, 2, 3])

    def test_original_is_unchanged_unless_inplace(self):
        f = make_features(4)
        f.sort()
        nt.assert_array_equal(f.id, [0, 1, 2, 3])

    def test_inplace(self):
        f = make_features(4)
        assert f.sort(inplace=True) is None
        nt.assert_array_equal(f.id, [3, 2, 1, 0])
        assert f.descriptor[0, 0] == 12

    def test_without_descriptors(self):
        f = make_features(4, descriptors=False)
        nt.assert_array_equal(f.sort().id, [3, 2, 1, 0])

    def test_bad_key(self):
        with pytest.raises(ValueError, match="bad sort method"):
            make_features().sort(by="colour")


class TestFilter:
    def test_scale_range(self):
        nt.assert_array_equal(make_features(5).filter(scale=(3, 5)).id, [1, 2, 3])

    def test_minscale_and_maxscale(self):
        nt.assert_array_equal(make_features(5).filter(minscale=4).id, [2, 3, 4])
        nt.assert_array_equal(make_features(5).filter(maxscale=4).id, [0, 1, 2])

    def test_strength_range(self):
        f = make_features(5).filter(strength=(0.15, 0.35))
        nt.assert_array_equal(f.id, [1, 2])

    def test_percentstrength(self):
        # at least 50% of the strongest (0.5)
        nt.assert_array_equal(make_features(5).filter(percentstrength=50).id, [2, 3, 4])

    def test_filters_combine(self):
        f = make_features(5).filter(minscale=3, maxscale=5)
        nt.assert_array_equal(f.id, [1, 2, 3])

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown filter"):
            make_features().filter(colour=1)

    @pytest.mark.xfail(
        strict=True,
        reason="minstrength keeps features weaker than the limit: the test is "
        "written limits >= v, not v >= limits",
    )
    def test_minstrength_keeps_the_stronger(self):
        f = make_features(5).filter(minstrength=0.25)
        nt.assert_array_equal(f.id, [2, 3, 4])


# --------------------------------------------------------------------------- #
# distance and match


class TestDistance:
    def test_metrics(self):
        a, b = make_features(3), make_features(3)
        # descriptors are 4 apart in each of 4 elements per step of one feature
        assert a.distance(b, "L2")[0, 1] == pytest.approx(8.0)
        assert a.distance(b, "L1")[0, 1] == pytest.approx(16.0)
        assert a.distance(b)[1, 1] == 0
        assert a.distance(b, "ncc")[0, 1] == pytest.approx(
            np.dot(a.descriptor[0], b.descriptor[1])
        )

    def test_single_features(self):
        a, b = make_features(3)[0], make_features(3)[2]
        assert a.distance(b).shape == (1, 1)
        assert a.distance(b)[0, 0] == pytest.approx(16.0)

    def test_needs_descriptors(self):
        with pytest.raises(ValueError, match="no descriptors"):
            make_features(descriptors=False).distance(make_features())

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="distance() also writes D[j, i], which is out of range unless "
        "both feature sets are the same size",
    )
    def test_different_sizes(self):
        D = make_features(3).distance(make_features(4))
        assert D.shape == (3, 4)


class TestMatch:
    @pytest.fixture
    def pair(self):
        # b holds the same descriptors as a, in reverse order, spread widely
        rng = np.random.default_rng(0)
        des = rng.normal(size=(8, 16)).astype(np.float32)
        a = make_features(8)
        a._descriptor = des
        b = make_features(8)
        b._descriptor = des[::-1].copy()
        return a, b

    @pytest.mark.parametrize("metric", ["L1", "L2"])
    def test_finds_the_reversed_order(self, pair, metric):
        a, b = pair
        m = a.match(b, metric=metric)
        assert len(m) == 8
        nt.assert_array_equal(m.id1 + m.id2, 7)
        assert (m.distance == 0).all()

    def test_binary_descriptors_default_to_hamming(self):
        rng = np.random.default_rng(1)
        des = rng.integers(0, 255, size=(6, 8), dtype=np.uint8)
        a, b = make_features(6), make_features(6)
        a._descriptor, b._descriptor = des, des[::-1].copy()
        m = a.match(b)
        assert len(m) == 6
        nt.assert_array_equal(m.id1 + m.id2, 5)
        assert len(a.match(b, metric="hamming2")) > 0

    def test_crosscheck(self, pair):
        a, b = pair
        assert len(a.match(b, crosscheck=True)) == 8

    def test_top_keeps_the_best(self, pair):
        a, b = pair
        m = a.match(b, top=3)
        assert len(m) == 3

    def test_unsorted(self, pair):
        a, b = pair
        m = a.match(b, sort=False)
        assert len(m) == 8

    def test_bad_metric(self, pair):
        a, b = pair
        with pytest.raises(ValueError, match="bad metric"):
            a.match(b, metric="cosine")


# --------------------------------------------------------------------------- #
# FeatureMatch


@pytest.fixture
def images():
    rng = np.random.default_rng(2)
    return (
        Image(rng.integers(0, 255, (60, 80), dtype=np.uint8)),
        Image(rng.integers(0, 255, (60, 80), dtype=np.uint8)),
    )


@pytest.fixture
def fmatch(images):
    """Four matches of feature i to feature 3-i, distances 0.5(i+1)."""
    f1 = make_features(4, image=images[0])
    f2 = make_features(4, image=images[1])
    pairs = [(i, 3 - i, 0.5 * (i + 1)) for i in range(4)]
    return FeatureMatch(pairs, f1, f2)


class TestFeatureMatchAccess:
    def test_len_and_correspondence(self, fmatch):
        assert len(fmatch) == 4
        nt.assert_array_equal(fmatch.correspondence(), [[0, 1, 2, 3], [3, 2, 1, 0]])

    def test_properties(self, fmatch):
        nt.assert_array_equal(fmatch.id1, [0, 1, 2, 3])
        nt.assert_array_equal(fmatch.id2, [3, 2, 1, 0])
        nt.assert_allclose(fmatch.distance, [0.5, 1.0, 1.5, 2.0])
        assert fmatch.p1.shape == (2, 4)
        nt.assert_array_equal(fmatch.p1[0], [10, 20, 30, 40])
        nt.assert_array_equal(fmatch.p2[0], [40, 30, 20, 10])
        assert len(fmatch.descriptor1) == 4
        assert len(fmatch.descriptor2) == 4

    def test_indexing(self, fmatch):
        assert len(fmatch[1]) == 1
        assert fmatch[1].id1 == 1
        assert len(fmatch[1:3]) == 2
        assert len(fmatch[np.array([True, False, True, False])]) == 2
        nt.assert_array_equal(fmatch[np.array([3, 0])].id1, [3, 0])

    def test_bad_index(self, fmatch):
        with pytest.raises(ValueError, match="bad index$"):
            fmatch["a"]
        with pytest.raises(ValueError, match="bad index dtype"):
            fmatch[np.array([0.5])]

    def test_by_id(self, fmatch):
        assert fmatch.by_id1(2).id2 == 1
        assert fmatch.by_id2(2).id1 == 1
        assert fmatch.by_id1(99) is None
        assert fmatch.by_id2(99) is None

    def test_subset(self, fmatch):
        assert fmatch.subset(N=10) is fmatch
        sub = fmatch.subset(N=2)
        assert len(sub) == 2
        nt.assert_array_equal(sub.id1, [0, 3])


class TestFeatureMatchInliers:
    def test_unset_is_an_error(self, fmatch):
        with pytest.raises(ValueError, match="not been set"):
            fmatch.inliers
        with pytest.raises(ValueError, match="not been set"):
            fmatch.outliers

    def test_inliers_and_outliers(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        nt.assert_array_equal(fmatch.inliers.id1, [0, 2, 3])
        nt.assert_array_equal(fmatch.outliers.id1, [1])

    def test_inliers_follow_indexing(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        assert fmatch[1].status == "-"
        assert fmatch[2].status == "+"
        assert len(fmatch[1:3]) == 2
        assert len(fmatch[np.array([True, False, True, False])]) == 2
        assert len(fmatch[np.array([0, 1])]) == 2

    def test_status_is_blank_when_unknown(self, fmatch):
        assert fmatch[0].status == ""

    def test_estimate_sets_inliers(self, fmatch):
        def solver(p1, p2, method, **kwargs):
            assert method == "ransac"
            assert kwargs == {"confidence": 0.9}
            return "solution", np.array([True, True, False, True])

        out = fmatch.estimate(solver, confidence=0.9)
        assert out == ("solution",)
        assert fmatch.inliers.id1.tolist() == [0, 1, 3]


class TestFeatureMatchPrinting:
    def test_one_match(self, fmatch):
        s = str(fmatch[1])
        assert s == "   1.00: (20.0, 25.0) <--> (30.0, 30.0)"
        assert repr(fmatch[1]) == s

    def test_one_match_shows_whether_it_is_an_inlier(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        assert str(fmatch[0]).startswith("+")
        assert str(fmatch[1]).startswith("-")

    def test_several(self, fmatch):
        assert str(fmatch) == "4 matches"

    def test_several_with_inliers(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        assert str(fmatch) == "4 matches, with 3 (75.0%) inliers"

    def test_list(self, fmatch, capsys):
        fmatch._inliers = np.array([True, False, True, True])
        fmatch.list()
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 4
        assert "+" in lines[0] and "-" in lines[1]
        assert "(10.0, 20.0) <--> (40.0, 35.0)" in lines[0]

    def test_list_without_inliers(self, fmatch, capsys):
        fmatch.list()
        assert len(capsys.readouterr().out.splitlines()) == 4

    def test_table(self, fmatch, capsys):
        fmatch._inliers = np.array([True, False, True, True])
        fmatch.table()
        out = capsys.readouterr().out
        assert "inlier" in out and "(10.0, 20.0)" in out

    def test_table_without_inliers(self, fmatch, capsys):
        fmatch.table()
        assert "(40.0, 35.0)" in capsys.readouterr().out


class TestFeatureMatchPlotting:
    def test_plot_draws_one_line_per_match(self, fmatch):
        fmatch.plot("r")
        ax = plt.gcf().axes[0]
        assert len(ax.lines) == 4
        # the second image sits to the right of the first, which is 80 wide;
        # feature 0 of the first is at u=10, matched to u=40 in the second
        first = ax.lines[0]
        assert first.get_xdata()[0] == 10
        assert 40 + 80 <= first.get_xdata()[1] <= 40 + 85

    def test_plot_into_given_axes(self, fmatch):
        fig, ax = plt.subplots()
        fmatch.plot(ax=ax)
        assert len(ax.lines) == 4

    def test_plot_correspondence(self, fmatch):
        plt.figure()
        fmatch.plot_correspondence("g", offset=(100, 0))
        ax = plt.gca()
        assert len(ax.lines) == 4
        assert ax.lines[0].get_xdata()[1] == pytest.approx(40 + 100)


# --------------------------------------------------------------------------- #
# drawing


class TestDrawing:
    @pytest.fixture
    def canvas(self):
        return Image(np.zeros((80, 80, 3), dtype=np.uint8))

    @pytest.fixture
    def one(self):
        # at (40, 40), size 10, angle 0, so a "hand" along +u of 10 pixels
        f = SIFTFeature(
            [cv2.KeyPoint(40.0, 40.0, 10.0, 0.0, 0.5, 0, 0)],
            np.zeros((1, 4), dtype=np.float32),
            scale=True,
            orient=True,
        )
        f._feature_type = "SIFT"
        return f

    def test_plot_markers(self):
        plt.figure()
        make_features(4).plot()
        ax = plt.gca()
        assert len(ax.lines) == 1
        nt.assert_array_equal(ax.lines[0].get_xdata(), [10, 20, 30, 40])

    def test_plot_filled_circles_and_hands(self):
        plt.figure()
        make_features(4).plot(filled=True, hand=True, color="r")
        ax = plt.gca()
        assert len(ax.patches) == 4
        assert len(ax.lines) == 4

    def test_plot_filled_without_hands(self):
        plt.figure()
        make_features(4).plot(filled=True)
        ax = plt.gca()
        assert len(ax.patches) == 4 and len(ax.lines) == 0

    def test_draw_marks_the_feature(self, canvas, one):
        one.draw(canvas)
        painted = np.argwhere(canvas.array.sum(axis=2) > 0)
        assert len(painted) > 0
        # marker is centred on the feature at (u, v) = (40, 40)
        assert abs(painted[:, 0].mean() - 40) < 8
        assert abs(painted[:, 1].mean() - 40) < 8

    def test_draw_filled_circle(self, canvas, one):
        one.draw(canvas, filled=True, color="r")
        red = canvas.array[:, :, 0] > 0
        ys, xs = np.nonzero(red)
        # circle of radius 10 around (40, 40)
        assert xs.min() >= 28 and xs.max() <= 52
        assert ys.min() >= 28 and ys.max() <= 52

    @pytest.mark.xfail(
        strict=True,
        reason="draw() passes the x pair and the y pair to draw_line as the "
        "start and end points, so the hand is vertical, not along the orientation",
    )
    def test_draw_hand_points_along_the_orientation(self, canvas, one):
        one.draw(canvas, filled=True, hand=True, color="r", handcolor="g")
        ys, xs = np.nonzero(canvas.array[:, :, 1] > 0)
        # orientation 0 is along +u, so a horizontal line on row 40
        assert (ys == 40).all()
        assert xs.min() >= 40 and xs.max() <= 50

    def test_draw_hand_is_drawn(self, canvas, one):
        one.draw(canvas, filled=True, hand=True, color="r", handcolor="g")
        assert (canvas.array[:, :, 1] > 0).any()

    def test_draw_needs_a_contiguous_image(self, one):
        view = np.zeros((80, 160, 3), dtype=np.uint8)[:, ::2, :]
        with pytest.raises(ValueError):
            one.draw(Image(view))

    @pytest.mark.parametrize("type", ["point", "rich", "not"])
    def test_draw2_types(self, canvas, one, type):
        out = one.draw2(canvas, color="r", type=type)
        assert out.shape == canvas.shape
        assert out.array.sum() > 0

    @pytest.mark.parametrize(
        "color",
        ["y", np.array([255, 0, 0]), (255, 0, 0), [0, 255, 0, 0], 0],
        ids=["name", "array", "tuple", "list", "other"],
    )
    def test_draw2_color_forms(self, canvas, one, color):
        out = one.draw2(canvas, color=color)
        assert out.array.sum() > 0

    def test_drawKeypoints(self, one):
        image = Image(np.full((80, 80), 50, dtype=np.uint8))
        out = one.drawKeypoints(image)
        assert out.shape == (80, 80, 3)
        assert (out.array != 50).any()

    def test_drawKeypoints_into_given_drawing(self, one):
        image = Image(np.full((80, 80), 50, dtype=np.uint8))
        drawing = np.zeros((80, 80, 3), dtype=np.uint8)
        out = one.drawKeypoints(image, drawing=drawing, isift=[0])
        assert out.shape == (80, 80, 3)

    def test_drawMatches(self, images):
        f1 = make_features(3, image=images[0])
        f2 = make_features(3, image=images[1])
        matches = [[cv2.DMatch(i, i, 1.0)] for i in range(3)]
        out = f1.drawMatches(images[0], f1, images[1], f2, matches)
        assert out.shape[:2] == (60, 160)


# --------------------------------------------------------------------------- #
# support region


class TestSupport:
    def test_extracts_a_normalised_patch(self):
        # a vertical edge at u = 40; the feature has angle 0 and size 10, so
        # the patch is rotated by 0 and scaled by N/2/size
        arr = np.zeros((80, 80), dtype=np.uint8)
        arr[:, 40:] = 200
        f = SIFTFeature(
            [cv2.KeyPoint(40.0, 40.0, 10.0, 0.0, 0.5, 0, 0)], scale=True, orient=True
        )
        patch = f.support(Image(arr), N=20)
        assert patch.shape == (20, 20)
        # left half dark, right half bright
        assert patch.array[:, :8].max() < 50
        assert patch.array[:, 12:].min() > 150

    def test_image_chosen_by_feature_id(self):
        dark = Image(np.zeros((80, 80), dtype=np.uint8))
        bright = Image(np.full((80, 80), 200, dtype=np.uint8))
        f = SIFTFeature(
            [cv2.KeyPoint(40.0, 40.0, 10.0, 0.0, 0.5, 0, 1)], scale=True, orient=True
        )
        assert f.support([dark, bright], N=10).array.min() == 200

    def test_one_feature_only(self):
        with pytest.raises(ValueError, match="single feature"):
            make_features(3).support(Image(np.zeros((80, 80), dtype=np.uint8)))


# --------------------------------------------------------------------------- #
# detectors


class TestDetectors:
    def test_harris_finds_the_checkerboard_corners(self, checkerboard):
        h = checkerboard.Harris()
        assert isinstance(h, HarrisFeature)
        assert h._feature_type == "Harris"
        # interior corners are at multiples of 40, five along each direction
        assert len(h) == 25
        offset = np.abs(h.p - 40 * np.round(h.p / 40))
        assert offset.max() <= 2

    def test_harris_nfeat_limits_the_count(self, checkerboard):
        assert len(checkerboard.Harris(nfeat=3)) == 3

    @pytest.mark.parametrize("name", ["SIFT", "ORB", "BRISK", "AKAZE"])
    def test_detectors_find_features_with_descriptors(self, checkerboard, name):
        f = getattr(checkerboard, name)()
        assert len(f) > 0
        assert f._feature_type == name
        assert f.descriptor.shape[0] == len(f)
        # keypoints lie within the image
        assert f.u.min() >= 0 and f.u.max() <= 240
        assert f.v.min() >= 0 and f.v.max() <= 240

    def test_orb_score_type(self, checkerboard):
        assert len(checkerboard.ORB(scoreType="fast")) > 0
        with pytest.raises(KeyError):
            checkerboard.ORB(scoreType="slow")

    def test_features_know_their_image(self, checkerboard):
        assert checkerboard.SIFT()._image is checkerboard

    def test_nfeat(self, checkerboard):
        f = checkerboard.SIFT(nfeat=5)
        assert len(f) == 5
        assert f.descriptor.shape[0] == 5

    def test_id_index(self, checkerboard):
        nt.assert_array_equal(checkerboard.SIFT(id="index", nfeat=4).id, [0, 1, 2, 3])

    def test_id_integer(self, checkerboard):
        nt.assert_array_equal(checkerboard.SIFT(id=7, nfeat=3).id, [7, 7, 7])

    def test_id_from_the_image(self):
        board = ((np.indices((240, 240)) // 40).sum(axis=0) % 2 * 200 + 30).astype(
            np.uint8
        )
        im = Image(cv2.GaussianBlur(board, (0, 0), 1.5), id=3)
        nt.assert_array_equal(im.SIFT(nfeat=3).id, [3, 3, 3])

    def test_bad_id(self, checkerboard):
        with pytest.raises(ValueError, match="bad id"):
            checkerboard.SIFT(id="name")

    def test_bad_algorithm(self, checkerboard):
        class BogusFeature(BaseFeature2D):
            pass

        with pytest.raises(ValueError, match="bad algorithm"):
            checkerboard._image2feature(BogusFeature)
