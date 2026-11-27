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
        assert str(make_features(3)) == "SIFT features, 3 points"
        assert repr(make_features(3)) == str(make_features(3))

    def test_one_with_scale_and_orientation(self):
        s = str(make_features(3)[1])
        assert s.startswith("SIFT feature: (20.0, 25.0), strength=0.20")
        # the orientation is 30 degrees; it was printed as radians (0.5) with a degree sign
        assert "scale=3.0" in s and "orient=30.0°" in s and s.endswith("id=1")

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

    def test_table_and_list_show_the_orientation_in_degrees(self, capsys):
        make_features(3).table()
        assert "30°" in capsys.readouterr().out  # feature 1 is at 30 degrees
        make_features(3).list()
        assert "orient=30.0°" in capsys.readouterr().out

    def test_without_strength(self, capsys):
        f = make_features(3)
        f._has_strength = False
        assert "strength" not in str(f[1])
        f.table()
        assert "strength" not in capsys.readouterr().out
        f.list()
        assert "strength" not in capsys.readouterr().out

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
        with pytest.raises(TypeError, match="cannot add int to a feature set"):
            make_features(3) + 5
        with pytest.raises(TypeError, match="cannot add a feature set to int"):
            5 + make_features(3)
        with pytest.raises(TypeError, match="cannot add a feature set to str"):
            "a" + make_features(3)
        with pytest.raises(TypeError):
            [1] + make_features(3)
        with pytest.raises(TypeError):
            True + make_features(3)

    def test_none_and_zero_on_the_left(self):
        a = make_features(3)
        # None cannot be on the left of + in Python, so call the method
        assert a.__radd__(None) is a
        assert (0 + a) is a

    def test_sum_of_feature_sets(self):
        a, b, c = make_features(3), make_features(2), make_features(4)
        total = sum([a, b, c])
        assert len(total) == 9
        assert total.descriptor.shape == (9, 4)
        # the inputs are not changed
        assert len(a) == 3 and len(b) == 2 and len(c) == 4
        nt.assert_array_equal(total.u[:3], a.u)
        nt.assert_array_equal(total.u[3:5], b.u)

    def test_sum_of_one_and_of_a_generator(self):
        a = make_features(3)
        assert sum([a]) is a
        assert len(sum(make_features(n) for n in (1, 2, 3))) == 6

    def test_an_empty_set_does_not_remove_the_descriptors(self):
        # an algorithm that finds no features gives a set with no descriptors
        a = make_features(3)
        empty = a._like()
        assert len(empty) == 0 and empty.descriptor is None
        for total in (a + empty, empty + a, sum([empty, a, empty])):
            assert len(total) == 3
            assert total.descriptor.shape == (3, 4)
            nt.assert_array_equal(total.descriptor, a.descriptor)

    def test_two_empty_sets_make_an_empty_set(self):
        empty = make_features(3)._like()
        total = empty + empty
        assert len(total) == 0 and total.descriptor is None

    def test_a_set_with_no_descriptors_is_still_not_given_misaligned_ones(self):
        # keypoints and descriptors must stay in step, so with features on both
        # sides and descriptors on only one, there are no descriptors
        a, b = make_features(3), make_features(2)
        b._descriptor = None
        total = a + b
        assert len(total) == 5 and total.descriptor is None

    def test_sum_of_different_types_is_an_error(self):
        a, b = make_features(3), make_features(3)
        b._feature_type = "ORB"
        with pytest.raises(TypeError, match="different feature types"):
            sum([a, b])


class TestGridifyPerCell:
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

    @staticmethod
    def by_hand(a, b, ord):
        """Distance from every descriptor of a to every descriptor of b."""
        return np.array([[np.linalg.norm(x - y, ord=ord) for y in b] for x in a])

    @pytest.mark.parametrize("sizes", [(3, 3), (3, 4), (4, 2), (1, 3), (3, 1)])
    @pytest.mark.parametrize("metric, ord", [("L1", 1), ("L2", 2)])
    def test_every_pair_of_different_sets(self, sizes, metric, ord):
        # different descriptors in each set, so the matrix is not symmetric and
        # D[i, j] must be the distance from a[i] to b[j]
        rng = np.random.default_rng(0)
        a = make_features(sizes[0])
        b = make_features(sizes[1])
        a._descriptor = rng.normal(size=(sizes[0], 5)).astype(np.float32)
        b._descriptor = rng.normal(size=(sizes[1], 5)).astype(np.float32)
        D = a.distance(b, metric)
        assert D.shape == sizes
        nt.assert_allclose(
            D, self.by_hand(a._descriptor, b._descriptor, ord), rtol=1e-5
        )

    def test_ncc_of_different_sets(self):
        rng = np.random.default_rng(1)
        a, b = make_features(3), make_features(4)
        a._descriptor = rng.normal(size=(3, 5)).astype(np.float32)
        b._descriptor = rng.normal(size=(4, 5)).astype(np.float32)
        nt.assert_allclose(
            a.distance(b, "ncc"), a._descriptor @ b._descriptor.T, rtol=1e-5
        )


class TestMatchNeedsDescriptors:
    def test_no_descriptors_is_a_clear_error(self):
        with pytest.raises(ValueError, match="no descriptors"):
            make_features(descriptors=False).match(make_features())
        with pytest.raises(ValueError, match="no descriptors"):
            make_features().match(make_features(descriptors=False))


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
        with pytest.raises(ValueError, match="index must be an int, slice"):
            fmatch["a"]
        with pytest.raises(ValueError, match="index array must be integer or bool"):
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

    @pytest.mark.parametrize(
        "index",
        [
            1,
            slice(1, 3),
            np.array([True, False, True, False]),
            np.array([0, 3]),
            [0, 3],
            (1, 2),
            [True, False, True, False],
        ],
    )
    def test_inlier_status_stays_an_array_whatever_the_index(self, fmatch, index):
        # indexing by an int used to leave a bare bool, and by an index array a
        # list, so inliers and outliers then failed
        fmatch._inliers = np.array([True, False, True, True])
        sub = fmatch[index]
        assert isinstance(sub._inliers, np.ndarray)
        assert sub._inliers.dtype == bool
        assert len(sub._inliers) == len(sub)
        assert len(sub.inliers) + len(sub.outliers) == len(sub)

    def test_inliers_and_outliers_of_a_selection(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        sub = fmatch[np.array([1, 2, 3])]
        nt.assert_array_equal(sub.inliers.id1, [2, 3])
        nt.assert_array_equal(sub.outliers.id1, [1])

    def test_outliers_of_one_match(self, fmatch):
        fmatch._inliers = np.array([True, False, True, True])
        assert len(fmatch[1].outliers) == 1
        assert len(fmatch[1].inliers) == 0
        assert len(fmatch[0].outliers) == 0

    def test_status_of_many_matches(self, fmatch):
        # status used to raise ValueError for more than one match
        fmatch._inliers = np.array([True, False, True, True])
        nt.assert_array_equal(fmatch.status, ["+", "-", "+", "+"])

    def test_status_of_many_matches_when_unknown(self, fmatch):
        nt.assert_array_equal(fmatch.status, ["", "", "", ""])

    def test_inliers_given_as_a_list(self, fmatch):
        match = FeatureMatch(
            [(0, 3, 0.5), (1, 2, 1.0)], fmatch._kp1, fmatch._kp2, [True, False]
        )
        assert isinstance(match._inliers, np.ndarray)
        assert len(match.inliers) == 1

    def test_index_by_a_list_or_tuple(self, fmatch):
        nt.assert_array_equal(fmatch[[3, 0]].id1, [3, 0])
        nt.assert_array_equal(fmatch[(1, 2)].id1, [1, 2])
        assert len(fmatch[[]]) == 0

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

    def test_draw_hand_points_along_the_orientation(self, canvas, one):
        one.draw(canvas, filled=True, hand=True, color="r", handcolor="g")
        ys, xs = np.nonzero(canvas.array[:, :, 1] > 0)
        # orientation 0 is along +u, so a horizontal line on row 40, from the
        # centre (40, 40) to the circumference at (50, 40)
        assert (ys == 40).all()
        assert xs.min() == 40 and xs.max() == 50

    def test_draw_hand_follows_a_quarter_turn(self, canvas):
        # orientation 90 degrees is along +v: a vertical line down column 40
        f = SIFTFeature(
            [cv2.KeyPoint(40.0, 40.0, 10.0, 90.0, 0.5, 0, 0)],
            np.zeros((1, 4), dtype=np.float32),
            scale=True,
            orient=True,
        )
        f.draw(canvas, filled=True, hand=True, color="r", handcolor="g")
        ys, xs = np.nonzero(canvas.array[:, :, 1] > 0)
        assert (xs == 40).all()
        assert ys.min() == 40 and ys.max() == 50

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
        with pytest.raises(ValueError, match="id must be .image., .index. or an int"):
            checkerboard.SIFT(id="name")

    def test_bad_algorithm(self, checkerboard):
        class BogusFeature(BaseFeature2D):
            pass

        with pytest.raises(ValueError, match="bad algorithm"):
            checkerboard._image2feature(BogusFeature)


class TestFeatureMatchEstimate:
    """estimate() keeps the last value of func as the inliers only if it is a mask."""

    MASK = np.array([True, False, True, True])

    def test_robust_method_keeps_the_mask_and_does_not_return_it(self, fmatch):
        result = fmatch.estimate(lambda p1, p2, method: ("F", 0.5, self.MASK))
        assert result == ("F", 0.5)
        nt.assert_array_equal(fmatch.inliers.id1, [0, 2, 3])

    def test_mask_given_as_a_list_is_not_a_mask(self, fmatch):
        # only a bool ndarray of one element per match is taken as the mask
        result = fmatch.estimate(lambda p1, p2, method: ("F", [True] * 4))
        assert result == ("F", [True] * 4)

    def test_method_without_a_mask_returns_everything(self, fmatch):
        # was: the residual was taken as the inliers and dropped from the result
        result = fmatch.estimate(lambda p1, p2, method: ("F", 0.5))
        assert result == ("F", 0.5)
        with pytest.raises(ValueError, match="not been set"):
            fmatch.inliers

    def test_single_array_result_is_returned(self, fmatch):
        # was: the estimate itself was taken as the inliers and dropped
        F = np.eye(3)
        (result,) = fmatch.estimate(lambda p1, p2, method: F)
        assert result is F
        with pytest.raises(ValueError, match="not been set"):
            fmatch.inliers

    def test_single_value_in_a_list_is_returned(self, fmatch):
        (result,) = fmatch.estimate(lambda p1, p2, method: ["F"])
        assert result == "F"

    def test_mask_of_the_wrong_length_is_not_a_mask(self, fmatch):
        result = fmatch.estimate(lambda p1, p2, method: ("F", np.array([True, False])))
        assert len(result) == 2

    def test_integer_array_is_not_a_mask(self, fmatch):
        result = fmatch.estimate(lambda p1, p2, method: ("F", np.array([1, 0, 1, 1])))
        assert len(result) == 2
        with pytest.raises(ValueError, match="not been set"):
            fmatch.inliers

    def test_a_later_estimate_without_a_mask_clears_the_old_one(self, fmatch):
        fmatch.estimate(lambda p1, p2, method: ("F", 0.5, self.MASK))
        assert len(fmatch.inliers) == 3
        fmatch.estimate(lambda p1, p2, method: ("F", 0.5))
        with pytest.raises(ValueError, match="not been set"):
            fmatch.inliers

    def test_method_and_options_are_passed_to_func(self, fmatch):
        seen = {}

        def func(p1, p2, method, **kwargs):
            seen.update(method=method, kwargs=kwargs, shapes=(p1.shape, p2.shape))
            return ("F",)

        fmatch.estimate(func, method="lmeds", confidence=0.9)
        assert seen["method"] == "lmeds"
        assert seen["kwargs"] == {"confidence": 0.9}
        assert seen["shapes"] == ((2, 4), (2, 4))


class TestFeatureMatchEstimateWithTheRealEstimators:
    """The real estimators return a mask only for ransac and lmeds."""

    @pytest.fixture
    def matches(self):
        v1 = Image.Read("eiffel-1.png", mono=True)
        v2 = Image.Read("eiffel-2.png", mono=True)
        return v1.ORB().match(v2.ORB())

    @pytest.mark.parametrize(
        "name, kwargs, n_returned, has_mask",
        [
            ("points2F", dict(method="ransac", seed=0), 2, True),
            ("points2F", dict(method="lmeds", seed=0), 2, True),
            ("points2F", dict(method="8p"), 2, False),
            ("points2F", dict(method="8p", residual=False), 1, False),
            ("points2H", dict(method="ransac", seed=0), 2, True),
            ("points2H", dict(method="leastsquares"), 2, False),
        ],
    )
    def test_estimator(self, matches, name, kwargs, n_returned, has_mask):
        from machinevisiontoolbox import CentralCamera

        result = matches.estimate(getattr(CentralCamera, name), **kwargs)
        assert len(result) == n_returned
        assert result[0].shape == (3, 3)  # the matrix is always returned
        if has_mask:
            assert 0 < len(matches.inliers) <= len(matches)
            assert len(matches.inliers) + len(matches.outliers) == len(matches)
        else:
            with pytest.raises(ValueError, match="not been set"):
                matches.inliers


# --------------------------------------------------------------------------- #
# scale, orientation and strength exist only for the features that have them


class TestPropertiesThatMayBeAbsent:
    @staticmethod
    def features(scale, orient, strength=True):
        f = make_features(3)
        f._has_scale, f._has_orient, f._has_strength = scale, orient, strength
        f._feature_type = "FAST"
        return f

    @pytest.mark.parametrize(
        "name, message",
        [
            ("scale", "FAST features have no scale"),
            ("orientation", "FAST features have no orientation"),
            ("strength", "FAST features have no strength"),
        ],
    )
    def test_raise_if_the_features_do_not_have_it(self, name, message):
        f = self.features(False, False, False)
        with pytest.raises(ValueError, match=message):
            getattr(f, name)
        with pytest.raises(ValueError, match=message):
            getattr(f[1], name)  # a single feature too

    def test_return_values_if_the_features_have_them(self):
        f = self.features(True, True, True)
        nt.assert_allclose(f.scale, [2, 3, 4])
        nt.assert_allclose(f.orientation, np.radians([0, 30, 60]))
        nt.assert_allclose(f.strength, [0.1, 0.2, 0.3])

    def test_each_is_independent_of_the_others(self):
        f = self.features(True, False)
        assert len(f.scale) == 3 and len(f.strength) == 3
        with pytest.raises(ValueError, match="orientation"):
            f.orientation

    def test_sort_and_filter_by_a_missing_quantity_raise(self):
        f = self.features(False, False)
        with pytest.raises(ValueError, match="no scale"):
            f.sort("scale")
        with pytest.raises(ValueError, match="no scale"):
            f.filter(minscale=1)
        f.sort("strength")  # the one it has still works
        f.filter(minstrength=0.15)

    def test_drawing_circles_needs_a_scale_and_a_hand_needs_an_orientation(self):
        f = self.features(False, False)
        f.plot()  # markers only: nothing needed
        with pytest.raises(ValueError, match="no scale"):
            f.plot(filled=True)
        g = self.features(True, False)
        g.plot(filled=True)
        with pytest.raises(ValueError, match="no orientation"):
            g.plot(filled=True, hand=True)
        self.features(True, True).plot(filled=True, hand=True)

    def test_flags_are_kept_by_slicing_adding_and_sorting(self):
        img = Image(np.zeros((50, 50), dtype=np.uint8))
        f = self.features(True, False, True)
        f._image = img
        for g in (
            f[1:],
            f[[0, 2]],
            f[np.array([True, False, True])],
            f + f,
            f.sort("scale"),
            f.subset(2),
        ):
            assert (g._has_scale, g._has_orient, g._has_strength) == (True, False, True)
            assert g._feature_type == "FAST"
            assert g._image is img
        assert len(f + f) == 6


# the algorithms the toolbox has, and what each really provides: the flags must agree
# with the keypoints, or the properties would raise for good data, or return junk
ALGORITHMS = [
    # name, has scale, has orientation
    ("SIFT", True, True),
    ("ORB", True, True),
    ("BRISK", True, True),
    ("AKAZE", True, True),
    ("Harris", False, False),
]


class TestFlagsAgreeWithTheData:
    @pytest.fixture
    def image(self):
        return Image.Read("eiffel-1.png", mono=True)

    @pytest.mark.parametrize("name, has_scale, has_orient", ALGORITHMS)
    def test_algorithm(self, image, name, has_scale, has_orient):
        f = getattr(image, name)()
        assert len(f) > 50
        assert (f._has_scale, f._has_orient, f._has_strength) == (
            has_scale,
            has_orient,
            True,
        )
        sizes = np.array([kp.size for kp in f._kp])
        angles = np.array([kp.angle for kp in f._kp])

        # strength is always there, and not all the same
        assert np.ptp(f.strength) > 0

        if has_scale:
            assert sizes.min() > 0 and np.ptp(sizes) > 0  # varies, so it is real
            assert len(f.scale) == len(f)
        else:
            assert np.ptp(sizes) == 0  # a constant, so it is not a scale
            with pytest.raises(ValueError, match="no scale"):
                f.scale

        if has_orient:
            assert np.ptp(angles) > 90  # spread round the circle, so it is real
            assert np.all(np.abs(f.orientation) <= 2 * np.pi)
        else:
            assert np.ptp(angles) == 0
            with pytest.raises(ValueError, match="no orientation"):
                f.orientation

    @pytest.mark.parametrize("name, has_scale, has_orient", ALGORITHMS)
    def test_a_sum_and_a_slice_keep_what_the_algorithm_has(
        self, image, name, has_scale, has_orient
    ):
        f = getattr(image, name)()
        for g in (f[:20], f[:20] + f[20:40], f.sort(inplace=False)):
            assert (g._has_scale, g._has_orient) == (has_scale, has_orient)
            assert g._image is not None or g is f

    def test_printing_an_orb_feature_shows_scale_and_orientation_in_degrees(
        self, image
    ):
        s = str(image.ORB()[0])
        assert "scale=" in s and "orient=" in s
        degrees = float(s.split("orient=")[1].split("°")[0])
        assert 0 <= degrees < 360


class TestHarrisOptions:
    """Harris has an option called ``scale``, which must reach the detector."""

    @pytest.fixture
    def image(self):
        return Image.Read("eiffel-1.png", mono=True)

    def test_scale_is_the_suppression_distance(self, image):
        near = image.Harris(scale=7)
        far = image.Harris(scale=40)
        # corners must be further apart, so there are fewer of them
        assert len(far) < len(near)

    def test_scale_does_not_claim_a_scale_per_feature(self, image):
        f = image.Harris(scale=15)
        assert (f._has_scale, f._has_orient) == (False, False)
        with pytest.raises(ValueError, match="no scale"):
            f.scale


class TestGridify:
    @pytest.fixture
    def orb(self):
        return Image.Read("eiffel-1.png", mono=True).ORB(nfeat=500)

    def test_result_has_keypoints_descriptors_and_flags(self, orb):
        g = orb.gridify(4, 5)
        assert 0 < len(g) < len(orb)
        assert all(isinstance(k, cv2.KeyPoint) for k in g._kp)
        assert g._descriptor.shape == (len(g), orb._descriptor.shape[1])
        assert (g._has_scale, g._has_orient, g._has_strength) == (True, True, True)
        assert g._feature_type == orb._feature_type
        assert g._image is orb._image
        assert len(g.scale) == len(g)

    def test_features_keep_their_own_descriptors(self, orb):
        g = orb.gridify(4, 5)
        # each kept feature is one of the originals, with its own descriptor row
        index = {tuple(k.pt): i for i, k in enumerate(orb._kp)}
        for k, d in zip(g._kp, g._descriptor):
            assert np.array_equal(d, orb._descriptor[index[tuple(k.pt)]])

    def test_result_can_be_matched(self, orb):
        g = orb.gridify(4, 5)
        assert len(g.match(g)) == len(g)

    def test_at_most_nfeat_per_cell(self, orb):
        w, h = orb._image.width, orb._image.height
        g = orb.gridify((4, 3), 2)
        cells = {}
        for k in g._kp:
            c = (int(k.pt[0] * 4 // w), int(k.pt[1] * 3 // h))
            cells[c] = cells.get(c, 0) + 1
        assert max(cells.values()) <= 2

    def test_features_at_the_far_edges(self):
        # 1280 is not a multiple of 13, so the right edge is past the last cell
        im = Image.Read("eiffel-1.png", mono=True)
        kps = [cv2.KeyPoint(x, y, 7) for x, y in [(0, 0), (1279, 850), (1279, 0)]]
        f = BaseFeature2D()
        f._kp = kps
        f._descriptor = np.zeros((3, 32), dtype=np.uint8)
        f._image = im
        assert len(f.gridify(13, 5)) == 3

    def test_no_image(self):
        with pytest.raises(ValueError, match="no image"):
            BaseFeature2D().gridify(4, 5)
