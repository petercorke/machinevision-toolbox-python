#!/usr/bin/env python

import unittest

import numpy as np

from machinevisiontoolbox import Image
from machinevisiontoolbox.ImagePointFeatures import (
    _COMBO_DETECTORS,
    _DESCRIPTORS,
    BRIEFFeature,
    FREAKFeature,
    LUCIDFeature,
    _unsupported_pair,
)


class TestComboFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = Image.Read("eiffel-1.png")

    def test_every_supported_pair_works(self):
        for det in _COMBO_DETECTORS:
            for des in _DESCRIPTORS:
                if _unsupported_pair(det, des) is not None:
                    continue
                with self.subTest(detector=det, descriptor=des):
                    f = self.img.ComboFeature(det, des, nfeat=200)
                    self.assertGreater(len(f), 0)
                    self.assertLessEqual(len(f), 200)
                    self.assertEqual(f.descriptor.shape[0], len(f))
                    self.assertIsInstance(f, _DESCRIPTORS[des][1])

    def test_result_is_of_the_class_of_the_descriptor(self):
        self.assertIsInstance(self.img.ComboFeature("FAST", "BRIEF"), BRIEFFeature)
        self.assertIsInstance(self.img.ComboFeature("GFTT", "FREAK"), FREAKFeature)
        self.assertIsInstance(self.img.ComboFeature("FAST", "LUCID"), LUCIDFeature)

    def test_options_reach_the_detector_and_descriptor(self):
        few = self.img.ComboFeature("ORB", "BRIEF", det_opts=dict(nfeatures=50))
        self.assertLessEqual(len(few), 50)
        short = self.img.ComboFeature("FAST", "BRIEF", des_opts=dict(bytes=16))
        self.assertEqual(short.descriptor.shape[1], 16)

    def test_nfeat_keeps_the_strongest(self):
        everything = self.img.ComboFeature("FAST", "BRIEF")
        some = self.img.ComboFeature("FAST", "BRIEF", nfeat=100)
        # a few can be lost, because the descriptor needs a margin around the point
        self.assertLessEqual(len(some), 100)
        self.assertGreater(len(some), 80)
        floor = min(f.strength for f in some)
        self.assertGreaterEqual(floor, np.sort(everything.strength)[-100])

    def test_scale_and_orientation_flags(self):
        # FAST finds neither, SIFT and ORB find both
        fast = self.img.ComboFeature("FAST", "BRIEF", nfeat=50)
        with self.assertRaises(ValueError):
            fast.scale
        with self.assertRaises(ValueError):
            fast.orientation
        sift = self.img.ComboFeature("SIFT", "BRIEF", nfeat=50)
        self.assertEqual(len(sift.scale), len(sift))
        self.assertEqual(len(sift.orientation), len(sift))
        # FREAK works out an orientation for each feature
        freak = self.img.ComboFeature("FAST", "FREAK", nfeat=50)
        self.assertEqual(len(freak.orientation), len(freak))
        with self.assertRaises(ValueError):
            freak.scale

    def test_matching_with_the_same_descriptor(self):
        a = self.img.ComboFeature("FAST", "BRIEF", nfeat=300)
        b = Image.Read("eiffel-2.png").ComboFeature("AGAST", "BRIEF", nfeat=300)
        m = a.match(b)
        self.assertGreater(len(m), 0)

    def test_matching_different_descriptors_says_so(self):
        a = self.img.ComboFeature("FAST", "BRIEF", nfeat=100)
        b = self.img.ComboFeature("FAST", "DAISY", nfeat=100)
        with self.assertRaisesRegex(ValueError, "different kinds of descriptor"):
            a.match(b)

    def test_id(self):
        f = self.img.ComboFeature("FAST", "BRIEF", nfeat=20, id="index")
        self.assertEqual([k.class_id for k in f._kp], list(range(len(f))))
        f = self.img.ComboFeature("FAST", "BRIEF", nfeat=20, id=7)
        self.assertTrue(all(k.class_id == 7 for k in f._kp))
        with self.assertRaises(ValueError):
            self.img.ComboFeature("FAST", "BRIEF", id="bad")

    def test_bad_names(self):
        with self.assertRaisesRegex(ValueError, "unknown detector"):
            self.img.ComboFeature("NOPE", "BRIEF")
        with self.assertRaisesRegex(ValueError, "unknown descriptor"):
            self.img.ComboFeature("FAST", "NOPE")

    def test_unsupported_pairs_say_why(self):
        with self.assertRaisesRegex(ValueError, "AKAZE detector"):
            self.img.ComboFeature("FAST", "AKAZE")

    def test_orb_describes_sift_keypoints_sensibly(self):
        # not just no error: the matches must agree with one epipolar geometry, to the
        # same standard as another pair that works
        import cv2

        def inlier_fraction(detector, descriptor):
            a = self.img.ComboFeature(detector, descriptor, nfeat=1500)
            b = Image.Read("eiffel-2.png").ComboFeature(
                detector, descriptor, nfeat=1500
            )
            m = a.match(b, ratio=0.8)
            self.assertGreater(len(m), 100)
            p1 = np.array(m.p1, dtype=np.float32).T
            p2 = np.array(m.p2, dtype=np.float32).T
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.99)
            return mask.sum() / len(m)

        self.assertGreater(inlier_fraction("SIFT", "ORB"), 0.7)
        self.assertGreater(inlier_fraction("FAST", "ORB"), 0.7)

    def test_lucid_needs_color(self):
        with self.assertRaisesRegex(ValueError, "color image"):
            self.img.mono().ComboFeature("FAST", "LUCID")

    def test_image_can_be_greyscale(self):
        f = self.img.mono().ComboFeature("FAST", "BRIEF", nfeat=50)
        self.assertGreater(len(f), 40)
        self.assertLessEqual(len(f), 50)


if __name__ == "__main__":
    unittest.main()
