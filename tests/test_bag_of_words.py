#!/usr/bin/env python
"""
Smoke tests for Visual Servo classes.
"""

import unittest
import warnings

from machinevisiontoolbox import FileCollection, BagOfWords
import numpy as np
import numpy.testing as nt


class TestBagOfWords(unittest.TestCase):

    def test_retrieve(self):

        images = FileCollection("campus/*.png", mono=True)

        features = []
        for image in images:
            features += image.SIFT()
        features.sort(by="scale", inplace=True)

        self.assertAlmostEqual(len(features), 42_213, delta=10)

        bag = BagOfWords(features, 2_000, seed=0)
        # self.assertEqual(len(bag), 42_213)

        # Regression test: nstopwords>0 can leave a word with zero
        # occurrences across every remaining image (ni=0), which used to
        # raise "RuntimeWarning: divide by zero" from idf = np.log(N/ni)
        # -- harmless (the inf it produces is always multiplied by 0
        # downstream) but not suppressed, unlike the equivalent case
        # right below it in the same function.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bag = BagOfWords(features, 2_000, nstopwords=50, seed=0)
        # self.assertEqual(len(bag), 34_997)

        query = FileCollection("campus/holdout/*.png", mono=True)

        S = bag.similarity(query)
        q = np.argmax(S, axis=1)
        # nt.assert_equal(q, [0, 10, 17, 17, 17])

        q = bag.retrieve(query[0])
        # self.assertEqual(q[0], 0)
        q = bag.retrieve(query[1])
        # self.assertEqual(q[0], 10)


if __name__ == "__main__":
    unittest.main()
