#!/usr/bin/env python

import unittest
import sys
import numpy as np
from machinevisiontoolbox.base.data import *
import inspect
from pathlib import Path


class TestData(unittest.TestCase):

    def test_matfile(self):

        data = mvtb_load_matfile('data/peakfit.mat')
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data['y'], np.ndarray)
        self.assertIsInstance(data['image'], np.ndarray)

        data = mvtb_load_matfile(Path('data/peakfit.mat'))
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data['y'], np.ndarray)
        self.assertIsInstance(data['image'], np.ndarray)

    def test_json(self):

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        data = mvtb_load_jsonfile(Path(filename).parent / 'test.json')
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 3)
        self.assertEqual(data['one'], 1)
        self.assertIsInstance(data['dict'], dict)
        self.assertEqual(len(data['dict']), 3)

        self.assertIsInstance(data['array'], list)
        self.assertEqual(data['array'], [1, 2, 3])

    def test_path(self):
        path = mvtb_path_to_datafile(Path('data/peakfit.mat'))
        self.assertIsInstance(path, Path)

        path = mvtb_path_to_datafile('data/peakfit.mat')
        self.assertIsInstance(path, Path)

        path = mvtb_path_to_datafile('data', 'peakfit.mat')
        self.assertIsInstance(path, Path)
        
        path = mvtb_path_to_datafile('data', 'peakfit.mat', string=True)
        self.assertIsInstance(path, str)

        path = mvtb_path_to_datafile(Path(__file__).parent / 'test.json')
        self.assertIsInstance(path, Path)

        path = mvtb_path_to_datafile(Path(__file__).parent / 'test.json', local=True)
        self.assertIsInstance(path, Path)

# ----------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()