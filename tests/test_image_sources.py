#!/usr/bin/env python

# test for Image input/output

import contextlib
import io
import os
import unittest
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.testing as nt

# import machinevisiontoolbox as mvt
from machinevisiontoolbox import Image, ImageCollection, ZipArchive, VideoFile
from machinevisiontoolbox.base import iread


class TestImageSources(unittest.TestCase):

    def test_filecollection(self):
        # single str with wild card for folder of images
        # print('test_wildcardstr')
        images = ImageCollection("campus/*.png")

        self.assertEqual(len(images), 20)
        self.assertIsInstance(images, Iterable)
        self.assertEqual(images[0], (426, 640, 3))
        self.assertEqual(images[0].dtype, "uint8")
        self.assertEqual(images[0].colororder_str, "R:G:B")
        self.assertEqual(images[0].nplanes, 3)

    def test_ziparchive(self):
        # zip archive with filter
        zf = ZipArchive("bridge-l.zip")

        self.assertEqual(len(zf), 253)
        self.assertIsInstance(zf[0], bytes)
        self.assertIsInstance(zf[1], Image)
        self.assertEqual(zf[1].shape, (488, 768))
        self.assertEqual(zf[1].dtype, "uint16")

        # test iteration
        count = 0
        for im in zf:
            count += 1
        self.assertEqual(count, 253)

        zf = ZipArchive("bridge-l.zip", filter="*.pgm")
        self.assertEqual(len(zf), 251)
        im = zf[0]
        self.assertIsInstance(im, Image)
        self.assertEqual(im.shape, (488, 768))
        self.assertEqual(im.dtype, "uint16")

        # test iteration
        count = 0
        for im in zf:
            self.assertIsInstance(im, Image)
            self.assertEqual(im.shape, (488, 768))
            self.assertEqual(im.dtype, "uint16")
            count += 1
        self.assertEqual(count, 251)

    def test_videofile(self):
        # video file

        # traffic_sequence.mpg
        v = VideoFile("traffic_sequence.mp4")
        self.assertEqual(v.nframes, 350)

        count = 0
        for im in v:
            self.assertIsInstance(im, Image)
            self.assertEqual(im.shape, (576, 704, 3))
            self.assertEqual(im.dtype, "uint8")
            self.assertEqual(im.colororder_str, "R:G:B")
            self.assertEqual(im.nplanes, 3)
            count += 1
        self.assertEqual(count, 350)


if __name__ == "__main__":
    unittest.main()
