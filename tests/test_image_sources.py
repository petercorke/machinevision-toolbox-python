#!/usr/bin/env python

# test for Image input/output

import contextlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import pytest

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import labelme  # noqa: F401

    _labelme_available = True
except ImportError:
    _labelme_available = False

try:
    import py7zr

    _py7zr_available = True
except ImportError:
    _py7zr_available = False

# import machinevisiontoolbox as mvt
import machinevisiontoolbox.Sources as sources
from machinevisiontoolbox import (
    FileArchive,
    FileCollection,
    Image,
    ImageCollection,
    VideoFile,
    ZipArchive,
)
from machinevisiontoolbox.base import iread, mvtb_path_to_datafile


def _has_file(*args):
    try:
        mvtb_path_to_datafile(*args)
        return True
    except ValueError:
        return False


class TestImageSources(unittest.TestCase):

    def test_filecollection(self):
        # single str with wild card for folder of images
        # print('test_wildcardstr')
        images = FileCollection("campus/*.png")

        self.assertEqual(len(images), 20)
        self.assertIsInstance(images, Iterable)
        self.assertEqual(images[0], (426, 640, 3))
        self.assertEqual(images[0].dtype, "uint8")
        self.assertEqual(images[0].colororder_str, "R:G:B")
        self.assertEqual(images[0].nplanes, 3)

    @pytest.mark.skipif(
        not _has_file("images", "bridge-l.zip"), reason="bridge-l.zip not available"
    )
    def test_ziparchive(self):
        # zip archive with filter
        zf = FileArchive("bridge-l.zip")

        # files are README, camera-2.dat, image, image...
        self.assertEqual(len(zf), 253)
        self.assertIsInstance(zf[0], bytes)
        self.assertIsInstance(zf[2], Image)
        self.assertEqual(zf[2].shape, (488, 768))
        self.assertEqual(zf[2].dtype, "uint16")

        # test iteration
        count = 0
        for im in zf:
            count += 1
        self.assertEqual(count, 253)

        zf = FileArchive("bridge-l.zip", filter="*.pgm")
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

    def test_videofile_next_cold_start(self):
        """VideoFile supports next() without explicit iter()."""
        v = VideoFile("traffic_sequence.mp4")
        im = next(v)
        self.assertIsInstance(im, Image)

    def test_imagesequence_next(self):
        """ImageSequence supports next() directly and via iteration."""
        from machinevisiontoolbox import ImageSequence

        images = FileCollection("campus/*.png")
        seq = ImageSequence(images[:3])
        first = next(seq)
        self.assertIsInstance(first, Image)
        self.assertEqual(len(list(seq)), 3)

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_imagesource_tensor_dtype_option(self):
        """ImageSource.tensor(dtype=...) returns requested tensor dtype."""
        images = FileCollection("campus/*.png")
        t = images.tensor(normalize=None, dtype=torch.float32)
        self.assertEqual(t.dtype, torch.float32)

    def test_deprecated_imagecollection_warns(self):
        """ImageCollection alias emits FutureWarning and still works."""
        with pytest.warns(FutureWarning, match="ImageCollection is deprecated"):
            images = ImageCollection("campus/*.png")
        self.assertEqual(len(images), 20)

    @pytest.mark.skipif(
        not _has_file("images", "bridge-l.zip"), reason="bridge-l.zip not available"
    )
    def test_deprecated_ziparchive_warns(self):
        """ZipArchive alias emits FutureWarning and still works."""
        with pytest.warns(FutureWarning, match="ZipArchive is deprecated"):
            zf = ZipArchive("bridge-l.zip")
        self.assertEqual(len(zf), 253)


class TestZipArchiveFormats(unittest.TestCase):

    @staticmethod
    def _sample_members(tmpdir: Path) -> dict[str, bytes]:
        pgm = b"P5\n2 2\n255\n" + bytes([0, 255, 255, 0])
        txt = b"hello from archive\n"

        (tmpdir / "frame.pgm").write_bytes(pgm)
        (tmpdir / "README.txt").write_bytes(txt)

        return {"frame.pgm": pgm, "README.txt": txt}

    @staticmethod
    def _write_zip(path: Path, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)

    @staticmethod
    def _write_tar(path: Path, mode: str, members: dict[str, bytes]) -> None:
        with tarfile.open(path, mode) as tf:
            for name in members:
                tf.add(path.parent / name, arcname=name)

    def _assert_archive(self, archive_path: Path) -> None:
        zf = FileArchive(str(archive_path))
        self.assertIn("README.txt", zf.files)
        self.assertIn("frame.pgm", zf.files)

        txt = zf[zf.files.index("README.txt")]
        self.assertIsInstance(txt, bytes)
        self.assertEqual(txt, b"hello from archive\n")

        img = zf[zf.files.index("frame.pgm")]
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, (2, 2))
        self.assertEqual(img.dtype, "uint8")

        zf_img = FileArchive(str(archive_path), filter="*.pgm")
        self.assertEqual(len(zf_img), 1)
        self.assertEqual(zf_img.files[0], "frame.pgm")
        self.assertIsInstance(zf_img[0], Image)

    def test_ziparchive_tar_and_zip_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            members = self._sample_members(tmpdir)

            archives: list[tuple[str, callable]] = [
                ("sample.zip", lambda p: self._write_zip(p, members)),
                ("sample.tar", lambda p: self._write_tar(p, "w", members)),
                ("sample.tgz", lambda p: self._write_tar(p, "w:gz", members)),
                ("sample.tar.bz2", lambda p: self._write_tar(p, "w:bz2", members)),
                ("sample.tar.xz", lambda p: self._write_tar(p, "w:xz", members)),
            ]

            for name, writer in archives:
                path = tmpdir / name
                writer(path)
                with self.subTest(name=name):
                    self._assert_archive(path)

    @pytest.mark.skipif(not _py7zr_available, reason="py7zr not installed")
    def test_ziparchive_7z_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            members = self._sample_members(tmpdir)
            archive = tmpdir / "sample.7z"

            with py7zr.SevenZipFile(archive, "w") as a:
                for name in members:
                    a.write(tmpdir / name, arcname=name)

            self._assert_archive(archive)

    def test_ziparchive_rar_shellout_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            members = self._sample_members(tmpdir)
            rar_path = tmpdir / "sample.rar"
            rar_path.write_bytes(b"not-a-real-rar")

            def _fake_which(name: str) -> str | None:
                if name in ("unar", "lsar"):
                    return f"/usr/bin/{name}"
                return None

            def _fake_run(cmd, **kwargs):
                exe = Path(cmd[0]).name
                if exe == "lsar":
                    payload = {
                        "lsarContents": [
                            {"XADFileName": "README.txt", "XADIsDirectory": False},
                            {"XADFileName": "frame.pgm", "XADIsDirectory": False},
                        ]
                    }
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=json.dumps(payload),
                        stderr="",
                    )

                if exe == "unar":
                    outdir = Path(cmd[cmd.index("-o") + 1])
                    name = cmd[-1]
                    target = outdir / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(members[name])
                    return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

                raise AssertionError(f"Unexpected executable {exe!r}")

            with patch.object(sources, "_unar_available", True):
                with patch.object(sources.shutil, "which", side_effect=_fake_which):
                    with patch.object(sources.subprocess, "run", side_effect=_fake_run):
                        zf = FileArchive(str(rar_path))
                        self.assertEqual(zf.files, ["README.txt", "frame.pgm"])
                        self.assertIsInstance(zf[0], bytes)
                        self.assertIsInstance(zf[1], Image)


@unittest.skipUnless(_torch_available, "PyTorch not installed")
class TestTensorStack(unittest.TestCase):

    def test_tensorstack_init_4d_rgb(self):
        """TensorStack initializes with (B, C, H, W) tensor."""
        batch = torch.randn(5, 3, 64, 64)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        self.assertEqual(len(ts), 5)

    def test_tensorstack_init_3d_mono(self):
        """TensorStack initializes with (B, H, W) mono tensor."""
        batch = torch.randn(10, 128, 128)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch)
        self.assertEqual(len(ts), 10)

    def test_tensorstack_invalid_tensor_type(self):
        """TensorStack rejects non-tensor input."""
        from machinevisiontoolbox import TensorStack

        with self.assertRaises(TypeError):
            TensorStack(np.random.randn(5, 3, 64, 64))

    def test_tensorstack_invalid_shape(self):
        """TensorStack rejects wrong tensor shapes."""
        from machinevisiontoolbox import TensorStack

        with self.assertRaises(ValueError):
            TensorStack(torch.randn(5))  # 1D

        with self.assertRaises(ValueError):
            TensorStack(torch.randn(5, 3))  # 2D

        with self.assertRaises(ValueError):
            TensorStack(torch.randn(5, 3, 64, 64, 2))  # 5D

    def test_tensorstack_indexing_rgb(self):
        """Indexing returns Image with correct shape."""
        batch = torch.randn(3, 3, 64, 64)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        img = ts[0]

        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, (64, 64, 3))  # (H, W, C) after permute

    def test_tensorstack_indexing_mono(self):
        """Indexing mono tensor returns (H, W) Image."""
        batch = torch.randn(5, 128, 256)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch)
        img = ts[2]

        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, (128, 256))

    def test_tensorstack_zero_copy(self):
        """Images are views into the original tensor (zero-copy)."""
        batch = torch.ones(2, 3, 32, 32) * 5
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        img = ts[0]

        # Verify image data matches tensor slice
        nt.assert_array_equal(img.A, batch[0].permute(1, 2, 0).numpy())

    def test_tensorstack_iteration(self):
        """Iteration yields all images in batch."""
        batch = torch.randn(4, 3, 32, 32)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        images = list(ts)

        self.assertEqual(len(images), 4)
        for img in images:
            self.assertIsInstance(img, Image)
            self.assertEqual(img.shape, (32, 32, 3))

    def test_tensorstack_next(self):
        """TensorStack supports next() directly."""
        batch = torch.randn(2, 3, 16, 16)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        img = next(ts)
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, (16, 16, 3))

    def test_tensorstack_mask_2d(self):
        """logits=True with 3D tensor does argmax on channel dim."""
        # Create logits: (B, C, H, W) where C is number of classes
        logits = torch.tensor(
            [[[[1.0, 2.0], [3.0, 1.0]], [[0.0, 5.0], [1.0, 2.0]]]],
            dtype=torch.float32,
        )  # (1, 2, 2, 2) - 1 batch, 2 classes, 2x2 image
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(logits, logits=True)
        mask = ts[0]

        # Expected argmax: pixel (0,0): class 0, (0,1): class 1, (1,0): class 0, (1,1): class 1
        expected = np.array([[0, 1], [0, 1]], dtype=np.int64)
        nt.assert_array_equal(mask.A, expected)

    def test_tensorstack_dtype_option(self):
        """dtype option is passed to output Image arrays."""
        batch = torch.randn(2, 3, 16, 16)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, dtype=np.float32)
        img = ts[0]

        self.assertEqual(img.A.dtype, np.float32)

    def test_tensorstack_index_out_of_bounds(self):
        """Indexing out of bounds raises IndexError."""
        batch = torch.randn(3, 3, 32, 32)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch)

        with self.assertRaises(IndexError):
            ts[3]  # Valid indices: 0, 1, 2

        with self.assertRaises(IndexError):
            ts[-1]

    def test_tensorstack_colororder_preserved(self):
        """Colororder is passed to Image constructor."""
        batch = torch.randn(2, 3, 32, 32)
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="BGR")
        img = ts[0]

        self.assertEqual(img.colororder_str, "B:G:R")

    def test_tensorstack_repr(self):
        """__repr__ returns informative string."""
        batch = torch.randn(10, 3, 64, 64).float()
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch)
        repr_str = repr(ts)

        self.assertIn("TensorStack", repr_str)
        self.assertIn("10", repr_str)
        self.assertIn("64", repr_str)

    def test_tensorstack_cuda_tensor(self):
        """TensorStack handles CUDA tensors correctly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        batch = torch.randn(2, 3, 32, 32).cuda()
        from machinevisiontoolbox import TensorStack

        ts = TensorStack(batch, colororder="RGB")
        img = ts[0]

        # Should be on CPU as numpy array
        self.assertIsInstance(img.A, np.ndarray)
        self.assertEqual(img.shape, (32, 32, 3))


@unittest.skipUnless(_labelme_available, "labelme not installed")
class TestLabelMeReader(unittest.TestCase):

    def test_labelme_read(self):
        """LabelMe returns image, polygons with attrs, and file flags."""
        from machinevisiontoolbox import LabelMeReader

        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "im.png")
            json_path = os.path.join(tmp, "ann.json")

            image_data = np.zeros((20, 30, 3), dtype=np.uint8)
            image_data[..., 0] = 128
            nt.assert_equal(image_data.shape, (20, 30, 3))

            import cv2 as cv

            assert cv.imwrite(
                image_path, image_data
            ), f"cv.imwrite failed: {image_path}"

            import base64

            with open(image_path, "rb") as fh:
                image_data_b64 = base64.b64encode(fh.read()).decode("ascii")

            payload = {
                "version": "5.0.0",
                "flags": {"scene": "test"},
                "shapes": [
                    {
                        "label": "poly",
                        "points": [[1, 1], [5, 1], [5, 4]],
                        "group_id": 7,
                        "shape_type": "polygon",
                        "flags": {"occluded": True},
                    },
                    {
                        "label": "rect",
                        "points": [[10, 10], [14, 13]],
                        "group_id": 3,
                        "shape_type": "rectangle",
                        "flags": {"hard": False},
                    },
                ],
                "imagePath": "im.png",
                "imageData": image_data_b64,
                "imageHeight": 20,
                "imageWidth": 30,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            reader = LabelMeReader(json_path)
            repr_str = repr(reader)
            self.assertIn("ann.json", repr_str)
            self.assertIn("nshapes=2", repr_str)

            img = reader.image
            self.assertIsInstance(img, Image)
            self.assertEqual(img.shape, (20, 30, 3))
            self.assertEqual(reader.flags, {"scene": "test"})

            shapes = reader.shapes
            self.assertEqual(len(shapes), 2)
            self.assertEqual(shapes[0].group_id, 7)
            self.assertEqual(shapes[0].flags, {"occluded": True})

            self.assertEqual(shapes[1].group_id, 3)
            self.assertEqual(shapes[1].flags, {"hard": False})


if __name__ == "__main__":
    unittest.main()
