import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import iwrite, mvtb_path_to_datafile
from machinevisiontoolbox.base.data import mvtb_load_jsonfile, mvtb_load_matfile

# test mvtb_path_to_datafile() which is used by a number of other functions
# to locate files in the MVTB data package


class TestDataFiles(unittest.TestCase):
    def test_datafile_local(self):
        # test for exception if file not found
        # test that exception raised if not present
        with self.assertRaises(FileNotFoundError):
            mvtb_path_to_datafile("notthere.png")

        # create a local test image file
        iwrite(np.zeros((5, 5)), "./test-image.png")
        path = mvtb_path_to_datafile("./test-image.png")
        self.assertTrue(os.path.isfile(path))
        # remove the test image file
        os.remove("./test-image.png")

    def test_datafile_mvtb(self):
        path = mvtb_path_to_datafile("images", "shark1.png")
        self.assertTrue(os.path.isfile(path))

        with self.assertRaises(FileNotFoundError):
            mvtb_path_to_datafile("images", "__shark100.png")

        path = mvtb_path_to_datafile("data", "bunny.ply")
        self.assertTrue(os.path.isfile(path))

        with self.assertRaises(FileNotFoundError):
            mvtb_path_to_datafile("data", "__bunny100.png")

    def test_missing_file_message(self):
        with self.assertRaises(FileNotFoundError) as cm:
            mvtb_path_to_datafile("images", "flowers.png")
        msg = str(cm.exception)
        # names the file as written, not as a tuple of path components
        self.assertTrue(msg.startswith("file 'flowers.png' not found"), msg)
        self.assertNotIn("PosixPath", msg)
        self.assertIn("mvtbdata/images", msg)
        # suggests the similar files that do exist
        self.assertIn("did you mean flowers1.png, flowers2.png, flowers3.png?", msg)

    def test_missing_file_no_suggestion(self):
        with self.assertRaises(FileNotFoundError) as cm:
            mvtb_path_to_datafile("images", "qqqqqqqqqq.png")
        self.assertNotIn("did you mean", str(cm.exception))

    def test_read_missing_file_traceback(self):
        import traceback

        from machinevisiontoolbox import Image

        # assertRaises discards the traceback, so catch the error by hand
        try:
            Image.Read("flowers.png")
        except FileNotFoundError as exc:
            error = exc
        else:
            self.fail("FileNotFoundError not raised")
        # one frame in the toolbox, not a chain of helper calls
        frames = traceback.extract_tb(error.__traceback__)
        self.assertEqual([f.name for f in frames][-1], "Read")
        self.assertEqual(len(frames), 2, [f.name for f in frames])  # test + Read
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)

    def test_iread_missing_file(self):
        from machinevisiontoolbox import iread

        with self.assertRaises(FileNotFoundError):
            iread("flowers.png")

    def tearDown(self):
        # remove the test image file
        try:
            os.remove("./test-image.png")
        except FileNotFoundError:
            pass


class TestDataLoaders(unittest.TestCase):
    """Read real data files with mvtb_load_matfile() and mvtb_load_jsonfile()."""

    def test_matfile_in_mvtbdata(self):
        # bunny.mat is in the mvtbdata package, and holds one array
        data = mvtb_load_matfile("data/bunny.mat")
        self.assertIsInstance(data["bunny"], np.ndarray)
        self.assertEqual(data["bunny"].ndim, 2)

    def test_matfile_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            mvtb_load_matfile("data/__not_there__.mat")

    def test_matfile_struct_becomes_namedtuple(self):
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.mat")
            savemat(path, {"s": {"a": 1.5, "b": np.arange(3)}, "x": np.eye(2)})
            data = mvtb_load_matfile(path)

        # a MATLAB struct is returned as a namedtuple, so its fields are attributes
        self.assertEqual(data["s"].a, 1.5)
        nt.assert_array_equal(data["s"].b, [0, 1, 2])
        nt.assert_array_equal(data["x"], np.eye(2))

    def test_matfile_struct_gives_no_deprecation_warning(self):
        # the mat_struct class used to be imported from scipy.io.matlab.mio5_params,
        # which is deprecated and will be removed in SciPy 2.0
        import warnings

        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.mat")
            savemat(path, {"s": {"a": 1.5}})
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                data = mvtb_load_matfile(path)
        self.assertEqual(data["s"].a, 1.5)

    def test_jsonfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.json")
            with open(path, "w") as f:
                json.dump({"name": "cube", "n": [1, 2, 3]}, f)
            data = mvtb_load_jsonfile(path)

        self.assertEqual(data, {"name": "cube", "n": [1, 2, 3]})

    def test_jsonfile_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            mvtb_load_jsonfile("__not_there__.json")


class TestDataPaths(unittest.TestCase):
    def test_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.dat"
            path.write_text("x")
            self.assertEqual(mvtb_path_to_datafile(path), path.resolve())
            self.assertEqual(
                mvtb_path_to_datafile(path, string=True), str(path.resolve())
            )

        # an absolute path that does not exist is an error, with no search of mvtbdata
        with self.assertRaises(FileNotFoundError):
            mvtb_path_to_datafile(Path(tmp) / "gone.dat")

    def test_string_result(self):
        path = mvtb_path_to_datafile("data", "bunny.ply", string=True)
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isfile(path))

    def test_local_file_string_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("local.dat").write_text("x")
                path = mvtb_path_to_datafile("local.dat", string=True)
                self.assertIsInstance(path, str)
                self.assertTrue(path.endswith("local.dat"))
            finally:
                os.chdir(cwd)


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    unittest.main()
