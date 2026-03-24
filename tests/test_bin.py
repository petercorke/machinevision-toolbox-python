#!/usr/bin/env python
"""
Smoke tests for command-line entry points in machinevisiontoolbox.bin.

``--help`` tests verify that imports and argument parsing work and the tool
exits cleanly.  Display tests verify that the tool reaches the blocking
display stage (``TimeoutExpired`` after a short timeout is treated as
success, since the tool is waiting for a window to be closed).
"""

import subprocess
import sys
import unittest

# An image bundled with mvtb-data that is always present.
_TEST_IMAGE = "monalisa.png"

# Seconds to wait for display tools before treating TimeoutExpired as success.
_DISPLAY_TIMEOUT = 5


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command via the current Python interpreter's entry-point module."""
    return subprocess.run(
        [sys.executable, "-m"] + args,
        capture_output=True,
        timeout=timeout,
    )


class TestMvtbtool(unittest.TestCase):

    def test_help(self):
        result = _run(["machinevisiontoolbox.bin.mvtbtool", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())


class TestImtool(unittest.TestCase):

    def test_help(self):
        result = _run(["machinevisiontoolbox.bin.imtool", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())

    def test_display_image(self):
        """Tool should reach its blocking display stage without error."""
        try:
            result = _run(
                ["machinevisiontoolbox.bin.imtool", _TEST_IMAGE],
                timeout=_DISPLAY_TIMEOUT,
            )
            # If it returns before the timeout, it must have exited cleanly.
            self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        except subprocess.TimeoutExpired:
            # Reached the blocking plt.show() — that is the expected success path.
            pass


class TestTagtool(unittest.TestCase):

    def test_help(self):
        result = _run(["machinevisiontoolbox.bin.tagtool", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())

    def test_display_image(self):
        """Tool should reach its blocking display stage without error."""
        try:
            result = _run(
                ["machinevisiontoolbox.bin.tagtool", _TEST_IMAGE],
                timeout=_DISPLAY_TIMEOUT,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        except subprocess.TimeoutExpired:
            pass


if __name__ == "__main__":
    unittest.main()
