#!/usr/bin/env python
"""Smoke tests for scripts in the examples folder.

These tests are intentionally lightweight and side-effect free:
- confirm example files are present
- confirm each example script parses and compiles

This catches syntax errors and merge-conflict artifacts without executing GUI,
camera, ROS, or network code from examples.
"""

from pathlib import Path
import unittest


class TestExamplesSmoke(unittest.TestCase):

    def test_examples_exist(self):
        examples_dir = Path(__file__).resolve().parents[1] / "examples"
        scripts = sorted(examples_dir.glob("*.py"))
        self.assertGreater(len(scripts), 0, msg="No example scripts found")

    def test_examples_compile(self):
        examples_dir = Path(__file__).resolve().parents[1] / "examples"
        scripts = sorted(examples_dir.glob("*.py"))

        failures = []
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            try:
                code_obj = compile(source, str(script), "exec")
                self.assertIsNotNone(code_obj)
            except SyntaxError as exc:
                failures.append(f"{script.name}:{exc.lineno}: {exc.msg}")

        if failures:
            self.fail("Example smoke test failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
