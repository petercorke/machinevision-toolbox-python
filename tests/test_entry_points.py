#!/usr/bin/env python

import importlib.util
import tomllib
import unittest
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def console_scripts(text: str) -> list[tuple[str, str, str]]:
    """Parse ``[project.scripts]`` out of pyproject.toml text.

    :param text: contents of a pyproject.toml file
    :return: list of (command name, module, function) tuples
    """
    scripts = tomllib.loads(text).get("project", {}).get("scripts", {})
    return [(name, *target.split(":")) for name, target in scripts.items()]


def missing_modules(text: str) -> list[str]:
    """Return the commands in ``[project.scripts]`` whose module cannot be found."""
    return [
        name
        for name, module, _ in console_scripts(text)
        if importlib.util.find_spec(module) is None
    ]


class TestEntryPoints(unittest.TestCase):
    def test_console_scripts_exist(self):
        """Every console script declared in pyproject.toml must have a module
        behind it -- regression test for ``labeltool`` being declared (and so
        installed by pip) in 2.4.0 while bin/labeltool.py was never committed,
        which made the command fail with ModuleNotFoundError"""
        scripts = console_scripts(PYPROJECT.read_text())
        self.assertGreater(len(scripts), 0, "no console scripts found; parser broken?")
        self.assertEqual(missing_modules(PYPROJECT.read_text()), [])

    def test_detects_missing_module(self):
        """The check itself must flag a script whose module does not exist"""
        text = (
            "[project.scripts]\n"
            'good = "machinevisiontoolbox.bin.imtool:main"\n'
            'bad = "machinevisiontoolbox.bin.no_such_tool:main"\n'
            "\n[build-system]\n"
        )
        self.assertEqual(missing_modules(text), ["bad"])


if __name__ == "__main__":
    unittest.main()
