#!/usr/bin/env python
"""Run the scripts in the examples folder headless, as a smoke test.

Each example runs in its own Python process with the non-interactive ``Agg``
matplotlib backend, so ``plt.show`` and ``block=True`` return at once.  A
``DeprecationWarning`` is turned into an error, so an example that uses a removed
or deprecated Toolbox API fails here rather than in front of a reader.

Examples that need a window, hardware, a server or large optional packages are
listed in ``SKIP`` with the reason.  An example that is not listed is run, so a
new example is tested by default.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SKIP = {
    "bunny.py": "opens an Open3D window",
    "showzip.py": "opens OpenCV windows and waits for key presses",
    "randomdots.py": "interactive threshold window",
    "ros_stream.py": "needs a rosbridge server",
    "torch-yolo.py": "needs ultralytics and downloads a model",
    "torch_seg.py": "needs torchvision and downloads a model",
}

# examples that fetch from the internet: a network failure skips, not fails
NETWORK = {"monalisa.py"}
NETWORK_ERRORS = ("URLError", "ConnectionError", "TimeoutError", "HTTPError", "gaierror")

# examples that run an external program: skipped when it is not installed
NEEDS_PROGRAM = {"blobs2.py": "dot"}  # Graphviz

# examples that redraw a figure many times: slow on a CI runner, so given longer
# than the 50 s per-test limit that CI applies to the rest of the suite
SLOW = {"morphdemo.py": 300}

ARGS = {"morphdemo.py": ["--delay", "0"]}


def _case(name):
    if name in SLOW:
        return pytest.param(name, marks=pytest.mark.timeout(SLOW[name]), id=name)
    return pytest.param(name, id=name)


@pytest.mark.parametrize("script", [_case(p.name) for p in sorted(EXAMPLES.glob("*.py"))])
def test_example_runs(script):
    if script in SKIP:
        pytest.skip(SKIP[script])
    if script in NEEDS_PROGRAM and shutil.which(NEEDS_PROGRAM[script]) is None:
        pytest.skip(f"needs {NEEDS_PROGRAM[script]} on the PATH")

    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", script, *ARGS.get(script, [])],
        cwd=EXAMPLES,
        env=env,
        capture_output=True,
        text=True,
        timeout=SLOW.get(script, 120) - 10,
    )
    if result.returncode != 0:
        if script in NETWORK and any(e in result.stderr for e in NETWORK_ERRORS):
            pytest.skip("network not available")
        pytest.fail(f"{script} exited with {result.returncode}:\n{result.stderr[-2000:]}")
