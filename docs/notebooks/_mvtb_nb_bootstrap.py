"""Environment bootstrap for machinevision-toolbox-python's Jupyter notebooks.

Installs the toolbox (and reports how) across the three environments a notebook in
this folder might run in: a local Jupyter/VS Code install, Google Colab, and
JupyterLite (Pyodide/WASM, in-browser).

This file is the single source of truth for that logic. Every notebook's own
bootstrap cell is a generated copy of this file's content, produced by
sync_bootstrap.py -- see docs/notebooks/README.md for the full explanation.
"""

import subprocess
import sys
from pathlib import Path


async def ensure_installed() -> bool:
    """Install machinevision-toolbox-python if needed, and report the environment.

    :returns: True if running on Google Colab, False otherwise.
    """
    if sys.platform == "emscripten":
        import micropip

        await micropip.install(
            [
                "opencv-python",
                "ansitable",
                "mvtb-data",
                "tqdm",
                "requests",
                "ipywidgets",
                # runtime dependencies of spatialmath-python and pgraph-python,
                # listed here because those two are installed with deps=False below
                "numpy",
                "scipy",
                "matplotlib",
                "typing_extensions",
            ]
        )
        # spatialmath-python <= 1.1.18 lists pre-commit (a dev tool) as a runtime
        # dependency; its virtualenv requires packaging>=26.3, which conflicts with
        # the packaging 24.2 built into Pyodide, so a normal resolve fails.
        # pgraph-python depends on spatialmath-python, so it needs the same
        # treatment.  Remove once a spatialmath release without that dependency is
        # out (rai-opensource/spatialmath-python#250) and install both normally.
        await micropip.install(
            ["spatialmath-python", "pgraph-python"],
            deps=False,
        )
        import cv2  # noqa: F401 - force cv2 into module registry before toolbox import

        wheels = sorted(Path("/pypi").glob("machinevision_toolbox_python-*.whl"))
        if wheels:
            # Prefer the wheel bundled with this JupyterLite site. Relative,
            # not absolute: a leading slash resolves against the origin, not
            # this site's own base URL (a GitHub Pages project subpath).
            await micropip.install(f"pypi/{wheels[-1].name}", deps=False)
        else:
            # Fall back to PyPI when running outside the published site layout.
            await micropip.install("machinevision-toolbox-python", deps=False)
        where, colab = "in browser", False
    else:
        try:
            import google.colab  # noqa: F401
        except ImportError:
            where, colab = "locally", False
        else:
            print("Installing machinevision-toolbox-python...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "machinevision-toolbox-python",
                ],
                check=True,
            )
            where, colab = "on Colab", True

    import cv2

    import machinevisiontoolbox

    version = getattr(machinevisiontoolbox, "__version__", "unknown")
    opencv_version = getattr(cv2, "__version__", "unknown")
    print(f"Running {where} using MVTB v{version} with OpenCV {opencv_version}")
    return colab
