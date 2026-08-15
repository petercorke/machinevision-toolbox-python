#!/usr/bin/env python3
"""
Interactive Machine Vision Toolbox shell — starts an IPython session with
NumPy, MVTB, and SpatialMath pre-imported.

Usage::

    $ mvtbtool
    $ mvtbtool street.png
    $ mvtbtool street.png --run=myscript.py
"""

import argparse
import os
import shlex
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version
from math import pi  # lgtm [py/unused-import]

import numpy as np
from matplotlib import image
import cv2
from spatialmath import *  # lgtm [py/polluting-import]
from spatialmath.base import *  # lgtm [py/polluting-import]

from machinevisiontoolbox import *  # lgtm [py/unused-import]
from machinevisiontoolbox.bin._bintools import LineWrapRawTextDefaultsHelpFormatter

try:
    from colored import Fore, Style

    _colored = True
    # print('using colored output')
except ImportError:
    # print('colored not found')
    _colored = False

# setup defaults
np.set_printoptions(
    linewidth=120,
    formatter={"float": lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"},
)
SE3._ansimatrix = True

_OPTIONS_ENVVAR = "MVTB_OPTIONS"


def env_arguments(parser):
    """Return command-line style options from the environment.

    :param parser: argument parser used for error reporting
    :type parser: :class:`argparse.ArgumentParser`
    :return: tokenised environment arguments
    :rtype: list[str]
    """
    options = os.environ.get(_OPTIONS_ENVVAR)
    if not options:
        return []

    try:
        return shlex.split(options)
    except ValueError as exc:
        parser.error(f"invalid {_OPTIONS_ENVVAR}: {exc}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Machine Vision Toolbox shell",
        formatter_class=LineWrapRawTextDefaultsHelpFormatter,
        epilog=(
            "options can be set via the environment variable MVTB_OPTIONS, "
            "for example:\n\n"
            "    $ export MVTB_OPTIONS=\"--backend TkAgg --prompt 'mvtb> ' "
            '--reload --torch --showassign"\n'
        ),
    )
    parser.add_argument(
        "-r",
        "--run",
        default=None,
        help="script to run at startup, but not displayed. Same as IPython's builtin -i option",
    )
    parser.add_argument(
        "-B",
        "--backend",
        default=None,
        metavar="BACKEND",
        help="specify %(metavar)s as the Matplotlib graphics backend (e.g. 'TkAgg', 'Qt5Agg', 'WebAgg', etc).  By default, the backend is chosen automatically by Matplotlib.",
    )
    parser.add_argument(
        "-t",
        "--theme",
        default="neutral",
        help="specify terminal color theme (neutral, lightbg, nocolor, linux), linux is for dark mode",
    )
    parser.add_argument(
        "-x",
        "--confirmexit",
        default=False,
        help="confirm exit",
        action="store_true",
    )
    parser.add_argument(
        "-P",
        "--prompt",
        default=">>> ",
        help="input prompt string",
    )

    parser.add_argument(
        "-a",
        "--showassign",
        action="store_true",
        default=False,
        help="automatically display the result of assignments, use ';' to suppress output",
    )

    parser.add_argument(
        "-R",
        "--resultprefix",
        default=None,
        help="execution result prefix, include {} for execution count number",
    )
    parser.add_argument(
        "--reload",
        default=False,
        action="store_true",
        help="enable autoreload of any imported modules, same as IPython's builtin %%autoreload 2",
    )
    parser.add_argument(
        "-b",
        "--base",
        default=False,
        action="store_true",
        help="'from machinevisiontoolbox.base import *', otherwise it is an alias 'mvb'.",
    )
    parser.add_argument(
        "--torch",
        default=False,
        action="store_true",
        help="import torch and torchvision if installed",
    )

    parser.add_argument(
        "images",
        nargs="*",
        help="images to load on startup. These appear in the variable img; or img[0], img[1], ... if multiple are specified",
    )

    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="non-interactive environment smoke test: print package versions, "
        "exercise one real numeric code path per package, exit 0/1 "
        "instead of starting an interactive shell",
    )

    argv = env_arguments(parser) + sys.argv[1:]
    return parser.parse_known_args(argv)


def optional_torch_imports(enable):
    """Optionally import torch and torchvision.

    :param enable: if ``True``, attempt optional imports
    :type enable: bool
    :return: tuple of imported modules dictionary and warning messages
    :rtype: tuple(dict, list)
    """
    modules = {}
    warnings = []

    if not enable:
        return modules, warnings

    try:
        import torch as _torch

        modules["torch"] = _torch
    except ImportError:
        warnings.append("PyTorch (torch) not found")

    try:
        import torchvision as _torchvision

        modules["torchvision"] = _torchvision
    except ImportError:
        warnings.append("TorchVision (torchvision) not found")

    return modules, warnings


def get_versions() -> list[str]:
    """Package version strings shown in the banner and by --test."""
    versions = [
        f"Python=={sys.version.split('|')[0].strip()}",
        f"MVTB=={version('machinevision-toolbox-python')}",
        f"SMTB=={version('spatialmath-python')}",
        f"NumPy=={version('numpy')}",
        f"SciPy=={version('scipy')}",
        f"Matplotlib=={version('matplotlib')}",
        f"OpenCV=={cv2.__version__}",
    ]
    try:
        versions.append(f"Open3D=={version('open3d')}")
    except PackageNotFoundError:
        versions.append("Open3D==not installed")
    return versions


def make_banner(args, optional_modules=None):
    if optional_modules is None:
        optional_modules = {}

    versions = get_versions()

    if "torch" in optional_modules:
        versions.append(
            f"PyTorch=={getattr(optional_modules['torch'], '__version__', 'unknown')}"
        )
    if "torchvision" in optional_modules:
        versions.append(
            "TorchVision=="
            f"{getattr(optional_modules['torchvision'], '__version__', 'unknown')}"
        )

    # create banner
    versions = "You're running: " + ", ".join(versions)

    # print the banner
    # https://patorjk.com/software/taag/#p=display&f=Cybermedium&t=Robotics%20Toolbox%0A

    banner = r"""_  _ ____ ____ _  _ _ _  _ ____    _  _ _ ____ _ ____ _  _ 
|\/| |__| |    |__| | |\ | |___    |  | | [__  | |  | |\ | 
|  | |  | |___ |  | | | \| |___     \/  | ___] | |__| | \| 
                                                        
___ ____ ____ _    ___  ____ _  _                          
|   |  | |  | |    |__] |  |  \/                           
|   |__| |__| |___ |__] |__| _/\_  

for Python

"""

    w = "\n".join(
        textwrap.wrap(
            versions,
            break_long_words=False,
            subsequent_indent=" " * len("You" "re running:  "),
            width=80,
        )
    )

    banner += w

    banner += "\n\nfrom machinevisiontoolbox import *\n"
    if args.base:
        banner += "from machinevisiontoolbox.base import *\n"
    else:
        # this line not strictly true, but it is what the import * line does
        banner += "import machinevisiontoolbox.base as mvb\n"
    banner += """
from spatialmath import *

matplotlib interactive mode on

func/object?       - show brief help
help(func/object)  - show detailed help
func/object??      - show source code"""
    if _colored:
        print(Fore.yellow + banner + Style.reset)
    else:
        print(banner)


def run_smoke_test() -> bool:
    """Non-interactive environment sanity check, used by --test.

    Not a substitute for the pytest suite -- a fast, human- or script-run
    "did this environment actually come together correctly" check: real
    versions, and one real numeric result per OpenCV- and Open3D-backed
    code path, checked against a sanity condition rather than just "it
    didn't raise". Missing optional dependencies (e.g. Open3D) are
    reported as a FAIL with the reason, not silently skipped.
    """
    print(", ".join(get_versions()))

    checks: list[tuple[str, bool]] = []

    try:
        img = Image.Read("monalisa.png", mono=True)
        smoothed = img.smooth(sigma=2)
        ok = smoothed.shape == img.shape and not np.array_equal(
            smoothed.array, img.array
        )
        checks.append(("Image.Read + smooth() (OpenCV-backed)", ok))
    except Exception as e:
        checks.append((f"Image.Read + smooth() (OpenCV-backed): {e}", False))

    try:
        import open3d  # noqa: F401  -- presence check; PointCloud does the real work
    except ImportError as e:
        checks.append(
            (f"PointCloud.Read('bunny.ply') + voxel_grid(): Open3D not installed ({e})", False)
        )
    else:
        try:
            bunny = PointCloud.Read("bunny.ply")
            voxels = bunny.voxel_grid(voxel_size=0.01)
            ok = len(bunny) > 0 and len(voxels._voxels.get_voxels()) > 0
            checks.append(("PointCloud.Read('bunny.ply') + voxel_grid() (Open3D-backed)", ok))
        except Exception as e:
            checks.append(
                (f"PointCloud.Read('bunny.ply') + voxel_grid() (Open3D-backed): {e}", False)
            )

    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    n_passed = sum(1 for _, passed in checks if passed)
    print(f"mvtbtool --test: {n_passed}/{len(checks)} checks passed")
    return n_passed == len(checks)


def main():
    args, ipython_args = parse_arguments()

    if args.test:
        sys.exit(0 if run_smoke_test() else 1)

    try:
        import IPython
        from IPython.terminal.prompts import Prompts
        from pygments.token import Token
        from traitlets.config import Config
    except ImportError as e:
        sys.exit(
            f"mvtbtool requires IPython and pygments, which are not "
            f"installed ({e}).\nInstall them with:\n\n"
            "    pip install machinevision-toolbox-python[tool]\n"
        )

    torch_modules, torch_warnings = optional_torch_imports(args.torch)

    if args.backend is not None:
        print(f"Using matplotlib backend {args.backend}")
        plt.use(args.backend)

    make_banner(args, torch_modules)

    if torch_warnings:
        for warning in torch_warnings:
            print(f"Warning: {warning}")

    # if args.script is not None:
    #     path = Path(args.script)
    #     if not path.exists():
    #         raise ValueError(f"script does not exist: {args.script}")
    #     exec(path.read_text())

    ## drop into IPython
    class MyPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            return [(Token.Prompt, args.prompt)]

        def out_prompt_tokens(self, cli=None):
            if args.resultprefix is None:
                # traditional behaviour
                return [
                    (Token.OutPrompt, "Out["),
                    (Token.OutPromptNum, str(self.shell.execution_count)),
                    (Token.OutPrompt, "]: "),
                ]
            else:
                return [
                    (Token.Prompt, args.resultprefix.format(self.shell.execution_count))
                ]

    # set configuration options, there are lots, see
    # https://ipython.readthedocs.io/en/stable/config/options/terminal.html
    c = Config()
    c.InteractiveShellEmbed.colors = args.theme
    c.InteractiveShell.confirm_exit = args.confirmexit
    # c.InteractiveShell.prompts_class = ClassicPrompts
    c.InteractiveShell.prompts_class = MyPrompt
    if args.showassign:
        c.InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

    code = [
        f"%matplotlib{' '+args.backend if args.backend is not None else ''}",
        "import matplotlib.pyplot as plt",
        "_prec = get_ipython().run_line_magic('precision', '%.3g'); "
        "print(f'Default numeric formatting: {_prec}')",
    ]
    if args.base:
        code.append("from machinevisiontoolbox.base import *")
    if args.reload:
        code.append("%load_ext autoreload")
        code.append("%autoreload 2")

    namespace = {k: v for k, v in globals().items() if not k.startswith("__")}
    namespace.update(torch_modules)
    # load images if specified on the command line

    if _colored:
        print(Fore.green)
    try:
        if len(args.images) > 0:
            print("Loading images...")
            images = []
            for i, filename in enumerate(args.images, 1):
                if len(args.images) > 1:
                    print(f"  {filename} --> img[{i-1}]")
                else:
                    print(f"  {filename} --> img")
                images.append(Image.Read(filename))
            if len(images) == 1:
                namespace["img"] = images[0]
            elif len(images) > 1:
                namespace["img"] = images
            print()

        if args.run is not None:
            if _colored:
                print(Fore.yellow)
            print(f"%run -i {args.run}")
            code.append(f"%run -i '{args.run}'")
    except Exception as e:
        if _colored:
            print(Fore.red)
        print(f"Error loading images or running script: {e}")
        print("Dropping into IPython without images or script.")

    if _colored:
        print(Style.reset)

    c.InteractiveShellApp.exec_lines = code

    # Clear argv so IPython doesn't try to execute our image filenames as scripts
    sys.argv = sys.argv[:1]

    IPython.start_ipython(config=c, user_ns=namespace, argv=ipython_args)


if __name__ == "__main__":
    main()
