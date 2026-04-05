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


def make_banner(args, optional_modules=None):
    if optional_modules is None:
        optional_modules = {}

    versions = []

    versions.append(f"Python=={sys.version.split('|')[0].strip()}")
    versions.append(f"MVTB=={version('machinevision-toolbox-python')}")
    versions.append(f"SMTB=={version('spatialmath-python')}")
    versions.append(f"NumPy=={version('numpy')}")
    versions.append(f"SciPy=={version('scipy')}")
    versions.append(f"Matplotlib=={version('matplotlib')}")
    versions.append(f"OpenCV=={cv.__version__}")
    try:
        versions.append(f"Open3D=={version('open3d')}")
    except PackageNotFoundError:
        pass

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

    banner += r"""

from machinevisiontoolbox import *
from spatialmath import *

matplotlib interactive mode on

func/object?       - show brief help
help(func/object)  - show detailed help
func/object??      - show source code

    """
    if _colored:
        print(Fore.yellow + banner + Style.reset)
    else:
        print(banner)


def main():
    args, ipython_args = parse_arguments()
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
    import IPython
    from IPython.terminal.prompts import ClassicPrompts, Prompts
    from pygments.token import Token
    from traitlets.config import Config

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
        "import matplotlib.pyplot as plt",
        f"%matplotlib{' '+args.backend if args.backend is not None else ''}",
        "_precision = %precision %.3g;",
    ]
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
