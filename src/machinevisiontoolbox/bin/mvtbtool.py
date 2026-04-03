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
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version
from math import pi  # lgtm [py/unused-import]

import numpy as np
from matplotlib import image
from spatialmath import *  # lgtm [py/polluting-import]
from spatialmath.base import *  # lgtm [py/polluting-import]

from machinevisiontoolbox import *  # lgtm [py/unused-import]
from machinevisiontoolbox.bin._bintools import CustomDefaultsHelpFormatter

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Machine Vision Toolbox shell",
        formatter_class=CustomDefaultsHelpFormatter,
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
        "images",
        nargs="*",
        help="images to load on startup. These appear in the variable img; or img[0], img[1], ... if multiple are specified",
    )

    return parser.parse_known_args()


def make_banner(args):
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

    if args.backend is not None:
        print(f"Using matplotlib backend {args.backend}")
        plt.use(args.backend)

    make_banner(args)

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
