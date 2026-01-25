#!/usr/bin/env python3

# a simple Machine Vision Toolbox "shell", runs Python3 and loads in NumPy, MVTB, SMTB
#
# Run it from the shell by
#
#  % mvtbtool
#
# using an wrapper script built during package installation.

# import stuff
import sys
import argparse
from math import pi  # lgtm [py/unused-import]
import numpy as np

from machinevisiontoolbox import *  # lgtm [py/unused-import]
from spatialmath import *  # lgtm [py/polluting-import]
from spatialmath.base import *  # lgtm [py/polluting-import]
from importlib.metadata import version, PackageNotFoundError
import textwrap

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
    parser = argparse.ArgumentParser("Machine Vision Toolbox shell")
    parser.add_argument("script", default=None, nargs="?", help="specify script to run")
    parser.add_argument(
        "-B", "--backend", default=None, help="specify graphics backend"
    )
    parser.add_argument(
        "-c",
        "--color",
        default="neutral",
        help="specify terminal color scheme (neutral, lightbg, nocolor, linux), linux is for dark mode",
    )
    parser.add_argument("-x", "--confirmexit", default=False, help="confirm exit")
    parser.add_argument("-p", "--prompt", default=">>> ", help="input prompt")

    parser.add_argument(
        "-a", "--showassign", default=False, help="display the result of assignments"
    )
    parser.add_argument("script", default=None, nargs="?", help="specify script to run")

    parser.add_argument(
        "-r",
        "--resultprefix",
        default=None,
        help="execution result prefix, include {} for execution count number",
    )

    args = parser.parse_args()
    return args


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
    args = parse_arguments()

    if args.backend is not None:
        print(f"Using matplotlib backend {args.backend}")
        plt.use(args.backend)

    make_banner(args)

    if args.script is not None:
        path = Path(args.script)
        if not path.exists():
            raise ValueError(f"script does not exist: {args.script}")
        exec(path.read_text())

    ## drop into IPython
    import IPython
    from traitlets.config import Config
    from IPython.terminal.prompts import ClassicPrompts
    from IPython.terminal.prompts import Prompts
    from pygments.token import Token

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
    c.InteractiveShellEmbed.colors = args.color
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
    c.InteractiveShellApp.exec_lines = code
    IPython.start_ipython(config=c, user_ns=globals())


if __name__ == "__main__":
    main()
