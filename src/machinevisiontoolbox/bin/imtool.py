#!/usr/bin/env python3
"""
Command-line tool to display images using the Machine Vision Toolbox.

Usage::

    $ imtool street.png monalisa.png
"""

import argparse

from colored import Fore, Style

from machinevisiontoolbox import Image
from machinevisiontoolbox.bin._bintools import CustomDefaultsHelpFormatter, MVTB_LINK
from ansitable import ANSITable, Column
import matplotlib
import numpy as np

# Set by figure keypress callback when the user requests immediate exit.
exit_requested = False


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=CustomDefaultsHelpFormatter,
        description=f"Display an image using Machine Vision Toolbox for Python.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="list of image files to view, files can also include those distributed with machinevision toolbox, eg. 'monalisa.png'",
    )
    parser.add_argument(
        "--colorspace",
        "-c",
        default=None,
        help="colorspace to display pixel value in (default RGB)",
    )
    parser.add_argument(
        "--block",
        "-b",
        action="store_true",
        default=False,
        help="block after each image",
    )
    # -m show metadata
    parser.add_argument(
        "--metadata", "-m", help="Print image metadata to stdout", action="store_true"
    )
    # -p pick points
    parser.add_argument("--points", "-p", help="Pick points", action="store_true")
    # csv output for pick points
    parser.add_argument(
        "--csv", help="Output picked points as CSV to stdout", action="store_true"
    )
    # -g show grid
    parser.add_argument(
        "--grid", "-g", help="Overlay grid on images", action="store_true"
    )
    # -v verbose show image details
    parser.add_argument(
        "--verbose", "-v", help="Show image details", action="store_true"
    )
    parser.add_argument(
        "--backend",
        "-B",
        default=None,
        metavar="BACKEND",
        help="Matplotlib backend to use, e.g. TkAgg, Qt5Agg, MacOSX (default: system default)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        action="store_true",
        default=False,
        help="Read and display the alpha channel if present (default: strip alpha)",
    )

    return parser.parse_args()


def visualize_image(image, args, block):
    import matplotlib.pyplot as plt
    from machinevisiontoolbox.bin._iminteractive import ImageInteractor

    if args.verbose:
        print(image)

    if args.metadata:
        md = image.metadata()
        table = ANSITable(
            Column("key", width=20),
            Column("value", width=58, headalign="<", colalign="<"),
        )

        # sort the metadata by key
        if md is not None:
            for key in sorted(md.keys()):
                table.row(key, md[key])
            table.print()

    ## pick points and display coordinates as well as delta between points, in an ANSITable
    if args.points:
        image.disp(block=False, grid=args.grid)
        print(f"""\
    {Fore.yellow}Click on points in the image (first click might just select the window and get lost)
    * left click to add a point
    * right click to remove point
    * Enter when done
    * you can zoom in using the magnifier button at bottom{Style.reset}
    """)
        points = plt.gcf().ginput(n=-1, timeout=0)

        if args.csv:
            print("u,v")
            for p in points:
                print(f"{p[0]},{p[1]}")
            return
        else:
            # define the table
            fmt = "{:.1f}"
            table = ANSITable(
                Column("u", headalign="^", colalign="<", fmt=fmt),
                Column("v", headalign="^", colalign="<", fmt=fmt),
                Column("Δu", headalign="^", colalign="<"),
                Column("Δv", headalign="^", colalign="<"),
                Column("|Δ|", headalign="^", colalign="<"),
            )

            # add rows, first row is just the point, subsequent rows have deltas
            for i, p in enumerate(points):
                if i == 0:
                    u0, v0 = p
                    table.row(u0, v0, "", "", "")
                else:
                    u1, v1 = p
                    du, dv = u1 - u0, v1 - v0
                    u0, v0 = u1, v1
                    table.row(
                        u1,
                        v1,
                        fmt.format(du),
                        fmt.format(dv),
                        fmt.format((du**2 + dv**2) ** 0.5),
                    )
            table.print()
    else:
        global cs_image, colorspace, exit_requested

        if args.colorspace is not None:
            cs_image = (
                image.to("float32")
                .gamma_decode("sRGB")
                .colorspace(args.colorspace)
                .array
            )
            colorspace = args.colorspace
        else:
            cs_image = image.array
            colorspace = image.colororder_str

        # put the filename into the title, if it's too long, truncate it and add ellipsis at the start
        name = image.name
        if len(name) > 40:
            name = "..." + name[-37:]
        image.disp(block=False, grid=args.grid, title=name, coordformat=format_coord)

        # 'q' keeps existing figure-close behaviour; 'x' exits all remaining files.
        fig = plt.gcf()
        ax = fig.gca()

        # Attach interactive rectangle / line-profile overlay.
        _nav_help = (
            "  x   close image and advance to next file\n" "  q   close window\n"
        )
        ImageInteractor(fig, ax, image, nav_help=_nav_help)

        def _on_exit(event):
            global exit_requested
            if event.key == "x":
                exit_requested = True
                plt.close(event.canvas.figure)

        fig.canvas.mpl_connect("key_press_event", _on_exit)

        if block:
            plt.show(block=True)


# format the pixel value display
def format_coord(u: float, v: float) -> str:
    global cs_image, colorspace

    u = int(u + 0.5)
    v = int(v + 0.5)

    try:
        if cs_image.ndim == 2:
            # monochrome image
            x = cs_image[v, u]
            if isinstance(x, np.integer):
                val = f"{x:d}"
            elif isinstance(x, np.floating):
                val = f"{x:.3f}"
            elif isinstance(x, (np.bool_, bool)):
                val = f"{x}"
            else:
                print(f"unknown pixel type {type(x)}")

            return f"({u}, {v}): {val}"
        else:
            # color image
            x = cs_image[v, u, :]
            if np.issubdtype(x.dtype, np.integer):
                val = [f"{_:d}" for _ in x]
            elif np.issubdtype(x.dtype, np.floating):
                val = [f"{_:.3f}" for _ in x]
            else:
                val = [str(_) for _ in x]
            val = "[" + ", ".join(val) + "]"

            return f"({u}, {v}): {val} {colorspace}, {x.dtype}"
    except Exception as e:
        return "bad"


def main():
    global exit_requested

    args = getargs()

    if args.backend is not None:
        matplotlib.use(args.backend)

    import matplotlib.pyplot as plt  # noqa: F401 — initialises backend, used via globals
    from machinevisiontoolbox.bin._iminteractive import ImageInteractor  # noqa: F401

    if len(args.files) > 10:
        args.block = True

    for i, file in enumerate(args.files):
        try:
            img = Image.Read(file, alpha=args.alpha)
            print(img)
        except ValueError:
            print(f"File {file} not found")

        if i == len(args.files) - 1:
            # last one
            block = True
        else:
            block = args.block

        visualize_image(img, args, block)

        if exit_requested:
            plt.close("all")
            break


if __name__ == "__main__":
    main()
