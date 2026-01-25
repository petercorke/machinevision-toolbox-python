#!/usr/bin/env python3

# interactive tool to browse images using machinevisiontoolbox.Image.disp()
#
# Run it from the shell by
#
#  % imtool
#


# import stuff
import argparse
from machinevisiontoolbox import *  # lgtm [py/unused-import]
from colored import Fore, Style
import textwrap as _textwrap


def getargs():
    # https://stackoverflow.com/questions/40419276/python-how-to-print-text-to-console-as-hyperlink
    def link(uri, label=None):
        if label is None:
            label = uri
        parameters = ""

        # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
        escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

        return escape_mask.format(parameters, uri, label)

    class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _split_lines(self, text, width):
            text = self._whitespace_matcher.sub(" ", text).strip()
            return _textwrap.wrap(text, 80)

    parser = argparse.ArgumentParser(
        formatter_class=LineWrapRawTextHelpFormatter,
        description=f"Display an image using {link('https://github.com/petercorke/machinevision-toolbox-python', 'Machine Vision Toolbox for Python')}",
    )
    parser.add_argument(
        "files",
        default=None,
        nargs="+",
        help="list of image files to view, files can also include those distributed with machinevision toolbox, eg. 'monalisa.png'",
    )
    parser.add_argument(
        "--block",
        "-b",
        action="store_true",
        default=False,
        help="block after each image",
    )
    # -m show metadata
    parser.add_argument("--metadata", "-m", help="Show metadata", action="store_true")
    # -p pick points
    parser.add_argument("--points", "-p", help="Pick points", action="store_true")
    # csv output for pick points
    parser.add_argument(
        "--csv", "-c", help="Output picked points as CSV", action="store_true"
    )
    # -g show grid
    parser.add_argument("--grid", "-g", help="Show grid", action="store_true")
    # -v verbose show image details
    parser.add_argument(
        "--verbose", "-v", help="Show image details", action="store_true"
    )

    return parser.parse_args()


def visualize_image(image, args, block):
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
        print(
            f"""\
    {Fore.yellow}Click on points in the image (first click might just select the window and get lost)
    * left click to add a point
    * right click to remove point
    * CR when done
    * you can zoom in using the magnifier button at bottom{Style.reset}
    """
        )
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
        image.disp(block=block, grid=args.grid)


def main():
    args = getargs()

    if len(args.files) > 10:
        args.block = True

    for i, file in enumerate(args.files):
        try:
            img = Image.Read(file)

            if i == len(args.files) - 1:
                # last one
                block = True
            else:
                block = args.block

            visualize_image(img, args, block)

        except ValueError:
            print(f"File {file} not found")


if __name__ == "__main__":
    main()
