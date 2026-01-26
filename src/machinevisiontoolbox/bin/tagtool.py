#!/usr/bin/env python3

# tool to parse AR tags using machinevisiontoolbox.Image.disp()
#
# Run it from the shell by
#
#  % tagtool image
#


# import stuff
import argparse
from machinevisiontoolbox import *  # lgtm [py/unused-import]
from colored import Fore, Style
import textwrap
from spatialmath import Polygon2
from spatialmath.base import plot_text, plot_point
import matplotlib.pyplot as plt


def getargs():
    # https://stackoverflow.com/questions/40419276/python-how-to-print-text-to-console-as-hyperlink
    def link(uri, label=None):
        if label is None:
            label = uri
        parameters = ""

        # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
        escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

        return escape_mask.format(parameters, uri, label)

    class CustomHelpFormatter(argparse.HelpFormatter):
        def _split_lines(self, text, width):
            # Allow custom manual breaks (e.g., \n)
            lines = text.splitlines()
            # Wrap each line that is too long, avoiding breaking words
            wrapped_lines = []
            for line in lines:
                wrapped_lines.extend(
                    textwrap.wrap(
                        line, width, break_long_words=False, break_on_hyphens=False
                    )
                )
            return wrapped_lines
            print(wrapped_lines)
            return wrapped_lines

    parser = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter,
        description=f"Display AR tags in image using "
        f"{link('https://github.com/petercorke/machinevision-toolbox-python', 'Machine Vision Toolbox for Python')}.  "
        f"AR tags are highlighted with their IDs and the canonic top-left corner is marked.",
        epilog="A camera model is required to determine poses, this requires that focal length is specified.",
    )
    parser.add_argument(
        "files",
        default=None,
        nargs="+",
        help="list of image files to view, files can also include those distributed with machinevision toolbox, eg. 'lab-scene.png'",
    )
    parser.add_argument(
        "-b",
        "--block",
        action="store_true",
        default=False,
        help="block after each image",
    )

    # -g show grid
    parser.add_argument("-g", "--grid", help="Show grid", action="store_true")
    # -v verbose show image details
    parser.add_argument(
        "-v", "--verbose", help="Show image details", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--dict",
        type=str,
        default="4x4_50",
        help="Aruco dictionary to use, default is %(default)s",
    )
    parser.add_argument(
        "-s",
        "--side",
        type=int,
        default=25,
        help="Tag side length, default is %(default)s",
    )
    parser.add_argument(
        "-f",
        "--focallength",
        type=str,
        default=None,
        help="Focal length in units of pixels: f | fu,fv, default is %(default)s",
    )
    parser.add_argument(
        "-p",
        "--principalpoint",
        type=str,
        default=None,
        help="Principal in units of pixels: pu,pv. If not specified use image centre, default is %(default)s",
    )
    parser.add_argument(
        "-a",
        "--axes",
        help="Show axes on the image",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gamma-correction",
        action="store_true",
        default=False,
        help="Apply gamma decode to image, default is %(default)s",
    )
    parser.add_argument(
        "--channel",
        choices=["r", "g", "b"],
        help="Color channel, default is %(default)s",
    )

    return parser.parse_args()


def visualize_image(image, args, block):
    if args.verbose:
        print(image)

    # get parameters for a camera model
    if args.focallength is not None:
        f = args.focallength.split(",")
        if len(f) == 1:
            f = [float(f[0]), float(f[0])]
        elif len(f) == 2:
            f = [float(f[0]), float(f[1])]

        if args.principalpoint is not None:
            pp = args.principalpoint.split(",")
            if len(pp) == 1:
                pp = [float(pp[0]), float(pp[0])]
            elif len(pp) == 2:
                pp = [float(pp[0]), float(pp[1])]
        else:
            pp = [image.width / 2, image.height / 2]

        camera = CentralCamera(f=f, pp=pp)
        if args.verbose:
            print(camera)
    else:
        camera = None

    # find the tags and sort them
    if camera is None:
        tags = image.fiducial(dict=args.dict)
    else:
        tags = image.fiducial(dict=args.dict, K=camera.K, side=args.side)
    tags.sort(key=lambda x: x.id)

    # display the coordinate frames
    if args.axes:
        for tag in tags:
            tag.draw(image, length=10, thick=20)
        image.disp()

    # highlight the tags in the image
    image.disp()

    if camera is not None:
        table = ANSITable(
            Column("id", headalign="^", colalign=">"),
            Column("RMSE (pix)", headalign="^", colalign=">", fmt="{:.2f}"),
            Column("Rmax (pix)", headalign="^", colalign=">", fmt="{:.2f}"),
            Column("pose", headalign="^", colalign="<"),
            border="thin",
        )

    for i, tag in enumerate(tags):
        if camera is not None:
            # compute residuals
            p3d = tag.p3d.squeeze().T
            p2d = camera.project_point(p3d, objpose=tag.pose)
            resid = p2d - tag.corners
            print(tag.corners)
            rmse = np.sqrt(np.mean(resid**2))
            rmax = np.max(np.abs(resid))

            # add row to table
            table.row(
                tag.id,
                rmse,
                rmax,
                tag.pose.strline(),
                bgcolor="red" if rmax >= 1 else None,
            )
            outline_color = "red" if rmax >= 1 else "blue"

        else:
            if i == 0:
                print("tag IDs:", tag.id, end="")
            else:
                print(f", {tag.id}", end="")
            outline_color = "blue"

        # create an outline around the tag
        polygon = Polygon2(tag.corners)
        polygon.plot(facecolor="white", alpha=0.8, edgecolor=outline_color, linewidth=2)
        plot_text(
            polygon.centroid(),
            str(tag.id),
            color=outline_color,
            fontsize=12,
            horizontalalignment="center",
        )
        # mark the top left corner
        plot_point(tag.corners[:, 0], color=outline_color, marker="o", markersize=5)

    if camera is not None:
        table.print()
    else:
        print()

    print()

    plt.show(block=block)


def main():
    args = getargs()

    if len(args.files) > 10:
        args.block = True

    for i, file in enumerate(args.files):
        try:
            img = Image.Read(file, rgb=False)
        except ValueError:
            print(f"File {file} not found")

        if i == len(args.files) - 1:
            # last one
            block = True
        else:
            block = args.block

        visualize_image(img, args, block)


if __name__ == "__main__":
    main()
