#!/usr/bin/env python3
"""
Command-line tool to inspect and display images from ROS bag files.

Usage::

    $ bagtool mybag.bag
"""

import argparse

import matplotlib.pyplot as plt
from colored import Fore, Style

from machinevisiontoolbox import Image, RosBag
from machinevisiontoolbox.bin._bintools import CustomHelpFormatter, MVTB_LINK


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter,
        description=f"Display AR tags in image using {MVTB_LINK}.  "
        "AR tags are highlighted with their IDs and the canonic top-left corner is marked.",
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

    # if len(args.files) > 10:
    #     args.block = True

    #     visualize_image(img, args, block)

    for filename in args.files:
        bag = RosBag(filename)
        print(f"{Fore.CYAN}{Style.BOLD}{bag}{Style.RESET}")
        bag.print()


if __name__ == "__main__":
    main()
