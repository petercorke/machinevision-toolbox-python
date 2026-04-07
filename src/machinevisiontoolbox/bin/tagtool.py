#!/usr/bin/env python3
"""
Command-line tool to detect and display AR tags (ArUco/AprilTag) in images.

Usage::

    $ tagtool lab-scene.png
"""

import argparse
import json

import matplotlib.pyplot as plt
from colored import Fore, Style
from spatialmath import Polygon2
from spatialmath.base import plot_point, plot_text

from machinevisiontoolbox import *  # lgtm [py/unused-import]
from machinevisiontoolbox.bin._bintools import (
    CustomDefaultsHelpFormatter,
    MVTB_LINK,
)


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=CustomDefaultsHelpFormatter,
        description=f"Display AR tags in image using Machine Vision Toolbox for Python."
        "AR tags are highlighted with their IDs and the canonic top-left corner is marked.",
        epilog="A camera model is required to determine poses, this requires that focal length is specified.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="list of image files to view, files can also include those distributed with machinevision toolbox, eg. 'lab-scene.png'",
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
        type=float,
        default=25,
        help="Tag side length, default is %(default)s",
    )
    parser.add_argument(
        "-f",
        "--focallength",
        type=str,
        default=None,
        help="Focal length in units of pixels or metres if rho is specified: f | fu,fv. Required for tag pose estimation",
    )
    parser.add_argument(
        "-p",
        "--principalpoint",
        type=str,
        default=None,
        help="Principal point coordinate in units of pixels: pu,pv. Required for tag pose estimation. If not specified use image centre",
    )
    parser.add_argument(
        "-r",
        "--rho",
        help="Pixel pitch in units of m/pixel, required for tag pose estimation if focal length is specified in metres",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-b",
        "--block",
        action="store_true",
        default=False,
        help="block after each image",
    )
    # -a show coordinate frames
    parser.add_argument(
        "-a", "--axes", help="Show coordinate frames", action="store_true"
    )
    # -j, --json output tag data to JSON file
    parser.add_argument(
        "-j",
        "--json",
        type=str,
        default=None,
        metavar="FILE",
        help="Output tag data to JSON file",
    )
    # --no-display do not display images, just output JSON data
    parser.add_argument(
        "--no-display",
        help="Do not display images, just output JSON data",
        action="store_true",
    )
    # --no-table do not display table of tag data
    parser.add_argument(
        "--no-table",
        help="Do not display table of tag data",
        action="store_true",
    )
    # --rmax=threshold for highlighting tags with large reprojection error
    parser.add_argument(
        "--rmax",
        type=float,
        default=0,
        help="Threshold for highlighting tags with large reprojection error (in pixels) in red, 0 for no highlighting",
    )
    # -g show grid
    parser.add_argument(
        "-g", "--grid", help="Overlay grid on images", action="store_true"
    )
    # -v verbose show image details
    parser.add_argument(
        "-v", "--verbose", help="Show image details", action="store_true"
    )
    return parser.parse_args()


def make_plot(tags, args, camera, image):
    # display the coordinate frames
    length = max(image.size) / 50
    if args.axes:
        for tag in tags:
            tag.draw(image, length=length, thick=20)

    # highlight the tags in the image
    image.disp()

    for tag in tags:
        if camera is not None and args.rmax > 0 and tag.rmax >= args.rmax:
            outline_color = "red"
        else:
            outline_color = "blue"
        # create an outline around the tag
        polygon = Polygon2(tag.corners)
        polygon.plot(facecolor="white", alpha=0.5, edgecolor=outline_color, linewidth=2)
        plot_text(
            polygon.centroid(),
            str(tag.id),
            color=outline_color,
            fontsize=12,
            horizontalalignment="center",
        )
        # mark the top left corner
        plot_point(tag.corners[:, 0], color=outline_color, marker="o", markersize=5)


def make_table(tags, args):
    table = ANSITable(
        Column("id", headalign="^", colalign=">"),
        Column("RMSE (pix)", headalign="^", colalign=">", fmt="{:.2f}"),
        Column("Rmax (pix)", headalign="^", colalign=">", fmt="{:.2f}"),
        Column("pose", headalign="^", colalign="<"),
        border="thin",
    )
    for tag in tags:
        # add row to table
        if args.rmax > 0 and tag.rmax >= args.rmax:
            bg = "red"
        else:
            bg = None
        table.row(
            tag.id,
            tag.rmse,
            tag.rmax,
            tag.pose.strline(),
            bgcolor=bg,
        )
    print(table)


def make_json(tags, args, camera):
    data = []
    for tag in tags:
        item = {
            "id": int(tag.id),
            "corners": [
                tuple(float(v) for v in col) for col in np.asarray(tag.corners).T
            ],
        }
        if camera is not None and getattr(tag, "pose", None) is not None:
            item["pose"] = np.asarray(tag.pose.A).tolist()
            item["rmse"] = tag.rmse
            item["rmax"] = tag.rmax
        data.append(item)
    # write to JSON file if specified
    # corners are a list of 2-tuples representing the (x, y) coordinates of each corner
    # in the image, and pose if present is a 4x4 list representing the homogeneous transformation
    # matrix of the tag's pose in the camera frame
    if args.json is not None:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def process_image(image, args, block):
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

        camera = CentralCamera(
            f=f, pp=pp, rho=args.rho if args.rho is not None else 1.0
        )
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

    if camera is not None:
        for tag in tags:
            # compute residuals
            p3d = tag.p3d.squeeze().T
            p2d = camera.project_point(p3d, objpose=tag.pose)
            resid = p2d - tag.corners
            rmse = np.sqrt(np.mean(resid**2))
            rmax = np.max(np.abs(resid))
            tag.rmse = rmse
            tag.rmax = rmax

    if camera is None:
        for i, tag in enumerate(tags):
            if i == 0:
                print("tag IDs:", tag.id, end="")
            else:
                print(f", {tag.id}", end="")
        print()
    elif not args.no_table:
        make_table(tags, args)

    if not args.no_display:
        make_plot(tags, args, camera, image)

    if args.json is not None:
        make_json(tags, args, camera)


def main():
    args = getargs()
    json_data = []

    if len(args.files) > 10:
        args.block = True

    for i, file in enumerate(args.files):
        try:
            img = Image.Read(file, rgb=False)
        except ValueError:
            print(f"File {file} not found")
            continue

        if i == len(args.files) - 1:
            # last one
            block = True
        else:
            block = args.block

        process_image(img, args, block)

        if not args.no_display:
            plt.show(block=block)


if __name__ == "__main__":
    main()
