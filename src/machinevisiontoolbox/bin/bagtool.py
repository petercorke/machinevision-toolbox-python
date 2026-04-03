#!/usr/bin/env python3
"""
Command-line tool to inspect and display images from ROS bag files.

Usage::

    $ bagtool mybag.bag
"""

import argparse
import atexit
import itertools
import os
import sys
import tempfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from colored import Fore, Style

from machinevisiontoolbox import (
    Image,
    PointCloud,
    RosBag,
    ImageSequence,
    PointCloudSequence,
)
from machinevisiontoolbox.bin._bintools import (
    CustomDefaultsHelpFormatter,
    MVTB_LINK,
)


def _download_bag(url: str, keep: bool) -> tuple[str, bool]:
    """
    Download a ROS bag file from *url*.

    :param url: HTTP or HTTPS URL to download.
    :param keep: if ``True``, save the file in the current directory using the
        remote filename; otherwise write to a temporary file.
    :return: ``(local_path, is_temp)`` — *local_path* is the path to the
        downloaded file, *is_temp* is ``True`` when the file is temporary.
    """
    remote_name = os.path.basename(urlparse(url).path) or "download.bag"

    if keep:
        local_path = remote_name
        if os.path.exists(local_path):
            print(
                f"Error: '{local_path}' already exists in the current directory. "
                "Remove it or run without --keep.",
                file=sys.stderr,
            )
            sys.exit(1)
        is_temp = False
    else:
        suffix = os.path.splitext(remote_name)[1] or ".bag"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        local_path = tmp.name
        tmp.close()
        is_temp = True

    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            open(local_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=remote_name,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    return local_path, is_temp


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=CustomDefaultsHelpFormatter,
        description=f"Display images or pointclouds from a ROS bag file using Machine Vision Toolbox for Python.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="list of ROS bag files to view.  URLs (http:// or https://) are also supported and will be downloaded before viewing, see --keep option below.",
    )

    msgtype = parser.add_mutually_exclusive_group()
    msgtype.add_argument(
        "-i",
        "--image",
        action="store_true",
        default=False,
        help="only display image messages (Image / CompressedImage), same as --msgfilter=Image",
    )
    msgtype.add_argument(
        "-p",
        "--pointcloud",
        action="store_true",
        default=False,
        help="only display point cloud messages (PointCloud2), same as --msgfilter=PointCloud2",
    )

    # -t, --topic only display images from the specified topic, this is a filter string and must appear in the name of the desired topic
    parser.add_argument(
        "-t",
        "--topic",
        metavar="FILTER",
        help="Only display messages from topics containing %(metavar)s",
        default=None,
    )
    # -m, --message only display messages of the specified type, this is a filter string and must appear in the message type, eg. 'image' to show only image messages
    parser.add_argument(
        "-m",
        "--message",
        metavar="FILTER",
        help="Only display messages of type containing %(metavar)s",
        default=None,
    )

    # -v, --view display images
    parser.add_argument(
        "-v", "--view", help="Display images in bag file", action="store_true"
    )
    # -l, --list only filtered messages in bag file, do not display images
    parser.add_argument(
        "-l", "--list", help="List topics in bag file", action="store_true"
    )

    parser.add_argument(
        "-b",
        "--block",
        action="store_true",
        default=False,
        help="block after each image",
    )

    # -a, --animate display images in bag file as an animation
    parser.add_argument(
        "-a", "--animate", help="Animate images in bag file", action="store_true"
    )

    # -g show grid
    parser.add_argument(
        "-g", "--grid", help="Overlay grid on images", action="store_true"
    )

    parser.add_argument(
        "--colororder",
        help="Override the default color order for the image messages",
    )

    parser.add_argument(
        "--dtype",
        help="Override the default data type for the image messages",
    )

    parser.add_argument(
        "-k",
        "--keep",
        action="store_true",
        default=False,
        help="when a file argument is a URL, save the downloaded bag in the current directory",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="disable the tqdm progress bar when scanning bag metadata",
    )

    return parser.parse_args()


def main():
    args = getargs()

    # Resolve any URL arguments to local files
    for i, f in enumerate(args.files):
        if f.startswith(("http://", "https://")):
            local, is_temp = _download_bag(f, args.keep)
            args.files[i] = local
            if is_temp:
                atexit.register(os.unlink, local)

    if not args.view and not args.animate and not args.list:
        for filename in args.files:
            bag = RosBag(filename, topicfilter=args.topic)
            print(f"{Fore.CYAN}{Style.BOLD}{bag}{Style.RESET}")
            bag.print(progress=not args.no_progress)
        return

    for filename in args.files:
        # Derive msgfilter from -i/-p flags when --message is not explicitly given
        if args.image:
            msgfilter = "Image"
        elif args.pointcloud:
            msgfilter = "PointCloud2"
        else:
            msgfilter = args.message

        bag = RosBag(
            filename,
            topicfilter=args.topic,
            msgfilter=msgfilter,
            colororder=args.colororder,
            dtype=args.dtype,
        )

        # Check for topic ambiguity when --topic is not specified (skip for --list)
        if args.topic is None and not args.list:
            all_topics = bag.topics()
            if msgfilter == "Image":
                relevant = {t: m for t, m in all_topics.items() if "Image" in m}
            elif msgfilter == "PointCloud2":
                relevant = {t: m for t, m in all_topics.items() if "PointCloud2" in m}
            else:
                relevant = {
                    t: m
                    for t, m in all_topics.items()
                    if "Image" in m or "PointCloud2" in m
                }
            if len(relevant) == 0:
                print(f"  {filename}: no matching topics found", file=sys.stderr)
                continue
            elif len(relevant) > 1:
                print(
                    f"  {filename}: multiple matching topics, use --topic to select one:",
                    file=sys.stderr,
                )
                for t, m in relevant.items():
                    print(f"    {t}  ({m})", file=sys.stderr)
                continue

        bag_iter = iter(bag)
        first = next(bag_iter, None)
        if first is None:
            print(f"No matching messages in {filename}", file=sys.stderr)
            continue
        frames = list(itertools.chain([first], bag_iter))

        if args.list:
            for frame in frames:
                ts = getattr(frame, "timestamp", None)
                prefix = RosBag.format_local_time(ts) if ts is not None else ""
                print(f"{prefix}  {frame}")
        elif args.pointcloud or (not args.image and not isinstance(first, Image)):
            PointCloudSequence(frames).disp(
                animate=args.animate,
                title=os.path.basename(filename),
            )
        else:
            ImageSequence(frames).disp(
                animate=args.animate,
                title=filename,
                grid=args.grid,
                badcolor="red",
            )


if __name__ == "__main__":
    main()
