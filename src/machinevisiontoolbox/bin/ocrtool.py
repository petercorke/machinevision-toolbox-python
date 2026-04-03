#!/usr/bin/env python3
"""
Command-line tool to perform optical character recognition (OCR) on images.

Usage::

    $ ocrtool text.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from machinevisiontoolbox import Image
from machinevisiontoolbox.bin._bintools import CustomDefaultsHelpFormatter


def getargs():
    parser = argparse.ArgumentParser(
        formatter_class=CustomDefaultsHelpFormatter,
        description=f"Display text words found in image using Machine Vision Toolbox for Python.  "
        "Words are written to stdout or a JSON file, but can also be highlighted in the image.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="list of image files to view, files can also include those distributed with machinevision toolbox, eg. '.png'",
    )
    # -L, --light text on dark backround (default), -D, --dark unsets light mode
    light_group = parser.add_mutually_exclusive_group()
    light_group.add_argument(
        "-L",
        "--lightbg",
        help="Look for light background with dark text (default)",
        action="store_true",
    )
    light_group.add_argument(
        "-D",
        "--darkbg",
        help="Look for dark background with light text",
        action="store_true",
    )

    # -c, --confidence minimum confidence for OCR text to be displayed (%)
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=50.0,
        help="Minimum confidence for OCR text to be displayed (%%)",
    )

    # -l, --long long listing (include bounding box coordinates and confidence in output)
    parser.add_argument(
        "-l",
        "--long",
        help="Long listing (include bounding box coordinates and confidence in output)",
        action="store_true",
    )
    # -j, --json output as JSON instead of plain text, has filename
    parser.add_argument(
        "-j",
        "--json",
        metavar="FILE",
        help="Output results in JSON format to FILE: word, confidence, LTRB bounding box coordinates, and dimensions",
        default=None,
    )

    parser.add_argument(
        "-v",
        "--view",
        help="Overlay recognised word boxes on image",
        action="store_true",
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
    return parser.parse_args()


def main():
    args = getargs()

    if len(args.files) > 10:
        args.block = True

    for i, file in enumerate(args.files):
        try:
            img = Image.Read(file, rgb=False)
        except ValueError:
            print(f"File {file} not found")
            continue

        # Pytesseract (and Tesseract OCR) works best with dark text on a light
        # background. While Tesseract can internally process dark backgrounds (inverted
        # images), for optimal accuracy and to avoid errors, it is highly recommended to
        # preprocess images to be black text on a white background.
        if args.darkbg:
            img = img.invert()

        if args.view:
            img.disp(title=file)
        words = img.ocr(minconf=args.confidence, plot=args.view)
        if args.grid:
            plt.grid()

        if args.long:
            c = np.array([word.conf for word in words])
            print(
                f"# {file}: {len(words)} words; confidence: {c.min():.1f} - {c.max():.1f}%, mean {c.mean():.1f}%"
            )

        if args.json is not None:
            import json

            output = []
            for word in words:
                output.append(
                    {
                        "text": word.text,
                        "conf": word.conf,
                        "ltrb": word.ltrb,
                        "wh": [word.w, word.h],
                    }
                )
            with open(args.json, "w") as f:
                json.dump(output, f, indent=2)

        # print the words to stdout, one per line, with optional confidence and bounding box if long listing is enabled
        if args.long:
            print("# word, confidence, (left, top), width x height")

        for word in words:
            if args.long:
                print(
                    f"{word.text}, {word.conf:.1f}%, ({word.l}, {word.t}), {word.w}x{word.h}"
                )
            else:
                print(f"{word.text}")

        if i == len(args.files) - 1:
            # last one
            plt.show(block=True)
        else:
            plt.show(block=args.block)


if __name__ == "__main__":
    main()
