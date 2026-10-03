#!/usr/bin/env python
"""
Create block icons from TeX math, using only matplotlib and Pillow.

Run it from anywhere to (re)create the icons listed in :data:`ICONS`::

    $ python make_icons.py                     # all icons, written next to this script
    $ python make_icons.py visjac_p            # just one
    $ python make_icons.py --outdir /tmp/icons # somewhere else

Each icon is a 250x250 RGBA PNG: black "ink" on a transparent background, the
size and style used by the bdsim block icons (see the README in this folder).

The math is rendered by matplotlib's ``mathtext`` engine with the Computer
Modern font set, so no LaTeX installation is needed.  ``mathtext`` implements a
subset of TeX, and not the custom macros (``\\mat``, ``\\pose``, ...) from the
``rvc-notation`` file that ``bdtex2icon`` (part of bdsim) uses, so the entries
in :data:`ICONS` are written with those macros expanded.
"""

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import mathtext, rc_context
from PIL import Image

IMSIZE = 250

#: icon file stem -> TeX math (without the surrounding ``$``)
ICONS: dict[str, str] = {
    "visjac_p": r"\mathbf{J}_p",
    "estpose_p": r"\xi(p, P)",
}


def tex_icon(
    tex: str, path: Path, size: int = IMSIZE, margin: float = 0.08, dpi: int = 600
) -> None:
    """Render TeX math to a square, transparent, black-ink PNG icon.

    :param tex: TeX math expression, without the surrounding ``$``
    :param path: output file
    :param size: width and height of the icon in pixels, defaults to 250
    :param margin: blank border as a fraction of ``size``, on each side,
        defaults to 0.08
    :param dpi: resolution of the intermediate rendering, defaults to 600

    The expression is rendered, cropped to its ink, scaled to fit inside the
    margin, and centered on a transparent canvas.
    """
    buf = io.BytesIO()
    with rc_context({"mathtext.fontset": "cm"}):  # Computer Modern, as LaTeX
        mathtext.math_to_image(f"${tex}$", buf, dpi=dpi, format="png", color="black")

    # math_to_image saves a figure, so it is black ink on an *opaque white*
    # background.  The ink is black, so how dark a pixel is gives its coverage,
    # and that is all the icon needs: black everywhere, with this as the alpha.
    # Working on one channel also avoids mixing colors into anti-aliased edges.
    alpha = 255 - np.asarray(Image.open(buf).convert("L"))
    ys, xs = np.nonzero(alpha)
    ink = Image.fromarray(alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])

    scale = size * (1 - 2 * margin) / max(ink.size)
    ink = ink.resize(
        (max(1, round(ink.width * scale)), max(1, round(ink.height * scale))),
        Image.LANCZOS,
    )

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(ink, ((size - ink.width) // 2, (size - ink.height) // 2))

    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.putalpha(canvas)
    icon.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help=f"icons to create, one or more of {', '.join(ICONS)}; default is all",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory, defaults to the directory containing this script",
    )
    args = parser.parse_args()

    unknown = [n for n in args.names if n not in ICONS]
    if unknown:
        parser.error(f"unknown icon(s) {unknown}, choose from {list(ICONS)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in args.names or ICONS:
        path = args.outdir / f"{name}.png"
        tex_icon(ICONS[name], path)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
