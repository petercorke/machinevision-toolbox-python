#!/usr/bin/env python
"""
Generate the stripe test fixture images committed to ``tests/data/``.

Run to regenerate the fixtures if they are ever lost or need to be recreated
from scratch::

    $ python tests/data/make_stripes.py
    $ python tests/data/make_stripes.py --outdir /tmp/stripes   # elsewhere

Images created:

``stripes.png``
    RGB, 1000 × 400 px, uint8.  Five vertical stripes each 200 px wide:
    black | red | green | blue | white

``stripes_a.png``
    RGBA, 1000 × 400 px, uint8.  Same stripes as ``stripes.png`` with a fully
    opaque alpha plane (alpha = 255 throughout).

These images are used by:

- ``tests/base/test_io.py``  -- ``TestBaseStripeIO`` (``iread`` / ``idisp``)
- ``tests/test_image_io.py`` -- ``TestStripeImageIO`` (``Image.Read`` / ``Image.disp``)

The images are written with OpenCV directly (not via ``iwrite``) so they are
independent of any MVTB I/O logic.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

HEIGHT = 400
WIDTH = 1000
STRIPE_W = 200

# Stripes in BGR order (OpenCV native write order)
STRIPES_BGR = [
    (0, 0, 0),  # black
    (0, 0, 255),  # red   (B=0, G=0, R=255 in BGR)
    (0, 255, 0),  # green
    (255, 0, 0),  # blue  (B=255, G=0, R=0 in BGR)
    (255, 255, 255),  # white
]


def make_stripes(out_dir: Path) -> None:
    """Write ``stripes.png`` and ``stripes_a.png`` into ``out_dir``.

    :param out_dir: directory to write the images to, created if necessary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- stripes.png (BGR → written as RGB by PNG codec) -------------------
    im_bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i, bgr in enumerate(STRIPES_BGR):
        im_bgr[:, i * STRIPE_W : (i + 1) * STRIPE_W] = bgr

    path_rgb = out_dir / "stripes.png"
    cv2.imwrite(str(path_rgb), im_bgr)
    print(f"Written: {path_rgb}")

    # --- stripes_a.png (BGRA) ----------------------------------------------
    im_bgra = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    im_bgra[:, :, :3] = im_bgr
    im_bgra[:, :, 3] = 255  # fully opaque

    path_rgba = out_dir / "stripes_a.png"
    cv2.imwrite(str(path_rgba), im_bgra)
    print(f"Written: {path_rgba}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory, defaults to the directory containing this script",
    )
    make_stripes(parser.parse_args().outdir)
