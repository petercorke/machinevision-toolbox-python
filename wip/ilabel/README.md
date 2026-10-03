# ilabel (work in progress)

Pixel-based, rather than contour-based, region labelling / segmentation.
Based on the `ilabel` C extension written for the MATLAB Machine Vision
Toolbox (Copyright (C) 1995-2009, Peter I. Corke).

| File | What it is |
|---|---|
| `ilabel.c` | Cython 3.0.11 output (2024-12). Generated from `ilabel.pyx`, which is not in the repo; Cython embeds the source as `/* "ilabel.pyx":NN */` comments, so the `.pyx` can be reconstructed from this file. |
| `ilabel.py-unopt` | Unoptimised pure-Python port of the MATLAB/C code, produced with Copilot (2024-12). |

Not wired into the package yet. See the tech-debt issue for the plan, and the
`# TODO [l,ml,p,c] = ilabel(im);` in `tests/test_image_processing.py`.
