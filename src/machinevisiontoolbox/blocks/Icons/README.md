# Block icons

Icons for the bdsim blocks in this toolbox. They follow the conventions of
[bdsim's `Icons` folder](https://github.com/petercorke/bdsim/blob/main/src/bdsim/blocks/Icons/README.md):
250x250 RGBA PNG, black "ink" on a transparent background, named as the
lower-case version of the block's class name.

| Icon | How it is made |
|---|---|
| `visjac_p.png`, `estpose_p.png` | TeX math, rendered by `make_icons.py` (below). The files in git were made with `bdtex2icon` from bdsim, which needs LaTeX; `make_icons.py` makes the same glyphs, with slightly different size and placement. |
| `camera.png`, `imageplane.png` | not made by `make_icons.py` |

## Creating icons from TeX

```
python make_icons.py                # all icons in the table at the top of the script
python make_icons.py visjac_p       # one icon
python make_icons.py --outdir /tmp  # write somewhere else, e.g. to compare
```

This needs only matplotlib and Pillow, which this toolbox already depends on,
and no LaTeX installation. To add an icon, add a line to the `ICONS` table in
`make_icons.py`. The math is rendered by matplotlib's `mathtext`, which
implements a subset of TeX and does not know the custom macros of the RVC
notation file (`\mat`, `\pose`, ...), so write those expanded, for example
`\mathbf{J}_p` rather than `\mat{J}_p`.

Running the script overwrites the icons of the same name in this folder.
