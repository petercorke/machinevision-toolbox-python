#!/usr/bin/env python3
"""
Set the width attribute of the logo <img> tag to 200 in a Jupyter notebook.

The logo is identified by its filename ``VisionToolboxLogo``.

Usage::

    $ set_logo_width.py <filename> [<width>]

width defaults to 200.
"""

import json
import re
import sys


def set_logo_width(path: str, width: int = 200) -> None:
    with open(path) as f:
        nb = json.load(f)

    pattern = re.compile(
        r'(<img\s[^>]*VisionToolboxLogo[^>]*\s)width="[^"]*"',
        re.IGNORECASE,
    )
    replacement = rf'\1width="{width}"'

    changed = False
    for cell in nb["cells"]:
        src_lines = cell.get("source", [])
        src = "".join(src_lines)
        if "VisionToolboxLogo" in src:
            new_src, n = pattern.subn(replacement, src)
            if n:
                cell["source"] = new_src.splitlines(keepends=True)
                changed = True

    if not changed:
        print(f"No logo <img> found in {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Set logo width to {width} in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file> [<width>]", file=sys.stderr)
        sys.exit(1)
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    set_logo_width(sys.argv[1], w)
