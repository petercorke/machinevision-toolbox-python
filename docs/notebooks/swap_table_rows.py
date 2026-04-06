#!/usr/bin/env python3
"""
Swap the two <td> columns in the first <tr> of the first <table> in a file.

Handles both plain HTML/text files and Jupyter notebook (.ipynb) files.
For notebooks the cell source list is parsed via the ``json`` module so
escaped characters are not an issue.

Usage::

    $ swap_table_rows.py <filename>
"""

import json
import re
import sys


_PATTERN = re.compile(
    r"(<tr[^>]*>\s*)"       # opening <tr> + leading whitespace  → group 1
    r"(<td.*?</td>)"        # first  <td>…</td>                  → group 2
    r"(\s*)"                # whitespace between the two cells    → group 3
    r"(<td.*?</td>)"        # second <td>…</td>                  → group 4
    r"(\s*</tr>)",          # trailing whitespace + </tr>         → group 5
    re.DOTALL,
)


def _swap(m: re.Match) -> str:
    return m.group(1) + m.group(4) + m.group(3) + m.group(2) + m.group(5)


def _swap_in_text(text: str, path: str) -> str:
    new_text, n = _PATTERN.subn(_swap, text, count=1)
    if n == 0:
        print(f"No two-column <tr> found in {path}", file=sys.stderr)
        sys.exit(1)
    return new_text


def swap_first_table_columns(path: str) -> None:
    if path.endswith(".ipynb"):
        with open(path) as f:
            nb = json.load(f)

        swapped = False
        for cell in nb["cells"]:
            src_lines = cell.get("source", [])
            src = "".join(src_lines)
            if not swapped and "<table>" in src:
                new_src = _swap_in_text(src, path)
                # Preserve the list-of-strings format; split on newlines,
                # re-adding the \n that join() consumed.
                lines = new_src.splitlines(keepends=True)
                cell["source"] = lines
                swapped = True
                break

        if not swapped:
            print(f"No <table> cell found in {path}", file=sys.stderr)
            sys.exit(1)

        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    else:
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(_swap_in_text(text, path))

    print(f"Swapped table columns in {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file>", file=sys.stderr)
        sys.exit(1)
    swap_first_table_columns(sys.argv[1])
