# Pyodide install check

A real-Pyodide, real-WASM check that the JupyterLite install path actually works —
not a mock, not a guess. Runs Pyodide under Node.js (the `pyodide` npm package),
which is the same WASM runtime a real browser uses, just without a browser or a
Web Worker around it.

## What this catches

machinevision-toolbox-python 2.2.0 shipped a broken JupyterLite demo: `tqdm` was
added as a real dependency, but nobody added it to the notebook's install cell, and
nothing tested it before release. This check exists so that class of bug fails CI
instead of shipping — it runs the real `ensure_installed()` from
`docs/notebooks/_mvtb_nb_bootstrap.py`, installs every seed package for real, and
confirms `import machinevisiontoolbox` actually succeeds afterward.

It reuses that function unmodified rather than re-listing the seed packages here —
a second copy of that list would just be a second place for it to drift.

## What this deliberately does NOT catch

Whether the *real* notebook's relative-URL wheel-install path (`pypi/<wheel>`)
resolves correctly inside a real browser's Pyodide Web Worker. Node has no
equivalent of a Worker's base URL, and faking one convincingly requires already
knowing the right answer — that question needs a real browser, and was already
answered that way (build the site locally with `jupyter lite build` +
`jupyter lite serve`, open it in an actual browser tab — see
`docs/notebooks/README.md`). This check sidesteps that entirely: it mounts the
wheel directly into Pyodide's own virtual filesystem and installs from there
(`emfs:/<wheel>`, a micropip scheme that reads straight from Pyodide's filesystem,
no HTTP involved) via a narrow, clearly-commented monkeypatch in
`check_pyodide_install.mjs`. Deliberate, not a shortcut — see that file's own
docstring for the reasoning.

If you ever suspect the relative-URL path itself is broken, this check will not
tell you — go build and serve the real site and check it in a real browser instead.

## Usage

```bash
python -m build --wheel --outdir docs/lite/pypi
npm --prefix docs/lite/wasm_check install   # once per checkout, or after a version bump
node docs/lite/wasm_check/check_pyodide_install.mjs docs/lite/pypi/*.whl
```

Exits 0 and prints `PASS: Pyodide install check succeeded.` on success. On failure,
prints a diagnostic block explaining the likely cause (almost always: a new
dependency needs adding to `_mvtb_nb_bootstrap.py`'s seed list) and how to fix it,
followed by the real traceback.

## Known flakiness

Each run installs ~35 packages fresh from a CDN (jsdelivr). An occasional network
hiccup can produce a failure (e.g. `AbortError`) unrelated to any real regression —
if a CI run fails here, a straight re-run is a reasonable first move before
assuming something's actually broken. `pyodide`'s own local package cache
(`node_modules/.cache` after the first run) makes repeated local runs faster and
more reliable than the first one.

## Version pinning

`package.json` pins the `pyodide` npm package to `0.27.6`, matching
`docs.yml`'s deliberate `jupyterlite-pyodide-kernel==0.6.1` pin (see that
workflow's own comment for the full reasoning: newer kernel releases bundle a
JSPI-dependent Pyodide that either crashes or silently races on browsers
without JSPI support, e.g. Safari as of 2026-08). These two pins have to move
together — if one changes without the other, this check silently stops
representing what the real deployed site actually runs, which could let it
pass while the real site fails, or vice versa. Once that kernel pin is
revisited, this one needs updating in the same change.

## Reusing this for another toolbox

This is meant to be a template — RTB, bdsim, etc. can copy this pattern for their
own JupyterLite setups (RTB's is compiled-extension-aware, per its own pyodide-wheel
notes, but the check-a-real-install-path idea transfers directly). Adjust the
`ensure_installed()` import path and the seed-package assumptions; the `emfs:`
redirect trick and the diagnostic-message design should carry over unchanged.
