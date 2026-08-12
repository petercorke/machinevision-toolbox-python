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

`package.json` pins the `pyodide` npm package to the same version confirmed
working in real-browser testing (`314.0.3`, corresponding to the Pyodide runtime
that `jupyterlite-pyodide-kernel` currently bundles). If that drifts out of sync
with what the real deployed site actually uses, this check could pass while the
real site fails, or vice versa — see the toolbox-maintainer notes on JupyterLite
version pairing for why that pairing matters and isn't self-healing.

## Reusing this for another toolbox

This is meant to be a template — RTB, bdsim, etc. can copy this pattern for their
own JupyterLite setups (RTB's is compiled-extension-aware, per its own pyodide-wheel
notes, but the check-a-real-install-path idea transfers directly). Adjust the
`ensure_installed()` import path and the seed-package assumptions; the `emfs:`
redirect trick and the diagnostic-message design should carry over unchanged.
