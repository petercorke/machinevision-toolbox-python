#!/usr/bin/env node
/**
 * Dependency-completeness check for the Pyodide/JupyterLite install path.
 *
 * Scope, deliberately narrow: this confirms that every package
 * _mvtb_nb_bootstrap.py's Pyodide branch installs actually resolves, and
 * that `import machinevisiontoolbox` succeeds afterward -- catching exactly
 * the class of bug that shipped in machinevision-toolbox-python 2.2.0 (a
 * new pyproject.toml dependency added but never added to the Pyodide
 * seed-install list, breaking the JupyterLite demo silently).
 *
 * What this deliberately does NOT test: whether the real notebook's
 * relative-URL wheel-install path resolves correctly inside a real
 * browser's Pyodide Web Worker. Node has no equivalent of a Worker's base
 * URL, and faking one convincingly requires already knowing the right
 * answer -- that question needs a real browser, and was already answered
 * that way (see docs/notebooks/README.md). This check mounts the wheel
 * directly into Pyodide's own virtual filesystem instead (the `emfs:`
 * micropip scheme) and installs from there, sidestepping the HTTP/URL
 * layer entirely -- deliberately, not as a shortcut around it.
 *
 * Reuses the real, unmodified ensure_installed() from _mvtb_nb_bootstrap.py
 * rather than re-listing the seed packages here -- that would just be a
 * second, driftable copy of the exact list this check exists to keep
 * honest.
 *
 * Usage:
 *   node check_pyodide_install.mjs <path-to-wheel>
 */

import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";

const wheelPath = process.argv[2];
if (!wheelPath) {
  console.error("Usage: node check_pyodide_install.mjs <path-to-wheel>");
  process.exit(2);
}

const wheelName = path.basename(wheelPath);
const bootstrapPath = new URL("../../notebooks/_mvtb_nb_bootstrap.py", import.meta.url);

const pyodide = await loadPyodide();
await pyodide.loadPackage("micropip");

pyodide.FS.writeFile("/" + wheelName, fs.readFileSync(wheelPath));
pyodide.FS.writeFile("/_mvtb_nb_bootstrap.py", fs.readFileSync(bootstrapPath));

const driver = `
import sys
sys.path.insert(0, "/")

import micropip
import _mvtb_nb_bootstrap as bootstrap

# Test-only scaffolding: redirect the one HTTP/relative-URL install call to
# the wheel already mounted in Pyodide's own virtual filesystem, so this
# check doesn't need to fake a browser Web Worker's base URL (see this
# file's module docstring for why that's out of scope here). Everything
# else -- the seed-package installs, which is what this check actually
# exists to catch drift in -- runs completely unmodified.
_real_install = micropip.install
async def _redirect_wheel_install(requirements, **kwargs):
    if isinstance(requirements, str) and requirements.startswith("pypi/"):
        requirements = "emfs:/${wheelName}"
    return await _real_install(requirements, **kwargs)
micropip.install = _redirect_wheel_install

await bootstrap.ensure_installed()

from machinevisiontoolbox import Image  # noqa: F401
print("import machinevisiontoolbox: OK")
`;

try {
  await pyodide.runPythonAsync(driver);
  console.log("\nPASS: Pyodide install check succeeded.");
} catch (err) {
  console.error(`
============================================================
FAIL: the Pyodide/JupyterLite install check failed.

This means _mvtb_nb_bootstrap.py's Pyodide branch doesn't actually
install and import cleanly. Most likely cause: a new dependency was
added to pyproject.toml (or a notebook now needs a new package) without
being added to the seed-install list in _mvtb_nb_bootstrap.py -- the
same class of bug that broke the JupyterLite demo in 2.2.0.

To reproduce locally:
    python -m build --wheel --outdir docs/lite/pypi
    node docs/lite/wasm_check/check_pyodide_install.mjs docs/lite/pypi/*.whl

To fix a missing-dependency error specifically:
  1. Add the missing package name to the micropip.install([...]) list
     near the top of docs/notebooks/_mvtb_nb_bootstrap.py.
  2. Run: python docs/notebooks/sync_bootstrap.py
  3. Commit the regenerated notebooks along with the bootstrap file change.

If the error isn't a missing package, it may be a real Pyodide
compatibility problem with one of the dependencies -- not something this
script can fix for you, but the traceback below should say which package.

Original error below.
============================================================
`);
  console.error(err);
  process.exitCode = 1;
}
