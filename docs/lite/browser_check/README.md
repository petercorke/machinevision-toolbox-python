# JupyterLite browser check

Opens the JupyterLite site in real browsers (Playwright: Chromium and WebKit, the
engine behind Safari), waits for the Pyodide kernel, runs `intro.ipynb`'s bootstrap
cell, and requires its `Running in browser using MVTB v...` line.  It also fails if
JupyterLab itself does not start (`Plugin '...' failed to activate`), which happens
before any kernel exists.

It complements `../wasm_check/`, which tests the dependency install in Node without
a browser or JupyterLab.

| Where | When |
|---|---|
| `docs.yml`, after `jupyter lite build` | every publish (`push` to main, `workflow_dispatch`), not pull requests; blocks the deploy on failure |
| `lite-smoke.yml` | after each Docs Publish on main, weekly, or manually; checks the live site and can only alert |

Run it yourself:

```bash
pip install playwright && playwright install chromium webkit
python docs/lite/browser_check/check_lite_in_browser.py \
    --base-url https://petercorke.github.io/machinevision-toolbox-python/lite
```

What it does **not** reproduce: your own browser's stored state.  A browser holding
files from an earlier deploy can fail where a fresh Playwright context passes; see
"Troubleshooting JupyterLite" in the docs.
