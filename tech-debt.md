# Technical Debt

## `Histogram.plot`'s `type` docstring doesn't match its actual accepted values

Found 2026-07-30 while adding type annotations to `Histogram.plot`
(`ImageWholeFeatures.py:1200`). The docstring says `type` accepts
`'frequency'` [default], `'cdf'`, or `'ncdf'`. The actual dispatch logic
in the method body accepts a different, larger set:
`'frequency'`, `'pdf'`/`'probability'`, `'cf'`/`'cumulative'`,
`'cdf'`/`'normalized'` — and does **not** handle `'ncdf'` at all (it
would fall through to the `else: raise ValueError("unknown type")`
branch). Left `type` annotated as plain `str` rather than a `Literal`
enum for this reason — using `Literal` would mean either copying the
stale docstring's wrong values or silently fixing behavior/docs as a
drive-by, both out of scope for an annotations-only pass.

### Fix

Reconcile the docstring with the real accepted values (or vice versa,
if `'ncdf'` was meant to work and was dropped by accident — check git
blame). Once settled, `type` can become
`Literal["frequency", "pdf", "probability", "cf", "cumulative", "cdf", "normalized"]`.

## [HIGH PRIORITY] mypy is not run anywhere in CI or dev tooling

Found 2026-07-29 while fixing the `_ImageBase` Protocol gaps in
`ImageBlobs.py`/`ImageRegionFeatures.py`/`ImagePointFeatures.py`
(see git history — three mixins were missing the
`class XMixin(_ImageBase if TYPE_CHECKING else object)` pattern every
other mixin uses). That fix prompted the question: if a new method is
added to a mixin and it's accessed via `self.` from a *different*
mixin, and it's not yet declared in `_image_typing.py`'s `_ImageBase`
Protocol, what actually catches that? Answer: **nothing, currently**.
`mypy` is not in `pyproject.toml`'s `dev` extra and does not run in any
`.github/workflows/*.yml` — confirmed by grep, zero hits. So a Protocol
gap like this doesn't fail a build or even show a warning; it just
silently produces incomplete/wrong type information for anyone using an
editor with type-checking (Pylance, mypy in an IDE), with no automated
signal anywhere. This is the quietest version of a pattern that's
already bitten this project twice today in more visible forms (the
`image_class.rst` autosummary list silently going stale for the whole
`Image` sidebar, and `ci.yml`'s conda `create-args` list silently
missing `pgraph-python`/drifting on `opencv`) — a hand-maintained
shadow list with no automated check that it stays in sync with reality.

**Ran `mypy src/machinevisiontoolbox --ignore-missing-imports` fresh,
2026-07-29** (superseding the stale, less-categorized April audit in
`NOTES`): **524 errors in 31 files** (checked 49 source files). By
category:

| Code | Count |
|---|---|
| `attr-defined` | 84 |
| `assignment` | 82 |
| `union-attr` | 76 |
| `index` | 75 |
| `arg-type` | 53 |
| `misc` | 26 |
| `name-defined` | 24 |
| `var-annotated` | 18 |
| `valid-type` | 15 |
| `operator` | 15 |
| `call-overload` | 14 |
| `return-value` | 13 |
| `has-type` | 11 |
| `return` | 6 |
| `no-redef` | 5 |
| `override` | 4 |
| `method-assign` | 2 |
| `call-arg` | 1 |

**Correcting an initial hypothesis**: expected most `attr-defined`
errors to trace to incomplete `_ImageBase` coverage (only ~96 of
`Image`'s ~290 public members are declared — by design, since the
Protocol only needs to cover attributes actually cross-referenced
between mixins, not the full public API). Checked the real breakdown
instead of assuming: **none** of the current `attr-defined` errors are
actually `_ImageBase` gaps. The two real dominant causes are unrelated:
- 23 of 84: `machinevisiontoolbox/base/__init__.py` re-exports every
  submodule via wildcard `from X import *` (9 submodules) with no
  explicit `__all__`/direct re-export list; mypy can't reliably resolve
  names through that chain, so every file that does
  `from machinevisiontoolbox.base import (draw_circle, plot_labelbox,
  findpeaks2d, ...)` gets a false "module has no attribute" even though
  these work fine at runtime.
- 8 of 84: `cv2.<X>_create` dynamic dispatch (`getattr(cv2, ...)` /
  `getattr(cv2.xfeatures2d, ...)` patterns in `ImagePointFeatures.py`'s
  feature-detector dict) — mypy can't type-check dynamic attribute
  access, expected and low-value to fix.
- The remaining ~53 are scattered; `VisualServo.py` alone accounts for
  44 of the 84 `attr-defined` errors (a mix of real typos like
  `"plotpose"; maybe "plot_pose"?` and missing `machinevisiontoolbox.base`
  attributes via the same wildcard-import issue) and is worth its own
  look independent of the mixin-Protocol question that prompted this
  audit.

The `_ImageBase`-completeness risk described above is still real, just
currently *latent* rather than demonstrated by a live error — worth
re-running this same `mypy` audit after any future mixin refactor to
catch it if it does start manifesting.

### Fix

Two independent pieces, roughly in priority order:
1. Fix `machinevisiontoolbox/base/__init__.py`'s wildcard re-exports
   (add explicit `__all__` composed from each submodule's own `__all__`,
   or switch to explicit `from X import (name1, name2, ...)` — either
   should immediately clear ~23+ of the `attr-defined` count and is a
   mechanical, low-risk change).
2. Wire `mypy` into CI (even just as a non-blocking/advisory job at
   first, given 524 existing errors) so future drift is visible instead
   of silent. Add `mypy` to `pyproject.toml`'s `dev` extra either way.
Do not attempt to fix all 524 errors in one pass — triage by category
(the `[call-arg]`/`[override]`/`[return]` categories are more likely to
be real bugs; `assignment`/`arg-type`/`index` are more likely the
`ArrayLike`-union-too-broad pattern the April `NOTES` audit already
identified).

## `Image.ncdf` is documented as deprecated but never warns

`ncdf` (`src/machinevisiontoolbox/ImageWholeFeatures.py:511-523`) has a
`.. deprecated:: 2.0.0` docstring note pointing at `hist().cdf`, but unlike
every other deprecated method/property in this codebase (`rank`, `image`,
`A`, `to_int`, `to_float`, `thresh`, `ithresh`, `adaptive_threshold`,
`column`, etc. — all of which call `warnings.warn(..., DeprecationWarning,
stacklevel=2)`), `ncdf`'s body just returns `hist.cdf` directly with no
warning call. Callers get no runtime signal that they're using a
deprecated API.

Found 2026-07-29 while auditing why `rankfilter()` was missing from the
Sphinx sidebar (unrelated bug, since fixed — see git history for
`docs/source/image_class.rst`, which also added a "Deprecated aliases"
section that now gives `ncdf` its own docs page like every other
deprecated alias). So this is purely a runtime-warning gap, not a docs
bug.

### Fix

Add the standard warning to `ncdf`, matching its sibling `cdf` property's
migration note:

```python
warnings.warn(
    "Deprecated in 2.0.0: use hist().cdf instead of ncdf.",
    DeprecationWarning,
    stacklevel=2,
)
```

This is a real code change (not docs-only), so bundle it with a `fix:`
commit when picked up rather than folding it into a docs-only change.

## `docs/requirements.txt` pinned `sphinx-codeautolink` to an unmerged branch

Added 2026-07-29: `docs/requirements.txt` pinned
`sphinx-codeautolink @ git+https://github.com/petercorke/sphinx-codeautolink.git@support-typing-self`
instead of the stock PyPI release, to get `typing.Self` return-annotation
resolution (needed for cross-linking chained method calls like
`Image.Random(...).print()`) ahead of upstream PR
[felix-hilden/sphinx-codeautolink#202](https://github.com/felix-hilden/sphinx-codeautolink/pull/202)
merging.

### Resolved 2026-07-29

PR #202 merged upstream and shipped in the `sphinx-codeautolink` 0.19.0
PyPI release the same day. Verified directly (downloaded and inspected
the 0.19.0 wheel, confirmed the `Self`-handling code, rebuilt the docs
against it — chained calls now cross-link correctly). Switched
`pyproject.toml`'s `docs` extra to `sphinx-codeautolink>=0.19.0` and
dropped the git-branch override from `docs/requirements.txt` entirely.

## GitHub Actions versions are stale across most workflows

Audited 2026-07-29 (prompted by a similar finding in another toolbox
repo). `docs.yml`'s actions have been bumped to current majors as part of
the same change that fixed the `Image` sidebar bug (see git history), but
the rest of `.github/workflows/` was deliberately left alone — bumping
`release.yml` touches the real PyPI publish pipeline and deserves its own
careful pass (per the release-safety rule in
`~/.claude/CLAUDE.md`: verify the actual release workflow file, don't
just bump and hope), not a drive-by alongside a docs fix.

| Action | Pinned | Latest (2026-07-29) | Where | Gap |
|---|---|---|---|---|
| `actions/download-artifact` | v4 | v8 | `release.yml` | 4 majors |
| `actions/upload-artifact` | v4 | v7 | `release.yml` | 3 majors |
| `actions/checkout` | v6 | v7 | `ci.yml`, `release.yml` | 1 major |
| `actions/setup-python` | v6 | v7 | `release.yml` | 1 major |
| `googleapis/release-please-action` | v4 | v5 | `release-please.yml` | 1 major |
| `amannn/action-semantic-pull-request` | v5 | v6 | `commitlint.yml` | 1 major |
| `mamba-org/setup-micromamba` | v2 | v3 | `ci.yml` | 1 major |
| `codecov/codecov-action` | v6 | v7 | `ci.yml` | 1 major |
| `pypa/gh-action-pypi-publish` | `release/v1` | — | `release.yml` | none — floating tag, already tracks latest v1.x |

`download-artifact` and `upload-artifact` are the standouts — 3-4 majors
behind, both used in `release.yml`'s build→publish artifact handoff. Most
of these `actions/*` majors turned out to be low-risk (mainly Node.js
runtime bumps: v24 requires Actions Runner ≥ v2.327.1, a non-issue on
GitHub-hosted runners), confirmed while bumping `docs.yml`, but
`download-artifact`/`upload-artifact` v4→v7/v8 haven't been checked for
breaking input/output changes yet — do that before bumping `release.yml`.

### Fix

For each remaining workflow file, check that action's release notes
between the pinned and latest major for actual breaking changes (not just
Node runtime bumps), then bump. Do `release.yml` last and most carefully
— it's the one that actually publishes to PyPI. Re-run
`.github/workflows/ci.yml` on a real PR after bumping it, since it's the
main test gate.

## `ci.yml` uses conda/micromamba; every sibling toolbox uses plain pip

Observed 2026-07-29, prompted directly by the opencv5 pin incident above
(a conda-forge-specific dependency-drift failure that plain pip installs
wouldn't have hit the same way, since `pyproject.toml`'s own
`opencv-python<5.0.0` / `opencv-contrib-python<5.0.0` pins would have
been honoured). `machinevision-toolbox-python/.github/workflows/ci.yml`
is the only one of Peter's toolbox CI configs that uses
`mamba-org/setup-micromamba` + a hand-maintained `create-args` package
list. Checked directly:

| Repo | CI setup |
|---|---|
| robotics-toolbox-python | `actions/setup-python` + `pip install .[dev]` |
| bdsim | `actions/setup-python` + `pip install .[dev,bdedit]` |
| spatialmath-python | `actions/setup-python` + `pip install .[dev]` |
| **machinevision-toolbox-python** | **`mamba-org/setup-micromamba` + `create-args` package list** |

This is a genuine outlier, apparently introduced by a conda-preferring
contributor at some point, not a deliberate MVTB-specific technical
requirement (MVTB's own `pyproject.toml` dependencies are ordinary PyPI
packages — `opencv-python`, `opencv-contrib-python`, etc. — nothing here
actually needs conda). Consequences of the mismatch, beyond one-off
annoyance:

- Dependency pins in `pyproject.toml` (the pip-installable, publishable
  package spec) don't apply to CI at all, since `ci.yml` never runs `pip
  install .` against the conda env's packages the normal way — it
  pre-installs everything via `create-args`, then does
  `pip install .[dev] --no-deps --no-build-isolation` (explicitly
  `--no-deps`, so pip's own resolver never even sees the pins). That's
  exactly how CI silently drifted onto conda-forge's opencv 5.0.0 despite
  `pyproject.toml` saying `<5.0.0` — see the opencv5 entry above.
- Doubles the maintenance surface for CI dependency changes: an
  `environment.yml`-style `create-args` list to keep in sync with
  `pyproject.toml`'s `dependencies`/`docs`/`dev` extras by hand, instead
  of one source of truth.
- Inconsistent with every sibling repo, so fixes/conventions that get
  worked out on RTB/bdsim/SMTB's CI don't transfer here without
  translation, and vice versa (see `~/.claude/toolbox-infrastructure.md`'s
  shared-infrastructure convention).

**Second live example, 2026-07-29**: PR #24 (`feat/tools-extra`, adding a
new `tool` extra to `pyproject.toml` for optional `IPython`/`pygments`
support) failed CI with `mvtbtool requires IPython and pygments, which
are not installed (No module named 'IPython')` — because `ci.yml`'s
`create-args` package list was never updated to include them. A plain
`pip install .[dev,tool]`-style CI setup would have picked up the new
extra automatically; the hand-maintained conda list requires a manual,
easy-to-forget edit in a second place every time `pyproject.toml` gains a
new extra or dependency. Patched directly (added `ipython`/`pygments` to
`create-args`) and merged 2026-07-29 — the underlying architectural gap
below is still open.

**Third example, found 2026-07-29 (pre-existing, not a new incident)**:
`pgraph-python` is a real, declared dependency in `pyproject.toml`'s
`dependencies` list, but is completely absent from `ci.yml`'s
`create-args` — not merely unpinned, never installed at all. The only
file that imports it, `BundleAdjust.py`, does so inside a bare
`except:` (see the bare-except finding elsewhere in this file) that
silently falls back when the import fails. So every CI run has been
exercising `BundleAdjust`'s pgraph-dependent code in "not installed"
fallback mode this whole time, with nothing surfacing it. Not fixed
here — decided (2026-07-29) to leave the conda→pip migration itself as
deferred tech debt rather than keep patching individual missing
packages one at a time; noting this one so it's not lost.

**Decision, 2026-07-29**: known who introduced the conda/micromamba
setup and why (a well-intentioned contributor's preference for conda,
not a technical requirement) — confirmed not urgent enough to fix now.
Branch protection on `main` (`All tests passed` + `build` required,
strict/up-to-date-with-base required) now contains the *blast radius*
of a future incident like this — it'll block just the one PR that hits
it, rather than silently landing on `main` and confusing every other
open PR at once, which was the actually painful part of the opencv5
incident. The root architectural gap remains real and will very likely
surface again the same way (a fourth missing/drifted package), but
that's an acceptable trade for now given the fix requires careful
cross-OS testing (libegl/Xvfb handling, `matplotlib-base` vs
`matplotlib` naming) rather than a quick patch.

### Fix

Normalize to the same `actions/setup-python` + `pip install .[dev]`
pattern the other three repos use, dropping `mamba-org/setup-micromamba`
entirely. Before doing so, check *why* conda was introduced here in the
first place — search git blame/log on `ci.yml` for context — in case
there's a real reason (e.g. a native dependency that's painful via pip
on some platform) rather than just contributor preference. If no real
reason turns up, this is a straightforward rip-and-replace: swap the
`mamba-org/setup-micromamba` step for `actions/setup-python`, replace
`create-args` with `pip install .[dev]` (defining a `dev` extra in
`pyproject.toml` if one doesn't already exist, matching RTB/bdsim), and
drop the separate `libegl` conda-forge install step (find the pip/apt
equivalent, or confirm it's no longer needed).

## opencv5 migration is in progress but not finished

Discovered 2026-07-29 via CI failures on unrelated PRs (#25, and an
`ImageConstants.py` `Self`-import fix). `pyproject.toml` pins
`opencv-python<5.0.0` / `opencv-contrib-python<5.0.0`, i.e. the pip-based
install path deliberately caps below opencv5 because the codebase isn't
ready for it yet. But `.github/workflows/ci.yml`'s conda/micromamba
install used the bare `opencv` conda-forge package with no version
constraint — conda-forge has since published opencv 5.0.0, so CI silently
started testing against opencv5 while the actual pip-installable package
still targets opencv4. Result: CI now fails across the whole test matrix
(every OS × Python version) on API surface that changed between opencv4
and opencv5 — confirmed causes: `cv2.BRISK_create` moved/renamed,
`cv2.aruco.estimatePoseSingleMarkers` removed/renamed, MSER indexing
return shape changed. Last known-green run on `main` was 2026-06-16;
conda-forge's opencv5 release landed sometime after that, so this wasn't
caused by any code change, just dependency drift.

Fixed for now on branch `ci/pin-opencv-below-5`: pinned both `opencv`
occurrences in `ci.yml` (`test` and `codecov` jobs) to `opencv<5`,
matching `pyproject.toml`'s existing pip constraint. This unblocks CI but
does not do any opencv5 migration work itself.

There is a separate, not-yet-finished branch (`opencv5`, this repo's
current working branch as of 2026-07-29) actively migrating the codebase
to support opencv5 — e.g. `src/machinevisiontoolbox/ImagePointFeatures.py`
has an uncommitted change from `cv2.BRISK_create` to
`cv2.xfeatures2d.BRISK_create`, presumably chasing opencv5's API
reorganization. **Do not casually bump the `ci.yml` opencv pin back up**
until that migration branch is actually merged and the full test suite
passes against opencv5 — re-check `pyproject.toml`'s pin at the same
time, since both need to move together.

## Repo root is full of untracked scratch/junk files

Observed 2026-07-29, pre-existing (not from this session's work). `git
status` on `opencv5` shows ~50 untracked files/dirs at the repo root and
scattered through `src/`, `docs/`, `examples/`, `tests/`, e.g.: stray
scratch scripts (`findimages.py`, `phone.py`, `sunday.py`, `readbag.py`,
`fmtparser.py`, `audit_typing.py`, `inspect_bag.py`,
`src/machinevisiontoolbox/cvfuncs.py`, `docbugs.py`, `newcameras.py`,
`test_skimage.py`, `testblobplots.py`), planning notes
(`CODEAUTOLINK_FORK_PLAN.md`, `MIGRATION.md`, `NOTES`,
`OPENCV_FUNCTIONS.md`/`OPENCV_FUNCTIONS-original.md`,
`plot_call_inventory.md`), build artifacts (`machinevision-toolbox-python.pdf`/`.svg`,
`machinevisiontoolbox.pdf`, `aruco0.pdf`, `aruco50.pdf`), stray media/data
(`flowers.jpg`, `xx.mp4`, `ss`, `bags/`, `examples/bus.jpg`,
`examples/street_scene.jpg`, `examples/yolo26n.pt`,
`packages/mvtb-data/mvtbdata/data/bunny.dat`, `.../images/tags.png`),
old Sphinx warning-log captures (`docs/warnings*.txt`, referenced by
`CODEAUTOLINK_FORK_PLAN.md`'s regression-corpus methodology — may still
be wanted), and a couple of loose `x.json` files.

One item worth checking rather than just sweeping up:
`release-please-config.json` is untracked despite `release-please.yml`
requiring it (`config-file: release-please-config.json`) — if that's
really never been committed, `release-please` may only be working by
accident (whatever's on disk locally) and would break for anyone else's
checkout / a fresh CI runner. Worth confirming before cleanup, not after.

Also 4 pre-existing modified-but-uncommitted tracked files as of
2026-07-29: `README.md`, `src/machinevisiontoolbox/ImagePointFeatures.py`,
`src/machinevisiontoolbox/bin/imtool.py`, `tests/base/test_graphics.py`
— not touched this session. `ImagePointFeatures.py`'s change is very
likely in-progress work for the opencv5 migration (see below), not
junk — check before assuming any of these four are safe to discard.

### Fix

Triage into: (a) delete outright (build artifacts, one-off scratch
scripts that are clearly done), (b) commit properly if still wanted
(`release-please-config.json` almost certainly belongs in git), (c) move
to `.gitignore` if it's a recurring local-only output (e.g. the
`docs/warnings*.txt` capture files, if that workflow continues). Don't
bulk `git clean -xdf` without a human eyeballing the list first — some of
this may be in-progress work, not junk.
