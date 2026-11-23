# machinevision-toolbox-python — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `machinevision-toolbox-python` |
| Nickname | MVTB |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; direct push to `main` at Peter's discretion |

## What the Toolbox is for

The Toolbox has three goals, in this order: **pedagogy** (code that is clear and easy to
read, and that follows the explanations in *Robotics, Vision & Control*), **functionality**
(the range of what it covers), and **performance** (speed and memory).

A faster implementation is welcome alongside a readable one, not instead of it. A change that
gains speed by making the code harder to understand will not be accepted. Nor will a change to
an algorithm that the book explains if the results differ from the book; a restructured
implementation that gives the same results is fine, with a comment pointing to the book
section. New algorithms beyond the book are welcome.

If your change trades clarity for speed, or changes what an algorithm does, say so in the PR
description and ask first.

## Notes specific to this repo

- Pure Python — no compiled extensions, builds with Hatch (`hatchling`) directly.
- Depends on `spatialmath`, `pgraph-python`, `ansitable`, plus the third-party `opencv-python`/
  `opencv-contrib-python` and `open3d`.
- `open3d` is consistently a Python version or two behind — never assume it supports the
  latest Python release; check compatibility before bumping this repo's Python floor/ceiling.
- OpenCV 5.x support was just added alongside the existing 4.x support (a major version jump,
  not a minor bump). This is fresh work — aim to keep both working, but treat 4.x/5.x
  compatibility friction as expected teething problems to fix, not a regression.
- Tech-debt tracked as GitHub Issues labelled `tech-debt` (migrated 2026-08-02) — this repo
  was first to migrate off `tech-debt.md`.
- This repo's `.github/pull_request_template.md` and `.github/ISSUE_TEMPLATE/*.yml` are the
  canonical source other repos' templates were copied from — check here first if updating
  template content ecosystem-wide.
- Codacy is wired in (badge + automated PR comments); grade A is the aspiration. Real backlog
  exists, tracked via `tech-debt`-labelled issues (including a "mypy not in CI" entry) — see
  ecosystem `AGENTS.md` §5 for how that target is framed.
- Type checking: `mypy` (config in `pyproject.toml` under `[tool.mypy]`; run it from the repo
  root with no arguments). The package passes with zero errors and must stay that way. CI runs
  it (the `typecheck` job, part of the required `All tests passed` check), so run it yourself
  before opening a PR. Write
  ignores as `# type: ignore[code]` with the specific error code; unused ignores are reported.
  Import from the defining submodule (for example `machinevisiontoolbox.base.graphics`), not
  through the package's star imports, which mypy cannot follow.
- Formatting and linting: `ruff`, configured in `pyproject.toml` (line length 88, as in RTB,
  bdsim and sphinx-pyrunblock). Format with `ruff format src tests` and lint with
  `ruff check src tests`; do not use `black`. Format only the files you change, so that a
  feature PR carries no unrelated reformatting. CI does not run ruff yet, but the tree is
  formatted, so `ruff format --check src tests` passes. `E501` is ignored: the formatter
  does not wrap long strings or comments.
- New and changed code uses modern annotation syntax: `X | Y`, `X | None`, `list[X]`,
  `dict[K, V]`, not `Union`, `Optional`, `List` or `Dict`. For array arguments use `ArrayLike`
  (or `ArrayLike2` for a 2-element value) from `machinevisiontoolbox.mvtb_types`, and put the
  shape in the docstring (`:type:`) only where it adds information.
- New and changed docstrings are reST with `:param:` first. An example is a `>>>` block under
  `.. runblock:: pycon`, headed `.. rubric:: Examples` (always plural), and cross-references
  are `.. seealso:: :meth:`other``, not the `:seealso:` field. A `.. versionadded::`,
  `.. versionchanged::` or `.. deprecated::` names the release it ships in. Much existing code
  still uses `Example:` and `:seealso:`; it is being migrated gradually, so follow the new
  style in anything you touch rather than rewriting whole files.

## If you are an AI assistant preparing a contribution

- **Check for existing work before you write code.** Several agents often work on the same
  open Issue at once. Search first: `gh pr list --state open --search "#<issue>"`, and check
  the Issue's linked pull requests and assignee. If a PR exists or the Issue is assigned to
  someone else, comment on it rather than opening a competing PR, and comment on the Issue to
  claim it before you start.
- **Say so in the PR.** If an AI assistant wrote or substantially helped with the change,
  state that, and which tool, in the PR description. You, the human, remain responsible and
  should have read the diff.
- **Sign the commits.** An AI-assisted commit ends with a `Co-Authored-By:` trailer naming
  the tool and model, and the PR description ends with the tool's attribution line. The
  repository squash-merges, so the PR description is what survives; keep the disclosure there
  too.
- Branch names are `type/short-description` (`fix/`, `feat/`, `docs/`, `chore/`, `refactor/`,
  `test/`). The required checks are `All tests passed` and the PR title.
- Keep the PR to the one Issue or fix, and say what you ran and what you could not run.
