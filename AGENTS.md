# machinevision-toolbox-python — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `machinevision-toolbox-python` |
| Nickname | MVTB |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; direct push to `main` at Peter's discretion |

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
