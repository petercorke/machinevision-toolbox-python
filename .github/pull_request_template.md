Thanks for contributing to MVTB!

## Summary

<!-- What does this PR do, and why? -->

## Related issue

<!-- Fixes #123 / Closes #123 — if applicable -->

## Checklist

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`type: description`) — checked automatically, see the "Check PR title" status
- [ ] Tests pass locally (`pytest`)
- [ ] Added/updated tests for this change, if applicable
- [ ] New/changed code is type-hinted with modern syntax (`X | Y`, `list[X]`, not `Union`/`Optional`/`List`)
- [ ] Docstrings updated (reST style: `:param:`, `:returns:`; type hints in the signature cover types now, `:type:`/`:rtype:` are rarely needed)

> **Automated bot comments to expect:** Codacy will comment with style/coverage findings — most of it is pre-existing backlog (see `tech-debt.md`), not something your PR introduced, so don't be alarmed. If this PR adds or bumps a dependency, Dependabot may separately comment flagging a known vulnerability — if so, use a newer patched version if one exists, or just note it in this PR and a maintainer will decide; you don't need to solve it yourself.

<!-- Target branch is `main` — release-please handles versioning automatically. -->

