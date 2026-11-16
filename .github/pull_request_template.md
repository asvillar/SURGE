<!--
Thanks for contributing to SURGE. Please fill in this template — short
answers are fine. Delete sections that don't apply.
-->

## Summary

<!-- One paragraph, user-visible framing: what changes, why it matters. -->

## Type of change

- [ ] `feat` — new user-visible behaviour
- [ ] `fix` — bug fix (no API change)
- [ ] `refactor` — internal cleanup, no behaviour change
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `ci` — CI / build / tooling
- [ ] `chore` — housekeeping

## Test plan

<!-- How did you verify this locally? Paste the relevant pytest output
or CLI banner. At minimum: -->

- [ ] `pytest -q tests/` passes locally
- [ ] `pytest -q tests/test_e2e_release_smoke.py` passes locally
- [ ] New public APIs have at least one direct unit test
- [ ] The `[surge.fit] ...` banner still fires exactly once per model
      train (if training code was touched)

## Backwards compatibility

- [ ] No public API change
- [ ] Public API change, documented above and in the docs

## Checklist

- [ ] No absolute cluster paths (home or scratch directories)
      added to tracked source
- [ ] No large binary / data blobs added to `data/` or elsewhere
- [ ] Commit messages use imperative, present-tense style
