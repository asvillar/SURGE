# Contributing to SURGE

Thank you for your interest in contributing. SURGE is in an early
pre-1.0 phase; the goal for the `0.1.0` DOE open-source release is a
small, well-tested, well-documented core. Contributions that keep the
public surface narrow and the release docs honest are especially
welcome.

For security-sensitive issues, please use the private channel described
in [`SECURITY.md`](./SECURITY.md) rather than a public issue.

## Quickstart for developers

```bash
git clone https://github.com/asvillar/SURGE.git
cd SURGE
python -m pip install -e ".[dev]"
pytest -q tests/
```

The test suite should report `N passed, M skipped, 0 failed`. The
skipped tests are legacy pre-refactor tests tracked in
[`docs/REFACTORING_PLAN.md`](./docs/REFACTORING_PLAN.md) §1.9; they are
out of scope for `0.1.0`.

## What "ready to merge" looks like

1. `pytest -q tests/` is green (or the PR justifies every new skip).
2. New user-visible behaviour is documented in `docs/` and, when
   relevant, in the `README.md`.
3. New public APIs carry docstrings and type hints.
4. No new absolute NERSC-style paths (`/path/to/data`,
   `/path/to/scratch`) land in tracked source. Use relative paths or
   environment variables. See [`docs/AUDIT_NIGHT1.md`](./docs/AUDIT_NIGHT1.md)
   for the disclosure-safety rationale.
5. The `[surge.fit] ...` banner still appears exactly once per model
   train. If a PR changes that, the reason belongs in the commit
   message.
6. CI is green on both the `test` and `e2e-regression` jobs defined in
   `.github/workflows/ci.yml`.

## Pull request process

1. Fork the repo and create a topic branch from `main`:
   `feature/<slug>`, `refactor/<slug>`, `fix/<slug>`, or `docs/<slug>`.
2. Keep each PR focused. A PR that ships both a refactor and a new
   feature will usually be asked to split.
3. Use imperative commit messages (`feat: ...`, `fix: ...`,
   `refactor: ...`, `docs: ...`, `chore: ...`, `ci: ...`, `test: ...`).
   The template at `.github/pull_request_template.md` will walk you
   through the rest.
4. Reference the relevant section of `docs/ROADMAP.md` or
   `docs/REFACTORING_PLAN.md` when the change is a scheduled item.

## Coding style

- Python 3.10+.
- `ruff check surge/ tests/` should stay clean (non-blocking pre-`0.1.0`,
  blocking after).
- Prefer explicit keyword-only constructors; `dataclass(frozen=True)`
  for value objects; `logging` over `print` in library code.
- New model adapters must declare a `resource_profile` (see
  `surge/model/base.py` and `surge/hpc/policy.py`) so the
  `[surge.fit]` banner reports honest numbers.

## Reporting issues

Use `.github/ISSUE_TEMPLATE/bug_report.md` for defects and
`feature_request.md` for proposals. For security issues please use the
channel in [`SECURITY.md`](./SECURITY.md) instead.

## Licensing

By submitting a contribution you agree that your work will be released
under the project's BSD-3-Clause license (see [`LICENSE`](./LICENSE)).
