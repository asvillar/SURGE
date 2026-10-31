# Releasing surge-ml to PyPI

The package is PyPI-ready: `uv build` produces a clean sdist + wheel
(runtime data `surge/benchmarks/metadata.yaml` ships via package-data; the
sdist does not ship `tests/`, `docs/`, or `examples/`). `README_PYPI.md` is
the published long description (plain Markdown, no relative-asset links
that break off-GitHub).

Publishing is manual: the maintainer runs `uv publish` locally (holding the
PyPI/TestPyPI tokens), **then** pushes the version tag. That order matters —
if the upload fails, no tag is left dangling. `.github/workflows/release.yml`
builds and smoke-tests every `v*` tag and attempts a Trusted Publishing
upload, but that job is `continue-on-error: true` and uses `skip-existing`,
so it is a CI safety net, not the primary path, until Trusted Publishing is
configured (see below). A tag pushed after a successful manual `uv publish`
makes the CI publish step a no-op (the version already exists on PyPI).

## One-time setup

1. Create the **surge-ml** project on PyPI and on TestPyPI
   (https://test.pypi.org), under the maintainer account.
2. Generate an API token scoped to each project. Never commit a token:
   export it as `UV_PUBLISH_TOKEN` in your shell for the duration of the
   publish command (TestPyPI token for the rehearsal, PyPI token for the
   real publish), or pass `--token` inline; do not put either in a
   tracked `.env` or config file.
3. Trusted Publishing (OIDC, no long-lived token in CI) — register
   **surge-ml** as a *pending publisher* on pypi.org (Publishing →
   Add a new pending publisher) with:
   - PyPI project name: `surge-ml`
   - Owner: `asvillar`
   - Repository: `SURGE`
   - Workflow name: `release.yml`
   - Environment: `pypi`

   A pending publisher activates on the first successful run of the
   `publish` job in `release.yml`; until then that job's upload attempt
   fails harmlessly (`continue-on-error: true`) and publishing stays
   manual.

## Release procedure

Decided order: **build → audit → fresh-venv smoke → TestPyPI rehearsal →
`uv publish` → push the tag.** The tag is pushed last on purpose.

```bash
# 0. bump the version — single source of truth is surge/__init__.__version__
#    (pyproject.toml reads it dynamically); update CHANGELOG.md and
#    CITATION.cff (date-released, both occurrences) to match.

# 1. clean build
rm -rf dist && uv build

# 2. content audit — fail this step rather than publish something wrong
tar tzf dist/*.tar.gz | grep -vE 'surge/.*\.py$'   # expect: metadata.yaml,
                                                     # schema, licence/readme —
                                                     # no tests/docs/examples
unzip -l dist/*.whl | grep -vE '\.py$'              # same expectation
uvx twine check dist/*                              # long_description renders

# 3. fresh-venv smoke (installed wheel only, nothing from the checkout)
uv venv /tmp/relcheck
VIRTUAL_ENV=/tmp/relcheck uv pip install dist/*.whl
/tmp/relcheck/bin/surge version
# -I: isolated mode, so ./surge in the checkout is not imported instead of the wheel
SURGE_STRICT_REGISTRY=1 /tmp/relcheck/bin/python -I -c "import surge"
/tmp/relcheck/bin/surge models --verbose
/tmp/relcheck/bin/surge bench -b synthetic.regression_1d \
    -m sklearn.random_forest --no-save

# 3b. toy-CSV surge init -> validate -> run (the non-interactive wizard path)
/tmp/relcheck/bin/python -c "
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
X = rng.standard_normal((60, 2))
y = X @ [1.5, -2.0] + 0.05 * rng.standard_normal(60)
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y
df.to_csv('/tmp/toy.csv', index=False)
"
/tmp/relcheck/bin/surge init --data /tmp/toy.csv --target y \
    --budget smoke --yes --out /tmp/spec.yaml
/tmp/relcheck/bin/surge validate /tmp/spec.yaml
/tmp/relcheck/bin/surge run /tmp/spec.yaml

# 3c. report renders from a saved result (an empty reports dir exits 1 by design)
/tmp/relcheck/bin/surge bench -b synthetic.regression_1d \
    -m sklearn.random_forest --save-dir /tmp/relcheck/reports
/tmp/relcheck/bin/surge report --reports /tmp/relcheck/reports \
    --out /tmp/relcheck/lb.html
grep -c -e 'audit/' -e 'docs/assets' -e '<video' /tmp/relcheck/lb.html  # expect 0

# 4. TestPyPI rehearsal — pin the exact version so an old TestPyPI upload
#    is never silently picked up
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
uv venv /tmp/relcheck-testpypi
VIRTUAL_ENV=/tmp/relcheck-testpypi uv pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ surge-ml==<version>
/tmp/relcheck-testpypi/bin/surge version
/tmp/relcheck-testpypi/bin/surge models --verbose

# 5. the real thing (manual; holds the PyPI token)
uv publish dist/*
pip index versions surge-ml   # confirm it is live under the new version

# 6. tag AFTER a successful publish, then push
git tag -a v<version> -m "surge-ml <version>"
git push origin v<version>

# 7. GitHub release, body = the CHANGELOG section for this version
awk '/^## \[<version>\]/{flag=1; next} /^## \[/{flag=0} flag' \
    CHANGELOG.md > /tmp/release-notes.md
gh release create v<version> --draft --title "surge-ml <version>" \
    --notes-file /tmp/release-notes.md

# 8. DOE CODE record (Code ID 179819, DOI 10.11578/dc.20260422.5 unchanged) —
#    update the release date on the OSTI/DOE CODE record whenever
#    CITATION.cff's date-released changes.
```

## Pre-publish checklist

- [ ] `pytest -q` green on 3.10–3.12 (CI matrix)
- [ ] Version bumped in `surge/__init__.__version__` only; `CHANGELOG.md`
      and both `date-released` fields in `CITATION.cff` match
- [ ] `rm -rf dist && uv build` — clean sdist + wheel
- [ ] `tar tzf dist/*.tar.gz | grep -vE 'surge/.*\.py$'` and
      `unzip -l dist/*.whl | grep -vE '\.py$'` show no `tests/`, `docs/`,
      or `examples/` content
- [ ] `uvx twine check dist/*` passes (long description renders)
- [ ] Fresh-venv smoke: `surge version`,
      `SURGE_STRICT_REGISTRY=1 python -c "import surge"` raises nothing,
      `surge models --verbose`,
      `surge bench -b synthetic.regression_1d -m sklearn.random_forest --no-save`,
      toy-CSV `surge init --budget smoke --yes` → `surge validate` → `surge run`,
      `surge bench … --save-dir <dir>` then `surge report --reports <dir> --out lb.html`
      with no `audit/`, `docs/assets`, or `<video` strings in the output
- [ ] TestPyPI rehearsal installs and runs with `surge-ml==<version>` pinned
- [ ] `uv publish dist/*` succeeds; `pip index versions surge-ml` shows it
- [ ] Tag pushed only after the publish above succeeded
- [ ] GitHub release drafted from the matching `CHANGELOG.md` section
- [ ] DOE CODE record date updated to match `CITATION.cff`
