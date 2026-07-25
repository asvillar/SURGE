# Releasing surge-ml to PyPI

The package is PyPI-ready: `uv build` produces a clean sdist + wheel
(runtime data `surge/benchmarks/metadata.yaml` ships via package-data;
no notebooks, canvases, or datasets leak into the artifacts).

## One-time setup

1. Create the PyPI project (and TestPyPI counterpart) for **surge-ml**
   under the maintainer account.
2. Generate an API token scoped to the project; store as
   `UV_PUBLISH_TOKEN` (or configure GitHub Trusted Publishing for CI).

## Release procedure

```bash
# 1. bump version in pyproject.toml (and surge/__init__.__version__)
# 2. clean build
rm -rf dist && uv build
# 3. rehearse against TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
uv venv /tmp/relcheck && VIRTUAL_ENV=/tmp/relcheck uv pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ surge-ml
/tmp/relcheck/bin/surge version && /tmp/relcheck/bin/surge models
# 4. the real thing
uv publish dist/*
# 5. tag
git tag v<version> && git push origin v<version>
```

## Pre-publish checklist

- [ ] `pytest -q` green on 3.10–3.12 (CI matrix)
- [ ] `SURGE_STRICT_REGISTRY=1 python -c "import surge"` raises nothing
- [ ] wheel smoke: fresh venv, install wheel, `surge version`,
      `surge models --verbose`, one `surge bench … --no-save`,
      one `surge run <spec>` on a toy CSV
- [ ] README badges/links render on PyPI (`long_description` = README.md)
- [ ] CITATION.cff / DOE CODE metadata match the version
