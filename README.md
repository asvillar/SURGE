# SURGE

**Surrogate Unified Robust Generation Engine** — a Python framework for building
tabular scientific surrogates: declarative workflows, an extensible model registry
(HPO, artifacts, metrics, optional UQ). See [`docs/SURGE_OVERVIEW.md`](docs/SURGE_OVERVIEW.md)
for a full map of the codebase.

## Public repository URL

**Set this after you create the GitHub (or lab) repository** and replace the
placeholder in `pyproject.toml` under `[project.urls]`:

| | |
|---|---|
| **Repository (for DOE CODE / publications)** | `https://github.com/<org>/<repo>` |

You can send that **single HTTPS URL** to your publications office so they can
register the software and publish the open-source statement.

## Install (from a clone)

```bash
cd /path/to/SURGE
python -m pip install -e ".[torch]"   # core + PyTorch MLP; omit [torch] if not needed
```

Or from a published tag (after you push to GitHub):

```bash
python -m pip install "surge-ml==0.1.0"
```

(`surge-ml` is the PyPI distribution name to avoid clashes with unrelated
packages named “surge”; the import name remains `surge`.)

## Smoke test

```bash
python -c "import surge; print(surge.__version__)"
```

Run tests (with dev deps): `python -m pip install -e ".[dev]" && pytest tests/`

## Documentation

- [docs/README.md](docs/README.md) — index
- [docs/PRERELEASE.md](docs/PRERELEASE.md) — scope and roadmap
- Sphinx: `make -C docs html` (requires `docs` extra)

## License and funding

- **License:** BSD 3-Clause — see [LICENSE](LICENSE).
- **Notice:** see [NOTICE](NOTICE) (DOE funding / contract text).

## Citation

After DOE CODE registration, cite using the **identifier and URL** your
publications office provides (and/or the repository’s **Cite this repository**
button on GitHub once enabled).

---

*Internal development history may live on a private remote; the **public** remote
should receive only policy-approved content. See [docs/PRERELEASE.md](docs/PRERELEASE.md).*
