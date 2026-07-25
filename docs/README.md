# SURGE documentation (Sphinx / Read the Docs)

## Current status (v0.1.0)

- **Source:** public repository [github.com/S-Villar/SURGE](https://github.com/S-Villar/SURGE).
- **DOE record:** [DOE Code 179819](https://www.osti.gov/doecode/biblio/179819), DOI [10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5).
- **Build:** `.readthedocs.yaml` at the repo root; Python 3.11; install path + `[docs]` extra.
- **Formal citation:** see [citation.rst](citation.rst) (*Citing SURGE*) and the root [CITATION.cff](../CITATION.cff).

## Where to start

| Page | Role |
|------|------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | **Start here** — install → first surrogate → reports, with troubleshooting |
| [gallery.md](gallery.md) | Results gallery — what SURGE produces per problem type |
| [index.rst](index.rst) | Landing page, status, pipeline overview |
| [installation.rst](installation.rst) | `surge-ml`, extras, verify |
| [quickstart.rst](quickstart.rst) | CLI + `examples/quickstart.py` |
| [BUILD_YOUR_OWN_SURROGATE.md](BUILD_YOUR_OWN_SURROGATE.md) | Train on your data + embed new models |
| [CUSTOM_DATASET_TUTORIAL.md](CUSTOM_DATASET_TUTORIAL.md) | Redirect → BUILD_YOUR_OWN_SURROGATE |
| [citation.rst](citation.rst) | OSTI / MLA and DOE Code |
| [SURGE_OVERVIEW.md](SURGE_OVERVIEW.md) | Deeper tour of the codebase |
| [setup/INSTALLATION.md](setup/INSTALLATION.md) | Long-form install and tips |
| [BENCHMARK_VERIFICATION_BRIEF.md](BENCHMARK_VERIFICATION_BRIEF.md) | Benchmark definitions, citations, verification status |
| [design/](design/) | Architecture roadmap and benchmark-restructure design (maintainers) |

## Build locally

```bash
cd /path/to/SURGE
python -m pip install -e ".[docs]"
make -C docs html
# Open docs/_build/html/index.html
```

## Navigation index

Internal planning and prerelease checklists (PRERELEASE, ROADMAP, etc.) are **excluded** in `conf.py` so they are not published on Read the Docs; they remain in the repository for maintainers.
