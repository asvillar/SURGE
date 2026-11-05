"""Tests for the PyPI long description (``README_PYPI.md``).

``pyproject.toml`` points ``readme`` at this file instead of the main
``README.md`` because the main README carries relative image/link paths and
``<picture>`` tags that don't resolve on pypi.org. These tests guard the
contract: no such relative/HTML leftovers, every GitHub URL pinned to the
package's own version with a path that actually exists, and every extra it
advertises is a real ``pyproject.toml`` extra.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import surge

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PYPI = REPO_ROOT / "README_PYPI.md"

FORBIDDEN_SNIPPETS = ("](docs/", 'src="docs/', "](data/", "<picture")

# Matches e.g. https://raw.githubusercontent.com/asvillar/SURGE/v0.1.0/docs/assets/readme/x.png
RAW_URL_RE = re.compile(
    r"raw\.githubusercontent\.com/asvillar/SURGE/v(?P<v>[^/]+)/(?P<path>\S+?)[)\"\s]"
)
# Matches e.g. https://github.com/asvillar/SURGE/blob/v0.1.0/docs/GETTING_STARTED.md
BLOB_URL_RE = re.compile(
    r"github\.com/asvillar/SURGE/blob/v(?P<v>[^/]+)/(?P<path>[^)#\s]+)"
)
# The `Extra` column of the install table: `| `torch` | ... |`
EXTRA_CELL_RE = re.compile(r"\|\s*`([a-zA-Z0-9_-]+)`\s*\|")


def _text() -> str:
    return README_PYPI.read_text(encoding="utf-8")


def test_readme_pypi_exists():
    assert README_PYPI.is_file(), "README_PYPI.md must exist (pyproject readme target)"


def test_no_relative_doc_or_picture_leftovers():
    text = _text()
    found = [snippet for snippet in FORBIDDEN_SNIPPETS if snippet in text]
    assert not found, (
        f"README_PYPI.md contains relative-path/HTML snippets that don't "
        f"render on PyPI: {found}"
    )


def test_github_urls_pin_current_version_with_existing_paths():
    text = _text()
    matches = list(RAW_URL_RE.finditer(text)) + list(BLOB_URL_RE.finditer(text))
    assert matches, "expected at least one raw.githubusercontent/github blob URL"
    for m in matches:
        version = m.group("v")
        path = m.group("path")
        assert version == surge.__version__, (
            f"URL pinned to v{version} but surge.__version__ is "
            f"{surge.__version__!r}: {m.group(0)!r}"
        )
        target = REPO_ROOT / path
        assert target.exists(), f"README_PYPI.md references a path that doesn't exist: {path}"


def test_extras_in_install_table_exist_in_pyproject():
    tomllib = pytest.importorskip("tomllib")  # stdlib 3.11+; skip on 3.10 CI leg
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    known_extras = set(pyproject["project"]["optional-dependencies"])

    extras_in_table = set(EXTRA_CELL_RE.findall(_text()))
    assert extras_in_table, "expected at least one `extra` cell in the install table"
    missing = extras_in_table - known_extras
    assert not missing, (
        f"README_PYPI.md advertises extras that aren't in pyproject.toml: {missing}"
    )
