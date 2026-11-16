"""The README's "built on" blocks must agree with pyproject.toml and each other."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPI = (ROOT / "README_PYPI.md").read_text(encoding="utf-8")

HEADING = "## Built on open-source software"
# Declared in pyproject.toml but not imported by the package, so the table
# does not claim it is used.
DECLARED_NOT_IMPORTED = {"scikit-optimize"}


def _name(spec: str) -> str:
    return re.split(r"[<>=!~;\[\s]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


@pytest.fixture(scope="module")
def project():
    if tomllib is None:
        pytest.skip("neither tomllib nor tomli is available")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _section() -> str:
    assert HEADING in README, "README is missing the built-on section"
    body = README.split(HEADING, 1)[1]
    return body.split("\n## ", 1)[0]


def _table_rows():
    rows = []
    for line in _section().splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Software"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 4, line
        rows.append(cells)
    return rows


def _headline_block() -> str:
    m = re.search(r"<p align=\"center\"><sub>Built on</sub>.*?</p>", README, re.S)
    assert m, "README is missing the headline logo row"
    return m.group(0)


def _pypi_block() -> str:
    m = re.search(r"\*\*Built on\*\*\n\n(.+?)\n\n", PYPI, re.S)
    assert m, "README_PYPI.md is missing the Built on line"
    return m.group(1)


def test_section_exists():
    assert _table_rows()


def test_table_distributions_are_declared(project):
    core = {_name(d) for d in project["dependencies"]}
    extras = {k: {_name(d) for d in v} for k, v in project["optional-dependencies"].items()}
    for _software, _use, dist, installed in _table_rows():
        names = [_name(n) for n in re.findall(r"`([^`]+)`", dist)]
        extra = re.findall(r"`([^`]+)`", installed)
        if installed == "core":
            for n in names:
                assert n in core, f"{n} is not a core dependency"
        else:
            assert len(extra) == 1 and extra[0] in extras, f"unknown extra {installed}"
            if not names:  # e.g. gpflow: installed separately, not declared
                continue
            for n in names:
                assert n in extras[extra[0]], f"{n} not in extra {extra[0]}"


def test_every_core_dependency_has_a_row(project):
    listed = set()
    for _s, _u, dist, installed in _table_rows():
        if installed == "core":
            listed |= {_name(n) for n in re.findall(r"`([^`]+)`", dist)}
    missing = {_name(d) for d in project["dependencies"]} - listed - DECLARED_NOT_IMPORTED
    assert not missing, f"core dependencies without a row: {sorted(missing)}"


def _slugs(block: str) -> set[str]:
    return set(re.findall(r"logo=([a-z0-9]+)", block))


def test_headline_and_pypi_name_same_logos():
    a, b = _slugs(_headline_block()), _slugs(_pypi_block())
    assert len(a) == 9
    assert a == b


def test_badges_are_remote_only():
    for block in (_headline_block(), _pypi_block(), _section()):
        srcs = re.findall(r'<img[^>]*src="([^"]+)"', block)
        srcs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", block)
        assert srcs
        for src in srcs:
            assert src.startswith("https://img.shields.io/"), src
