"""Tests for transparent adapter registration (surge.model registration log)."""
from __future__ import annotations

import pytest

from surge.model import (
    MODEL_REGISTRY,
    _register_optional,
    list_models,
    registration_report,
    registration_table,
)
from surge.model.registry import REGISTRATION_LOG


def test_every_known_adapter_is_accounted_for():
    """Each registration attempt is logged with a valid status."""
    recs = registration_report()
    assert recs, "registration log must not be empty"
    assert {r.status for r in recs} <= {"registered", "skipped", "error"}
    # every registered record is actually resolvable; skipped ones are not
    for r in recs:
        if r.status == "registered":
            assert r.key in MODEL_REGISTRY
        elif r.status == "skipped":
            assert r.key not in MODEL_REGISTRY
            assert r.reason, f"skip without reason for {r.key}"


def test_core_sklearn_adapters_always_register():
    models = list_models()
    for key in ("sklearn.random_forest", "sklearn.mlp", "sklearn.gpr",
                "sklearn.ridge", "sklearn.logistic_regression"):
        assert key in models


def test_missing_dependency_records_skip_with_reason():
    before = len(REGISTRATION_LOG)

    def loader():
        raise ImportError("No module named 'not_a_real_backend'")

    _register_optional(("fake.model",), ("not_a_real_backend",), loader)
    rec = REGISTRATION_LOG[-1]
    del REGISTRATION_LOG[before:]
    assert rec.key == "fake.model"
    assert rec.status == "skipped"
    assert "not_a_real_backend" in rec.reason


def test_dependency_own_error_type_counts_as_skip():
    before = len(REGISTRATION_LOG)

    class FakeDepError(Exception):
        pass
    FakeDepError.__module__ = "fakedep.core"

    def loader():
        raise FakeDepError("native library could not be loaded")

    _register_optional(("fake.native",), ("fakedep",), loader)
    rec = REGISTRATION_LOG[-1]
    del REGISTRATION_LOG[before:]
    assert rec.status == "skipped"
    assert "fakedep broken" in rec.reason


def test_internal_bug_is_error_and_warns_never_silent():
    before = len(REGISTRATION_LOG)

    def loader():
        raise ValueError("a genuine SURGE bug")

    with pytest.warns(RuntimeWarning, match="internal error"):
        _register_optional(("buggy.model",), ("torch",), loader)
    rec = REGISTRATION_LOG[-1]
    del REGISTRATION_LOG[before:]
    assert rec.status == "error"
    assert "ValueError" in rec.reason


def test_strict_mode_raises_on_internal_bug(monkeypatch):
    import surge.model as m

    monkeypatch.setattr(m, "_STRICT", True)
    before = len(REGISTRATION_LOG)

    def loader():
        raise ValueError("boom")

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError):
        _register_optional(("buggy.strict",), (), loader)
    del REGISTRATION_LOG[before:]


def test_registration_table_is_printable():
    table = registration_table()
    assert table.startswith("Adapter registration:")
    assert "registered" in table
