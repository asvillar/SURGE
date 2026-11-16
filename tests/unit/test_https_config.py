"""HTTPS default-context configuration in surge.benchmarks.tasks."""

import ssl

import pytest

pytest.importorskip("certifi")

from surge.benchmarks import tasks  # noqa: E402


@pytest.fixture
def restore_factory():
    orig = ssl._create_default_https_context
    yield
    ssl._create_default_https_context = orig


def test_default_verifies(monkeypatch, restore_factory):
    monkeypatch.delenv("SURGE_INSECURE_SSL", raising=False)
    tasks._configure_https()
    ctx = ssl._create_default_https_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_env_var_disables_with_warning(monkeypatch, restore_factory):
    monkeypatch.setenv("SURGE_INSECURE_SSL", "1")
    with pytest.warns(UserWarning, match="verification is disabled"):
        tasks._configure_https()
    assert ssl._create_default_https_context().verify_mode == ssl.CERT_NONE
