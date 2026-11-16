"""``_ensure_hf_ssl_env`` is opt-in: no keychain access without SURGE_USE_SYSTEM_CA=1."""

from __future__ import annotations

import os

import pytest

from surge.benchmarks.loaders import constellaration as cl


def _boom(*args, **kwargs):  # pragma: no cover - must never run
    raise AssertionError("subprocess.run must not be called")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SURGE_USE_SYSTEM_CA", raising=False)
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cl.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cl.subprocess, "run", _boom)
    monkeypatch.setattr(cl, "_default_data_dir", lambda: tmp_path / "constellaration")
    return tmp_path


def test_default_does_nothing(clean_env):
    before = dict(os.environ)
    cl._ensure_hf_ssl_env()
    assert dict(os.environ) == before
    assert not (clean_env / "constellaration").exists()


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_other_values_do_nothing(clean_env, monkeypatch, value):
    monkeypatch.setenv("SURGE_USE_SYSTEM_CA", value)
    cl._ensure_hf_ssl_env()
    assert "SSL_CERT_FILE" not in os.environ


def test_optin_runs_keychain_read(clean_env, monkeypatch):
    calls = []

    class _R:
        stdout = "ROOTS\n"

    monkeypatch.setattr(cl.subprocess, "run", lambda *a, **k: calls.append(a) or _R())
    monkeypatch.setenv("SURGE_USE_SYSTEM_CA", "1")
    cl._ensure_hf_ssl_env()
    assert len(calls) == 1
    assert os.environ["SSL_CERT_FILE"].endswith(".ca_bundle.pem")
