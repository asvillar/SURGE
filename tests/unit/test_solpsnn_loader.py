"""SOLPS-NN loader tests: synthetic layout, fake HTTP range server (no network)."""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pytest

from surge.benchmarks.loaders import solpsnn as sn

N = 6


def _write_split(root, split, n=N):
    d = root / sn.ZIP_PREFIX / split
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 9.0, size=(n, 8))
    X[:, 3:6] = 10.0 ** rng.uniform(19, 22, size=(n, 3))
    np.save(d / "X_units.npy", X)
    np.save(d / "X_legacy.npy", X.copy())
    np.save(d / "completed.npy", np.arange(n))
    np.save(d / "failed.npy", np.array([n + 1]))
    for k in sn.SCALAR_TARGETS:
        np.save(d / f"{k}.npy", rng.uniform(size=n))
    ramp = np.linspace(1.0, 100.0, 104 * 50).reshape(1, 104, 50) * np.ones((n, 1, 1))
    np.save(d / "te.npy", ramp * sn.EV)
    return X


@pytest.fixture
def layout(tmp_path):
    X = _write_split(tmp_path, "train")
    return tmp_path, X


def test_load_shapes_and_ev(layout):
    root, X = layout
    s = sn.load_solpsnn("train", data_dir=root, download=False)
    assert s.X.shape == (N, 8) and s.te.shape == (N, *sn.GRID_SHAPE)
    assert s.te.dtype == np.float32
    assert s.te.max() == pytest.approx(100.0, rel=1e-5)
    assert set(s.scalars) == set(sn.SCALAR_TARGETS)
    assert s.outer_target_te().shape == (N, 48)
    np.testing.assert_allclose(s.X, X)


def test_x_log_only_log_columns(layout):
    root, X = layout
    s = sn.load_solpsnn("train", data_dir=root, download=False)
    XL = s.X_log
    keep = [i for i in range(8) if i not in sn.LOG_INPUT_COLUMNS]
    np.testing.assert_array_equal(XL[:, keep], X[:, keep])
    np.testing.assert_allclose(XL[:, 3:6], np.log10(X[:, 3:6]))
    np.testing.assert_array_equal(s.X, X)


def test_load_regression(layout):
    root, _ = layout
    Xl, y = sn.load_regression("train", "psol", data_dir=root, download=False)
    assert Xl.shape == (N, 8) and y.shape == (N,)
    _, f = sn.load_regression("train", "te_field", data_dir=root, download=False)
    assert f.shape == (N, 104 * 50)
    with pytest.raises(ValueError):
        sn.load_regression("train", "nope", data_dir=root, download=False)


def test_missing_and_bad_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="19237127"):
        sn.load_solpsnn("test", data_dir=tmp_path, download=False)
    with pytest.raises(ValueError):
        sn.load_solpsnn("val", data_dir=tmp_path, download=False)


def test_default_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    assert sn.default_data_dir() == tmp_path / "solpsnn"


class _FakeResp:
    def __init__(self, data, headers):
        self._b = io.BytesIO(data)
        self.headers = headers

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_zip(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    payload = rng.bytes(3_000_000)  # incompressible
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("pre/big.bin", payload)
        zf.writestr("pre/small.txt", b"hello")
    blob = buf.getvalue()

    def fake_urlopen(req, timeout=None, context=None):
        m = re.match(r"bytes=(\d+)-(\d+)", req.get_header("Range"))
        a, b = int(m[1]), min(int(m[2]), len(blob) - 1)
        return _FakeResp(
            blob[a : b + 1], {"Content-Range": f"bytes {a}-{b}/{len(blob)}"}
        )

    monkeypatch.setattr(sn.urllib.request, "urlopen", fake_urlopen)
    return payload, blob


def test_zip_members_and_download(fake_zip, tmp_path):
    payload, blob = fake_zip
    names = [i.filename for i in sn.zip_members("http://x/z")]
    assert names == ["pre/big.bin", "pre/small.txt"]
    out = sn.download_members(
        ["pre/small.txt"], tmp_path / "d", url="http://x/z", progress=False
    )
    assert out[0].read_bytes() == b"hello"
    out = sn.download_members(
        ["pre/big.bin"], tmp_path / "d", url="http://x/z", progress=False
    )
    assert out[0].read_bytes() == payload
    # second call is a no-op skip
    sn.download_members(
        ["pre/big.bin"], tmp_path / "d", url="http://x/z", progress=False
    )


def test_rangefile_reads_little(fake_zip):
    _, blob = fake_zip
    rf = sn._RangeFile("http://x/z")
    with zipfile.ZipFile(rf) as zf:
        assert zf.read("pre/small.txt") == b"hello"
    assert rf.size == len(blob)
    assert rf.bytes_read < len(blob) / 10
