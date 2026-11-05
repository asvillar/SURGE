"""Loader for the SOLPS-NN tokamak edge-plasma surrogate dataset.

Data: Zenodo record 19237127 (CC BY 4.0). Citations:

* Dasbach, S. & Wiesen, S., "Towards fast surrogate models for interpolation of
  tokamak edge plasmas", Nuclear Materials and Energy 34 (2023) 101396.
* Dasbach, S., PhD thesis, HHU Duesseldorf (2023) (shipped as ``thesis.pdf``
  in the record).

The dataset is ~17.5 GB in a single zip. The server honours HTTP Range
requests, so :func:`download_members` reads the zip central directory and
extracts only the requested members without fetching the archive.

Design: 8 SOLPS-ITER inputs (R, B, P, D_puff, N_puff, D_core, D_perp,
chi_perp); the train split (5756 runs) is a Sobol design, the test split
(1465 runs) a random design. Crashed runs are listed in ``failed.npy``.

Grid convention: B2 grid of ``nx=104`` poloidal x ``ny=50`` radial cells.
Index 0 and 103 (poloidal) and 0 and 49 (radial) are guard cells, leaving 48
physical radial cells. The outer divertor target is poloidal row 102, the
inner target row 1. Temperatures ship in Joules; the loader returns eV.
"""

from __future__ import annotations

import io
import os
import ssl
import sys
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths

ZENODO_RECORD = "19237127"
ZENODO_ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/dataset.zip/content"
)
ZIP_PREFIX = "solpsnn_dataset_zenodo"

PARAM_NAMES = ("R", "B", "P", "D_puff", "N_puff", "D_core", "D_perp", "chi_perp")
PARAM_UNITS = ("m", "T", "W", "atoms/s", "atoms/s", "atoms/s", "m^2/s", "m^2/s")
PARAM_RANGES = (
    (1.0, 10.0),
    (1.0, 10.0),
    (1e7, 2e8),
    (1e18, 1e24),
    (1e18, 1e23),
    (1e19, 1e24),
    (0.1, 2.0),
    (0.1, 2.0),
)
LOG_INPUT_COLUMNS = (3, 4, 5)

SCALAR_TARGETS = {
    "psol": ("Power crossing the separatrix", "W"),
    "pwmxap": ("Peak heat flux at the outer target", "W/m^2"),
    "fnixap": ("Integrated D+ flux to the outer target", "atoms/s"),
}

GRID_SHAPE = (104, 50)
OUTER_TARGET_ROW = 102
INNER_TARGET_ROW = 1
EV = 1.602176634e-19
SPLITS = ("train", "test")

_SPLIT_FILES = (
    "X_units.npy",
    "X_legacy.npy",
    "completed.npy",
    "failed.npy",
    "simulations.db",
    "psol.npy",
    "pwmxap.npy",
    "fnixap.npy",
    "te.npy",
)
_GEOMETRY_FILES = ("geometry/crx.npy", "geometry/cry.npy")
MINIMAL_MEMBERS = tuple(
    [f"{ZIP_PREFIX}/{s}/{f}" for s in SPLITS for f in _SPLIT_FILES]
    + [f"{ZIP_PREFIX}/{f}" for f in _GEOMETRY_FILES]
    + [f"{ZIP_PREFIX}/Readme.md", f"{ZIP_PREFIX}/LICENSE.txt"]
)

_CHUNK = 8 * 1024 * 1024


def default_data_dir() -> Path:
    """Return ``<datasets_root>/solpsnn`` (honours ``SURGE_DATA_DIR``)."""
    return paths.datasets_root() / "solpsnn"


# --------------------------------------------------------------------------
# HTTP range access to the remote zip
# --------------------------------------------------------------------------
def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SURGE_CA_BUNDLE")
    if not cafile:
        try:
            import certifi

            cafile = certifi.where()
        except ImportError:
            cafile = None
    return ssl.create_default_context(cafile=cafile)


def _remote_size(url: str) -> int:
    """Total size of ``url`` via a one-byte range request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        cr = resp.headers.get("Content-Range")
        if not cr or "/" not in cr:
            raise OSError(
                f"{url}: server did not return Content-Range; Range unsupported"
            )
        return int(cr.rsplit("/", 1)[1])


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests."""

    def __init__(self, url: str, size: int | None = None):
        super().__init__()
        self.url = url
        self.size = int(size) if size is not None else _remote_size(url)
        self._pos = 0
        self.bytes_read = 0
        self._ctx = _ssl_context()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, self._pos)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""
        end = self._pos + size - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=120, context=self._ctx) as resp:
            data = resp.read()
        self._pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def zip_members(
    url: str = ZENODO_ZIP_URL, size: int | None = None
) -> list[zipfile.ZipInfo]:
    """List the members of the remote zip (reads only the central directory)."""
    with zipfile.ZipFile(_RangeFile(url, size)) as zf:
        return zf.infolist()


def download_members(
    members: Iterable[str],
    dest: Path,
    *,
    url: str = ZENODO_ZIP_URL,
    size: int | None = None,
    progress: bool = True,
) -> list[Path]:
    """Extract ``members`` from the remote zip into ``dest / member``.

    Members already present with the right size are skipped.
    """
    dest = Path(dest)
    rf = _RangeFile(url, size)
    out: list[Path] = []
    with zipfile.ZipFile(rf) as zf:
        for name in members:
            info = zf.getinfo(name)
            target = dest / name
            out.append(target)
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            done = 0
            with zf.open(info) as src, open(tmp, "wb") as dst:
                while chunk := src.read(_CHUNK):
                    dst.write(chunk)
                    done += len(chunk)
                    if progress:
                        sys.stderr.write(
                            f"\r{name}: {done / 1e6:.0f}/{info.file_size / 1e6:.0f} MB"
                        )
            tmp.replace(target)
            if progress:
                sys.stderr.write("\n")
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class SolpsnnSplit:
    """One split of the SOLPS-NN dataset (completed runs only)."""

    X: np.ndarray  # (n, 8) float64, physical units
    X_legacy: np.ndarray  # (n, 8)
    ids: np.ndarray  # (n,) completed simulation ids
    failed_ids: np.ndarray  # (m,) crashed simulation ids
    scalars: dict[str, np.ndarray]
    te: np.ndarray | None  # (n, 104, 50) electron temperature in eV
    split: str

    @property
    def X_log(self) -> np.ndarray:
        """Copy of ``X`` with log10 applied to ``LOG_INPUT_COLUMNS``."""
        out = np.array(self.X, dtype=np.float64, copy=True)
        out[:, list(LOG_INPUT_COLUMNS)] = np.log10(out[:, list(LOG_INPUT_COLUMNS)])
        return out

    def outer_target_te(self) -> np.ndarray:
        """Te at the outer target over the physical radial cells, ``(n, 48)`` eV."""
        if self.te is None:
            raise ValueError("Te was not loaded (with_te=False)")
        return self.te[:, OUTER_TARGET_ROW, 1:-1]


def _root(data_dir: Path | str | None) -> Path:
    return Path(data_dir) if data_dir is not None else default_data_dir()


def _ensure(root: Path, rels: list[str], download: bool) -> None:
    missing = [r for r in rels if not (root / ZIP_PREFIX / r).is_file()]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(
            f"SOLPS-NN files missing under {root / ZIP_PREFIX}: {missing}. Dataset: Zenodo "
            f"record {ZENODO_RECORD}. Fetch with surge.benchmarks.loaders.solpsnn."
            f"download_members(MINIMAL_MEMBERS, dest=Path('{root}')) or pass download=True."
        )
    download_members([f"{ZIP_PREFIX}/{r}" for r in missing], root)


def load_geometry(data_dir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(crx, cry)``, each ``(104, 50, 4)``: cell-corner x/y in metres."""
    root = _root(data_dir)
    _ensure(root, list(_GEOMETRY_FILES), download=True)
    g = root / ZIP_PREFIX / "geometry"
    return (
        np.load(g / "crx.npy", allow_pickle=False),
        np.load(g / "cry.npy", allow_pickle=False),
    )


def load_solpsnn(
    split: str = "train",
    *,
    data_dir: Path | str | None = None,
    with_te: bool = True,
    download: bool = True,
) -> SolpsnnSplit:
    """Load one split; downloads only the needed members when missing."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = _root(data_dir)
    files = [
        f for f in _SPLIT_FILES if f != "simulations.db" and (with_te or f != "te.npy")
    ]
    _ensure(root, [f"{split}/{f}" for f in files], download)
    d = root / ZIP_PREFIX / split

    def ld(name: str, **kw) -> np.ndarray:
        return np.load(d / name, allow_pickle=False, **kw)

    te = None
    if with_te:
        te = np.asarray(ld("te.npy", mmap_mode="r"), dtype=np.float64)
        te = (te / EV).astype(np.float32)
    return SolpsnnSplit(
        X=ld("X_units.npy").astype(np.float64),
        X_legacy=ld("X_legacy.npy"),
        ids=ld("completed.npy"),
        failed_ids=ld("failed.npy"),
        scalars={k: ld(f"{k}.npy") for k in SCALAR_TARGETS},
        te=te,
        split=split,
    )


def load_regression(
    split: str,
    target: str,
    *,
    data_dir: Path | str | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_log, y)`` for a scalar target or ``"te_field"`` (n, 5200) eV."""
    if target != "te_field" and target not in SCALAR_TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose from {[*SCALAR_TARGETS, 'te_field']}"
        )
    s = load_solpsnn(
        split, data_dir=data_dir, with_te=target == "te_field", download=download
    )
    if target == "te_field":
        return s.X_log, s.te.reshape(len(s.te), -1)
    return s.X_log, s.scalars[target]
