"""Per-benchmark dataset preview figures for the leaderboard report.

Each preview is a small themed matplotlib figure showing what the raw data
of a benchmark actually looks like (sample images, field pairs, attractor
trajectories, feature/target densities). Previews are rendered ONLY from
locally cached data: all loaders run inside a socket guard so report
generation can never trigger a download — a benchmark whose cache is
missing simply gets no preview.
"""
from __future__ import annotations

import gzip
import io
import pickle
import socket
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from surge.viz.theme import density_cmap, surge_theme

_REPO = Path(__file__).resolve().parents[2]
_BENCH_DATA = _REPO / "data" / "datasets" / "benchmarks"

_FIGSIZE = (3.6, 2.5)


@contextmanager
def _no_network():
    """Hard-block sockets so cached-loader fallbacks can't download."""
    real = socket.socket

    def _blocked(*a, **k):
        raise OSError("network disabled during report generation")

    socket.socket = _blocked  # type: ignore[misc]
    try:
        yield
    finally:
        socket.socket = real  # type: ignore[misc]


def _fig_to_svg(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None},
                bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


# ------------------------------------------------------------- renderers

def _tabular_regression(X, y, mode, xlabel="strongest input", ylabel="target"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[keep], y[keep]
    if len(y) > 20000:
        sel = np.random.default_rng(0).choice(len(y), 20000, replace=False)
        X, y = X[sel], y[sel]
    # most-correlated input carries the story
    with np.errstate(invalid="ignore"):
        corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                         if X[:, j].std() > 0 else 0.0
                         for j in range(X.shape[1])])
    j = int(np.nanargmax(corr))

    with surge_theme(mode) as p:
        cmap = density_cmap(mode)
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        ax.set_facecolor(cmap.get_under())
        h = ax.hist2d(X[:, j], y, bins=44, cmap=cmap,
                      norm=LogNorm(vmin=1), cmin=1)
        cb = fig.colorbar(h[3], ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("counts", fontsize=6.5, color=p["muted"])
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
        ax.set_xlabel(f"{xlabel} (|ρ|={corr[j]:.2f})", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.5)
        return fig


def _tabular_classification(X, y, mode, class_names=None):
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    classes = np.unique(y)[:8]
    if len(y) > 6000:
        sel = np.random.default_rng(0).choice(len(y), 6000, replace=False)
        X, y = X[sel], y[sel]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # top-2 principal components separate classes better than raw columns
    _, _, vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    pc = Xs @ vt[:2].T

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for i, c in enumerate(classes):
            m = y == c
            label = str(class_names[int(c)]) if class_names is not None else str(c)
            ax.scatter(pc[m, 0], pc[m, 1], s=7, alpha=0.6, linewidths=0,
                       color=p["series"][i % 8], label=label[:14])
        ax.set_xlabel("PC 1", fontsize=7.5); ax.set_ylabel("PC 2", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ncol = 2 if len(classes) > 4 else 1
        ax.legend(fontsize=5.8, ncol=ncol, markerscale=1.6,
                  handletextpad=0.3, columnspacing=0.8, frameon=False)
        return fig


def _image_grid(imgs, labels, mode, gray=True):
    import matplotlib.pyplot as plt

    with surge_theme(mode):
        fig, axes = plt.subplots(2, 4, figsize=_FIGSIZE)
        for ax, img, lab in zip(axes.ravel(), imgs, labels):
            ax.imshow(img, cmap="gray_r" if gray and mode == "light"
                      else "gray" if gray else None,
                      interpolation="nearest")
            ax.set_title(str(lab), fontsize=6.5, pad=2)
            ax.axis("off")
        return fig


def _field_pairs(X, Y, mode, n=3):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    idx = rng.choice(len(X), n, replace=False)
    xg = np.arange(X.shape[1])
    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for k, i in enumerate(idx):
            c = p["series"][k]
            ax.plot(xg, X[i], color=c, lw=1.0, ls=(0, (3, 2)), alpha=0.8)
            ax.plot(xg, Y[i], color=c, lw=1.7)
        ax.plot([], [], color=p["ink2"], ls=(0, (3, 2)), lw=1.0,
                label="u(x, 0)")
        ax.plot([], [], color=p["ink2"], lw=1.7, label="u(x, T)")
        ax.set_xlabel("x index", fontsize=7.5); ax.set_ylabel("u", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.5, frameon=False)
        return fig


def _lorenz(X, Y, mode, n=4):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), n, replace=False)
    with surge_theme(mode) as p:
        fig = plt.figure(figsize=_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(p["surface"])
        for i in idx:
            a = X[i].reshape(-1, 3)
            b = Y[i].reshape(-1, 3)
            full = np.vstack([a, b])
            ax.plot(*full.T, color=p["series"][0], lw=0.7, alpha=0.35)
            ax.plot(*b.T, color=p["series"][1], lw=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x", fontsize=7, labelpad=-12)
        ax.set_ylabel("y", fontsize=7, labelpad=-12)
        ax.set_zlabel("z", fontsize=7, labelpad=-12)
        ax.grid(False)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(p["surface"]); pane.set_edgecolor(p["grid"])
        return fig


# ----------------------------------------------------------- data loaders

def _mnist_samples():
    raw = _BENCH_DATA / "vision" / "MNIST" / "raw"
    fi = raw / "t10k-images-idx3-ubyte"
    fl = raw / "t10k-labels-idx1-ubyte"
    if not fi.exists():
        gz = fi.with_suffix(fi.suffix + ".gz")
        if not gz.exists():
            return None
        data = gzip.decompress(gz.read_bytes())
        labels = gzip.decompress(fl.with_suffix(fl.suffix + ".gz").read_bytes())
    else:
        data, labels = fi.read_bytes(), fl.read_bytes()
    _, n, rows, cols = struct.unpack(">IIII", data[:16])
    imgs = np.frombuffer(data, np.uint8, offset=16).reshape(n, rows, cols)
    labs = np.frombuffer(labels, np.uint8, offset=8)
    return imgs[:8], labs[:8]


def _cifar_samples():
    batch = _BENCH_DATA / "vision" / "cifar-10-batches-py" / "data_batch_1"
    meta = _BENCH_DATA / "vision" / "cifar-10-batches-py" / "batches.meta"
    if not batch.exists():
        return None
    d = pickle.loads(batch.read_bytes(), encoding="bytes")
    names = pickle.loads(meta.read_bytes(), encoding="bytes")[b"label_names"]
    imgs = d[b"data"][:8].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labs = [names[i].decode() for i in d[b"labels"][:8]]
    return imgs, labs


def _npz(relpath: str):
    f = _BENCH_DATA / relpath
    if not f.exists():
        return None
    return np.load(f, allow_pickle=True)


# --------------------------------------------------------------- dispatch

def preview_figure(key: str, mode: str = "dark"):
    """Build the preview figure for one benchmark key, or None."""
    if key in ("vision.mnist",):
        s = _mnist_samples()
        return _image_grid(*s, mode, gray=True) if s else None
    if key == "vision.cifar10":
        s = _cifar_samples()
        return _image_grid(*s, mode, gray=False) if s else None
    if key == "pde.burgers_1d":
        d = _npz("pde/burgers_1d.npz")
        return _field_pairs(d["X"], d["y"], mode) if d is not None else None
    if key == "sequence.lorenz63":
        d = _npz("sequence/lorenz63.npz")
        return _lorenz(d["X"], d["y"], mode) if d is not None else None
    if key == "plasma.qlknn_transport":
        d = _npz("plasma/qlknn_transport.npz")
        if d is None:
            return None
        return _tabular_regression(d["X"], d["y"], mode,
                                   xlabel="Ati (norm. ∇Ti)",
                                   ylabel="efeITG [gB]")
    if key == "plasma.cmod_density_limit":
        d = _npz("plasma/cmod_density_limit.npz")
        return (_tabular_classification(d["X"], d["y"], mode,
                                        ["stable", "disrupted"])
                if d is not None else None)
    if key == "classification.plasma_stability":
        d = _npz("classification/plasma_stability.npz")
        return (_tabular_classification(d["X"], d["y"], mode,
                                        ["unstable", "stable"])
                if d is not None else None)

    if key == "tabular.california_housing":
        # loader redirects data_home to the repo cache; also accept the
        # default sklearn cache (~/scikit_learn_data) before giving up
        try:
            from sklearn.datasets import fetch_california_housing
            with _no_network():
                X, y = fetch_california_housing(return_X_y=True)
            return _tabular_regression(X, y, mode,
                                       xlabel="median income",
                                       ylabel="house value [$100k]")
        except Exception:  # noqa: BLE001 — cache miss => no preview
            return None

    # the paper/multioutput ConStellaration variants share the base dataset
    if key.startswith("plasma.constellaration"):
        key = "plasma.constellaration"

    # generic path: any loader that works offline (sklearn built-ins,
    # inline synthetic generators, warm sklearn/OpenML caches)
    try:
        from surge.benchmarks.leaderboard import _load_dataset
        with _no_network():
            loaded = _load_dataset(key)
    except Exception:  # noqa: BLE001 — no cache/no loader => no preview
        return None
    if loaded is None:
        return None
    X, y = loaded[0], loaded[1]
    y_arr = np.asarray(y)
    is_clf = (y_arr.dtype.kind in "iub" and len(np.unique(y_arr)) <= 20)
    if key == "tabular.digits":
        X_arr = np.asarray(X, dtype=float)
        return _image_grid(X_arr[:8].reshape(-1, 8, 8), y_arr[:8].astype(int),
                           mode, gray=True)
    if is_clf:
        return _tabular_classification(X, y_arr, mode)
    if y_arr.ndim > 1 and y_arr.shape[1] > 8:  # field-like target
        return _field_pairs(np.asarray(X, dtype=float),
                            y_arr.astype(float), mode)
    if y_arr.ndim > 1:
        y_arr = y_arr[:, 0]
    return _tabular_regression(X, y_arr, mode)


def preview_svg(key: str, mode: str = "dark") -> str:
    """Inline SVG preview for one benchmark, or '' if unavailable."""
    try:
        fig = preview_figure(key, mode)
    except Exception:  # noqa: BLE001 — a broken preview must not kill reports
        return ""
    return _fig_to_svg(fig) if fig is not None else ""
