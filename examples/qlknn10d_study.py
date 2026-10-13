#!/usr/bin/env python3
"""Reproduce QLKNN from its own public training data (QLKNN10D).

Task: predict the QuaLiKiz ITG leading flux (ion heat flux efiITG, GB
units) on ITG-unstable points of the 9-D input grid — the task the
published QLKNN networks solve (van de Plassche et al., PoP 27, 022310
(2020)). Data: 2.4M-row strided subsample of the 290M-row public table
(Zenodo 10.5281/zenodo.3497066, CC-BY 4.0), cached by the
``plasma.qlknn10d`` benchmark loader.

Reference line: the published QLKNN_7_11 network (DeepMind
`fusion_surrogates`) evaluated on the same held-out rows. Caveats
stated on the figure: that model is trained on a later data generation
with different flux normalisation, so a single least-squares scale
factor (fit on training rows) is applied and the line is labelled
"rescaled"; residual mismatch is convention, not necessarily quality.

Usage:
    SURGE_DEVICE=auto python examples/qlknn10d_study.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from surge.viz.theme import density_cmap, save_figure, surge_theme

OUT = _REPO / "examples" / "viz_gallery_output"
COLS = ["Zeff", "Ati", "Ate", "An", "q", "smag", "x", "Ti_Te", "Nustar"]


def qlknn_reference(X9: np.ndarray, normni: float = 0.9):
    """QLKNN_7_11 efiITG on 9-D rows (normni approximated — caveat)."""
    from fusion_surrogates.qlknn import qlknn_model
    import jax.numpy as jnp

    model = qlknn_model.QLKNNModel.load_default_model()
    An = X9[:, 3]
    X10 = np.column_stack([
        X9[:, 1], X9[:, 2], An, An, X9[:, 4], X9[:, 5], X9[:, 6],
        X9[:, 7], np.log10(np.maximum(X9[:, 8], 1e-10)),
        np.full(len(X9), normni, dtype=np.float32)])
    pred = model.predict(jnp.array(X10, dtype=jnp.float32))
    return np.array(pred["efiITG"]).ravel()


def main() -> None:
    from sklearn.metrics import r2_score

    from surge.benchmarks.leaderboard import _load_qlknn10d
    from surge.model import MODEL_REGISTRY

    X, y = _load_qlknn10d()
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(X))
    n_te = 200_000
    te, tr = perm[:n_te], perm[n_te:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"[data] {len(tr):,} train / {len(te):,} test "
          f"(ITG-unstable rows of QLKNN10D)")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    # ── SURGE models ──────────────────────────────────────────────────
    for label, key, params, log_target in (
        ("ridge", "sklearn.ridge", {}, False),
        ("residual MLP", "pytorch.residual_mlp",
         {"hidden_layers": [512, 512, 256], "n_epochs": 60,
          "batch_size": 4096, "patience": 15, "patience_window": 5,
          "random_state": 0}, False),
        ("residual MLP (log)", "pytorch.residual_mlp",
         {"hidden_layers": [512, 512, 256], "n_epochs": 60,
          "batch_size": 4096, "patience": 15, "patience_window": 5,
          "random_state": 0}, True),
    ):
        model = MODEL_REGISTRY.create(key, **params)
        target = np.log10(ytr) if log_target else ytr
        t_mu, t_sd = target.mean(), target.std()
        t0 = time.time()
        model.fit(((Xtr - mu) / sd), (target - t_mu) / t_sd)
        raw = np.asarray(model.predict((Xte - mu) / sd)).ravel() * t_sd + t_mu
        p = 10 ** raw if log_target else raw
        dt = time.time() - t0
        r2 = r2_score(yte, p)
        results[label] = {"r2": float(r2), "runtime_s": dt}
        preds[label] = p
        print(f"[done] {label:20s} R2 {r2:.4f} ({dt:.0f}s)")

    # ── published QLKNN reference (rescaled; convention caveat) ──────
    try:
        ref_raw = qlknn_reference(Xte)
        ok = np.isfinite(ref_raw) & (ref_raw > 0)
        # single LS scale factor fit on TRAIN rows (fair: fixes pure
        # normalisation difference between data generations)
        ref_tr = qlknn_reference(Xtr[:100_000])
        oktr = np.isfinite(ref_tr) & (ref_tr > 0)
        lam = float(np.dot(ytr[:100_000][oktr], ref_tr[oktr])
                    / np.dot(ref_tr[oktr], ref_tr[oktr]))
        r2_ref = r2_score(yte[ok], lam * ref_raw[ok])
        results["QLKNN_7_11 (rescaled)"] = {
            "r2": float(r2_ref), "runtime_s": 0.0, "scale": lam,
            "coverage": float(ok.mean())}
        preds["QLKNN_7_11 (rescaled)"] = np.where(ok, lam * ref_raw, np.nan)
        print(f"[ref ] QLKNN_7_11 x{lam:.2f}: R2 {r2_ref:.4f} "
              f"(coverage {ok.mean():.2f})")
    except Exception as exc:  # noqa: BLE001 - reference is optional
        print(f"[ref ] QLKNN_7_11 unavailable: {exc}")

    # ── figure ────────────────────────────────────────────────────────
    best = max((k for k in preds if "QLKNN" not in k),
               key=lambda k: results[k]["r2"])
    for mode in ("light", "dark"):
        with surge_theme(mode) as p:
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8),
                                     width_ratios=[1.0, 1.15])
            ax = axes[0]
            order = sorted(results, key=lambda k: results[k]["r2"])
            ypos = np.arange(len(order))
            for i, k in enumerate(order):
                col = (p["series"][2] if "QLKNN" in k else p["series"][0])
                ax.barh(i, results[k]["r2"], height=0.6, color=col,
                        alpha=0.9)
                rt = results[k]["runtime_s"]
                lbl = f" {results[k]['r2']:.3f}"
                if rt:
                    lbl += f" · {rt:.0f}s"
                ax.text(results[k]["r2"], i, lbl, va="center", fontsize=7.5,
                        color=p["ink2"])
            ax.set_yticks(ypos)
            ax.set_yticklabels(order, fontsize=8)
            ax.set_xlabel("held-out R² (raw flux)")
            ax.set_xlim(0, 1.15)
            ax.set_title("(a) 2.2M-row training, 200k held out", fontsize=9)

            ax = axes[1]
            cmap = density_cmap(mode)
            ax.set_facecolor(cmap.get_under())
            lim = np.percentile(yte, 99.5)
            bins = np.linspace(0, lim, 60)
            ax.hist2d(yte, preds[best], bins=[bins, bins], cmap=cmap,
                      norm=LogNorm(vmin=1), cmin=1)
            ax.plot([0, lim], [0, lim], color=p["ink2"], lw=1.1,
                    ls=(0, (5, 3)))
            ax.set_xlabel(r"QuaLiKiz  $q_i^{ITG}$  [GB]")
            ax.set_ylabel(f"{best} prediction")
            ax.set_title(f"(b) parity — {best}, "
                         f"R² {results[best]['r2']:.3f}", fontsize=9)

            fig.suptitle(
                "Reproducing QLKNN from its public training data "
                "(QLKNN10D, 290M QuaLiKiz fluxes; ITG leading flux)",
                fontsize=10.5, fontweight="bold")
            for path in save_figure(fig, OUT / f"qlknn10d_{mode}"):
                print("wrote", path)
            plt.close(fig)

    (OUT / "qlknn10d_summary.json").write_text(json.dumps({
        "results": results,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "note": ("QLKNN_7_11 reference trained on a later data "
                 "generation; single LS rescale applied (fit on train); "
                 "residual mismatch partly convention."),
    }, indent=2))
    print(json.dumps({k: round(v["r2"], 4) for k, v in results.items()},
                     indent=2))


if __name__ == "__main__":
    main()
