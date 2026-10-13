#!/usr/bin/env python3
"""Probabilistic forecasting of the turbulent layer — CRPS, not RMSE.

The seven-lever battery established that pointwise deterministic error
on TRL-2D at dt=8 is chaos-limited. This study changes the question the
way the weather-forecasting literature does: predict a calibrated
DISTRIBUTION of future fields and score it with the Continuous Ranked
Probability Score,

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|,   X, X' ~ F,

estimated from ensemble samples (per pixel, averaged). A deep ensemble
(``pytorch.mlp_ensemble``) is trained on POD mode coefficients — modal
uncertainty maps back to field-space ensemble members via the POD
inverse. Baselines: a deterministic forecast treated as a point mass
(its CRPS = its MAE) and the persistence point mass.

Also reports the spread-skill relation (binned ensemble spread vs
actual error) — the calibration diagnostic.

Usage:
    SURGE_DEVICE=auto python examples/thewell_probabilistic_study.py
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

from surge.preprocessing import pod_fit, pod_inverse, pod_transform
from surge.viz.theme import save_figure, surge_theme

OUT = _REPO / "examples" / "viz_gallery_output"
K = 64
N_MEMBERS = 8


def crps_ensemble(members: np.ndarray, truth: np.ndarray) -> float:
    """Ensemble CRPS averaged over pixels+samples. members: (M, n, D)."""
    M = members.shape[0]
    term1 = np.abs(members - truth[None]).mean()
    term2 = 0.0
    for a in range(M):
        for b in range(a + 1, M):
            term2 += np.abs(members[a] - members[b]).mean()
    term2 = 2.0 * term2 / (M * M)          # E|X - X'| with self-pairs zero
    return float(term1 - 0.5 * term2)


def main() -> None:
    from examples.thewell_turbulence_study import load_trl2d
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_trl2d(800, 100, seed=0, horizon=8)
    npz = np.load(OUT / "thewell_turbulence_preds.npz")
    assert np.allclose(npz["yte"], yte, atol=1e-5)
    unet_pred = npz["pred__U-Net"].reshape(yte.shape)
    nte = len(yte)
    yf = yte.reshape(nte, -1)

    bx = pod_fit(Xtr.reshape(len(Xtr), -1), K)
    by = pod_fit(ytr.reshape(len(ytr), -1), K)
    cx_tr = pod_transform(bx, Xtr.reshape(len(Xtr), -1)).astype("float32")
    cy_tr = pod_transform(by, ytr.reshape(len(ytr), -1)).astype("float32")
    cx_te = pod_transform(bx, Xte.reshape(nte, -1)).astype("float32")
    mu, sd = cy_tr.mean(0), cy_tr.std(0) + 1e-8

    ens = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble", n_ensembles=N_MEMBERS, hidden_dim=256,
        n_layers=2, n_epochs=150, patience=25, random_state=0)
    t0 = time.time()
    ens.fit(cx_tr, (cy_tr - mu) / sd)
    print(f"[fit] {N_MEMBERS}-member ensemble on {K} POD modes: "
          f"{time.time() - t0:.0f}s")

    raw = np.asarray(ens._model._predict_raw(cx_te))   # (M, n, K)
    members = np.stack([
        pod_inverse(by, raw[m] * sd + mu) for m in range(N_MEMBERS)])

    # raw deep-ensemble spread is underdispersed (same finding as the
    # QLKNN UQ study) — inflate member deviations by a factor tuned on
    # half the test set, evaluate everything on the held-out half
    cal, hold = slice(0, nte // 2), slice(nte // 2, nte)
    m_mean = members.mean(0)
    lams = np.linspace(1.0, 4.0, 13)
    lam = float(min(lams, key=lambda L: crps_ensemble(
        m_mean[None, cal] + L * (members[:, cal] - m_mean[None, cal]),
        yf[cal])))
    members_cal = m_mean[None] + lam * (members - m_mean[None])
    print(f"[calibration] spread inflation lambda = {lam:.2f} (tuned on half)")

    crps_ens = crps_ensemble(members[:, hold], yf[hold])
    crps_ens_cal = crps_ensemble(members_cal[:, hold], yf[hold])
    crps_unet = float(np.abs(unet_pred.reshape(nte, -1)[hold] - yf[hold]).mean())
    crps_pers = float(np.abs(Xte.reshape(nte, -1)[hold] - yf[hold]).mean())
    crps_mean = float(np.abs(m_mean[hold] - yf[hold]).mean())
    print(f"[CRPS held-out] raw ens {crps_ens:.4f} | calibrated ens "
          f"{crps_ens_cal:.4f} | ens-mean point {crps_mean:.4f} | U-Net "
          f"point {crps_unet:.4f} | persistence {crps_pers:.4f}")

    # spread-skill on the held-out half, with calibrated spread
    spread = members_cal[:, hold].std(0).mean(axis=1)
    err = np.abs(m_mean[hold] - yf[hold]).mean(axis=1)
    order = np.argsort(spread)
    nb = 10
    bins = np.array_split(order, nb)
    sp_b = [spread[b].mean() for b in bins]
    er_b = [err[b].mean() for b in bins]
    corr = float(np.corrcoef(spread, err)[0, 1])
    print(f"[spread-skill] corr(spread, error) = {corr:.3f}")

    for mode in ("light", "dark"):
        with surge_theme(mode) as p:
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))

            ax = axes[0]
            labels = ["persistence\n(point)", "U-Net\n(point)",
                      "ens. mean\n(point)", "raw\nensemble",
                      f"calibrated\nens. (×{lam:.1f})"]
            vals = [crps_pers, crps_unet, crps_mean, crps_ens, crps_ens_cal]
            cols = [p["series"][2], p["axis"], p["series"][1],
                    p["series"][0], p["good"]]
            ax.bar(range(5), vals, color=cols, alpha=0.9)
            for i, v in enumerate(vals):
                ax.text(i, v, f" {v:.3f}", ha="center", va="bottom",
                        fontsize=7)
            ax.set_xticks(range(5))
            ax.set_xticklabels(labels, fontsize=6.3)
            ax.set_ylabel("CRPS (lower is better)")
            ax.set_title("(a) probabilistic skill", fontsize=9)

            ax = axes[1]
            ax.plot(sp_b, er_b, marker="o", ms=4.5, lw=1.6,
                    color=p["series"][0], label="binned test samples")
            lo = min(min(sp_b), min(er_b))
            hi = max(max(sp_b), max(er_b))
            ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=1.1,
                    color=p["axis"], label="perfect calibration")
            ax.set_xlabel("ensemble spread")
            ax.set_ylabel("error of ensemble mean")
            ax.set_title(f"(b) spread–skill (r = {corr:.2f})", fontsize=9)
            ax.legend(fontsize=6.5)

            ax = axes[2]
            i = int(np.argsort(spread)[-1])            # most uncertain held-out sample
            px = np.linspace(0, 1, members.shape[-1])
            H, W = yte.shape[1:]
            col = W // 2
            for m in range(N_MEMBERS):
                ax.plot(members_cal[m, hold][i].reshape(H, W)[:, col], lw=0.8,
                        color=p["series"][0], alpha=0.35)
            ax.plot(yte[hold][i][:, col], lw=1.6, color=p["good"],
                    label="truth")
            ax.plot(m_mean[hold][i].reshape(H, W)[:, col], lw=1.4,
                    color=p["series"][1], ls=(0, (4, 2)), label="ens. mean")
            ax.set_xlabel("y index (mid-column profile)")
            ax.set_ylabel(r"$\log_{10}\rho$")
            ax.set_title("(c) members fan out where it matters", fontsize=9)
            ax.legend(fontsize=6.5)

            fig.suptitle(
                "Probabilistic turbulence forecasting — deep ensemble on "
                f"{K} POD modes, TRL-2D Δt = 8 (chaos scored honestly: CRPS)",
                fontsize=10.5, fontweight="bold")
            for path in save_figure(fig, OUT / f"thewell_crps_{mode}"):
                print("wrote", path)
            plt.close(fig)

    (OUT / "thewell_crps_summary.json").write_text(json.dumps({
        "crps_heldout": {"raw_ensemble": crps_ens,
                         "calibrated_ensemble": crps_ens_cal,
                         "ensemble_mean_point": crps_mean,
                         "unet_point": crps_unet,
                         "persistence_point": crps_pers},
        "spread_inflation": lam,
        "spread_skill_corr": corr, "k_modes": K, "members": N_MEMBERS,
    }, indent=2))


if __name__ == "__main__":
    main()
