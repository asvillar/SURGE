#!/usr/bin/env python3
"""Simformer (all-in-one SBI) correctness demo on an analytic problem.

Linear-Gaussian simulator — θ ~ N(0, I₂), x = Aθ + 0.1ε — is one of the
few SBI problems whose posterior is known in closed form, which makes it
the honest validation: train ``pytorch.simformer`` on (θ, x) pairs and
compare its sampled posterior against the analytic N(μ*, Σ*) for a held
-out observation. The same trained network also samples the likelihood
p(x | θ) and the joint.

Output: examples/viz_gallery_output/simformer_sbi_{light,dark}.png

Usage:
    SURGE_DEVICE=auto python examples/simformer_sbi_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from surge.viz.theme import save_figure, surge_theme


def analytic_posterior(A, x_obs, noise_var):
    prec = np.eye(A.shape[1]) + A.T @ A / noise_var
    cov = np.linalg.inv(prec)
    mean = cov @ (A.T @ x_obs / noise_var)
    return mean, cov


def cov_ellipse(ax, mean, cov, n_sigma, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    w, h = 2 * n_sigma * np.sqrt(vals[-1]), 2 * n_sigma * np.sqrt(vals[0])
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, **kw))


def main() -> None:
    from surge.model import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.standard_normal((n, 2)).astype("float32")
    A = np.array([[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7]], dtype="float32")
    x = theta @ A.T + 0.1 * rng.standard_normal((n, 3)).astype("float32")

    model = MODEL_REGISTRY.create(
        "pytorch.simformer", n_epochs=200, d_model=64, n_layers=3,
        batch_size=256, random_state=0)
    model.fit(x, theta)

    th_true = np.array([1.0, -0.5], dtype="float32")
    x_obs = th_true @ A.T
    S = model.sample_posterior(x_obs, n_samples=1500)
    mu, cov = analytic_posterior(A, x_obs, 0.01)
    print("analytic mean", np.round(mu, 3), "| simformer", np.round(S.mean(0), 3))

    L = model.sample_likelihood(th_true, n_samples=800)
    J = model._model.sample_joint(1500)

    for mode in ("light", "dark"):
        with surge_theme(mode) as p:
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))

            ax = axes[0]
            ax.scatter(S[:, 0], S[:, 1], s=4, alpha=0.25,
                       color=p["series"][0], label="Simformer posterior")
            for k in (1, 2):
                cov_ellipse(ax, mu, cov, k, edgecolor=p["ink"], lw=1.3,
                            ls=(0, (4, 3)))
            ax.plot(*th_true, marker="*", ms=14, color=p["warning"],
                    markeredgecolor=p["ink"], label="true θ")
            ax.plot([], [], color=p["ink"], ls=(0, (4, 3)),
                    label="analytic 1σ/2σ")
            ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
            ax.set_title("(a) posterior p(θ|x) vs analytic", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper left")

            ax = axes[1]
            ax.scatter(L[:, 0], L[:, 1], s=4, alpha=0.25,
                       color=p["series"][1], label="p(x|θ) samples")
            ax.plot(x_obs[0], x_obs[1], marker="*", ms=14,
                    color=p["warning"], markeredgecolor=p["ink"],
                    label="noise-free Aθ")
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            ax.set_title("(b) same network, likelihood", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper left")

            ax = axes[2]
            ax.scatter(J[:, 0], J[:, 2], s=4, alpha=0.2,
                       color=p["series"][2], label="joint samples")
            ax.scatter(theta[:800, 0], x[:800, 0], s=4, alpha=0.2,
                       color=p["muted"], label="training data")
            ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$x_1$")
            ax.set_title("(c) same network, joint p(θ, x)", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper left")

            fig.suptitle(
                "Simformer — one score-based transformer, every conditional "
                "(linear-Gaussian benchmark with known posterior)",
                fontsize=10.5, fontweight="bold")
            out = _REPO / "examples" / "viz_gallery_output"
            for path in save_figure(fig, out / f"simformer_sbi_{mode}"):
                print("wrote", path)
            plt.close(fig)


if __name__ == "__main__":
    main()
