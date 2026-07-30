#!/usr/bin/env python3
"""Rollout physics statistics — does the surrogate capture the turbulence?

The literature's honest long-horizon claim for chaotic systems is
STATISTICAL fidelity, not pointwise accuracy: after many autoregressive
steps the trajectory diverges (it must), but a good surrogate keeps the
right mean profiles, distributions, and spectra. This study rolls the
Well-protocol ConvNeXt U-Net (trained by thewell_wellprotocol_study.py)
over a held-out TRL-2D trajectory and compares against the truth on:

  (a) mean vertical density profile  <log rho>(y)   (mixing structure)
  (b) density PDF                                     (phase distribution)
  (c) density power spectrum along x                  (turbulent scales)
  (d) cold-phase mass fraction vs time                (the TRL observable:
      radiative mixing drives net hot->cold mass flux)

Usage (after the protocol study):
    SURGE_DEVICE=auto python examples/thewell_rollout_stats.py [--steps 40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surge.viz.theme import save_figure, sequential_cmap, surge_theme

OUT = _REPO / "examples" / "viz_gallery_output"
N_HIST = 4
N_FIELDS = 4
DENS = 0                     # field order: density, pressure, vx, vy


def load_trajectory(traj_idx: int = 0):
    """One full valid trajectory: (T, 4, 128, 384)."""
    import glob

    import h5py

    root = (Path.home() / ".surge/data/thewell/datasets"
            / "turbulent_radiative_layer_2D/data/valid")
    f = sorted(glob.glob(str(root / "*.hdf5")))[0]
    with h5py.File(f) as h:
        fields = []
        for name in ("density", "pressure", "velocity_x", "velocity_y"):
            for group in ("t0_fields", "t1_fields", "t2_fields"):
                if name in h.get(group, {}):
                    fields.append(h[f"{group}/{name}"][traj_idx])
                    break
    return np.stack(fields, axis=1).astype("float32")   # (T, 4, H, W)


def load_model():
    import joblib
    import torch

    from surge.model.backends.convnext_unet import _CNextUNet
    from surge.utils import resolve_device

    d = joblib.load(OUT / "cunet_wellprotocol.joblib")
    cfg = d["config"]
    net = _CNextUNet(N_HIST * N_FIELDS, N_FIELDS, cfg["base_channels"],
                     cfg["depth"], cfg["blocks_per_stage"])
    net.load_state_dict(d["net_state"])
    dev = resolve_device(None)
    net.to(dev).eval()
    return net, dev


def rollout(net, dev, traj, n_steps, mu, sd):
    """Autoregressive rollout; returns (n_steps, 4, H, W) predictions."""
    import torch

    hist = [traj[i] for i in range(N_HIST)]              # each (4, H, W)
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = np.concatenate(hist[-N_HIST:], axis=0)   # (16, H, W)
            xn = (x - np.tile(mu[0], (N_HIST, 1, 1))) / \
                np.tile(sd[0], (N_HIST, 1, 1))
            xt = torch.from_numpy(xn[None]).to(dev)
            y = net(xt).cpu().numpy()[0] * sd[0] + mu[0]
            preds.append(y)
            hist.append(y)
    return np.stack(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    npz = np.load(OUT / "wellprotocol_preds.npz")
    mu, sd = npz["norm_mu"], npz["norm_sd"]

    traj = load_trajectory(0)
    print(f"[traj] {traj.shape}")
    net, dev = load_model()
    preds = rollout(net, dev, traj, args.steps, mu, sd)
    truth = traj[N_HIST:N_HIST + args.steps]
    print(f"[rollout] {args.steps} steps on {dev}")

    lr_t = np.log10(np.maximum(truth[:, DENS], 1e-10))
    lr_p = np.log10(np.maximum(preds[:, DENS], 1e-10))

    # cold-phase mass fraction (density above the geometric mean of the
    # two phases — TRL densities span ~[0.1..100] in code units)
    thresh = 10 ** (0.5 * (lr_t.min() + lr_t.max()))
    frac_t = [(truth[k, DENS] > thresh).mean() for k in range(args.steps)]
    frac_p = [(preds[k, DENS] > thresh).mean() for k in range(args.steps)]

    def spectrum(f):
        # mean power spectrum of log-density along x, averaged over y+time
        F = np.fft.rfft(f, axis=-1)
        return (np.abs(F) ** 2).mean(axis=(0, 1))

    for mode in ("light", "dark"):
        with surge_theme(mode) as p:
            fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.1))

            ax = axes[0]
            ax.plot(lr_t.mean(axis=(0, 2)), lw=1.7, color=p["good"],
                    label="truth")
            ax.plot(lr_p.mean(axis=(0, 2)), lw=1.5, color=p["series"][0],
                    ls=(0, (4, 2)), label="rollout")
            ax.set_xlabel("y index")
            ax.set_ylabel(r"$\langle \log_{10}\rho \rangle$")
            ax.set_title("(a) mean mixing profile", fontsize=9)
            ax.legend(fontsize=6.5)

            ax = axes[1]
            bins = np.linspace(lr_t.min(), lr_t.max(), 60)
            ax.hist(lr_t.ravel(), bins=bins, density=True, alpha=0.55,
                    color=p["good"], label="truth")
            ax.hist(lr_p.ravel(), bins=bins, density=True, alpha=0.55,
                    color=p["series"][0], label="rollout")
            ax.set_xlabel(r"$\log_{10}\rho$")
            ax.set_yticks([])
            ax.set_title("(b) density PDF (two phases)", fontsize=9)
            ax.legend(fontsize=6.5)

            ax = axes[2]
            k = np.arange(1, lr_t.shape[-1] // 2 + 1)
            ax.loglog(k, spectrum(lr_t)[1:], lw=1.7, color=p["good"],
                      label="truth")
            ax.loglog(k, spectrum(lr_p)[1:], lw=1.5, color=p["series"][0],
                      ls=(0, (4, 2)), label="rollout")
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel("power")
            ax.set_title("(c) density spectrum", fontsize=9)
            ax.legend(fontsize=6.5)

            ax = axes[3]
            ax.plot(frac_t, lw=1.7, color=p["good"], label="truth")
            ax.plot(frac_p, lw=1.5, color=p["series"][0], ls=(0, (4, 2)),
                    label="rollout")
            ax.set_xlabel("rollout step")
            ax.set_ylabel("cold-phase area fraction")
            ax.set_title("(d) mass-flux observable", fontsize=9)
            ax.legend(fontsize=6.5)

            fig.suptitle(
                f"Turbulence statistics under {args.steps}-step "
                "autoregressive rollout — ConvNeXt U-Net, Well protocol "
                "(pointwise divergence is expected; statistics are the claim)",
                fontsize=10, fontweight="bold")
            for path in save_figure(fig, OUT / f"thewell_rollout_{mode}"):
                print("wrote", path)
            plt.close(fig)


if __name__ == "__main__":
    main()
