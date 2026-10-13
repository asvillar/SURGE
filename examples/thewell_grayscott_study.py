#!/usr/bin/env python3
"""TheWell Gray-Scott surrogate study: FNO-2D vs U-Net vs DeepONet vs ridge.

Task: operator forecasting on the Gray-Scott reaction–diffusion system
from TheWell (Ohana et al., NeurIPS 2024) — predict the species-B field
160 stored timesteps ahead. A persistence baseline anchors the
leaderboard: consecutive frames are so similar that the single-step task
is trivial (persistence rel-L2 ~0.002), so the horizon is chosen where
"predict no change" visibly fails (~0.28).

Requires the dataset cache created by
``surge.benchmarks.loaders.thewell.download_thewell("gray_scott")``
(~132 GB on disk: 117 GB train + 15 GB valid).
Fields are downsampled 128 -> 64 and a single species channel is used so
every registered operator model consumes the same flat (B, 64*64) task.

Usage (from repo root):
    python examples/thewell_grayscott_study.py \
        [--n-train 400] [--n-test 80] [--side 64] [--epochs 60] \
        [--out examples/viz_gallery_output]
"""
from __future__ import annotations

import argparse
import json
import re
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
from matplotlib.colors import CenteredNorm

from surge.viz.theme import (
    diverging_cmap,
    fmt_metric,
    save_figure,
    sequential_cmap,
    surge_theme,
)

SPECIES = 1          # Gray-Scott channel: 0 = A (feed), 1 = B (pattern-forming)

# --- direct-prediction anchor from the headline study (README, this file's
# --horizon 160 run): U-Net (residual) median rel-L2 at a single 160-step
# jump, used as the honest comparison point for the rollout curve below.
DIRECT_160_RELL2 = 0.206
DIRECT_160_PERSISTENCE = 0.265


def load_grayscott(n_train: int, n_test: int, side: int, seed: int = 0,
                   horizon: int = 160):
    """(X, y) pairs: species-B field at t -> t + horizon stored steps.

    Consecutive stored frames are nearly identical (persistence scores
    rel-L2 ~0.002 at horizon 1), so a single-step task is trivial and every
    model loses to "predict no change". At horizon 160 persistence degrades
    to ~0.34 and the surrogate must learn actual pattern dynamics.
    """
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "gray_scott_reaction_diffusion" / "data"
    out = {}
    for split, needed in (("train", n_train), ("valid", n_test)):
        ds = WellDataset(
            path=str(root / split),
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,
            min_dt_stride=horizon,
            max_dt_stride=horizon,
        )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(needed, len(ds)), replace=False)
        X_list, y_list = [], []
        stride = None
        for i in idx:
            item = ds[int(i)]
            xin = np.asarray(item["input_fields"])[0, ..., SPECIES]
            yout = np.asarray(item["output_fields"])[0, ..., SPECIES]
            if stride is None:
                stride = max(1, xin.shape[0] // side)
            X_list.append(xin[::stride, ::stride][:side, :side])
            y_list.append(yout[::stride, ::stride][:side, :side])
        X = np.stack(X_list).astype(np.float32)
        y = np.stack(y_list).astype(np.float32)
        out[split] = (X.reshape(len(X), -1), y.reshape(len(y), -1))
    return out["train"], out["valid"]


def load_grayscott_trajectories(side: int, seed: int, n_traj: int,
                                return_meta: bool = False):
    """``n_traj`` full held-out species-B trajectories from the valid split.

    Read directly from the cached HDF5 (bypassing ``WellDataset``'s pair
    sampler) so every stored timestep is available for an autoregressive
    rollout, not just one ``(t, t+horizon)`` pair per trajectory. Same
    128 -> ``side`` spatial downsampling as :func:`load_grayscott`. Draws
    from a different ``.hdf5`` file per trajectory when enough files are
    available, so the rollout isn't scored on one regime's dynamics.

    ``return_meta=True`` additionally returns a list of ``{"file",
    "traj_index"}`` dicts (one per trajectory) so callers can label figures
    by regime (see ``--rollout-gif``).
    """
    import glob

    import h5py

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "gray_scott_reaction_diffusion" / "data" / "valid"
    files = sorted(glob.glob(str(root / "*.hdf5")))
    rng = np.random.default_rng(seed + 1)  # offset from the pair-sampling seed
    # sample distinct files without replacement up to len(files), then wrap
    file_order = []
    while len(file_order) < n_traj:
        file_order.extend(rng.permutation(len(files)).tolist())
    trajs, metas = [], []
    for k in range(n_traj):
        f = files[file_order[k]]
        with h5py.File(f) as h:
            B = h["t0_fields/B"]
            # a handful of Gray-Scott parameter regimes decay to a uniform
            # B=0 field partway through the trajectory (the reaction dies
            # out); such a trajectory makes every downstream rel-L2
            # denominator ~0 and its ratio blows up, corrupting the median.
            # Resample within this file (up to len(B) tries) for one whose
            # late-trajectory norm is non-degenerate.
            n_avail = B.shape[0]
            for _attempt in range(n_avail):
                i = int(rng.integers(n_avail))
                field = np.asarray(B[i])          # (T, 128, 128)
                if float(np.linalg.norm(field[-1])) > 1e-6:
                    break
            else:
                raise RuntimeError(
                    f"every trajectory in {f} decays to a degenerate "
                    "(near-zero) field; cannot sample a usable trajectory")
        stride = max(1, field.shape[-1] // side)
        field = field[:, ::stride, ::stride][:, :side, :side]
        trajs.append(field.astype(np.float32))
        metas.append({"file": f, "traj_index": i})
    if return_meta:
        return trajs, metas
    return trajs


def load_grayscott_pairs_with_t0(side: int, seed: int, n_pairs: int,
                                 horizon: int, split: str,
                                 t0_min: int = 0, t0_max: int | None = None):
    """``n_pairs`` (X, y, t0) samples drawn uniformly over (file, traj, t0).

    Unlike :func:`load_grayscott` (which uses ``WellDataset``'s pair sampler
    and discards the start index), this reads the cached HDF5 directly so
    each pair's stored start step ``t0`` is recorded — the regime split
    (``--regime-split``) needs it to separate transient (``t0 < 200``) from
    settled (``t0 >= 200``) test pairs. ``t0_min``/``t0_max`` restrict the
    sampled start window (e.g. ``t0_min=200`` for a settled-only training
    set); default samples the whole trajectory.
    """
    import glob

    import h5py

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "gray_scott_reaction_diffusion" / "data" / split
    files = sorted(glob.glob(str(root / "*.hdf5")))
    rng = np.random.default_rng(seed)
    handles = [h5py.File(f, "r") for f in files]
    try:
        X_list, y_list, t0_list = [], [], []
        stride = None
        for _ in range(n_pairs):
            fi = int(rng.integers(len(handles)))
            B = handles[fi]["t0_fields/B"]
            n_traj, traj_len = B.shape[0], B.shape[1]
            hi = (traj_len - horizon - 1) if t0_max is None else min(
                t0_max, traj_len - horizon - 1)
            if hi < t0_min:
                raise ValueError(
                    f"t0 window [{t0_min}, {hi}] is empty for "
                    f"traj_len={traj_len}, horizon={horizon}")
            ti = int(rng.integers(n_traj))
            t0 = int(rng.integers(t0_min, hi + 1))
            xin = np.asarray(B[ti, t0])
            yout = np.asarray(B[ti, t0 + horizon])
            if stride is None:
                stride = max(1, xin.shape[-1] // side)
            X_list.append(xin[::stride, ::stride][:side, :side])
            y_list.append(yout[::stride, ::stride][:side, :side])
            t0_list.append(t0)
    finally:
        for h in handles:
            h.close()
    X = np.stack(X_list).astype(np.float32).reshape(n_pairs, -1)
    y = np.stack(y_list).astype(np.float32).reshape(n_pairs, -1)
    t0 = np.array(t0_list, dtype=np.int64)
    return X, y, t0


def run_study(n_train: int, n_test: int, side: int, epochs: int, seed: int = 0,
              horizon: int = 160):
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_grayscott(n_train, n_test, side, seed,
                                            horizon)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}  horizon {horizon}")

    # honesty anchor: every model must beat "predict no change"
    rel0 = (np.linalg.norm(Xte - yte, axis=1)
            / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
    results = [{
        "key": "baseline.persistence", "label": "Persistence B(t)",
        "residual_target": False, "runtime_s": 0.0,
        "rel_l2_median": float(np.median(rel0)),
        "pred": Xte.copy(), "rel": rel0,
    }]
    print(f"[done] {'Persistence B(t)':16s} median rel-L2 "
          f"{np.median(rel0):.4f} (0s)")

    # residual=True trains on the *increment* B(t+1) − B(t) and adds the
    # input back at prediction time — the standard fix for architectures
    # without local spatial bias (DeepONet's low-rank global basis cannot
    # represent "sharpen each filament in place", but the small residual
    # field is far easier to expand)
    candidates = [
        ("pytorch.fno2d", "FNO-2D", {"n_modes": 24, "n_epochs": epochs}, False),
        ("pytorch.fno2d", "FNO-2D (residual)",
         {"n_modes": 24, "n_epochs": epochs}, True),
        ("pytorch.unet", "U-Net", {"n_epochs": epochs}, False),
        ("pytorch.unet", "U-Net (residual)", {"n_epochs": epochs}, True),
        ("pytorch.deeponet", "DeepONet", {"n_epochs": epochs}, False),
        ("pytorch.deeponet", "DeepONet (residual)", {"n_epochs": epochs}, True),
        ("sklearn.ridge", "Ridge (linear)", {}, False),
    ]
    for key, label, params, residual in candidates:
        try:
            model = MODEL_REGISTRY.create(key, **params)
        except KeyError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr - Xtr if residual else ytr)
            pred = np.asarray(model.predict(Xte)).reshape(len(Xte), -1)
            if residual:
                pred = pred + Xte
        except Exception as exc:  # noqa: BLE001 - study reports, never dies
            print(f"[fail] {label}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        rel = (np.linalg.norm(pred - yte, axis=1)
               / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
        # median only: early-time Gray-Scott frames have near-zero species-B
        # norm, so the per-sample relative error (and its mean) diverges
        results.append({
            "key": key, "label": label, "residual_target": residual,
            "runtime_s": dt,
            "rel_l2_median": float(np.median(rel)),
            "pred": pred, "rel": rel,
        })
        print(f"[done] {label:16s} median rel-L2 {np.median(rel):.4f} "
              f"({fmt_metric(dt, 'runtime')})")
    return (Xte, yte), results


def study_figure(data, results, side: int, mode: str = "light",
                 horizon: int = 160):
    Xte, yte = data
    results = sorted(results, key=lambda r: r["rel_l2_median"])
    # median-difficulty sample judged by the best model
    best = results[0]
    i = int(np.argsort(best["rel"])[len(best["rel"]) // 2])
    truth_img = yte[i].reshape(side, side)

    # the persistence "prediction" is the input field itself — keep it in
    # the comparison bars but don't show the same image twice; cap the image
    # row at the four best models so the figure stays readable
    panels = [r for r in results if r["key"] != "baseline.persistence"][:4]

    with surge_theme(mode) as p:
        n_models = len(panels)
        fig = plt.figure(figsize=(2.9 * (n_models + 2), 5.6))
        gs = fig.add_gridspec(2, n_models + 2, height_ratios=[1.15, 1.0])

        # one shared color scale anchored to the truth: models that produce
        # out-of-range noise must LOOK wrong, not get their own flattering
        # auto-scaled colorbar
        vmin = float(min(truth_img.min(), Xte[i].min()))
        vmax = float(max(truth_img.max(), Xte[i].max()))
        field_kw = dict(cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax)

        field_axes = []
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(Xte[i].reshape(side, side), **field_kw)
        ax.set_title(r"input  $B(t)$", fontsize=9.5)
        field_axes.append(ax)

        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(truth_img, **field_kw)
        ax.set_title(r"truth  $B(t+\Delta t)$", fontsize=9.5)
        field_axes.append(ax)

        for k, r in enumerate(panels):
            ax = fig.add_subplot(gs[0, k + 2])
            ax.imshow(r["pred"][i].reshape(side, side), **field_kw)
            ax.set_title(f"{r['label']}\nrel-L2 {r['rel'][i]:.3f}", fontsize=9)
            field_axes.append(ax)

        for ax in field_axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cbar = fig.colorbar(im, ax=field_axes, fraction=0.012, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.set_label(r"$B$  (shared scale)", fontsize=8.5)

        # bottom: error map of best model + rel-L2 comparison bars
        ax = fig.add_subplot(gs[1, 0:2])
        err = best["pred"][i].reshape(side, side) - truth_img
        vmax = float(np.abs(truth_img).max()) * 0.1 + 1e-9
        im = ax.imshow(err, cmap=diverging_cmap(mode),
                       norm=CenteredNorm(halfrange=vmax))
        ax.set_title(f"error map — {best['label']}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)

        ax = fig.add_subplot(gs[1, 2:])
        ypos = np.arange(len(results))[::-1]
        meds = [r["rel_l2_median"] for r in results]
        colors = [p["series"][2] if r["key"] == "baseline.persistence"
                  else p["series"][0] for r in results]
        ax.barh(ypos, meds, height=0.6, color=colors, alpha=0.9)
        for y, r in zip(ypos, results):
            label = f"  {r['rel_l2_median']:.4f}"
            if r["key"] != "baseline.persistence":
                label += f" · {fmt_metric(r['runtime_s'], 'runtime')}"
            ax.text(r["rel_l2_median"], y, label,
                    va="center", fontsize=8, color=p["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in results], fontsize=9)
        ax.set_xlabel("median rel-L2 (lower is better)")
        ax.set_title("model comparison", fontsize=9.5)

        task = ("next-step surrogates" if horizon == 1
                else "forecast surrogates")
        fig.suptitle(
            f"TheWell · Gray-Scott reaction–diffusion — {task}"
            r"  $B(t) \mapsto B(t+\Delta t)$"
            f", $\\Delta t$ = {horizon} stored step{'s' if horizon > 1 else ''}"
            f"  ({side}×{side}, species B)",
            fontsize=11.5, fontweight="bold")
        return fig


SETTLE_STEPS = 200  # stored steps before Gray-Scott is out of the pattern-
                     # formation transient (see module docstring / task brief)


def run_rollout_study(n_train: int, epochs: int, side: int, seed: int,
                      dt_candidates: tuple[int, ...] = (20, 40, 80),
                      total_horizon: int = 160, n_traj: int = 5):
    """Pick a single-step Δt (smallest that beats persistence), train one
    U-Net (residual target) at it, and roll it out autoregressively to
    ``total_horizon`` stored steps starting from settled-regime states on
    ``n_traj`` held-out trajectories.

    Returns ``(chosen_dt, dt_table, beats_persistence, model, rollout_dict,
    trained)`` where ``rollout_dict`` has ``horizons``, ``rel_l2`` (median
    rollout over trajectories), ``rel_l2_iqr``, ``persistence`` (median),
    per-trajectory curves, per-trajectory frame arrays (``per_traj_initial``/
    ``per_traj_truths``/``per_traj_preds``), ``traj_meta`` (source file +
    index per trajectory), and the single representative GIF trajectory's
    frames. ``trained`` is ``{dt: fitted_model}`` for every candidate Δt, so
    a caller that wants a *specific* Δt's model (not necessarily the one
    ``choose_dt`` picked) doesn't need to retrain it.
    """
    from surge.inference.rollout import (
        autoregressive_rollout,
        choose_dt,
        persistence_rel_l2,
        sample_rollout_starts,
    )
    from surge.model import MODEL_REGISTRY

    # 1. sweep Δt: for each candidate, train the residual U-Net and measure
    #    BOTH the model's and persistence's median rel-L2 on 100 held-out
    #    pairs. Persistence at these strides is computed on the same pairs
    #    (not settled-state trajectories) since it's a single-step sweep to
    #    pick Δt, not the rollout itself.
    dt_table: dict[int, tuple[float, float]] = {}
    trained: dict[int, object] = {}
    for dt in dt_candidates:
        (Xtr, ytr), (Xte, yte) = load_grayscott(n_train, n_test=100, side=side,
                                                seed=seed, horizon=dt)
        model = MODEL_REGISTRY.create("pytorch.unet", n_epochs=epochs)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr - Xtr)
        dt_fit = time.perf_counter() - t0
        pred = model.predict(Xte).reshape(len(Xte), -1) + Xte
        rel = (np.linalg.norm(pred - yte, axis=1)
               / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
        rel_persist = (np.linalg.norm(Xte - yte, axis=1)
                       / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
        model_med, persist_med = float(np.median(rel)), float(np.median(rel_persist))
        dt_table[dt] = (model_med, persist_med)
        trained[dt] = model
        print(f"[single-step] dt={dt:3d}  model {model_med:.4f}  "
              f"persistence {persist_med:.4f}  ({fmt_metric(dt_fit, 'runtime')})")

    chosen_dt, beats_persistence = choose_dt(dt_table)
    model = trained[chosen_dt]
    model_med, persist_med = dt_table[chosen_dt]
    if beats_persistence:
        print(f"[chosen] dt={chosen_dt}: model {model_med:.4f} < "
              f"persistence {persist_med:.4f}")
    else:
        print(f"[chosen] dt={chosen_dt}: NO candidate Δt beat persistence "
              f"(model {model_med:.4f} vs persistence {persist_med:.4f} "
              "— rolling out anyway, honestly flagged)")

    # 2. autoregressive rollout on n_traj held-out trajectories, each started
    #    from a settled-regime state (t_start in [SETTLE_STEPS, T -
    #    total_horizon - 1]) so the rollout isn't scored on the pattern-
    #    formation transient the training pairs never sampled.
    n_apps = total_horizon // chosen_dt
    assert n_apps * chosen_dt == total_horizon, (
        f"chosen_dt={chosen_dt} must divide total_horizon={total_horizon}")
    trajs, traj_meta = load_grayscott_trajectories(
        side, seed, n_traj=n_traj, return_meta=True)
    t_starts = sample_rollout_starts(
        n_traj, trajectory_length=trajs[0].shape[0],
        total_horizon=total_horizon, settle_steps=SETTLE_STEPS, seed=seed)

    def predict_fn(state_img):
        flat = np.asarray(state_img, dtype=np.float32).reshape(1, -1)
        out = model.predict(flat)
        return np.asarray(out).reshape(side, side)

    horizons = [k * chosen_dt for k in range(1, n_apps + 1)]
    per_traj_rel_l2, per_traj_persist = [], []
    per_traj_frames = []  # (initial_state, truths, preds) per trajectory
    for traj, t_start in zip(trajs, t_starts):
        t_start = int(t_start)
        truths = np.stack([traj[t_start + k * chosen_dt]
                           for k in range(1, n_apps + 1)])
        initial_state = traj[t_start]
        preds, rel_l2 = autoregressive_rollout(predict_fn, initial_state, truths,
                                               residual=True)
        persistence = persistence_rel_l2(initial_state, truths)
        per_traj_rel_l2.append(rel_l2)
        per_traj_persist.append(persistence)
        per_traj_frames.append((initial_state, truths, preds))

    per_traj_rel_l2 = np.stack(per_traj_rel_l2)      # (n_traj, n_apps)
    per_traj_persist = np.stack(per_traj_persist)    # (n_traj, n_apps)
    rel_l2_median = np.median(per_traj_rel_l2, axis=0)
    rel_l2_iqr = (np.percentile(per_traj_rel_l2, 75, axis=0)
                 - np.percentile(per_traj_rel_l2, 25, axis=0))
    persistence_median = np.median(per_traj_persist, axis=0)

    # trajectory whose own median error is closest to the overall median —
    # used for the GIF so it's representative, not cherry-picked
    traj_medians = np.median(per_traj_rel_l2, axis=1)
    gif_idx = int(np.argmin(np.abs(traj_medians - np.median(traj_medians))))
    gif_initial, gif_truths, gif_preds = per_traj_frames[gif_idx]

    rollout = {
        "horizons": horizons,
        "rel_l2": rel_l2_median.tolist(),
        "rel_l2_iqr": rel_l2_iqr.tolist(),
        "persistence": persistence_median.tolist(),
        "per_traj_rel_l2": per_traj_rel_l2.tolist(),
        "per_traj_persistence": per_traj_persist.tolist(),
        "t_start_per_traj": [int(t) for t in t_starts],
        "n_traj": n_traj,
        "initial_state": gif_initial,
        "truths": gif_truths,
        "preds": gif_preds,
        # the GIF trajectory's OWN per-step error (not the cross-trajectory
        # median) — used for the per-frame title so it matches what's shown
        "gif_rel_l2": per_traj_rel_l2[gif_idx].tolist(),
        "traj_meta": traj_meta,
        "per_traj_initial": [f[0] for f in per_traj_frames],
        "per_traj_truths": [f[1] for f in per_traj_frames],
        "per_traj_preds": [f[2] for f in per_traj_frames],
    }
    return chosen_dt, dt_table, beats_persistence, model, rollout, trained


def rollout_gif(rollout: dict, side: int, out_path: Path, mode: str = "dark",
                fps: int = 10) -> Path:
    """Assemble the truth | rollout | signed-error GIF (see make_helmholtz_gif.py
    for the retired pattern this follows: one PNG buffer per frame, palette
    quantized, assembled with PIL)."""
    import io

    from PIL import Image

    truths = rollout["truths"]
    preds = rollout["preds"]
    horizons = rollout["horizons"]
    rel_l2 = rollout.get("gif_rel_l2", rollout["rel_l2"])
    init = rollout["initial_state"]

    frames_data = np.concatenate([init[None], truths, preds])
    vmin_field, vmax_field = float(frames_data.min()), float(frames_data.max())
    vmax_err = float(np.abs(truths - preds).max()) + 1e-9

    with surge_theme(mode) as p:
        buf_frames = []
        for k in range(len(horizons)):
            fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
            err = preds[k] - truths[k]
            for ax, img, title, cmap, norm in (
                (axes[0], truths[k], "truth", sequential_cmap(mode),
                 dict(vmin=vmin_field, vmax=vmax_field)),
                (axes[1], preds[k], "U-Net rollout", sequential_cmap(mode),
                 dict(vmin=vmin_field, vmax=vmax_field)),
                (axes[2], err, "signed error", diverging_cmap(mode),
                 dict(norm=CenteredNorm(halfrange=vmax_err))),
            ):
                ax.imshow(img, cmap=cmap, **norm)
                ax.set_title(title, fontsize=8.5, color=p["ink"])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
            fig.suptitle(
                f"Gray-Scott B(t) rollout — step {horizons[k]}/{horizons[-1]}"
                f"  ·  rel-L2 {rel_l2[k]:.3f}",
                fontsize=9.5, fontweight="bold", color=p["ink"])
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90,
                       facecolor=p["surface"])
            plt.close(fig)
            buf.seek(0)
            buf_frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    buf_frames[0].save(out_path, save_all=True, append_images=buf_frames[1:],
                       duration=duration_ms, loop=0, optimize=True)
    return out_path


def rollout_curve_figure(rollout: dict, mode: str = "light"):
    """rel-L2 vs horizon: median rollout + IQR band, median persistence,
    direct-160 anchor and its persistence counterpart."""
    horizons = np.asarray(rollout["horizons"])
    rel_l2 = np.asarray(rollout["rel_l2"])
    iqr = np.asarray(rollout["rel_l2_iqr"])
    persistence = rollout["persistence"]
    n_traj = rollout.get("n_traj", 1)

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.fill_between(horizons, rel_l2 - iqr / 2, rel_l2 + iqr / 2,
                        color=p["series"][0], alpha=0.18, linewidth=0,
                        label="rollout IQR")
        ax.plot(horizons, rel_l2, marker="o", ms=4, lw=1.8,
               color=p["series"][0], label="U-Net rollout (median)")
        ax.plot(horizons, persistence, ls=(0, (4, 2)), lw=1.6,
               color=p["series"][2], label="persistence (median)")
        ax.scatter([160], [DIRECT_160_RELL2], marker="*", s=180,
                  color=p["warning"], edgecolor=p["ink"], linewidth=0.6,
                  zorder=5,
                  label=f"direct single-shot @160 ({DIRECT_160_RELL2:.3f})")
        ax.scatter([160], [DIRECT_160_PERSISTENCE], marker="X", s=90,
                  color=p["ink2"], edgecolor=p["ink"], linewidth=0.5,
                  zorder=5,
                  label=f"persistence @160 ({DIRECT_160_PERSISTENCE:.3f})")
        ax.set_xlabel("horizon (stored steps)")
        ax.set_ylabel("rel-L2")
        ax.set_title(
            "Gray-Scott autoregressive rollout\n"
            f"settled-state starts, median of {n_traj} trajectories",
            fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        return fig


def regime_label(file_path: str) -> str:
    """``".../gray_scott_reaction_diffusion_maze_F_0.029_k_0.057.hdf5"`` ->
    ``"maze F=0.029 k=0.057"`` for figure row/legend labels."""
    stem = Path(file_path).stem
    m = re.search(
        r"gray_scott_reaction_diffusion_(?P<regime>\w+?)_F_(?P<F>[\d.]+)_k_(?P<k>[\d.]+)$",
        stem)
    if not m:
        return stem
    return f"{m['regime']} F={m['F']} k={m['k']}"


def multicase_rollout_gif(rollout: dict, out_path: Path, mode: str = "dark",
                          fps: int = 8) -> Path:
    """4-case x 4-panel (truth | rollout | persistence | signed error) GIF,
    one frame per application, row label = regime (see :func:`regime_label`).
    """
    import io

    from PIL import Image

    from surge.inference.rollout import layout_case_grid_frames

    truths = np.stack(rollout["per_traj_truths"])        # (n_cases, n_apps, H, W)
    preds = np.stack(rollout["per_traj_preds"])
    persistence = np.stack(rollout["per_traj_initial"])   # (n_cases, H, W)
    horizons = rollout["horizons"]
    per_case_rel_l2 = np.asarray(rollout["per_traj_rel_l2"])   # (n_cases, n_apps)
    per_case_persist = np.asarray(rollout["per_traj_persistence"])
    labels = [regime_label(m["file"]) for m in rollout["traj_meta"]]
    n_cases = truths.shape[0]

    frames = layout_case_grid_frames(truths, preds, persistence)  # (n_apps, n_cases, 4, H, W)
    panel_titles = ["truth", "U-Net rollout", "persistence", "signed error"]

    field_vmin = float(min(truths.min(), preds.min(), persistence.min()))
    field_vmax = float(max(truths.max(), preds.max(), persistence.max()))
    err_vmax = float(np.abs(preds - truths).max()) + 1e-9

    with surge_theme(mode) as p:
        buf_frames = []
        for k in range(len(horizons)):
            fig, axes = plt.subplots(n_cases, 4, figsize=(6.4, 1.6 * n_cases))
            for row in range(n_cases):
                for col in range(4):
                    ax = axes[row, col]
                    img = frames[k, row, col]
                    if col == 3:
                        ax.imshow(img, cmap=diverging_cmap(mode),
                                 norm=CenteredNorm(halfrange=err_vmax))
                    else:
                        ax.imshow(img, cmap=sequential_cmap(mode),
                                 vmin=field_vmin, vmax=field_vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    if row == 0:
                        ax.set_title(panel_titles[col], fontsize=8,
                                    color=p["ink"])
                    if col == 0:
                        ax.set_ylabel(labels[row], fontsize=6.5,
                                     color=p["ink2"])
            median_roll = float(np.median(per_case_rel_l2[:, k]))
            median_pers = float(np.median(per_case_persist[:, k]))
            fig.suptitle(
                f"step {horizons[k]} / {horizons[-1]}  ·  rollout rel-L2 "
                f"(median over cases) {median_roll:.3f}  ·  "
                f"persistence {median_pers:.3f}",
                fontsize=9, fontweight="bold", color=p["ink"])
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, facecolor=p["surface"])
            plt.close(fig)
            buf.seek(0)
            buf_frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    buf_frames[0].save(out_path, save_all=True, append_images=buf_frames[1:],
                       duration=duration_ms, loop=0, optimize=True)
    return out_path


def multicase_rollout_curve_figure(rollout: dict, mode: str = "light"):
    """rel-L2 vs horizon: one thin line per case + bold median rollout +
    dashed median persistence, with the direct-160 anchor and its
    persistence counterpart marked."""
    horizons = np.asarray(rollout["horizons"])
    per_case_rel_l2 = np.asarray(rollout["per_traj_rel_l2"])
    per_case_persist = np.asarray(rollout["per_traj_persistence"])
    labels = [regime_label(m["file"]) for m in rollout["traj_meta"]]
    median_roll = np.median(per_case_rel_l2, axis=0)
    median_pers = np.median(per_case_persist, axis=0)

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for i, label in enumerate(labels):
            ax.plot(horizons, per_case_rel_l2[i], lw=0.9, alpha=0.5,
                   color=p["series"][0], label=f"{label} (rollout)"
                   if i == 0 else None)
        ax.plot(horizons, median_roll, marker="o", ms=4, lw=2.2,
               color=p["series"][0], label="median rollout")
        ax.plot(horizons, median_pers, ls=(0, (4, 2)), lw=1.8,
               color=p["series"][2], label="median persistence")
        ax.scatter([160], [DIRECT_160_RELL2], marker="*", s=180,
                  color=p["warning"], edgecolor=p["ink"], linewidth=0.6,
                  zorder=5,
                  label=f"direct single-shot @160 ({DIRECT_160_RELL2:.3f})")
        ax.scatter([160], [DIRECT_160_PERSISTENCE], marker="X", s=90,
                  color=p["ink2"], edgecolor=p["ink"], linewidth=0.5,
                  zorder=5,
                  label=f"persistence @160 ({DIRECT_160_PERSISTENCE:.3f})")
        ax.set_xlabel("horizon (stored steps)")
        ax.set_ylabel("rel-L2")
        ax.set_title(
            "Gray-Scott multi-regime rollout\n"
            f"{len(labels)} held-out cases from distinct regimes",
            fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper left")
        return fig


def run_rollout_gif_cases(args) -> None:
    """``--rollout-gif``: 4-case (4 distinct regimes), long-horizon (480-step,
    24-application) rollout grid GIF + curve + numbers table."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = 0
    total_horizon = 480
    dt = 20
    n_apps = total_horizon // dt

    _chosen_dt, dt_table, beats_persistence, _model, rollout, _trained = run_rollout_study(
        args.n_train, args.epochs, args.side, seed,
        dt_candidates=(dt,), total_horizon=total_horizon, n_traj=4)
    print(f"[trained] dt={dt} model {dt_table[dt][0]:.4f} vs "
          f"persistence {dt_table[dt][1]:.4f}")

    labels = [regime_label(m["file"]) for m in rollout["traj_meta"]]
    print(f"[cases] {labels}")

    gif_path = multicase_rollout_gif(
        rollout, out / "grayscott_rollout_cases.gif", mode="dark", fps=8)
    size_mb = gif_path.stat().st_size / 1e6
    print(f"wrote {gif_path} ({size_mb:.2f} MB, {n_apps} frames)")
    if size_mb > 6.0:
        print(f"[warn] gif is {size_mb:.2f} MB, over the 6 MB budget")

    for mode in ("light", "dark"):
        fig = multicase_rollout_curve_figure(rollout, mode)
        for path in save_figure(fig, out / f"grayscott_rollout_cases_{mode}"):
            print("wrote", path)
        plt.close(fig)

    per_case = []
    for i, meta in enumerate(rollout["traj_meta"]):
        per_case.append({
            "file": meta["file"], "traj_index": meta["traj_index"],
            "t_start": rollout["t_start_per_traj"][i],
            "horizons": rollout["horizons"],
            "rollout_rel_l2": rollout["per_traj_rel_l2"][i],
            "persistence_rel_l2": rollout["per_traj_persistence"][i],
        })
    summary = {
        "dt": dt,
        "n_apps": n_apps,
        "total_horizon": total_horizon,
        "beats_persistence_at_dt": beats_persistence,
        "cases": per_case,
        "median_rollout_rel_l2": rollout["rel_l2"],
        "median_persistence_rel_l2": rollout["persistence"],
        "horizons": rollout["horizons"],
        "direct_160_rel_l2": DIRECT_160_RELL2,
        "direct_160_persistence": DIRECT_160_PERSISTENCE,
    }
    (out / "grayscott_rollout_cases.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "grayscott_rollout_cases.json")



# ---------------------------------------------------------------------------
# --rollout-longjump: roll out the *headline* 160-step model recursively
# ---------------------------------------------------------------------------

def identify_headline_case(side: int, seed: int, n_train: int, n_test: int,
                           epochs: int, horizon: int = 160, out_dir=None):
    """Recover ``(file, traj_index, t0)`` of the README's headline sample.

    Reproduces the headline study's test set and best model, picks the same
    median-difficulty sample as :func:`study_figure`, then brute-force
    matches its input field against every valid-split frame with
    :func:`surge.inference.rollout.match_frame`. Returns the identity dict
    and the trained horizon-``horizon`` residual U-Net (kept so the caller
    can roll it out without retraining).
    """
    import glob

    import h5py

    from surge.benchmarks.loaders.thewell import _well_base_path
    from surge.inference.rollout import match_frame
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte_new, yte_new) = load_grayscott(n_train, n_test, side, seed, horizon)
    model = MODEL_REGISTRY.create("pytorch.unet", n_epochs=epochs)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr - Xtr)
    print(f"[headline] U-Net (residual) trained @h={horizon} in "
          f"{fmt_metric(time.perf_counter() - t0, 'runtime')}")
    # Identity of the README figure's sample must be DETERMINISTIC: prefer the
    # headline run's own saved test set + predictions (thewell_grayscott_preds.npz),
    # which is exactly what study_figure rendered. Re-training and re-picking the
    # median sample lands on a different trajectory every run (training is not
    # bit-reproducible), so that path is only a fallback.
    npz = Path(out_dir) / "thewell_grayscott_preds.npz" if out_dir else None
    sj = Path(out_dir) / "thewell_grayscott_summary.json" if out_dir else None
    if npz is not None and npz.exists() and sj.exists():
        d = np.load(npz)
        Xte, yte = d["Xte"], d["yte"]
        summ = json.loads(sj.read_text())
        best_run = min((r for r in summ if r["key"] != "baseline.persistence"),
                       key=lambda r: r["rel_l2_median"])
        pred = d[f"pred__{best_run['label']}"]
        source = f"saved README run · {best_run['label']}"
    else:
        Xte, yte = Xte_new, yte_new
        pred = np.asarray(model.predict(Xte)).reshape(len(Xte), -1) + Xte
        source = "retrained model (README preds npz not found)"
    rel = (np.linalg.norm(pred - yte, axis=1)
           / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
    i = int(np.argsort(rel)[len(rel) // 2])
    target = Xte[i].reshape(side, side)
    print(f"[headline] median-difficulty sample i={i} from {source}, "
          f"rel-L2 {rel[i]:.4f} (study median {np.median(rel):.4f})")

    root = _well_base_path(None) / "gray_scott_reaction_diffusion" / "data" / "valid"
    best = None
    for f in sorted(glob.glob(str(root / "*.hdf5"))):
        with h5py.File(f) as h:
            B = h["t0_fields/B"]
            for ti in range(B.shape[0]):
                idx, diff = match_frame(target, np.asarray(B[ti]), stride=2, side=side)
                if best is None or diff < best["max_abs_diff"]:
                    best = {"file": f, "traj_index": ti, "t0": idx,
                            "max_abs_diff": diff}
                if diff < 1e-6:
                    break
        if best is not None and best["max_abs_diff"] < 1e-6:
            break
    assert best is not None and best["max_abs_diff"] < 1e-5, (
        f"could not identify the headline sample (best diff {best})")
    best.update({
        "regime": regime_label(best["file"]),
        "identity_source": source,
        "transient_start": bool(best["t0"] < SETTLE_STEPS),
        "sample_index": i, "sample_rel_l2_direct": float(rel[i]),
        "study_median_rel_l2": float(np.median(rel)),
        "horizon": horizon, "n_train": n_train, "n_test": n_test,
        "epochs": epochs, "seed": seed,
    })
    print(f"[headline] identified: {best['regime']} traj {best['traj_index']} "
          f"t0={best['t0']} (diff {best['max_abs_diff']:.2e}, "
          f"transient={best['transient_start']})")
    return best, model


def _load_case_trajectory(file: str, traj_index: int, side: int) -> np.ndarray:
    import h5py

    with h5py.File(file) as h:
        field = np.asarray(h["t0_fields/B"][traj_index])
    stride = max(1, field.shape[-1] // side)
    return field[:, ::stride, ::stride][:, :side, :side].astype(np.float32)


def _rollout_cases(model, cases, side: int, dt: int, n_apps: int):
    """Roll ``model`` (residual predictor at step ``dt``) out on each case."""
    from surge.inference.rollout import autoregressive_rollout, persistence_rel_l2

    def predict_fn(state_img):
        flat = np.asarray(state_img, dtype=np.float32).reshape(1, -1)
        return np.asarray(model.predict(flat)).reshape(side, side)

    out = []
    for c in cases:
        traj = c["traj"]; t0 = c["t_start"]
        truths = np.stack([traj[t0 + k * dt] for k in range(1, n_apps + 1)])
        preds, rel = autoregressive_rollout(predict_fn, traj[t0], truths, residual=True)
        pers = persistence_rel_l2(traj[t0], truths)
        out.append({"truths": truths, "preds": preds, "rel_l2": rel,
                    "persistence": pers, "initial": traj[t0]})
    return out


def longjump_curve_figure(res: dict, mode: str = "light"):
    """rel-L2 vs horizon: short-step rollout (thin per case, bold median) vs
    long-jump rollout (markers at 160/320/480); headline case highlighted.
    No persistence line — its medians go in the subtitle."""
    labels = res["labels"]; hl = res["headline_index"]
    sh = np.asarray(res["short"]["per_case_rel_l2"]); sH = res["short"]["horizons"]
    lj = np.asarray(res["long"]["per_case_rel_l2"]); lH = res["long"]["horizons"]
    pers = np.asarray(res["long"]["per_case_persistence"])
    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        for i in range(sh.shape[0]):
            is_hl = i == hl
            ax.plot(sH, sh[i], color=p["series"][0], lw=2.2 if is_hl else 0.9,
                    alpha=1.0 if is_hl else 0.35, zorder=4 if is_hl else 2,
                    label="README case · short-step (Δt = 20)" if is_hl else None)
        ax.plot(sH, np.median(sh, axis=0), color=p["series"][0], lw=2.6, ls="--",
                label="short-step rollout · median of 5", zorder=5)
        for i in range(lj.shape[0]):
            is_hl = i == hl
            ax.plot(lH, lj[i], color=p["series"][1], marker="o",
                    ms=7 if is_hl else 4.5, lw=2.2 if is_hl else 0.9,
                    alpha=1.0 if is_hl else 0.45, zorder=6 if is_hl else 3,
                    label="README case · long-jump (Δt = 160)" if is_hl else None)
        ax.plot(lH, np.median(lj, axis=0), color=p["series"][1], marker="s", ms=8,
                lw=2.6, ls="--", label="long-jump rollout · median of 5", zorder=7)
        ax.scatter([160], [DIRECT_160_RELL2], marker="*", s=220, color=p["warning"],
                   edgecolors=p["ink"], zorder=8, label=f"direct single-shot @160 ({DIRECT_160_RELL2})")
        ax.set_xlabel("horizon (stored steps)"); ax.set_ylabel("rel-L2")
        ax.set_xticks([0, 80, 160, 240, 320, 400, 480])
        ax.set_title("Gray-Scott · recursive 160-step U-Net vs Δt = 20 stepping\n"
                     f"{len(labels)} held-out cases · persistence median "
                     f"{np.median(pers[:,0]):.2f} @160, {np.median(pers[:,-1]):.2f} @480 "
                     "(not drawn)", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
    return fig


def longjump_strip_figure(res: dict, mode: str = "light"):
    """rows = cases; cols = truth | long-jump @160, @320, @480."""
    labels = res["labels"]; hl = res["headline_index"]
    lH = res["long"]["horizons"]; n_h = len(lH)
    truths = res["long"]["truths"]; preds = res["long"]["preds"]
    rel = np.asarray(res["long"]["per_case_rel_l2"])
    n = len(labels)
    with surge_theme(mode) as p:
        fig, axes = plt.subplots(n, 2 * n_h, figsize=(1.55 * 2 * n_h, 1.55 * n + 0.5))
        for r in range(n):
            vmin = float(truths[r].min()); vmax = float(truths[r].max())
            for j in range(n_h):
                for c, img in ((2 * j, truths[r][j]), (2 * j + 1, preds[r][j])):
                    ax = axes[r, c]
                    ax.imshow(img, cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    if r == 0:
                        ax.set_title(f"truth @{lH[j]}" if c % 2 == 0 else f"long-jump @{lH[j]}",
                                     fontsize=8, color=p["ink"])
                    if c % 2 == 1:
                        ax.text(0.04, 0.06, f"{rel[r, j]:.2f}", transform=ax.transAxes,
                                fontsize=7.5, color="white", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))
            axes[r, 0].set_ylabel(("README case\n" if r == hl else "") + labels[r],
                                  fontsize=7, color=p["ink2"])
        fig.suptitle("Gray-Scott · headline 160-step U-Net applied recursively · "
                     "rel-L2 per panel", fontsize=9.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def shortstep_cases_gif(res: dict, out_path: Path, mode: str = "dark", fps: int = 8) -> Path:
    """rows = 5 cases; cols = truth | Δt=20 rollout | signed error (no persistence)."""
    import io

    from PIL import Image

    from surge.inference.rollout import stack_case_panels

    labels = res["labels"]; hl = res["headline_index"]
    truths = np.stack(res["short"]["truths"]); preds = np.stack(res["short"]["preds"])
    H = res["short"]["horizons"]; rel = np.asarray(res["short"]["per_case_rel_l2"])
    frames = stack_case_panels(truths, preds, preds - truths)   # (n_apps, n_cases, 3, H, W)
    n = truths.shape[0]
    vmin, vmax = float(truths.min()), float(truths.max())
    err_vmax = float(np.abs(preds - truths).max()) + 1e-9
    with surge_theme(mode) as p:
        bufs = []
        for k in range(len(H)):
            fig, axes = plt.subplots(n, 3, figsize=(4.9, 1.6 * n))
            for r in range(n):
                for c in range(3):
                    ax = axes[r, c]; img = frames[k, r, c]
                    if c == 2:
                        ax.imshow(img, cmap=diverging_cmap(mode), norm=CenteredNorm(halfrange=err_vmax))
                    else:
                        ax.imshow(img, cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    if r == 0:
                        ax.set_title(["truth", "Δt = 20 rollout", "signed error"][c], fontsize=8, color=p["ink"])
                    if c == 0:
                        ax.set_ylabel(("README case\n" if r == hl else "") + labels[r], fontsize=6.5, color=p["ink2"])
            fig.suptitle(f"step {H[k]} / {H[-1]}  ·  short-step rollout median rel-L2 "
                         f"{np.median(rel[:, k]):.3f}", fontsize=9, fontweight="bold", color=p["ink"])
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80, facecolor=p["surface"]); plt.close(fig)
            buf.seek(0); bufs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bufs[0].save(out_path, save_all=True, append_images=bufs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out_path


def run_rollout_longjump(args) -> None:
    """``--rollout-longjump``: identify the README headline case, then roll
    out (a) the headline 160-step U-Net recursively (160/320/480) and (b) a
    Δt = 20 U-Net (24 applications) on the headline case + the 4 regime
    cases of ``--rollout-gif``; persistence recorded but not plotted."""
    from surge.model import MODEL_REGISTRY

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed = 0; side = args.side; total = 480
    cache = out / "grayscott_headline_case.json"

    # 1. headline case + its model (always retrain the model; cache only the identity)
    headline, model160 = identify_headline_case(side, seed, args.n_train, args.n_test,
                                                args.epochs, horizon=160, out_dir=out)
    cache.write_text(json.dumps(headline, indent=2)); print("wrote", cache)

    # 2. cases: headline + the 4 regime cases (same traj/t_start as --rollout-gif)
    cases_json = out / "grayscott_rollout_cases.json"
    regime_cases = json.loads(cases_json.read_text())["cases"] if cases_json.exists() else []
    cases = [{"file": headline["file"], "traj_index": headline["traj_index"],
              "t_start": int(headline["t0"]),
              "label": f"{headline['regime']} (t0={headline['t0']}"
                       f"{', transient' if headline['transient_start'] else ''})"}]
    for c in regime_cases[:4]:
        cases.append({"file": c["file"], "traj_index": c["traj_index"],
                      "t_start": int(c["t_start"]), "label": regime_label(c["file"])})
    for c in cases:
        c["traj"] = _load_case_trajectory(c["file"], c["traj_index"], side)
        T = c["traj"].shape[0]
        if c["t_start"] + total >= T:          # keep room for 480 steps
            c["t_start"] = max(SETTLE_STEPS, T - total - 1)
    labels = [c["label"] for c in cases]
    print(f"[cases] {labels}  t_start={[c['t_start'] for c in cases]}")

    # 3a. long-jump rollout with the headline model
    long = _rollout_cases(model160, cases, side, dt=160, n_apps=3)
    # 3b. short-step model + rollout
    (Xtr, ytr), _ = load_grayscott(args.n_train, 100, side, seed, horizon=20)
    model20 = MODEL_REGISTRY.create("pytorch.unet", n_epochs=args.epochs)
    t0 = time.perf_counter(); model20.fit(Xtr, ytr - Xtr)
    print(f"[short] U-Net (residual) trained @dt=20 in {fmt_metric(time.perf_counter() - t0, 'runtime')}")
    short = _rollout_cases(model20, cases, side, dt=20, n_apps=24)

    res = {
        "headline": headline, "headline_index": 0, "labels": labels,
        "cases": [{k: c[k] for k in ("file", "traj_index", "t_start", "label")} for c in cases],
        "long": {"dt": 160, "horizons": [160, 320, 480],
                 "per_case_rel_l2": [r["rel_l2"].tolist() for r in long],
                 "per_case_persistence": [r["persistence"].tolist() for r in long],
                 "truths": [r["truths"] for r in long], "preds": [r["preds"] for r in long]},
        "short": {"dt": 20, "horizons": [20 * k for k in range(1, 25)],
                  "per_case_rel_l2": [r["rel_l2"].tolist() for r in short],
                  "per_case_persistence": [r["persistence"].tolist() for r in short],
                  "truths": [r["truths"] for r in short], "preds": [r["preds"] for r in short]},
    }
    summary = {k: v for k, v in res.items() if k not in ("long", "short")}
    for key in ("long", "short"):
        d = {k: v for k, v in res[key].items() if k not in ("truths", "preds")}
        d["median_rel_l2"] = np.median(np.asarray(d["per_case_rel_l2"]), axis=0).tolist()
        d["median_persistence"] = np.median(np.asarray(d["per_case_persistence"]), axis=0).tolist()
        summary[key] = d
    summary["direct_160_rel_l2"] = DIRECT_160_RELL2
    summary["direct_160_persistence"] = DIRECT_160_PERSISTENCE
    (out / "grayscott_rollout_longjump.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "grayscott_rollout_longjump.json")
    print(f"{'horizon':>8} {'long-jump med':>14} {'short-step med':>15} {'persistence':>12}")
    sh_med = summary["short"]["median_rel_l2"]
    for j, h in enumerate(summary["long"]["horizons"]):
        print(f"{h:>8} {summary['long']['median_rel_l2'][j]:>14.3f} "
              f"{sh_med[summary['short']['horizons'].index(h)]:>15.3f} "
              f"{summary['long']['median_persistence'][j]:>12.3f}")

    for mode in ("light", "dark"):
        for name, fn in (("grayscott_rollout_longjump_curve", longjump_curve_figure),
                         ("grayscott_rollout_longjump_strip", longjump_strip_figure)):
            fig = fn(res, mode)
            for path in save_figure(fig, out / f"{name}_{mode}"):
                print("wrote", path)
            plt.close(fig)
    gif = shortstep_cases_gif(res, out / "grayscott_rollout_shortstep_cases.gif", mode="dark")
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.2f} MB)")


def run_rollout(args) -> None:
    """``--rollout``: sweep Δt (pick the smallest that beats persistence),
    train one U-Net at it, roll it out to a 160-step horizon on 5 settled-
    state trajectories, and write the gif + curve figures + numbers table."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = 0
    chosen_dt, dt_table, beats_persistence, _model, rollout, _trained = run_rollout_study(
        args.n_train, args.epochs, args.side, seed,
        dt_candidates=(20, 40, 80), total_horizon=160, n_traj=5)

    dt_table_list = [{"dt": dt, "model_rel_l2": m, "persistence_rel_l2": p}
                     for dt, (m, p) in sorted(dt_table.items())]
    print(f"{'dt':>6} {'model rel-L2':>14} {'persistence':>14}")
    for row in dt_table_list:
        print(f"{row['dt']:>6} {row['model_rel_l2']:>14.4f} "
              f"{row['persistence_rel_l2']:>14.4f}")

    table = [{"horizon": h, "rollout_rel_l2": r, "persistence_rel_l2": p,
             "rollout_iqr": iqr}
             for h, r, p, iqr in zip(rollout["horizons"], rollout["rel_l2"],
                                     rollout["persistence"],
                                     rollout["rel_l2_iqr"])]
    print(f"{'horizon':>8} {'rollout rel-L2':>16} {'persistence':>14} "
          f"{'iqr':>8}")
    for row in table:
        print(f"{row['horizon']:>8} {row['rollout_rel_l2']:>16.4f} "
              f"{row['persistence_rel_l2']:>14.4f} {row['rollout_iqr']:>8.4f}")

    summary = {
        "dt_table": dt_table_list,
        "chosen_dt": chosen_dt,
        "beats_persistence_at_dt": beats_persistence,
        "n_traj": rollout["n_traj"],
        "t_start_per_traj": rollout["t_start_per_traj"],
        "direct_160_rel_l2": DIRECT_160_RELL2,
        "direct_160_persistence": DIRECT_160_PERSISTENCE,
        "table": table,
        "per_traj_rel_l2": rollout["per_traj_rel_l2"],
        "per_traj_persistence": rollout["per_traj_persistence"],
    }
    if not beats_persistence:
        print(f"[warn] NO candidate Δt beat persistence — chose the largest "
              f"offered (dt={chosen_dt}) and flagged it in the json "
              "(beats_persistence_at_dt=false)")
    (out / "grayscott_rollout.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "grayscott_rollout.json")

    gif_path = rollout_gif(rollout, args.side, out / "grayscott_rollout.gif",
                           mode="dark", fps=10)
    size_mb = gif_path.stat().st_size / 1e6
    print(f"wrote {gif_path} ({size_mb:.2f} MB, {len(rollout['horizons'])} frames)")
    if size_mb > 6.0:
        print(f"[warn] gif is {size_mb:.2f} MB, over the 6 MB budget")

    for mode in ("light", "dark"):
        fig = rollout_curve_figure(rollout, mode)
        for path in save_figure(fig, out / f"grayscott_rollout_curve_{mode}"):
            print("wrote", path)
        plt.close(fig)


def regime_split_figure(summary: dict, mode: str = "light"):
    """Grouped bars: median rel-L2 for model vs persistence, on all /
    transient / settled test pairs, plus the settled-only-trained model on
    settled pairs."""
    cells = summary["cells"]
    groups = ["all", "transient", "settled"]
    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        width = 0.25
        x = np.arange(len(groups))
        model_vals = [cells[g]["model_rel_l2"] for g in groups]
        persist_vals = [cells[g]["persistence_rel_l2"] for g in groups]
        ax.bar(x - width / 2, model_vals, width, color=p["series"][0],
              label="U-Net (all-regime train)")
        ax.bar(x + width / 2, persist_vals, width, color=p["series"][2],
              label="persistence")
        settled_only = summary["cells"]["settled_only_model_on_settled"]
        ax.bar([x[groups.index("settled")] + 1.5 * width],
              [settled_only["model_rel_l2"]], width,
              color=p["series"][1], label="U-Net (settled-only train)")
        if "ridge_rel_l2" in cells["all"]:
            ridge_vals = [cells[g].get("ridge_rel_l2") for g in groups]
            ax.bar(x - 1.5 * width, ridge_vals, width, color=p["series"][3],
                  label="ridge (all-regime train)")
        for xi, v in zip(x - width / 2, model_vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                   color=p["ink2"])
        for xi, v in zip(x + width / 2, persist_vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                   color=p["ink2"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={cells[g]['n']})" for g in groups])
        ax.set_ylabel("median rel-L2 (lower is better)")
        ax.set_title(
            "Gray-Scott @160: headline vs transient/settled regime split\n"
            f"train fractions: transient {summary['train_fractions']['transient']:.2f}, "
            f"settled {summary['train_fractions']['settled']:.2f}",
            fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7.5)
        return fig


def run_regime_split(args) -> None:
    """``--regime-split``: does the headline 160-step advantage come mostly
    from the early pattern-formation transient (t0<200), where persistence
    is much worse? Score model/persistence/ridge on all/transient/settled
    test pairs, then retrain on settled-only pairs and re-score on settled."""
    from surge.inference.rollout import regime_masks
    from surge.model import MODEL_REGISTRY

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = 0
    horizon = 160
    n_train, n_test = 800, 400

    Xtr, ytr, t0_tr = load_grayscott_pairs_with_t0(
        args.side, seed, n_train, horizon, split="train")
    Xte, yte, t0_te = load_grayscott_pairs_with_t0(
        args.side, seed + 1000, n_test, horizon, split="valid")
    print(f"[data] train {Xtr.shape} test {Xte.shape}")

    transient_mask, settled_mask, train_fractions = regime_masks(t0_tr)
    _, test_settled_mask, test_fractions = regime_masks(t0_te)
    test_transient_mask = ~test_settled_mask
    print(f"[regime] train fractions {train_fractions}  "
          f"test fractions {test_fractions}")

    def score(pred, truth):
        rel = (np.linalg.norm(pred - truth, axis=1)
               / np.maximum(np.linalg.norm(truth, axis=1), 1e-12))
        return rel

    rel_persist_all = score(Xte, yte)

    t0 = time.perf_counter()
    model = MODEL_REGISTRY.create("pytorch.unet", n_epochs=args.epochs)
    model.fit(Xtr, ytr - Xtr)
    unet_dt = time.perf_counter() - t0
    pred_all = model.predict(Xte).reshape(len(Xte), -1) + Xte
    rel_model_all = score(pred_all, yte)
    print(f"[done] U-Net (all-regime) {fmt_metric(unet_dt, 'runtime')}")

    ridge_result = None
    t0 = time.perf_counter()
    try:
        ridge = MODEL_REGISTRY.create("sklearn.ridge")
        ridge.fit(Xtr, ytr - Xtr)
        ridge_dt = time.perf_counter() - t0
        if ridge_dt < 120:
            pred_ridge = ridge.predict(Xte).reshape(len(Xte), -1) + Xte
            rel_ridge_all = score(pred_ridge, yte)
            ridge_result = rel_ridge_all
            print(f"[done] Ridge {fmt_metric(ridge_dt, 'runtime')}")
        else:
            print(f"[skip] Ridge took {ridge_dt:.1f}s (> 2 min budget)")
    except Exception as exc:  # noqa: BLE001 - ridge is a nice-to-have here
        print(f"[skip] Ridge failed: {type(exc).__name__}: {exc}")

    def cell(mask, rel_model, rel_persist, rel_ridge=None):
        n = int(mask.sum())
        out_cell = {
            "n": n,
            "model_rel_l2": float(np.median(rel_model[mask])) if n else None,
            "persistence_rel_l2": float(np.median(rel_persist[mask])) if n else None,
        }
        if rel_ridge is not None:
            out_cell["ridge_rel_l2"] = float(np.median(rel_ridge[mask])) if n else None
        return out_cell

    cells = {
        "all": cell(np.ones(len(yte), dtype=bool), rel_model_all, rel_persist_all,
                   ridge_result),
        "transient": cell(test_transient_mask, rel_model_all, rel_persist_all,
                         ridge_result),
        "settled": cell(test_settled_mask, rel_model_all, rel_persist_all,
                       ridge_result),
    }
    for name, c in cells.items():
        extra = f" ridge {c['ridge_rel_l2']:.4f}" if c.get("ridge_rel_l2") is not None else ""
        print(f"[{name:9s}] n={c['n']:4d}  model {c['model_rel_l2']:.4f}  "
              f"persistence {c['persistence_rel_l2']:.4f}{extra}")

    # second model: trained on settled-only pairs (t0 >= 200), scored on the
    # same settled test pairs used above
    Xtr_s, ytr_s, t0_tr_s = load_grayscott_pairs_with_t0(
        args.side, seed + 1, n_train, horizon, split="train", t0_min=SETTLE_STEPS)
    _, _, settled_train_fracs = regime_masks(t0_tr_s)
    assert settled_train_fracs["transient"] == 0.0, (
        "settled-only training set leaked a transient-regime pair")
    t0 = time.perf_counter()
    model_settled = MODEL_REGISTRY.create("pytorch.unet", n_epochs=args.epochs)
    model_settled.fit(Xtr_s, ytr_s - Xtr_s)
    settled_dt = time.perf_counter() - t0
    Xte_settled = Xte[test_settled_mask]
    yte_settled = yte[test_settled_mask]
    pred_settled = model_settled.predict(Xte_settled).reshape(len(Xte_settled), -1) \
        + Xte_settled
    rel_settled_only = score(pred_settled, yte_settled)
    print(f"[done] U-Net (settled-only train) {fmt_metric(settled_dt, 'runtime')} "
          f"median rel-L2 {np.median(rel_settled_only):.4f}")

    cells["settled_only_model_on_settled"] = {
        "n": int(test_settled_mask.sum()),
        "model_rel_l2": float(np.median(rel_settled_only)),
        "persistence_rel_l2": cells["settled"]["persistence_rel_l2"],
    }

    summary = {
        "horizon": horizon,
        "cutoff": SETTLE_STEPS,
        "n_train": n_train,
        "n_test": n_test,
        "train_fractions": train_fractions,
        "test_fractions": test_fractions,
        "cells": cells,
    }
    (out / "grayscott_regime_split.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "grayscott_regime_split.json")

    for mode in ("light", "dark"):
        fig = regime_split_figure(summary, mode)
        for path in save_figure(fig, out / f"grayscott_regime_split_{mode}"):
            print("wrote", path)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=160,
                    help="prediction horizon in stored timesteps")
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    ap.add_argument("--rollout", action="store_true",
                    help="multi-step autoregressive rollout mode: train one "
                         "U-Net at a modest step and roll it out to >=160 "
                         "stored steps, instead of the single-shot study")
    ap.add_argument("--rollout-gif", action="store_true",
                    help="multi-case (4 distinct regimes), 480-step rollout "
                         "grid GIF + curve, instead of the single-shot study")
    ap.add_argument("--rollout-longjump", action="store_true",
                    help="identify the README headline case and roll out the "
                         "160-step U-Net recursively (160/320/480) vs a "
                         "dt=20 U-Net on 5 cases; persistence not plotted")
    ap.add_argument("--regime-split", action="store_true",
                    help="split the headline 160-step result into "
                         "transient/settled regimes by pair start t0, "
                         "instead of the single-shot study")
    args = ap.parse_args()

    if args.rollout:
        run_rollout(args)
        return
    if args.rollout_gif:
        run_rollout_gif_cases(args)
        return
    if args.regime_split:
        run_regime_split(args)
        return
    if args.rollout_longjump:
        run_rollout_longjump(args)
        return

    data, results = run_study(args.n_train, args.n_test, args.side,
                              args.epochs, horizon=args.horizon)
    if not results:
        raise SystemExit("no model produced results")
    out = Path(args.out)
    # horizon 160 is the headline task and keeps unsuffixed filenames;
    # other horizons (e.g. --horizon 1, the next-step task) get _h<N>
    suffix = "" if args.horizon == 160 else f"_h{args.horizon}"
    for mode in ("light", "dark"):
        fig = study_figure(data, results, args.side, mode,
                           horizon=args.horizon)
        for path in save_figure(fig, out / f"thewell_grayscott{suffix}_{mode}"):
            print("wrote", path)
        plt.close(fig)
    (Xte, yte) = data
    np.savez_compressed(
        out / f"thewell_grayscott{suffix}_preds.npz",
        Xte=Xte, yte=yte,
        **{f"pred__{r['label']}": r["pred"] for r in results})
    summary = [{k: v for k, v in r.items() if k not in ("pred", "rel")}
               for r in results]
    (out / f"thewell_grayscott{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
