#!/usr/bin/env python3
"""One point of the TheWell-protocol scaling sweep — HPC-ready.

Trains a single ConvNeXt U-Net configuration on turbulent_radiative_layer_2D
under TheWell's own benchmark protocol (next-step, 4-frame history, 4 fields,
128x384, VRMSE) and writes ``result.json`` + ``epochs.jsonl``.

Why this exists
---------------
The laptop study (``examples/thewell_wellprotocol_study.py``) reached VRMSE
0.486 against a published ConvNeXt-U-Net at 0.125, but three axes were
confounded: we used 1,400 of the 6,984 available training pairs, 3.4M of
~20M parameters, and 60 of ~500+ epochs. This script varies those axes
INDEPENDENTLY so the sweep answers *which one binds* — see
``scripts/hpc/grid.sh`` for the 2x2x2 factorial.

Wall-clock safety
-----------------
The largest configuration runs ~11 h on an A100, against a 12 h queue limit.
``--time-limit-s`` makes the job checkpoint and exit cleanly (status
``paused``) once the *next* epoch would not fit in the remaining time; the
submit script resubmits and ``--resume`` picks up mid-run. Normalization
statistics and the training-subset indices are stored in the checkpoint, so
a resumed run is identical to an uninterrupted one.

Usage
-----
    python scripts/hpc/wellprotocol_sweep.py \\
        --tag D_epochs --n-train 1400 --base-channels 40 --depth 3 \\
        --epochs 500 --out-dir $SCRATCH/well_sweep --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FIELDS = ["density", "pressure", "velocity_x", "velocity_y"]
N_HIST = 4
N_FIELDS = len(FIELDS)

# Published validation baselines, 15-20M params trained 12 h on an H100
# (Ohana et al. 2024, TheWell).
PUBLISHED = {"FNO": 0.4906, "TFNO": 0.4938, "U-Net": 0.2394,
             "CNextU-Net": 0.1247}


# ---------------------------------------------------------------------------
# Metric + data
# ---------------------------------------------------------------------------

def vrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """TheWell metric: variance-scaled RMSE, per sample+field, averaged.

    Identical to ``examples/thewell_wellprotocol_study.py`` — do not change
    one without the other, or the numbers stop being comparable.
    """
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = truth.reshape(truth.shape[0], truth.shape[1], -1)
    num = ((p - t) ** 2).mean(axis=-1)
    den = ((t - t.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1) + 1e-12
    return float(np.sqrt(num / den).mean())


def _split_root(split: str, data_root: str | None) -> Path:
    """Directory holding the HDF5 files of one split.

    ``--data-root`` points at a staged copy (e.g. ``$SCRATCH``); without it
    we fall back to the standard SURGE cache.
    """
    if data_root:
        return Path(data_root) / "turbulent_radiative_layer_2D" / "data" / split
    from surge.benchmarks.loaders.thewell import _well_base_path

    return (_well_base_path(None) / "turbulent_radiative_layer_2D"
            / "data" / split)


def _well_dataset(split: str, data_root: str | None):
    from the_well.data import WellDataset

    return WellDataset(path=str(_split_root(split, data_root)),
                       n_steps_input=N_HIST, n_steps_output=1,
                       use_normalization=False)


def _pair(ds, i: int):
    """(16, 128, 384) frame-major input, (4, 128, 384) target."""
    item = ds[int(i)]
    xin = np.asarray(item["input_fields"])       # (4, H, W, 4)
    yout = np.asarray(item["output_fields"])[0]  # (H, W, 4)
    x = np.moveaxis(xin, -1, 1).reshape(-1, *xin.shape[1:3])
    return x.astype("float32"), np.moveaxis(yout, -1, 0).astype("float32")


class StreamingPairs:
    """Lazy torch Dataset over WellDataset — HDF5 reads happen per item.

    RAM stays flat regardless of ``n``, which is what lets the full 6,984-pair
    split train on a single GPU node.
    """

    def __init__(self, split: str, idx: np.ndarray, mu: np.ndarray,
                 sd: np.ndarray, data_root: str | None) -> None:
        self.ds = _well_dataset(split, data_root)
        self.idx = idx
        self.x_mu = np.tile(mu, (1, N_HIST, 1, 1))[0]
        self.x_sd = np.tile(sd, (1, N_HIST, 1, 1))[0]
        self.y_mu, self.y_sd = mu[0], sd[0]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        import torch

        x, y = _pair(self.ds, self.idx[i])
        x = (x - self.x_mu) / self.x_sd
        y = (y - self.y_mu) / self.y_sd
        return torch.from_numpy(x), torch.from_numpy(y)


def load_eval(n_test: int, seed: int, data_root: str | None):
    """Eval pairs held in RAM — 873 max, comfortably small."""
    ds = _well_dataset("valid", data_root)
    idx = np.random.default_rng(seed).choice(
        len(ds), size=min(n_test, len(ds)), replace=False)
    X, Y = zip(*(_pair(ds, i) for i in idx))
    return np.stack(X), np.stack(Y)


def train_indices(n_train: int, seed: int, data_root: str | None):
    """Fixed subset of the training split — the DATA axis of the sweep.

    Same rng call as the laptop study, so ``--n-train 1400`` reproduces the
    exact pairs that produced the 0.486 baseline.
    """
    ds = _well_dataset("train", data_root)
    n_avail = len(ds)
    rng = np.random.default_rng(seed)
    return rng.choice(n_avail, size=min(n_train, n_avail),
                      replace=False), n_avail


def norm_stats(idx: np.ndarray, n_stat: int, data_root: str | None):
    """Per-field z-score stats from a bounded sample of the training subset.

    Well practice: frame channels of a field share that field's statistics.
    Bounded at ``n_stat`` so the 6,984-pair configs do not pay a full pass
    just to normalize.
    """
    ds = _well_dataset("train", data_root)
    take = idx[:min(n_stat, len(idx))]
    ys = np.stack([_pair(ds, i)[1] for i in take])
    mu = ys.mean(axis=(0, 2, 3), keepdims=True)
    sd = ys.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return mu, sd


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def evaluate(net, Xte, yte, mu, sd, device, batch_size, amp_dtype):
    """VRMSE in physical units (de-normalized predictions)."""
    import torch

    x_mu = np.tile(mu, (1, N_HIST, 1, 1))
    x_sd = np.tile(sd, (1, N_HIST, 1, 1))
    Xn = torch.from_numpy((Xte - x_mu) / x_sd)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = Xn[i:i + batch_size].to(device)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type,
                                    dtype=amp_dtype):
                    pred = net(xb)
            else:
                pred = net(xb)
            out.append(pred.float().cpu().numpy())
    net.train()
    return vrmse(np.concatenate(out) * sd + mu, yte)


def persistence_vrmse(Xte: np.ndarray, yte: np.ndarray) -> float:
    """Last input frame of each field — the free baseline to beat.

    Channel layout is FRAME-major: [t0f0, t0f1, t0f2, t0f3, t1f0, ...].
    """
    last = np.stack([Xte[:, (N_HIST - 1) * N_FIELDS + f]
                     for f in range(N_FIELDS)], axis=1)
    return vrmse(last, yte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="config name, used as subdir")
    ap.add_argument("--out-dir", default="well_sweep")
    ap.add_argument("--data-root", default=os.environ.get("WELL_DATA_ROOT"),
                    help="staged dataset dir (default: SURGE cache)")
    # sweep axes
    ap.add_argument("--n-train", type=int, default=6984)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=500)
    # fixed protocol / optimization
    ap.add_argument("--blocks-per-stage", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=873)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-stat", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    # runtime
    ap.add_argument("--device", default=None, help="default: cuda if present")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--no-channels-last", dest="channels_last",
                    action="store_false")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--time-limit-s", type=float, default=0.0,
                    help="checkpoint and exit before this many seconds "
                         "(0 = no limit)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    import torch

    from surge.model.backends.convnext_unet import _CNextUNet
    from surge.utils import resolve_device

    device = (torch.device(args.device) if args.device
              else resolve_device("auto"))
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    out = Path(args.out_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    log_path = out / "epochs.jsonl"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data -------------------------------------------------------------
    resumed = args.resume and ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu",
                      weights_only=False) if resumed else None
    if resumed:
        # reuse the EXACT subset + statistics, else resume changes the run
        idx, mu, sd = ckpt["idx"], ckpt["mu"], ckpt["sd"]
        n_avail = ckpt["n_avail"]
        print(f"[resume] {args.tag} from epoch {ckpt['epoch']}", flush=True)
    else:
        idx, n_avail = train_indices(args.n_train, args.seed, args.data_root)
        mu, sd = norm_stats(idx, args.n_stat, args.data_root)

    train_ds = StreamingPairs("train", idx, mu, sd, args.data_root)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0, drop_last=False)
    Xte, yte = load_eval(args.n_test, args.seed, args.data_root)
    print(f"[data] train {len(train_ds)}/{n_avail} pairs  "
          f"eval {len(Xte)}  device {device}  amp {args.amp}", flush=True)

    # ---- model ------------------------------------------------------------
    net = _CNextUNet(N_HIST * N_FIELDS, N_FIELDS, args.base_channels,
                     args.depth, args.blocks_per_stage).to(device)
    if args.channels_last and device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # bf16 has fp32 dynamic range — only fp16 needs loss scaling
    scaler = (torch.amp.GradScaler(device.type)
              if amp_dtype is torch.float16 else None)
    crit = torch.nn.MSELoss()

    start_epoch, best = 0, float("inf")
    if resumed:
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        torch.set_rng_state(ckpt["rng"])
        start_epoch, best = ckpt["epoch"], ckpt["best"]
    else:
        log_path.write_text("")
        print(f"[model] {n_par/1e6:.2f}M params  "
              f"base={args.base_channels} depth={args.depth}", flush=True)

    persistence = persistence_vrmse(Xte, yte)

    def save_ckpt(epoch: int, best_v: float) -> None:
        torch.save({"epoch": epoch, "best": best_v, "net": net.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "idx": idx, "mu": mu,
                    "sd": sd, "n_avail": n_avail, "args": vars(args)},
                   ckpt_path)

    # ---- train ------------------------------------------------------------
    status, epoch_times = "completed", []
    last_val = None
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.perf_counter()
        eloss, nseen = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = crit(net(xb), yb)
            else:
                loss = crit(net(xb), yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            eloss += loss.item() * len(xb)
            nseen += len(xb)
        sched.step()
        dt = time.perf_counter() - t_ep
        epoch_times.append(dt)

        rec = {"epoch": epoch + 1, "train_loss": eloss / max(nseen, 1),
               "lr": sched.get_last_lr()[0], "secs": round(dt, 2)}
        do_eval = ((epoch + 1) % args.eval_every == 0
                   or epoch + 1 == args.epochs)
        if do_eval:
            last_val = evaluate(net, Xte, yte, mu, sd, device,
                                args.batch_size, amp_dtype)
            rec["val_vrmse"] = last_val
            best = min(best, last_val)
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{args.tag}] epoch {epoch+1}/{args.epochs} "
              f"loss {rec['train_loss']:.5f} {dt:.1f}s"
              + (f" VRMSE {last_val:.4f}" if do_eval else ""), flush=True)

        if (epoch + 1) % args.checkpoint_every == 0:
            save_ckpt(epoch + 1, best)

        # stop before the queue kills us: budget the NEXT epoch + an eval
        if args.time_limit_s:
            elapsed = time.perf_counter() - t_start
            typical = float(np.median(epoch_times[-5:]))
            if elapsed + typical * 1.5 > args.time_limit_s:
                save_ckpt(epoch + 1, best)
                status = "paused"
                print(f"[{args.tag}] time limit — checkpointed at epoch "
                      f"{epoch+1}, resubmit with --resume", flush=True)
                break

    if status == "completed":
        if last_val is None:
            last_val = evaluate(net, Xte, yte, mu, sd, device,
                                args.batch_size, amp_dtype)
            best = min(best, last_val)
        save_ckpt(args.epochs, best)

    result = {
        "tag": args.tag,
        "status": status,
        "axes": {"n_train": len(train_ds), "params": n_par,
                 "epochs_requested": args.epochs,
                 "epochs_done": (args.epochs if status == "completed"
                                 else len(epoch_times) + start_epoch)},
        "config": {k: vars(args)[k] for k in
                   ("base_channels", "depth", "blocks_per_stage", "n_train",
                    "n_test", "batch_size", "lr", "weight_decay", "amp",
                    "seed")},
        "vrmse": {"final": last_val, "best": best,
                  "persistence": persistence},
        "published_12h_H100": PUBLISHED,
        "timing": {"median_epoch_s": (float(np.median(epoch_times))
                                      if epoch_times else None),
                   "wall_s": round(time.perf_counter() - t_start, 1),
                   "device": str(device)},
        "protocol": ("next-step, 4-frame history, 4 fields, 128x384, VRMSE"),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    # non-zero tells the submit script to resubmit for another window
    sys.exit(0 if status == "completed" else 42)


if __name__ == "__main__":
    main()
