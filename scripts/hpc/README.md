# TheWell-protocol scaling sweep (Perlmutter / Frontier)

**Status: prepared, never submitted.** Everything here is ready to run; no
allocation has been spent. Written 2026-07-31.

## The question this answers

Under TheWell's own benchmark protocol on `turbulent_radiative_layer_2D`
(next-step, 4-frame history, 4 fields, 128x384, VRMSE), our laptop-scale
ConvNeXt U-Net reached **0.486** against a published **0.125**. Our plain
U-Net reached 0.739 — *worse than persistence at 0.554*.

Three axes were confounded in that comparison:

| Axis | Laptop study | Published | Gap |
|---|---|---|---|
| Train pairs | 1,400 | **6,984** (the entire split) | 5x |
| Parameters | 3.43 M | 15-20 M | ~5x |
| Optimization | 60 epochs | 12 h on an H100 (~500-1000 epochs) | 50-100x |

Losing to persistence is an **underfitting** signature, not a data-starvation
one (starved models overfit), so the working hypothesis is that optimization
budget and capacity bind first and data barely matters. **That is a
hypothesis, not a result** — the sweep exists to test it, and it may well be
wrong. The failure mode we are guarding against is spending an allocation to
"scale everything up", getting a better number, and still not knowing which
axis produced it.

## The design: 2x2x2 factorial

Eight configurations, each axis at a low and high level (`grid.sh`):

| tag | n_train | capacity | epochs | what it isolates |
|---|---|---|---|---|
| `A_control` | 1400 | 3.4 M | 60 | reproduces the laptop 0.486 on GPU |
| `B_data` | **6984** | 3.4 M | 60 | data only |
| `C_capacity` | 1400 | **19.8 M** | 60 | capacity only |
| `D_epochs` | 1400 | 3.4 M | **500** | epochs only |
| `E_data_capacity` | **6984** | **19.8 M** | 60 | interaction |
| `F_data_epochs` | **6984** | 3.4 M | **500** | interaction |
| `G_capacity_epochs` | 1400 | **19.8 M** | **500** | interaction |
| `H_full` | **6984** | **19.8 M** | **500** | published-scale target |

`A_control` is the sanity check: if it does not land near 0.486, the port
changed something and nothing else is interpretable yet.

## Cost — extrapolated, verify before trusting

Measured on an Apple M2 Pro (16 GPU cores), fp32, batch 4, true protocol
shapes: 3.43 M params = 213 ms/sample; 19.8 M params = 330 ms/sample (a
full epoch at 6,984 pairs = 38 min, i.e. ~64 days for `H_full` locally).

Assuming a **30x** speedup to A100 with bf16 + channels_last + batch 32:

| | |
|---|---|
| `H_full` (longest single config) | ~10.7 GPU-h |
| `F_data_epochs` | ~6.9 GPU-h |
| all 8 configs | **~24 GPU-h** |
| Perlmutter (4 A100/node, 2 nodes) | **~12-22 node-hours** |
| Frontier (8 GCDs/node, 1 node) | **~11-12 node-hours** |

The 20-50x speedup assumption spans two vendors' hardware and is the weakest
number on this page — hence step 1 below. Cross-check that gives confidence:
12 h on an H100 is roughly 600-1000 epochs at this rate, consistent with the
paper's fixed 12 h budget, so `H_full` should be genuinely comparable to the
published run rather than an approximation of it.

Note `H_full` at ~10.7 h sits just under Perlmutter's 12 h regular-QOS limit.
The driver checkpoints before the wall and the submit scripts chain windows,
but this is why resume exists.

## Run order

1. **Calibrate** (~15 min, one GPU) — replaces the estimate above with
   measurement, and re-projects every config:
   ```bash
   python scripts/hpc/calibrate_throughput.py --out calibration.json
   ```
   If the total is wildly off ~24 GPU-h, re-scope the grid before submitting.

2. **Stage the data** (~6 GB, once). Compute nodes must not read `$HOME`:
   ```bash
   mkdir -p $SCRATCH/thewell
   cp -r ~/.surge/data/thewell/datasets $SCRATCH/thewell/
   ```

3. **Edit the account line** in the submit script (`-A CHANGEME`) and the
   environment block (module load vs. your own venv).

4. **Submit**:
   ```bash
   sbatch scripts/hpc/perlmutter_wellprotocol.slurm          # configs 1-4
   HALF=2 sbatch scripts/hpc/perlmutter_wellprotocol.slurm   # configs 5-8
   # or, on Frontier, all eight on one node:
   sbatch scripts/hpc/frontier_wellprotocol.slurm
   ```

5. **Collate** — NOT YET WRITTEN. Each config leaves
   `$OUT_DIR/<tag>/result.json` (final/best VRMSE, persistence baseline,
   timings) and `epochs.jsonl` (per-epoch loss + periodic val VRMSE). The
   remaining work is a script that reads all eight and produces the scaling
   figure plus the honest headline for the README results grid.

## Files

| File | Role |
|---|---|
| `calibrate_throughput.py` | measure ms/sample, project the grid cost |
| `wellprotocol_sweep.py` | train ONE config; checkpoint/resume; writes result.json |
| `grid.sh` | the 8 configs + launcher, shared by both machines |
| `perlmutter_wellprotocol.slurm` | 4 configs/node on A100s, self-chaining |
| `frontier_wellprotocol.slurm` | 8 configs/node on MI250X GCDs, ROCm caveats |

## Deliberate non-decisions

- **No DDP.** Every config fits one GPU, and with 4-8 GPUs per node the
  throughput that matters is *experiments per node-hour*, not time per
  experiment. Running 8 independent configs beats making one 8x faster.
  DDP becomes necessary for the 3D operator backend or the 132 GB
  `gray_scott` dataset (backlog P3), not here.
- **Frontier is untested.** The ROCm PyTorch build plus `h5py`/`the_well`
  needs verifying on a compute node before a long submission. Perlmutter is
  the lower-friction first target; Frontier is cheaper once it works.
- **Eval on the `valid` split**, not `test` — `test` was never downloaded,
  and the published baselines are validation numbers.
