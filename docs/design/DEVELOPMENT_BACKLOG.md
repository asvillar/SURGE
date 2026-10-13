# Development backlog

Single place for pending developments, updated 2026-07-26. Detailed
designs live in the sibling docs (`ARCHITECTURE_RECOMMENDATIONS.md`
R1–R14, `RESOURCE_MANAGEMENT.md` R15–R18, `FIGURE_UPGRADE_PLAN.md`).

## Done recently (for orientation)

- Figure-upgrade plan phases 1–6: theme-aware README (`<picture>` +
  `docs/assets/readme/dark/`), mission-control leaderboard, ensemble
  retold, ConStellaration showcase, HPO mission-control dashboard,
  MLflow per-epoch curves + nested HPO trial runs, Gray-Scott study as
  honest 160-step forecast with persistence gate + residual-target
  variants; U-Net flat-input contract fixed.

## P1 — near term

- ~~QLKNN10D benchmark~~ — DONE: plasma.qlknn10d trains on the public
  290M-row QuaLiKiz table; residual MLP R2 0.989 held-out (gate 0.95).
  Follow-ups: multi-output (efe+pfe channels, TEM/ETG modes), full-table
  chunked training, QLKNN11D (1B rows).
- ~~Streaming training path~~ — DONE (fit_from_loader + lazy Well
  pairs); NOTE: TRL-2D train split physically caps at 6,984 four-frame
  pairs — the published 12h-H100 baselines' data advantage is bounded;
  overnight all-pairs x14-epoch run in flight.

- **TheWell-protocol scaling sweep on HPC** — PREPARED, NOT SUBMITTED.
  `scripts/hpc/` holds a 2x2x2 factorial (data 1400/6984 x capacity
  3.4M/19.8M x epochs 60/500) that disentangles why our ConvNeXt U-Net
  sits at VRMSE 0.486 vs the published 0.125 on
  turbulent_radiative_layer_2D. Sweep driver (checkpoint/resume for the
  12h queue wall), calibration job, and Perlmutter + Frontier submit
  scripts are written and smoke-tested locally. Estimated ~24 GPU-hours
  total (~12-22 Perlmutter node-hours, ~11 on Frontier). NEXT: run
  `calibrate_throughput.py` on the target machine, stage the 6 GB
  dataset to $SCRATCH, set the account line, submit. Then write the
  collation script (step 5 in `scripts/hpc/README.md`) — the only piece
  deliberately left unbuilt.

- **Simformer follow-ups** (adapter `pytorch.simformer` SHIPPED and
  validated against the analytic linear-Gaussian posterior; private
  NSTX-U amortized-reconstruction demo lives in ../TokaMakerGen):
  (i) richer NSTX-U campaign with the full 273-probe EFIT set +
  profile parameters (ffp/pp alphas) for true KINETIC reconstruction;
  (ii) attention masks encoding dependency graphs (paper §3.2);
  (iii) function-valued θ via Fourier index embeddings (profiles);
  (iv) spec-level SBI task type so `surge run` can drive it from YAML.

- ~~R15 device resolution (GPU/MPS)~~ — DONE: `resolve_device` in all 19
  backends; MPS opt-in via `SURGE_DEVICE=auto` (LSTM unsafe on MPS).
  Follow-up: per-architecture MPS allow-list; record resolved device in
  run artifacts.
- ~~Parallel benchmark fan-out~~ — DONE: `surge bench --parallel N`
  (subprocess jobs, thread splitting, collision-safe result dirs).
- ~~`surge init` wizard~~ / ~~`surge validate`~~ / ~~spec JSON Schema~~ —
  DONE: interactive + non-interactive wizard (goal/budget-aware model
  slates), offline validation with did-you-mean suggestions (exit 0/2),
  schema generated from the dataclasses and shipped in package-data
  (editor autocomplete documented in GETTING_STARTED).
- **Publish 0.1.0rc1 → 0.1.0** (user-side: `uv publish`; then version
  bump, retag, Trusted Publishing on PyPI).
- **Verify README publication titles** (user-side).

- **PCA / POD training transforms (R3 slice)** — library slice DONE:
  `pod_fit/pod_transform/pod_inverse` in surge.preprocessing (unit-tested)
  + examples/thewell_pod_study.py (POD+ridge beats FNO-2D 11x on
  Helmholtz, edges U-Net on turbulence). REMAINING: spec-level wiring —
  `preprocessing: {pca: ...}` for inputs and `target_pca: k` with
  inverse-transform plumbing through scalers + inference so `surge run`
  drives it declaratively.

## P2 — medium term

- **R16/R17 remaining** — spec-level `parallel_models` inside `surge
  run`, Optuna `n_jobs`, memory-tier enforcement, `--dry-run` placement
  report.
- ~~TheWell second dataset~~ — DONE: `turbulence_2d` study shipped
  (all neural operators beat persistence: U-Net 0.250 vs 0.355, MPS).
- **Helmholtz staircase full study** — feasibility CONFIRMED (smoke:
  U-Net rel-L2 0.077 vs persistence 1.38 on a quarter-period phase
  advance, 31 s on MPS); `examples/thewell_helmholtz_study.py` written,
  runs once the ~80 GB download completes.
- ~~Turbulence improvement battery~~ — SEVEN levers now tested at the
  800-sample budget (examples/thewell_turbulence_improve.py), skill
  ratio = rel-L2/persistence: published U-Net 0.71; POD+ridge 0.66;
  POD-U-Net blend 0.65 (only real gain, +2%); 4-frame temporal context
  0.72; full 128x384 resolution 0.78 (worse — same data over 4x
  pixels); h=2 composed x4 0.79 (compounding); data 5x flat. Verdict:
  intrinsic predictability floor at dt=8 — further gains require
  changing the question (probabilistic forecasts + CRPS, spectral
  metrics, shorter horizons), not bigger fits.
- ~~Squeeze protocol~~ — RESULTS: constellaration was undertrained
  (60-epoch cap): wide [512,512,256] + 400 epochs lifts QI R2
  0.937 -> 0.951 (figure updated). Turbulence is CHAOS-LIMITED, not
  undertrained: 4-channel physics input, 2.5x epochs, and a 5x data
  scan (800/2000/4000 -> 0.250/0.258/0.254) all leave median rel-L2
  flat at ~0.25. At dt=8 the small scales decorrelate; pointwise rel-L2
  has a predictability floor. The right next steps are probabilistic
  forecasting (mlp_ensemble/CRPS), spectral or statistical evaluation
  metrics, and rollout-trained models — not bigger fits (P2).
- **DeepONet CNN branch** — the residual target halved its error but a
  convolutional branch is the real fix for field inputs.
- **Leaderboard preview special-cases** — constellaration benchmarks
  cache under a non-standard path; the leaderboard figure's data-preview
  inset falls back to "unavailable" for them (draw boundary
  cross-sections instead).
- **RTD dark mode** — README switches automatically; the Sphinx site is
  light-only (needs a theme with `prefers-color-scheme` support, e.g.
  furo, plus dark asset wiring in gallery.md).

## P3 — larger projects

- **R18 multi-GPU (DDP)** for large operator models.
- **3D operator backend** (FNO-3D / 3D U-Net) → unlocks TheWell `MHD_64`
  and other volumetric datasets.
- **Benchmark restructure** (`BENCHMARK_RESTRUCTURE.md`): suite/dataset/
  protocol split with status tracking.
- **PCA as training transform** (R3 pipeline work; analysis-only PCA
  shipped in preprocessing).
