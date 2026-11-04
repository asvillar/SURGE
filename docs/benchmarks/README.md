# SURGE benchmark ledger

Per-benchmark status, generated from `surge/benchmarks/metadata.yaml` and the newest `result.json` per benchmark. Do not edit by hand; regenerate with:

```bash
surge-benchmark --status-md docs/benchmarks/README.md
```

Statuses:

- **Ready** (`ready`): real data, real runs on file, gate meaningful.
- **In progress** (`in_progress`): the benchmark or its data pipeline is still being built.
- **Needs rerun** (`needs_rerun`): a result may exist but it is invalid, stale, synthetic or not produced by `surge bench`.
- **Tutorial / plumbing** (`tutorial`): small or self-referential dataset; checks plumbing, not scientific capability.
- **Sanity checks (not surrogate results)** (`sanity`): too easy as posed or measuring a proxy; not a surrogate result.

The on-file result is the best of each model's newest run on or after the benchmark's definition date (older runs are ignored even if better); a green check/cross reflects the benchmark gate at that run.

## Focus

- `pde.burgers_1d`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `plasma.cmod_density_limit`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `plasma.constellaration_multioutput`: MAybe look at the one its publsished already in the website?
- `plasma.constellaration_paper`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `plasma.qlknn10d`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `plasma.qlknn_transport`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `tabular.breast_cancer`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?
- `tabular.iris`: Need more explanation. Can we understand better parameters / visualize further the predictions we are making? How does it compare with State of the art models? Where is this published?

## Ready

| benchmark | task shape | discipline | data source + license | best on-file result (model, date) | gate | SOTA | note | visible | focus |
|---|---|---|---|---|---|---|---|---|---|
| `plasma.constellaration_multioutput` | 90 → 8 | Fusion & Plasma Physics | proxima-fusion/constellaration on HuggingFace; upstream dataset terms | r2_equilibrium_mean 0.917 ✓ (pytorch.residual_mlp, 2026-10-20) | equilibrium mean R² ≥ 0.85 | 0.99 R² (mean) [diff] | 90→8 equilibrium metrics; run 2026-09-25 mean R² 0.916, QI 0.935 | ✓ | ✓ |
| `plasma.qlknn10d` | 9 → 1 | Fusion & Plasma Physics | QLKNN10D Zenodo 10.5281/zenodo.3497066; CC-BY 4.0 | r2 0.990 ✓ (pytorch.residual_mlp, 2026-10-21) | R² ≥ 0.95 | — | surge bench run 2026-10-21: residual MLP R² 0.9898 on 200k held-out QuaLiKiz fluxes (gate 0.95); trained with the benchmark-declared hyperparameters (batch 4096, 60 epochs) | ✓ | ✓ |

## In progress

| benchmark | task shape | discipline | data source + license | best on-file result (model, date) | gate | SOTA | note | visible | focus |
|---|---|---|---|---|---|---|---|---|---|
| `plasma.constellaration_modb` | 2×64×32 → 64×32 | Fusion & Plasma Physics | Cadena et al. (2025) arXiv:2506.19583; VMEC — Hirshman & Whitson (1983) Phys. Fluids 26, 3553 | rel-L2 0.069 ✗ (sklearn.ridge, 2026-10-31) | rel-L2 ≤ 0.05 | — | field surrogates; 500/26,897 equilibria cached | — | ✓ |
| `plasma.constellaration_surfaces` | 2×64×32 → 2×16×64 | Fusion & Plasma Physics | Cadena et al. (2025) arXiv:2506.19583; VMEC — Hirshman & Whitson (1983) Phys. Fluids 26, 3553 | rel-L2 0.004 ✓ (sklearn.ridge, 2026-10-31) | rel-L2 ≤ 0.02 | — | field surrogates; 500/26,897 equilibria cached | — | ✓ |
| `plasma.solpsnn_exhaust_scalars` | 8 → 3 | Fusion & Plasma Physics | SOLPS-NN, Dasbach & Wiesen 2023, Zenodo 10.5281/zenodo.19237127; CC BY 4.0 | median rel. err. 8.4 % ✓ (kumo.tabular_regressor, 2026-11-04) | median rel. err. ≤ 0.20 (peak target heat flux) | 0.13 median rel. err. (q_peak) [same] | SOLPS-ITER exhaust scalars from the SOLPS-NN record (Dasbach & Wiesen 2023); fixed Sobol-train / random-test split. In-context Kumo Tabular reaches 8.6 % median relative error on the peak target heat flux against the published 13 % (five-fold tuned networks); the benchmark card, per-regime metrics and the gate are provisional. | ✓ | — |
| `plasma.solpsnn_te_field` | 8 → 5200 | Fusion & Plasma Physics | Dasbach & Wiesen (2023) Nucl. Mater. Energy 34, 101396; Dasbach PhD thesis (2023); Zenodo 10.5281/zenodo.19237127 (CC BY 4.0) | no result on file | median rel. err. ≤ 0.10 (all cells) | 0.065 median rel. err. (Te field) [?] | SOLPS-ITER electron-temperature field (104x50 B2 grid) from the SOLPS-NN record; hidden and provisional. Rel-L2 is computed on log10(Te); the gate uses median relative error on physical Te over all cells. | — | — |
| `sequence.lorenz63` | 3 → 3 (one step; scored as 10-λ rollout) | Nonlinear Dynamics & Chaos | Lorenz (1963) J. Atmospheric Sciences | vpt 0.770 λ ✗ (pytorch.residual_mlp, 2026-10-21) | VPT ≥ 2.0 λ | — | protocol upgraded to Lyapunov-time rollout. Sweep: residual MLP 0.91 λ, MLP 0.68, RF 0.61, ridge 0.20 — all below the 2 λ gate; capacity sweep shows bigger nets LOWER VPT (512-512-256: 0.36 λ) while improving one-step NRMSE, i.e. one-step accuracy is the wrong tuning target for chaotic rollouts (the ledger shows each model's newest saved run, so the residual MLP appears with its last capacity-sweep configuration, 0.77 λ, not the 0.91 λ default). 50 λ probe: rollouts stay bounded on the attractor (0/30 diverge, 100% still lobe-switching, climate error 0.16) — the surrogate learns the climate, not the weather. Next: rollout-aware training (noise injection, multi-step loss) and HPO on VPT instead of NRMSE. | — | — |

## Needs rerun

| benchmark | task shape | discipline | data source + license | best on-file result (model, date) | gate | SOTA | note | visible | focus |
|---|---|---|---|---|---|---|---|---|---|
| `classification.covertype` | 54 → 7 | Environmental Sciences | Blackard & Dean (1999) Computers & Electronics in Agriculture | accuracy 0.759 ✗ (sklearn.gradient_boosting_classifier, 2026-10-20) | Acc ≥ 0.85 | 0.95 Accuracy [diff] | run 2026-10-20 gradient boosting acc 0.760 FAIL on 20k subsample; full 581k not yet tried | — | ✓ |
| `fusion.m3dc1_sample` | 13 → 1 | Fusion & Plasma Physics | M3DC1 Group — NSTX-U experimental data, PPPL | r2 0.998 ✓ (synthetic fixture) (sklearn.ridge, 2026-05-22) | R² ≥ 0.85 | — | on-file result is the synthetic fallback fixture | — | — |
| `plasma.cmod_density_limit` | ~10 → 2 | Fusion & Plasma Physics | Greenwald et al. (2002) Plasma Phys. Control. Fusion | accuracy 0.997 ✓ (lgbm.classifier, 2026-05-18) | Acc ≥ 0.90 | — | accuracy on resampled data; field convention is ROC/FPR | ✓ | ✓ |
| `plasma.constellaration` | 90 → 1 | Fusion & Plasma Physics | Cadena et al. (2025) arXiv:2506.19583 — ConStellaration dataset | r2 0.899 ✓ (sklearn.mlp, 2026-05-23) | R² ≥ 0.85 | 0.982 R² [diff] | code gate fixed to 0.85 (was 0.50); rerun in progress 2026-10-20 | — | ✓ |
| `plasma.constellaration_paper` | 90 → 12 | Fusion & Plasma Physics | Cadena et al. (2025) arXiv:2506.19583 §A.4 — 12-metric protocol | r2_mean 0.928 ✗ (pytorch.residual_mlp, 2026-05-23) | mean R² ≥ 0.85 | 0.99 R² (mean) [diff] | paper protocol, mean R² 0.929 vs paper 0.97 — parity check, honest fail | ✓ | ✓ |

## Tutorial / plumbing

| benchmark | task shape | discipline | data source + license | best on-file result (model, date) | gate | SOTA | note | visible | focus |
|---|---|---|---|---|---|---|---|---|---|
| `classification.flow_regime` | 3 → 4 | Fluid Dynamics & PDE Operators | SURGE inline fixture — Mach / Re / AoA regime labeling | accuracy 0.944 ✓ (sklearn.random_forest_classifier, 2026-10-21) | Acc ≥ 0.85 | — | Small classification set (n < 1,000 effective); plumbing check | ✓ | — |
| `classification.plasma_stability` | 12 → 2 | Engineering | Arzamasov, Bohm & Jochem (2018) IEEE PMAPS | accuracy 0.936 ✓ (sklearn.gradient_boosting_classifier, 2026-10-19) | Acc ≥ 0.92 | 0.978 Accuracy [?] | Power-grid stability classification (not fusion); hidden tutorial-tier plumbing check | — | — |
| `multioutput.scm20d` | 61 → 20 | Operations Research | Spyromitros-Xioufis et al. (2016) Machine Learning | r2 0.879 ✓ (pytorch.gru, 2026-05-18) | avg R² ≥ 0.60 | — | Small multi-output set; plumbing / teaching benchmark | ✓ | — |
| `plasma.qlknn_transport` | 10 → 1 | Fusion & Plasma Physics | van de Plassche et al. (2020) Phys. Plasmas — QuaLiKiz surrogate | r2 0.950 ✓ (pytorch.residual_mlp, 2026-05-22) | R² ≥ 0.90 | — | ground truth is the QLKNN_7_11 network itself | ✓ | ✓ |
| `tabular.airfoil_noise` | 5 → 1 | Engineering | Brooks, Pope & Marcolini (1989) NASA TM 100514 | r2 0.986 ✓ (kumo.tabular_regressor, 2026-11-02) | R² ≥ 0.85 | 0.963 R² [?] | General tabular demo (UCI/NASA): pull-and-run tutorial; real runs on file, not a fusion result | ✓ | — |
| `tabular.breast_cancer` | 30 → 2 | Biology & Medicine | Mangasarian & Wolberg (1990) University of Wisconsin | accuracy 0.956 ✓ (pytorch.mlp_classifier, 2026-05-18) | Acc ≥ 0.95 | 0.99 Accuracy [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | ✓ |
| `tabular.california_housing` | 8 → 1 | Environmental Sciences | Pace & Barry (1997) Statistics & Probability Letters | r2 0.890 ✓ (kumo.tabular_regressor, 2026-11-02) | R² ≥ 0.75 | — | General tabular demo (OpenML, n ≈ 20k): pull-and-run tutorial for trying SURGE; real runs on file, not a fusion result | ✓ | — |
| `tabular.concrete_strength` | 8 → 1 | Materials Science | Yeh (1998) Cement and Concrete Research | r2 0.944 ✓ (kumo.tabular_regressor, 2026-11-02) | R² ≥ 0.80 | 0.94 R² [diff] | General tabular demo (UCI): pull-and-run tutorial; real runs on file, not a fusion result | ✓ | — |
| `tabular.diabetes` | 10 → 1 | Biology & Medicine | Efron, Hastie, Johnstone & Tibshirani (2004) Annals of Statistics | r2 0.535 ✓ (pytorch.mlp, 2026-05-18) | R² ≥ 0.40 | 0.5 R² [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.digits` | 64 → 10 | Computer Vision | Alpaydin & Kaynak (1998) UCI | accuracy 0.983 ✓ (pytorch.mlp_classifier, 2026-05-18) | Acc ≥ 0.97 | 0.99 Accuracy [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.energy_efficiency` | 8 → 1 | Engineering | Tsanas & Xifara (2012) Energy & Buildings | r2 0.997 ✓ (kumo.tabular_regressor, 2026-11-02) | R² ≥ 0.90 | 0.99 R² [?] | General tabular demo (UCI): pull-and-run tutorial; real runs on file, not a fusion result | ✓ | — |
| `tabular.iris` | 4 → 3 | Biology & Medicine | Fisher (1936) Annals of Eugenics | accuracy 0.947 ✓ (sklearn.logistic_regression, 2026-10-21) | Acc ≥ 0.95 | 1 Accuracy [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | ✓ |
| `tabular.superconductor` | 81 → 1 | Materials Science | Hamidieh (2018) Computational Materials Science | r2 0.941 ✓ (kumo.tabular_regressor, 2026-11-02) | R² ≥ 0.90 | 0.92 R² [?] | Materials-science tabular demo (UCI, n ≈ 21k): pull-and-run tutorial for trying SURGE on a larger table; real runs on file | ✓ | — |
| `tabular.wine` | 13 → 3 | Chemistry | Aeberhard, Coomans & De Vel (1992) UCI | accuracy 1.000 ✓ (sklearn.logistic_regression, 2026-05-18) | Acc ≥ 0.90 | 1 Accuracy [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.yacht_dynamics` | 6 → 1 | Engineering | Gerritsma, Onnink & Versluis (1981) CFD — SHIP DESIGN | r2 1.000 ✓ (gpflow.gpr, 2026-05-23) | R² ≥ 0.95 | 0.99 R² [?] | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `vision.cifar10` | 32×32×3 → 10 | Computer Vision | Krizhevsky (2009) — Learning Multiple Layers of Features from Tiny Images | accuracy 0.965 ✓ (pytorch.dino_linear_probe, 2026-10-21) | Acc ≥ 0.91 | 0.995 Accuracy [diff] | Natural-image classification with no surrogate-modelling content; kept as a plumbing/tutorial benchmark and a transfer-learning reference. surge bench run 2026-10-21: pytorch.dino_linear_probe (frozen DINOv2 ViT-S/14 + LogisticRegression, literature linear-probe protocol) reaches accuracy 0.965 (gate 0.91); the from-scratch pytorch.resnet20 remains the scratch-training reference | ✓ | — |
| `vision.mnist` | 28×28 → 10 | Computer Vision | LeCun, Bottou, Bengio & Haffner (1998) Proc. IEEE | accuracy 0.996 ✓ (pytorch.resnet20, 2026-05-26) | Acc ≥ 0.99 | 0.998 Accuracy [same] | MNIST is a plumbing check, not a surrogate result | ✓ | — |

## Sanity checks (not surrogate results)

| benchmark | task shape | discipline | data source + license | best on-file result (model, date) | gate | SOTA | note | visible | focus |
|---|---|---|---|---|---|---|---|---|---|
| `pde.burgers_1d` | 64 → 64 (field) | Fluid Dynamics & PDE Operators | SURGE finite-difference Burgers generator; project generated fixture | rel-L2 0.019 ✓ (pytorch.fno1d, 2026-10-31) | NRMSE ≤ 0.10 | 0.0068 rel-L2 [diff] | R² 1.000 at 64 points — too easy as posed vs FNO paper's 1024 | ✓ | ✓ |
