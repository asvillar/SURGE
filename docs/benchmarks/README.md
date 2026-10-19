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

The on-file result is the best model of the newest sweep (runs within 3 h of the newest one); older runs are ignored even if better; a green check/cross reflects the benchmark gate at that run.

## Ready

| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |
|---|---|---|---|---|---|---|---|
| `plasma.constellaration_multioutput` | 90 → 8 | proxima-fusion/constellaration on HuggingFace; upstream dataset terms | r2_equilibrium_mean 0.916 ✓ (pytorch.residual_mlp, 2026-09-25) | equilibrium mean R² ≥ 0.85 | 90→8 equilibrium metrics; run 2026-09-25 mean R² 0.916, QI 0.935 | ✓ | — |
| `tabular.airfoil_noise` | 5 → 1 | Brooks, Pope & Marcolini (1989) NASA TM 100514 | r2 0.979 ✓ (pytorch.kan, 2026-05-18) | R² ≥ 0.85 | Real UCI/OpenML data (n ≥ 1,000 train+test) with real runs on file | ✓ | — |
| `tabular.california_housing` | 8 → 1 | Pace & Barry (1997) Statistics & Probability Letters | r2 0.847 ✓ (xgboost.xgbregressor, 2026-05-18) | R² ≥ 0.75 | Real UCI/OpenML data (n ≥ 1,000 train+test) with real runs on file | ✓ | — |
| `tabular.concrete_strength` | 8 → 1 | Yeh (1998) Cement and Concrete Research | r2 0.930 ✓ (gpflow.gpr, 2026-05-23) | R² ≥ 0.80 | Real UCI/OpenML data (n ≥ 1,000 train+test) with real runs on file | ✓ | — |
| `tabular.energy_efficiency` | 8 → 1 | Tsanas & Xifara (2012) Energy & Buildings | r2 0.997 ✓ (gpflow.multi_kernel, 2026-05-23) | R² ≥ 0.90 | Real UCI/OpenML data (n ≥ 1,000 train+test) with real runs on file | ✓ | — |
| `tabular.superconductor` | 81 → 1 | Hamidieh (2018) Computational Materials Science | r2 0.930 ✓ (sklearn.random_forest, 2026-05-18) | R² ≥ 0.90 | Real UCI/OpenML data (n ≥ 1,000 train+test) with real runs on file | ✓ | — |

## In progress

| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |
|---|---|---|---|---|---|---|---|
| `plasma.constellaration_modb` | 2×64×32 → 64×32 | Cadena et al. (2025) arXiv:2506.19583; VMEC — Hirshman & Whitson (1983) Phys. Fluids 26, 3553 | no result on file | rel-L2 ≤ 0.05 | field surrogates; 500/26,897 equilibria cached | ✓ | — |
| `plasma.constellaration_surfaces` | 2×64×32 → 2×16×64 | Cadena et al. (2025) arXiv:2506.19583; VMEC — Hirshman & Whitson (1983) Phys. Fluids 26, 3553 | no result on file | rel-L2 ≤ 0.02 | field surrogates; 500/26,897 equilibria cached | ✓ | — |

## Needs rerun

| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |
|---|---|---|---|---|---|---|---|
| `classification.covertype` | 54 → 7 | Blackard & Dean (1999) Computers & Electronics in Agriculture | no result on file | Acc ≥ 0.85 | no result on file | ✓ | — |
| `classification.plasma_stability` | 12 → 2 | Arzamasov, Bohm & Jochem (2018) IEEE PMAPS | no result on file | Acc ≥ 0.92 | no result on file | ✓ | — |
| `fusion.m3dc1_sample` | 13 → 1 | M3DC1 Group — NSTX-U experimental data, PPPL | r2 0.998 ✓ (synthetic fixture) (sklearn.ridge, 2026-05-22) | R² ≥ 0.85 | on-file result is the synthetic fallback fixture | ✓ | — |
| `plasma.cmod_density_limit` | ~10 → 2 | Greenwald et al. (2002) Plasma Phys. Control. Fusion | accuracy 0.997 ✓ (lgbm.classifier, 2026-05-18) | Acc ≥ 0.90 | accuracy on resampled data; field convention is ROC/FPR | ✓ | — |
| `plasma.constellaration` | 90 → 1 | Cadena et al. (2025) arXiv:2506.19583 — ConStellaration dataset | r2 0.899 ✓ (sklearn.mlp, 2026-05-23) | R² ≥ 0.85 | docs gate 0.85 but code gate 0.50; rerun after fix | ✓ | — |
| `plasma.constellaration_paper` | 90 → 12 | Cadena et al. (2025) arXiv:2506.19583 §A.4 — 12-metric protocol | r2_mean 0.867 ✗ (pytorch.ft_transformer, 2026-05-23) | mean R² ≥ 0.85 | paper protocol, mean R² 0.929 vs paper 0.97 — parity check, honest fail | ✓ | — |
| `plasma.qlknn10d` | 9 → 1 | van de Plassche et al. (2020) Phys. Plasmas 27, 022310 — QLKNN10D, Zenodo 10.5281/zenodo.3497066 | no result on file | R² ≥ 0.95 | strongest result, R² 0.989 in examples/qlknn10d_study.py, no result.json via surge bench yet | ✓ | — |
| `vision.cifar10` | 32×32×3 → 10 | torchvision CIFAR-10; upstream dataset terms | accuracy 0.285 ✗ (pytorch.resnet20, 2026-06-16) | Acc ≥ 0.91 | best on file 0.285, invalid run | ✓ | — |

## Tutorial / plumbing

| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |
|---|---|---|---|---|---|---|---|
| `classification.flow_regime` | 3 → 4 | SURGE inline fixture — Mach / Re / AoA regime labeling | accuracy 0.944 ✓ (sklearn.random_forest_classifier, 2026-10-13) | Acc ≥ 0.85 | Small classification set (n < 1,000 effective); plumbing check | ✓ | — |
| `multioutput.scm20d` | 61 → 20 | Spyromitros-Xioufis et al. (2016) Machine Learning | r2 0.879 ✓ (pytorch.gru, 2026-05-18) | avg R² ≥ 0.60 | Small multi-output set; plumbing / teaching benchmark | ✓ | — |
| `plasma.qlknn_transport` | 10 → 1 | Google DeepMind fusion_surrogates QLKNN; upstream package/model terms | r2 0.747 ✗ (sklearn.random_forest, 2026-06-16) | R² ≥ 0.90 | ground truth is the QLKNN_7_11 network itself | ✓ | — |
| `tabular.breast_cancer` | 30 → 2 | Mangasarian & Wolberg (1990) University of Wisconsin | accuracy 0.956 ✓ (pytorch.mlp_classifier, 2026-05-18) | Acc ≥ 0.95 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.diabetes` | 10 → 1 | Efron, Hastie, Johnstone & Tibshirani (2004) Annals of Statistics | r2 0.519 ✓ (gpflow.gpr, 2026-05-23) | R² ≥ 0.40 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.digits` | 64 → 10 | Alpaydin & Kaynak (1998) UCI | accuracy 0.983 ✓ (pytorch.mlp_classifier, 2026-05-18) | Acc ≥ 0.97 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.iris` | 4 → 3 | Fisher (1936) Annals of Eugenics | accuracy 0.947 ✓ (sklearn.logistic_regression, 2026-10-13) | Acc ≥ 0.95 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.wine` | 13 → 3 | Aeberhard, Coomans & De Vel (1992) UCI | accuracy 1.000 ✓ (sklearn.logistic_regression, 2026-05-18) | Acc ≥ 0.90 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `tabular.yacht_dynamics` | 6 → 1 | Gerritsma, Onnink & Versluis (1981) CFD — SHIP DESIGN | r2 1.000 ✓ (gpflow.gpr, 2026-05-23) | R² ≥ 0.95 | Small dataset (n < 1,000); plumbing / teaching benchmark | ✓ | — |
| `vision.mnist` | 28×28 → 10 | LeCun, Bottou, Bengio & Haffner (1998) Proc. IEEE | accuracy 0.996 ✓ (pytorch.resnet20, 2026-05-26) | Acc ≥ 0.99 | MNIST is a plumbing check, not a surrogate result | ✓ | — |

## Sanity checks (not surrogate results)

| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |
|---|---|---|---|---|---|---|---|
| `pde.burgers_1d` | 64 → 64 (field) | Li et al. (2021) ICLR — FNO paper; solver: Burgers (1948) | r2 1.000 ✓ (pytorch.fno1d, 2026-05-22) | NRMSE ≤ 0.10 | R² 1.000 at 64 points — too easy as posed vs FNO paper's 1024 | ✓ | — |
| `sequence.lorenz63` | 3×20 → 3×20 (rollout) | Lorenz (1963) J. Atmospheric Sciences | nrmse 0.026 ✓ (sklearn.random_forest, 2026-10-13) | NRMSE ≤ 0.10 | fixed-window NRMSE, not Lyapunov-time rollout | ✓ | — |
