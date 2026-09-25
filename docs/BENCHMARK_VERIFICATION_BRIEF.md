# SURGE benchmark verification brief

Handoff document for verifying that each SURGE leaderboard benchmark is a
legitimate, correctly described scientific or ML task. This brief covers
**benchmark definitions only** — not SURGE model scores or pass thresholds as
performance claims.

> **Note (2026-07):** the Cursor canvas dashboards
> (`surge/viz/*.canvas.tsx`) referenced below were removed from the
> repository. Benchmark citations, shapes, and tiers are being consolidated
> into `surge/benchmarks/registry.py` and the artifact-driven leaderboard
> report; where this brief says "canvas", read the registry metadata.

**Primary sources**

| Artifact | Path |
|----------|------|
| Registry (keys, loaders, aliases) | `surge/benchmarks/registry.py` |
| Task runners | `surge/benchmarks/tasks.py` |
| Dataset loaders | `surge/benchmarks/leaderboard.py` |
| I/O documentation | `surge/benchmarks/dataset_io.py` |

**Related docs:** [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) (full report with
model grid), [`benchmarks/benchmark_policy.md`](benchmarks/benchmark_policy.md)
(CI tiers and commands).

---

## How to use this brief

For each benchmark below, a verifier should return:

```markdown
### <registry_key>
- **Verdict:** VALID | PARTIALLY VALID | MISLABELED | SYNTHETIC-ONLY
- **Citation match:** yes/no (+ corrected citation if needed)
- **Problem:** one-line confirmation or correction
- **Real vs synthetic:** ...
- **SURGE modifications:** subsample / mask / fallback / balancing
- **Flags:** ...
```

**SURGE tier legend**

| Tier | Meaning |
|------|---------|
| 0 | Inline or synthetic fixture; no external download |
| 1 | Published tabular/UCI/sklearn/OpenML data; network on first run |
| 2 | Larger domain-specific sets (vision, fusion/plasma); GPU often advised |

**Default SURGE protocol:** 80/20 train/test split, `seed=42`, unless noted.

---

## Scalar regression

### `tabular.california_housing`

| Field | Detail |
|-------|--------|
| Citation | Pace & Barry (1997), *Statistics & Probability Letters* — [DOI 10.1016/S0167-7160(97)00010-0](https://doi.org/10.1016/S0167-7160(97)00010-0) |
| Problem | Predict median house value in California census block groups from socioeconomic and geographic features (1990 US Census aggregates). |
| Task | Scalar regression |
| I/O | 8 inputs → 1 target (`MedHouseVal`, units of \$100k) |
| n | 20,640 |
| Source | `sklearn.datasets.fetch_california_housing` |
| Metric | R² |

### `tabular.concrete_strength`

| Field | Detail |
|-------|--------|
| Citation | Yeh (1998), *Cement and Concrete Research* — [DOI 10.1016/S0008-8846(98)00165-3](https://doi.org/10.1016/S0008-8846(98)00165-3) |
| Problem | Predict concrete compressive strength (MPa) from mixture composition and age. |
| Task | Scalar regression |
| I/O | 8 → 1 |
| n | 1,030 |
| Source | OpenML #4353 (`fetch_openml`) |
| Metric | R² |

### `tabular.energy_efficiency`

| Field | Detail |
|-------|--------|
| Citation | Tsanas & Xifara (2012), *Energy & Buildings* — [DOI 10.1016/j.enbuild.2012.03.003](https://doi.org/10.1016/j.enbuild.2012.03.003) |
| Problem | Predict building **heating load** from envelope, geometry, and HVAC features. |
| Task | Scalar regression |
| I/O | 8 → 1 (SURGE uses target **Y1 = Heating Load** only; dataset also has cooling load Y2) |
| n | 768 |
| Source | OpenML `energy-efficiency` v1 |
| Metric | R² |

### `tabular.airfoil_noise`

| Field | Detail |
|-------|--------|
| Citation | Brooks, Pope & Marcolini (1989), NASA TM 100514 — [UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) |
| Problem | Predict scaled sound pressure level of airfoil self-noise from aerodynamic and geometry parameters. |
| Task | Scalar regression |
| I/O | 5 → 1 |
| n | 1,503 |
| Source | OpenML `airfoil_self_noise` |
| Metric | R² |

### `tabular.yacht_dynamics`

| Field | Detail |
|-------|--------|
| Citation | Gerritsma, Onnink & Versluis (1981) — [UCI Yacht Hydrodynamics](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics) |
| Problem | Predict residuary resistance per unit weight of sailing yachts from hull geometry ratios. |
| Task | Scalar regression |
| I/O | 6 → 1 |
| n | 308 |
| Source | OpenML `yacht_hydrodynamics` |
| Metric | R² |

### `tabular.superconductor`

| Field | Detail |
|-------|--------|
| Citation | Hamidieh (2018), *Computational Materials Science* — [DOI 10.1016/j.commatsci.2018.07.052](https://doi.org/10.1016/j.commatsci.2018.07.052) |
| Problem | Predict superconducting critical temperature Tc from chemical/formula-derived features. |
| Task | Scalar regression |
| I/O | 81 → 1 |
| n | 21,263 |
| Source | OpenML `superconduct` |
| Metric | R² |

### `tabular.diabetes`

| Field | Detail |
|-------|--------|
| Citation | Efron, Hastie, Johnstone & Tibshirani (2004), *Annals of Statistics* — [DOI 10.1214/009053604000000067](https://doi.org/10.1214/009053604000000067) |
| Problem | Predict diabetes disease progression one year after baseline from 10 physiological variables. |
| Task | Scalar regression |
| I/O | 10 → 1 |
| n | 442 |
| Source | `sklearn.datasets.load_diabetes` |
| Metric | R² |
| Notes | High label noise; low R² is expected. Tier 1 but used mainly as a fast CI smoke test. |

---

## Multi-output regression

### `multioutput.scm20d`

| Field | Detail |
|-------|--------|
| Citation | Spyromitros-Xioufis et al. (2016), *Machine Learning* — [DOI 10.1007/s10994-016-5546-z](https://doi.org/10.1007/s10994-016-5546-z) |
| Problem | Joint prediction of 20 supply-chain management target variables from 61 input features. |
| Task | Multi-output regression |
| I/O | 61 → 20 |
| n | 8,966 (OpenML `scm20d` v2 — verify row count on OpenML) |
| Source | OpenML `scm20d` |
| Metric | Average R² across outputs |

---

## Tabular classification

### `tabular.breast_cancer`

| Field | Detail |
|-------|--------|
| Citation | Mangasarian & Wolberg (1990) — [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| Problem | Binary classification: malignant vs benign from cytology features. |
| Task | Binary classification |
| I/O | 30 → 2 |
| n | 569 |
| Source | `sklearn.datasets.load_breast_cancer` |
| Metric | Accuracy (F1, AUROC also logged) |

### `tabular.digits`

| Field | Detail |
|-------|--------|
| Citation | Alpaydin & Kaynak (1998) — [UCI Optical Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) |
| Problem | Classify 8×8 grayscale digit images (flattened to 64 features). |
| Task | 10-class classification |
| I/O | 64 → 10 |
| n | 1,797 |
| Source | `sklearn.datasets.load_digits` |
| Metric | Accuracy |

### `tabular.iris`

| Field | Detail |
|-------|--------|
| Citation | Fisher (1936), *Annals of Eugenics* — [DOI 10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) |
| Problem | Classify iris species from sepal and petal measurements. |
| Task | 3-class classification |
| I/O | 4 → 3 |
| n | 150 |
| Source | `sklearn.datasets.load_iris` |
| Metric | Accuracy |

### `tabular.wine`

| Field | Detail |
|-------|--------|
| Citation | Aeberhard, Coomans & De Vel (1992) — [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine) |
| Problem | Classify wine cultivar from chemical analysis. |
| Task | 3-class classification |
| I/O | 13 → 3 |
| n | 178 |
| Source | `sklearn.datasets.load_wine` |
| Metric | Accuracy |

---

## Scientific classification

### `tabular.covertype` (canvas key: `classification.covertype`)

| Field | Detail |
|-------|--------|
| Citation | Blackard & Dean (1999) — [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) |
| Problem | Classify forest cover type from cartographic and terrain variables. |
| Task | 7-class classification |
| I/O | 54 → 7 |
| n | **20,000 subsample** of full 581,012 rows |
| Source | UCI / OpenML |
| Metric | Accuracy |
| SURGE note | Subsampled for leaderboard speed; harder than the full dataset. |

### `tabular.plasma_stability` (canvas key: `classification.plasma_stability`)

| Field | Detail |
|-------|--------|
| Citation | Arzamasov, Bohm & Jochem (2018), IEEE PMAPS — [UCI Electrical Grid Stability](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) |
| Problem | Binary classification of **power-grid stability** from simulated operating points. |
| Task | Binary classification |
| I/O | 12 → 2 |
| n | 10,000 |
| Source | UCI electrical grid stability simulated data |
| Metric | Accuracy |
| Verify | Canvas labels this “plasma proxy”; data is **grid simulation**, not fusion plasma measurements. |

### `classification.flow_regime`

| Field | Detail |
|-------|--------|
| Citation | **None** — SURGE inline fixture |
| Problem | Classify CFD-like flow regime from Mach number, log₁₀(Reynolds), angle of attack. |
| Task | 4-class classification |
| I/O | 3 → 4 (subsonic laminar / subsonic turbulent / transonic / supersonic) |
| n | 800 |
| Source | **Fully synthetic** — rule-based labels + 5% label noise |
| Metric | Accuracy |
| Notes | Tier 0 smoke test only; not literature-backed. |

---

## Time series / forecasting

### `sequence.lorenz63`

| Field | Detail |
|-------|--------|
| Citation | Lorenz (1963), *J. Atmospheric Sciences* — [DOI 10.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2) |
| Problem | Short-horizon prediction on the Lorenz-63 chaotic attractor (σ=10, ρ=28, β=8/3). |
| Task | Windowed sequence forecasting |
| I/O | 3 state vars × 20 timesteps in → 3 × 20 out (60 → 60 flattened) |
| n | 1,200 trajectories |
| Source | **SURGE-generated** via RK-4 (`dt=0.01`, warmup=500 steps) |
| Metric | NRMSE |

---

## 1D PDE operator learning

### `pde.burgers_1d`

| Field | Detail |
|-------|--------|
| Citation | Li et al. (2021) FNO — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895); PDE: viscous Burgers |
| Problem | Learn operator u(x,0) → u(x,T) for 1D viscous Burgers: ∂u/∂t + u∂u/∂x = ν∂²u/∂x². |
| Task | Field-to-field operator regression |
| I/O | 64-point field → 64-point field |
| n | 1,024 simulation trajectories |
| Source | **SURGE inline finite-difference solver** (ν=0.01, nt=100, dt=0.001) |
| Metric | NRMSE / relative L² |
| Verify | Not PDEBench HDF5; coarser grid (n_x=64) than typical FNO paper setups (often 1024). |

---

## Vision

### `vision.mnist`

| Field | Detail |
|-------|--------|
| Citation | LeCun, Bottou, Bengio & Haffner (1998) — [MNIST](http://yann.lecun.com/exdb/mnist/) |
| Problem | Classify 28×28 handwritten digits. |
| Task | 10-class image classification |
| I/O | 28×28 grayscale → 10 classes |
| n | 70,000 (standard 60k/10k split) |
| Source | `torchvision` MNIST |
| Metric | Top-1 accuracy |

### `vision.cifar10`

| Field | Detail |
|-------|--------|
| Citation | Krizhevsky (2009) — [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Problem | Classify 32×32×3 natural images into 10 object categories. |
| Task | 10-class image classification |
| I/O | 32×32×3 → 10 |
| n | 60,000 |
| Source | `torchvision` CIFAR-10 |
| Metric | Top-1 accuracy |

---

## Scientific domain (fusion / plasma)

### `fusion.m3dc1_sample`

| Field | Detail |
|-------|--------|
| Citation | M3DC1 Group, PPPL — [m3dc1.pppl.gov](https://m3dc1.pppl.gov) |
| Problem | Tokamak MHD equilibrium surrogate: map equilibrium parameters to a stability/growth-rate metric. |
| Task | Scalar regression |
| I/O | 13 → 1 |
| n | Depends on HDF5; synthetic fallback uses 2,000 rows |
| Source | `m3dc1_sample.hdf5` if present; else **synthetic Gaussian linear fixture** |
| Metric | R² |
| Verify | Confirm whether a given run used real M3DC1 HDF5 or the synthetic fallback. |

### `plasma.cmod_density_limit`

| Field | Detail |
|-------|--------|
| Citation | Greenwald et al. (2002), *Plasma Phys. Control. Fusion* — [DOI 10.1088/0741-3335/44/8/325](https://doi.org/10.1088/0741-3335/44/8/325) |
| Problem | Predict density-limit disruption precursor (Greenwald limit) from Alcator C-Mod plasma signals. |
| Task | Binary classification |
| I/O | 6 plasma signals → 2 (`density_limit_phase`) |
| n | Raw: 264,385 time-slices; SURGE **balances to ≤40k** (20k per class) |
| Source | [MIT-PSFC/open_density_limit_database](https://github.com/MIT-PSFC/open_density_limit_database) |
| Metric | Accuracy |
| Notes | Raw data highly imbalanced (~1.4% positive). |

### `plasma.constellaration`

| Field | Detail |
|-------|--------|
| Citation | Cadena et al. (2025) — [arXiv:2506.19583](https://arxiv.org/abs/2506.19583) |
| Problem | Stellarator surrogate: VMEC/DESC boundary shape → log₁₀(qi) quasi-isodynamic quality. |
| Task | Scalar regression |
| I/O | 90 boundary Fourier coefficients → 1 target |
| n | **10k random subsample** from 26,897 filtered rows |
| Source | HuggingFace `proxima-fusion/constellaration` |
| Metric | R² |

### `plasma.constellaration_paper`

| Field | Detail |
|-------|--------|
| Citation | Cadena et al. (2025) §A.4 — [arXiv:2506.19583](https://arxiv.org/abs/2506.19583) |
| Problem | Paper protocol: **12 independent 90→1 models**, one per stellarator metric. |
| Task | 12 scalar regressions (reported as mean test R²) |
| I/O | 90 → 12 metrics |
| n | 26,897 after filters |
| Source | HF `proxima-fusion/constellaration` |
| Filters | nfp=3, optimised DESC/VMEC, 0.05% outlier clip per metric |
| Metric | Mean test R² across 12 models |
| Verify | Filtering matches Cadena et al. Appendix A.4. |

### `plasma.constellaration_multioutput`

| Field | Detail |
|-------|--------|
| Citation | Cadena et al. (2025) — [arXiv:2506.19583](https://arxiv.org/abs/2506.19583) |
| Problem | **Single joint surrogate** predicting all 12 stellarator metrics from boundary shape (alternative to the paper’s 12-model protocol). |
| Task | Multi-output regression |
| I/O | 90 → 12 |
| n | 26,897 (same filtered cache as paper benchmark) |
| Source | Same as `plasma.constellaration_paper` |
| Outputs | aspect_ratio, elongation, mirror ratios, triangularity, vacuum_well, log₁₀(qi), etc. — see `surge/benchmarks/dataset_io.py` |
| Target classes | **equilibrium** (8, need the VMEC++ solve): ι axis/edge, both mirror ratios, vacuum well, min L∇B, flux compression, log₁₀(qi) · **geometry** (3, closed-form from the boundary): aspect_ratio, max_elongation, average_triangularity · **derived** (1, algebraic ratio of two other outputs): aspect_ratio_over_edge_rotational_transform — see `output_kinds()` in `surge/benchmarks/dataset_io.py` |
| Metric | **Gate is the equilibrium-mean R²** (mean over the 8 equilibrium targets); geometry targets are reported as a sanity check (expected R² ≥ 0.99), not part of the gate; log₁₀(qi) also reported standalone |

### `plasma.qlknn_transport`

| Field | Detail |
|-------|--------|
| Citation | van de Plassche et al. (2020) — [DOI 10.1063/1.5134126](https://doi.org/10.1063/1.5134126) (canvas); code also cites *Physics of Plasmas* 27, 022310 |
| Problem | Emulate QLKNN/QuaLiKiz turbulent transport: predict electron ITG heat flux from normalised gyrokinetic parameters. |
| Task | Scalar regression |
| I/O | 10 plasma parameters → `efeITG` (gyroBohm-normalised electron heat flux) |
| n | ~7,475 after **efeITG > 0** mask (from 20k sampled points) |
| Source | Ground truth = **QLKNN_7_11** via Google DeepMind `fusion_surrogates` (surrogate-of-surrogate) |
| Inputs | Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni |
| Metric | R² |
| Verify | Reconcile primary citation (journal/year) across canvas and loader comments. |

---

## Pending / higher-tier (cited in canvas, not in main leaderboard table)

| Registry key | Citation | Problem | I/O (approx.) | Source |
|--------------|----------|---------|---------------|--------|
| `pdebench.burgers_1d` | Takamoto et al., NeurIPS 2022 (PDEBench) | Real 1D Burgers HDF5 operator learning | 1024 → 1024, n≈9,000 | PDEBench HDF5 download |
| `pdebench.darcy_2d` | Takamoto et al., NeurIPS 2022 | 2D Darcy flow permeability field | 128×128 → 128×128 | PDEBench HDF5 |
| `pdebench.shallow_water_2d` | Takamoto et al., NeurIPS 2022 | 2D shallow-water dynamics | 128×128 field | PDEBench HDF5 |
| `thewell.gray_scott` | Ohana et al., NeurIPS 2024 (The Well) | 2D Gray-Scott reaction–diffusion | 64×64×2 → 64×64×2 | `the-well` package |
| `thewell.turbulence_2d` | Ohana et al., NeurIPS 2024 | 2D homogeneous turbulence | 64×64×4 fields | `the-well` |
| `thewell.mhd` | Ohana et al., NeurIPS 2024 | 3D MHD turbulence | 64³×8 fields | `the-well` |

---

## Known discrepancies to verify

| Item | Canvas / doc | Code | Action |
|------|--------------|------|--------|
| Plasma stability registry key | `classification.plasma_stability` | `tabular.plasma_stability` | Confirm alias equivalence |
| Covertype registry key | `classification.covertype` | `tabular.covertype` | Confirm alias equivalence |
| `multioutput.scm20d` n | 8,966 | Registry once listed 9,803 | Check OpenML v2 row count |
| `fusion.m3dc1_sample` n | ~500 | Synthetic fallback n=2,000 | Confirm HDF5 vs fallback |
| `plasma.cmod_density_limit` I/O | Canvas ~10 → 2 | Loader 6 → 2 | Canvas may be stale |
| QLKNN citation | Phys. Plasmas DOI in canvas | Nuclear Fusion / PoP in code | Reconcile primary reference |
| `classification.flow_regime` | Grouped with scientific benchmarks | No external paper; synthetic | Tier 0 only |

---

## CLI aliases (for cross-checking registry keys)

```bash
surge list   # full key list
```

Short names include: `california`, `qlknn`, `constellaration`, `cmod`, `mnist`,
`cifar10`, `covertype`, `burgers`, etc. — mapped in `surge/benchmarks/registry.py`
under `_SHORT_ALIASES`.
