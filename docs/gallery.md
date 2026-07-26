# Results gallery

What SURGE produces, by problem type. Every figure below is generated
from machine-readable run artifacts by the SURGE visual system — nothing
is hand-drawn or hand-entered. Regenerate all of them (light **and**
dark variants, deterministic PNG/SVG/PDF) with:

```bash
python examples/viz_theme_gallery.py
python scripts/sync_readme_assets.py                        # curate into docs/
python -m surge.report.leaderboard --out leaderboard.html   # dashboard
```

## Tabular regression

Parity density in the SURGE signature style — reversed-plasma colormap,
log-scaled counts, train/test panels, R² boxes — here for the QLKNN
ITG heat-flux surrogate (random-forest baseline; the HPO-tuned residual
MLP reaches R² 0.96 on the same task).

![Regression parity density plots](assets/gallery/parity.png)

## Hyperparameter optimization

Per-model Optuna history: solid per-trial trace, dashed running best,
gold star at each model's optimum with its score.

![HPO history with starred bests](assets/gallery/hpo_convergence.png)

## Neural-network training

Per-epoch train/validation loss from the JSONL training logs: shaded
generalisation gap, best epoch marked, power-law convergence fit, and
smoothed early stopping (patience on the rolling-mean validation loss)
terminating training at true saturation.

![Training curves](assets/gallery/training_curves.png)

## Classification

ROC, precision–recall, confusion matrix (counts + shares), and a
reliability diagram with per-bin calibration gaps and ECE.

![Classification diagnostics](assets/gallery/classification.png)

## Operator learning (fields / PDEs)

Burgers' equation surrogate trained through the registry: best / median /
worst test samples, the signed-error field over the whole test set
(sorted by per-sample rel-L2), and the error distribution.

![Field operator diagnostics](assets/gallery/field_operator.png)

## 2D operator learning

FNO-2D learns the periodic Poisson solver (source → solution) at median
rel-L2 0.006 on 32×32 fields generated on the fly — U-Net covers the
same task shape.

![2D operator learning triptych](assets/gallery/field2d.png)

## External PDE benchmark — TheWell Gray-Scott

Operator forecasting on the Gray-Scott reaction–diffusion system from
[TheWell](https://polymathic-ai.org/the_well/) (Ohana et al., NeurIPS
2024): predict the 64×64 species-B field 160 stored steps ahead. The
horizon is set where the persistence baseline ("predict no change",
green bar) visibly fails — at single-step the task is trivial
(persistence rel-L2 0.002). U-Net with a residual target leads (median
rel-L2 0.206) and is the only architecture beating persistence (0.265);
FNO-2D blurs the sharp Turing interfaces at this horizon and DeepONet's
global low-rank basis cannot localize. Reproduce with:

```bash
# note: the full Gray-Scott archive is ~132 GB (117 train + 15 valid)
python -c "from surge.benchmarks.loaders.thewell import download_thewell; download_thewell('gray_scott')"
python examples/thewell_grayscott_study.py
```

![TheWell Gray-Scott surrogate study](assets/gallery/thewell_grayscott.png)

## Stellarator design — ConStellaration

One residual MLP maps stellarator plasma-boundary Fourier coefficients
(R_mn, Z_mn; n_fp = 3) to 12 equilibrium figures of merit across 26,897
QI-like configurations (Goodman et al. 2025, arXiv:2506.19583): real
rotating boundary cross-sections, log₁₀(QI) parity at R² 0.93, and
per-metric learnability.

![ConStellaration stellarator surrogate](assets/gallery/constellaration.png)

## HPO mission control

The per-trial artifacts of an HPO campaign rendered as a one-look
dashboard: all validation-loss curves with the winning trial highlighted,
search convergence with the starred best, run-summary card, best-trial
train/val detail, parameter sensitivity, and the tuned model's test
parity.

![HPO mission control dashboard](assets/gallery/mission_control.png)

## Multi-backend comparison

Random forest, PyTorch MLP, and an exact Gaussian process trained on
identical QLKNN splits through the one registry — the small-data regime
is where the GP earns its keep.

![Three backends compared](assets/gallery/trio.png)

## Deep-ensemble uncertainty

`pytorch.mlp_ensemble` mean ± 2σ with honest calibration: raw ensemble
spread is overconfident; σ-rescaling on a held-out split recovers
near-Gaussian coverage.

![Deep ensemble UQ](assets/gallery/ensemble.png)

## Uncertainty quantification

Gaussian-process surrogate with nested 68/95% credible bands — the band
widens where training data is absent; truth coverage is annotated.

![GP uncertainty bands](assets/gallery/uncertainty.png)

## Dataset characterization (pre-training)

Input distributions, target distribution, signal-to-noise, input–target
correlations, the strongest single relationship, and the PCA variance
spectrum with effective dimensionality.

![Dataset characterization panel](assets/gallery/characterization.png)

## Benchmark leaderboards

Score ± std across repeated runs against the published threshold, with
runtime on a separate log-scale panel. The interactive HTML dashboard
(`surge report`) adds spider charts, per-benchmark dataset previews,
citations, and sortable tables.

![Benchmark leaderboard](assets/gallery/leaderboard.png)
