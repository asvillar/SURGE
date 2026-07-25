# Results gallery

What SURGE produces, by problem type. Every figure below is generated
from machine-readable run artifacts by the SURGE visual system — nothing
is hand-drawn or hand-entered. Regenerate all of them (light **and**
dark variants, deterministic PNG/SVG/PDF) with:

```bash
python examples/viz_theme_gallery.py
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
generalisation gap, starred best epoch, and a power-law convergence
fit tracking the saturation.

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
