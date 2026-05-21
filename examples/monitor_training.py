"""
SURGE — Training Monitoring Demo
=================================

Demonstrates three monitoring modes for any PyTorch model:

1. Live tqdm progress bar (terminal)
2. Streaming JSONL log file (tail-able mid-training)
3. Post-training loss curve plots

Run:
    python examples/monitor_training.py

Requirements:
    pip install tqdm matplotlib  (both already in the standard SURGE env)
"""

from __future__ import annotations

import pathlib
import tempfile
import time

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

# ── ensure SURGE is importable from the repo root ──────────────────────────
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from surge.model.registry import MODEL_REGISTRY
from surge.model.plot_training import (
    compare_training_histories,
    plot_training_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

section("Loading dataset — UCI Concrete Strength (n=1030, d=8)")
from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=4353, as_frame=True, parser="auto")
df = data.frame
y = df.iloc[:, -1].values.astype(float)
X = df.iloc[:, :-1].values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")


# ---------------------------------------------------------------------------
# DEMO 1 — Live tqdm progress bar
# ---------------------------------------------------------------------------

section("Demo 1 — Live tqdm progress bar (verbose=True)")
print("  Watch the live loss bar below:\n")

model_verbose = MODEL_REGISTRY.create(
    "pytorch.residual_mlp",
    n_epochs=60,
    learning_rate=1e-3,
    hidden_layers=[256, 256],
    verbose=True,          # <── THIS enables the tqdm bar
)
t0 = time.perf_counter()
model_verbose.fit(X_train, y_train)
elapsed = time.perf_counter() - t0

y_pred = model_verbose.predict(X_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"\n  Done in {elapsed:.1f}s  |  test R² = {r2:.4f}")
print(f"  Epochs recorded in training_history: {len(model_verbose.training_history)}")
print(f"  Final train_loss: {model_verbose.training_history[-1]['train_loss']:.5f}")


# ---------------------------------------------------------------------------
# DEMO 2 — Log file (streamable)
# ---------------------------------------------------------------------------

section("Demo 2 — Streaming JSONL log file  (log_file=...)")

log_path = pathlib.Path("surge_training_log.jsonl")
print(f"  Writing epoch records to: {log_path.resolve()}")
print("  (You could open another terminal and run:  tail -f surge_training_log.jsonl)\n")

model_logged = MODEL_REGISTRY.create(
    "pytorch.ft_transformer",
    n_epochs=40,
    log_file=str(log_path),    # <── THIS writes a JSONL file per epoch
    verbose=True,
)
model_logged.fit(X_train, y_train)

import json
lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
epoch_lines = [l for l in lines if "epoch" in l]
print(f"\n  {len(epoch_lines)} epoch records written to {log_path}")
print(f"  First record : {epoch_lines[0]}")
print(f"  Last  record : {epoch_lines[-1]}")


# ---------------------------------------------------------------------------
# DEMO 3 — Load from log and plot mid-run history
# ---------------------------------------------------------------------------

section("Demo 3 — plot_training_history from log file")
print("  This simulates loading and plotting the log at any point during training.")

fig = plot_training_history(
    log_file=str(log_path),
    save_path="training_loss.png",
    title="FT-Transformer on Concrete Strength",
    show=False,              # set show=True for interactive display
    smoothing=0.6,
)
print(f"  Saved to training_loss.png  ({fig.get_figwidth():.0f}×{fig.get_figheight():.0f} in)")


# ---------------------------------------------------------------------------
# DEMO 4 — Compare multiple models on one plot
# ---------------------------------------------------------------------------

section("Demo 4 — compare_training_histories across models")

# Train a third model silently
model_kan = MODEL_REGISTRY.create(
    "pytorch.kan",
    n_epochs=40,
    hidden_layers=[64, 64],
    log_file="surge_kan_log.jsonl",
    verbose=True,
)
model_kan.fit(X_train, y_train)

fig2 = compare_training_histories(
    {
        "ResidualMLP (60 ep)": model_verbose._model,
        "FT-Transformer (40 ep)": model_logged._model,
        "KAN (40 ep)":  model_kan._model,
    },
    metric="train_loss",
    save_path="training_comparison.png",
    title="Training loss comparison — Concrete Strength",
    smoothing=0.5,
    show=False,
)
print(f"  Saved to training_comparison.png")


# ---------------------------------------------------------------------------
# DEMO 5 — adapter.plot_training_history() shortcut
# ---------------------------------------------------------------------------

section("Demo 5 — adapter.plot_training_history() shortcut")

fig3 = model_verbose.plot_training_history(
    save_path="residual_mlp_loss.png",
    show=False,
    title="ResidualMLP — training loss",
)
print(f"  model.plot_training_history() → residual_mlp_loss.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Summary")
print("""
  Created files:
    surge_training_log.jsonl   — JSONL stream (one line per epoch)
    surge_kan_log.jsonl        — same for KAN
    training_loss.png          — single-model loss plot
    training_comparison.png    — multi-model comparison
    residual_mlp_loss.png      — adapter shortcut

  Key API:
    MODEL_REGISTRY.create("pytorch.*", verbose=True)          → live tqdm bar
    MODEL_REGISTRY.create("pytorch.*", log_file="path.jsonl") → stream to file
    adapter.training_history                                   → list[dict]
    adapter.plot_training_history(save_path=...)               → matplotlib fig
    plot_training_history(log_file="path.jsonl")               → plot from file
    compare_training_histories({"A": m1, "B": m2})            → overlay plot

  CLI:
    python -m surge.benchmarks.run \\
        --benchmark tabular.concrete_strength \\
        --model pytorch.ft_transformer \\
        --verbose \\
        --train-log-file logs/training.jsonl
""")
