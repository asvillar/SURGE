#!/usr/bin/env python3
"""End-to-end demo for docs/BUILD_YOUR_OWN_SURROGATE.md (Part I).

Creates a small synthetic CSV (or PKL/H5 when requested), then runs the bundled
workflow spec. This script is the fastest way to verify the tutorial path.

Examples
--------
    # Generate sample data + train (artifacts under runs/custom_dataset_tutorial/)
    python examples/custom_dataset_tutorial.py

    # Only write examples/data/tutorial_sample.csv
    python examples/custom_dataset_tutorial.py --prepare-only

    # Exercise pickle and HDF5 loaders documented in the tutorial
    python examples/custom_dataset_tutorial.py --format pkl
    python examples/custom_dataset_tutorial.py --format h5 --run-tag custom_dataset_h5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DATA_DIR = _REPO / "examples" / "data"
DEFAULT_CSV = DATA_DIR / "tutorial_sample.csv"
DEFAULT_SPEC = _REPO / "examples" / "configs" / "custom_dataset_tutorial.yaml"
DEFAULT_META = _REPO / "examples" / "configs" / "custom_dataset_meta.yaml"


def _make_dataframe(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inputs = {
        "input_a": rng.normal(size=n_rows),
        "input_b": rng.normal(size=n_rows),
        "input_c": rng.normal(size=n_rows),
    }
    outputs = {
        "output_x": inputs["input_a"] * 0.5 + rng.normal(scale=0.05, size=n_rows),
        "output_y": inputs["input_b"] * -0.25 + rng.normal(scale=0.03, size=n_rows),
    }
    return pd.DataFrame({**inputs, **outputs})


def write_dataset(path: Path, fmt: str, n_rows: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _make_dataframe(n_rows=n_rows)

    fmt = fmt.lower()
    if fmt == "csv":
        out = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
        df.to_csv(out, index=False)
    elif fmt in ("pkl", "pickle"):
        out = path if path.suffix.lower() in (".pkl", ".pickle") else path.with_suffix(".pkl")
        df.to_pickle(out)
    elif fmt in ("h5", "hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDF5 output. Install with: pip install h5py"
            ) from exc
        out = path if path.suffix.lower() in (".h5", ".hdf5") else path.with_suffix(".h5")
        with h5py.File(out, "w") as handle:
            ds = handle.create_dataset("data", data=df.to_numpy())
            ds.attrs["column_names"] = np.array(
                [str(c).encode("utf-8") for c in df.columns],
                dtype="S",
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    print(f"[tutorial] wrote {len(df):,} rows -> {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["csv", "pkl", "h5"],
        default="csv",
        help="Dataset file format to generate (default: csv).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=200,
        help="Number of synthetic rows (default: 200).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the dataset file; do not run the workflow.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag passed to run_workflow.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO,
        help="Parent directory for runs/ (default: repo root).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = write_dataset(DEFAULT_CSV, args.format, args.n_rows)
    if args.prepare_only:
        return 0

    cmd = [
        sys.executable,
        str(_REPO / "examples" / "run_workflow.py"),
        "--spec",
        str(DEFAULT_SPEC),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.run_tag:
        cmd.extend(["--run-tag", args.run_tag])

    # Point the bundled spec at the file we just wrote (may be .pkl/.h5).
    import yaml

    spec_payload = yaml.safe_load(DEFAULT_SPEC.read_text(encoding="utf-8"))
    spec_payload["dataset_path"] = str(dataset_path.relative_to(_REPO))
    spec_payload["dataset_format"] = args.format if args.format != "csv" else "auto"
    if args.run_tag:
        spec_payload["run_tag"] = args.run_tag

    tmp_spec = DATA_DIR / f"tutorial_spec_{args.format}.yaml"
    tmp_spec.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    cmd[cmd.index(str(DEFAULT_SPEC))] = str(tmp_spec)

    print(f"[tutorial] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=_REPO, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
