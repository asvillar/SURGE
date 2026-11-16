"""SURGE datagenerator utilities.

Provides a simple DataGenerator class that can create parameter samples
using random uniform sampling or Latin Hypercube Sampling (LHS) and
write/replace values into input files. Variable replacement is done by a
built-in pure-Python routine by default. Optionally, an external helper script
(replace_var.sh) can be used by passing ``bin_dir`` or setting the
``SURGE_DATAGEN_BIN`` environment variable to the folder that holds it. Point
it only at a folder you trust; the script found there is executed.

API (high level):
  generator = DataGenerator(bin_dir=...)
  generator.generate(
      inpnames=[...],
      inputfilename='input.dat',
      ranges=[[min,max],...],
      integer_mask=[True,False,...],
      n_samples=10,
      method='lhs',
      out_dir='/path/to/param_cases',  # required
      base_case_dir='/path/to/case_default'
  )

The class will create case folders under out_dir named case_1, case_2, ...
and copy the base_case contents into each one before modifying the input
file with the built-in replacement (or the external script, if configured).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence, Optional
import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import qmc


class DataGenerator:
    """Generate parameter cases and modify their input files.

    By default variables are replaced with the built-in Python routine, which
    needs no external files. An external script is used only when ``bin_dir``
    or the ``SURGE_DATAGEN_BIN`` environment variable is set and
    ``use_python_replacement`` is not True. Point ``SURGE_DATAGEN_BIN`` only at
    a folder you trust; the script found there is executed.

    Args:
        bin_dir: optional directory containing an external replace script
        replace_script: script name to call for replacing variables (default: replace_var.sh)
        dry_run: print the replacement commands instead of running them
        use_python_replacement: force the built-in replacement even if a
            script directory is configured
    """

    def __init__(
        self,
        bin_dir: Optional[str] = None,
        replace_script: str = "replace_var.sh",
        dry_run: bool = False,
        use_python_replacement: bool = False,
    ):
        # Folder holding an optional external replace script (argument, else
        # SURGE_DATAGEN_BIN). None means the built-in replacement is used.
        self.bin_dir = bin_dir or os.environ.get("SURGE_DATAGEN_BIN") or None
        self.replace_script = replace_script
        # If dry_run is True, the generator will not execute external scripts and
        # will only print the replacement commands. Useful for testing on systems
        # that don't have the helper scripts available.
        self.dry_run = bool(dry_run)
        # If True, perform in-Python replacement (no external sed/perl) and
        # avoid creating editor backup files like file.py~.
        # Without an external script directory the built-in replacement is used.
        self.use_python_replacement = bool(use_python_replacement) or self.bin_dir is None
        # Tracks the most recent batch directory created by generate()
        self.last_batch_path = None

    def _call_replace(self, varname: str, value, filepath: str):
        script_path = os.path.join(self.bin_dir or "", self.replace_script)
        # Convert numpy types to python native
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()

        cmd = [script_path, str(varname), str(value), str(filepath)]

        if self.dry_run:
            # Just print the command and return
            print("[dry-run] would run:", " ".join(cmd))
            return

        if self.use_python_replacement:
            return self._replace_in_file(varname, value, filepath)

        # The external replace script is a bash script (replace_var.sh) and is
        # not portable to Windows. Steer callers to the pure-Python path instead
        # of failing with an opaque OS error.
        if os.name == "nt":
            raise RuntimeError(
                "The bash replace script is not supported on Windows. "
                "Pass use_python_replacement=True to DataGenerator(...) to use "
                "the built-in pure-Python variable replacement instead."
            )

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Replace script not found: {script_path}")

        # Use subprocess.run to execute the shell script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to run replace script: {e}")

    def _replace_in_file(self, varname: str, value, filepath: str):
        """Replace all occurrences of a variable assignment in a file.

        Matching rule: replace lines where the variable appears on the LHS of
        an assignment and the variable name is not part of a longer token. A
        valid boundary character before/after the name is start/end of string
        or one of: space, dot (.), or equals sign (=).
        """
        import re
        import tempfile

        # Convert numpy types and ensure integers are written without decimal points
        if isinstance(value, (np.floating, np.integer)):
            val = value.item()
            # If value is an integer (or very close to one), write as integer
            if isinstance(val, (int, np.integer)) or (isinstance(val, (float, np.floating)) and abs(val - round(val)) < 1e-10):
                value_to_write = str(int(round(val)))
            else:
                value_to_write = repr(val)
        elif isinstance(value, (int, float)):
            # Python native types - check if integer
            if isinstance(value, int) or (isinstance(value, float) and abs(value - round(value)) < 1e-10):
                value_to_write = str(int(round(value)))
            else:
                value_to_write = repr(value)
        else:
            value_to_write = repr(value)

        # Pattern to find the variable as a standalone-ish token in a LHS
        # part. We will split lines at the first '=' and inspect the LHS.
        # The pattern asserts that the char before (if any) is start, space,
        # dot or '=' and similarly for char after.
        var_pat = re.compile(rf"(?:(?<=^)|(?<=[\s.=])){re.escape(varname)}(?:(?=$)|(?=[\s.=]))")

        replaced_count = 0
        dirpath = os.path.dirname(filepath) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_replace_", dir=dirpath)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf, open(filepath, "r", encoding="utf-8") as srcf:
                for line in srcf:
                    if "=" in line:
                        lhs, rhs = line.split("=", 1)
                        if var_pat.search(lhs):
                            # reconstruct line with new RHS
                            tmpf.write(f"{lhs.strip()} = {value_to_write}\n")
                            replaced_count += 1
                            continue
                    tmpf.write(line)

            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

        if replaced_count == 0:
            raise RuntimeError(f"Variable '{varname}' not found (per matching rules) in {filepath}")

    def _make_samples(self, dim: int, n: int, ranges, integer_mask, method: str, seed: Optional[int] = None, log_space: Optional[Sequence[bool]] = None):
        """Create a (n x dim) array of samples using the requested method.

        Centralized helper so sampling logic isn't duplicated.
        
        Args:
            dim: Number of dimensions (parameters)
            n: Number of samples to generate
            ranges: List of [min, max] pairs for each parameter
            integer_mask: List of booleans indicating which parameters are integers
            method: Sampling method ('lhs' or 'random')
            seed: Random seed
            log_space: Optional list of booleans indicating which parameters should be
                     sampled in log space. For log-space sampling, samples are drawn
                     uniformly in log space then exponentiated. For integer parameters,
                     rounding happens after exponentiation.
        """
        if log_space is None:
            log_space = [False] * dim
        
        if len(log_space) != dim:
            raise ValueError(f"log_space length ({len(log_space)}) must match dim ({dim})")
        
        rng = np.random.default_rng(seed)
        if method.lower() == "lhs":
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            s = sampler.random(n=n)
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                if log_space[i]:
                    # Sample in log space: uniform in [log(lo), log(hi)]
                    log_lo, log_hi = np.log(lo), np.log(hi)
                    s[:, i] = s[:, i] * (log_hi - log_lo) + log_lo
                    # Exponentiate to get back to original space
                    s[:, i] = np.exp(s[:, i])
                else:
                    # Sample in linear space
                    s[:, i] = s[:, i] * (hi - lo) + lo
        elif method.lower() in ("random", "uniform"):
            s = np.empty((n, dim), dtype=float)
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                if log_space[i]:
                    # Sample in log space: uniform in [log(lo), log(hi)]
                    log_lo, log_hi = np.log(lo), np.log(hi)
                    log_samples = rng.uniform(log_lo, log_hi, size=n)
                    # Exponentiate to get back to original space
                    s[:, i] = np.exp(log_samples)
                else:
                    # Sample in linear space
                    s[:, i] = rng.uniform(lo, hi, size=n)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Apply integer rounding after all sampling (including log-space transformation)
        for i, is_int in enumerate(integer_mask):
            if is_int:
                s[:, i] = np.round(s[:, i]).astype(int)
        return s

    def _make_case(self, base_case_dir: str, out_dir: str, case_idx: int):
        """Create a case directory and copy only the template file to it.

        The template file name is provided by the caller (see generate).
        """
        wdir = os.path.join(out_dir, f"case{case_idx}")
        os.makedirs(wdir, exist_ok=False)

        if not os.path.isdir(base_case_dir):
            raise FileNotFoundError(f"Base case dir not found: {base_case_dir}")

        # Copy only the template file that will be modified (input file)
        # The caller passes the template filename and will place the file in
        # the correct place after creating the case directory.
        return wdir

    def generate(
        self,
        inpnames: Sequence[str],
        inputfilename: str,
        ranges: Sequence[Sequence[float]],
        integer_mask: Optional[Sequence[bool]] = None,
        n_samples: int = 10,
        method: str = "random",
        out_dir: Optional[str] = None,
        base_case_dir: Optional[str] = None,
        seed: Optional[int] = None,
        batch_root: Optional[str] = None,
        confirm_dirs: bool = False,
        save_plots: bool = False,
    ) -> List[dict]:
        """Generate parameter files.

        Samples parameter values, copies ``base_case_dir`` into one folder per
        sample and replaces the named variables in ``inputfilename``.

        Args:
            out_dir: required output folder; created if missing. Also the
                default location of the base case (``<out_dir>/case_default``).
            batch_root: folder that receives the numbered ``batchN`` folders.
                Defaults to ``<out_dir>/batches``.
            seed: seed for both the ``random`` and ``lhs`` methods; the same
                seed gives the same samples.
            confirm_dirs: ask on stdin before creating the batch folders.

        Returns a list of dicts with generated parameter values and case directory.
        """
        if integer_mask is None:
            integer_mask = [False] * len(inpnames)

        if len(inpnames) != len(ranges) or len(inpnames) != len(integer_mask):
            raise ValueError("Lengths of inpnames, ranges and integer_mask must match")

        if not out_dir:
            raise ValueError("out_dir is required: pass the folder that should receive the generated cases")
        base_case_dir = base_case_dir or os.path.join(out_dir, f"case_default")
        os.makedirs(out_dir, exist_ok=True)

        dim = len(inpnames)

        rng = np.random.default_rng(seed)

        if method.lower() == "lhs":
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            sample = sampler.random(n=n_samples)
            # scale to ranges using qmc.scale for each column
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                # sample[:, i] is in [0,1); scale manually to [lo, hi]
                sample[:, i] = sample[:, i] * (hi - lo) + lo
        elif method.lower() in ("random", "uniform"):
            sample = np.empty((n_samples, dim), dtype=float)
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                sample[:, i] = rng.uniform(lo, hi, size=n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Batch management: prompt for directory if not provided and create batch folder
        results = []
        samples = np.zeros((n_samples, dim), dtype=float)

        # Determine batch root (non-interactive optional)
        if batch_root is None:
            batch_root = os.path.join(out_dir, "batches")

        # Create a new numbered batch directory under batch_root
        if not self.dry_run:
            if confirm_dirs:
                try:
                    perm = input(f"Allow creating batch directory and subdirectories at {batch_root}? [Y/n]: ").strip()
                except Exception:
                    perm = "y"
                if perm and perm.lower() not in ("y", "yes", ""):
                    raise PermissionError("User declined creation of batch directories")

        os.makedirs(batch_root, exist_ok=True)
        existing = [d for d in os.listdir(batch_root) if os.path.isdir(os.path.join(batch_root, d)) and d.startswith("batch")]
        idxs = [int(d.replace("batch", "")) for d in existing if d.replace("batch", "").isdigit()]
        next_idx = max(idxs) + 1 if idxs else 1
        batch_path = os.path.join(batch_root, f"batch{next_idx}")
        if self.dry_run:
            print(f"[dry-run] would create batch directory: {batch_path}")
        else:
            os.makedirs(batch_path, exist_ok=False)
        # record last created batch dir
        self.last_batch_path = batch_path

        for idx in range(sample.shape[0]):
            caseidx = idx + 1
            # Create the case directory inside the batch_path (case names: case1, case2,...)
            target_parent = batch_path or out_dir
            if self.dry_run:
                wdir = os.path.join(target_parent, f"case{caseidx}")
                print(f"[dry-run] would create case directory: {wdir}")
            else:
                wdir = self._make_case(base_case_dir, target_parent, caseidx)

            # Copy the template file from base_case_dir into the new case dir
            src_template = os.path.join(base_case_dir, inputfilename)
            dst_template = os.path.join(wdir, inputfilename)
            if os.path.exists(src_template):
                if self.dry_run:
                    print(f"[dry-run] would copy template {src_template} -> {dst_template}")
                else:
                    shutil.copy2(src_template, dst_template)
            else:
                # If template not found, create an empty file so replacement fails clearly
                if self.dry_run:
                    print(f"[dry-run] template not found in base case: would create empty {dst_template}")
                else:
                    open(dst_template, "w", encoding="utf-8").close()
            target_file = os.path.join(wdir, inputfilename)

            assigned = {}
            for j, name in enumerate(inpnames):
                val = sample[idx, j]
                if integer_mask[j]:
                    val = int(round(val))
                samples[idx, j] = val
                # call replace to change value in target file
                try:
                    self._call_replace(name, val, target_file)
                except Exception as e:
                    # continue but record error
                    assigned[name] = {"value": val, "error": str(e)}
                else:
                    assigned[name] = {"value": val}

            results.append({"case": f"case{caseidx}", "dir": wdir, "params": assigned})

        # If batch_path provided, save samples and README
        if batch_path is not None and not self.dry_run:
            meta = {
                "inpnames": list(inpnames),
                "ranges": list([list(r) for r in ranges]),
                "integer_mask": list(integer_mask),
                "n_samples": n_samples,
                "method": method,
                "seed": seed,
                "base_case_dir": base_case_dir,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            }
            npz_path = os.path.join(batch_path, "samples.npz")
            np.savez_compressed(npz_path, X=samples, names=np.array(inpnames), integer_mask=np.array(integer_mask), meta=json.dumps(meta))

            # Write README with Markdown tables
            readme_path = os.path.join(batch_path, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                bname = os.path.basename(batch_path)
                f.write(f"# Batch {bname}\n\n")
                f.write(f"Created: {meta['created_at']}\n\n")

                # Parameters table
                f.write("## Parameters\n\n")
                f.write("| Variable | Range Min | Range Max | Integer |\n")
                f.write("|:--|--:|--:|:--:|\n")
                def _fmt(v, as_int=False):
                    try:
                        if as_int:
                            return str(int(round(float(v))))
                        # scientific notation with 4 significant figures
                        return f"{float(v):.3e}"
                    except Exception:
                        return str(v)

                for name, rng_, is_int in zip(inpnames, ranges, integer_mask):
                    f.write(
                        f"| {name} | {_fmt(rng_[0], as_int=is_int)} | {_fmt(rng_[1], as_int=is_int)} | {'Yes' if is_int else 'No'} |\n"
                    )

                # Sampling config table
                f.write("\n## Sampling\n\n")
                f.write("| Key | Value |\n")
                f.write("|:--|:--|\n")
                f.write(f"| method | {method} |\n")
                f.write(f"| n_samples | {n_samples} |\n")
                f.write(f"| seed | {seed} |\n")
                f.write(f"| base_case_dir | {base_case_dir} |\n")

                # Samples preview table (first 10)
                f.write("\n## Samples (first 10)\n\n")
                # header
                f.write("| " + " | ".join(inpnames) + " |\n")
                f.write("|" + "|".join([":-:" for _ in inpnames]) + "|\n")
                preview_rows = min(10, samples.shape[0])
                for i in range(preview_rows):
                    row = [
                        _fmt(samples[i, j], as_int=bool(integer_mask[j]))
                        for j in range(samples.shape[1])
                    ]
                    f.write("| " + " | ".join(row) + " |\n")

                # Summary statistics table
                f.write("\n## Summary stats (all samples)\n\n")
                f.write("| Variable | Min | Mean | Max |\n")
                f.write("|:--|--:|--:|--:|\n")
                for j, name in enumerate(inpnames):
                    col = samples[:, j]
                    f.write(
                        f"| {name} | {_fmt(col.min(), as_int=bool(integer_mask[j]))} | {_fmt(col.mean(), as_int=False)} | {_fmt(col.max(), as_int=bool(integer_mask[j]))} |\n"
                    )

            # Optional: save sampling plots
            if save_plots:
                try:
                    save_sampling_plots(samples, list(inpnames), batch_path, title=f"{os.path.basename(batch_path)}")
                except Exception as e:
                    print(f"[warn] failed to save plots: {e}")

        return results


# -------- Visualization helpers --------
def save_sampling_plots(X: np.ndarray, names: Sequence[str], out_dir: str, title: Optional[str] = None):
    """Save simple visualizations of the sampled dataset.

    Produces:
      - sampling_hist.png: 1D histograms for each variable
      - sampling_scatter.png: pairwise scatter matrix (up to 4 variables to avoid clutter)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    n, d = X.shape
    title = title or "Sampling"

    # 1) Histograms
    cols = min(4, d)
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 3.2 * rows), squeeze=False)
    for j in range(d):
        ax = axes[j // cols][j % cols]
        ax.hist(X[:, j], bins=12, color="#4472c4", alpha=0.85)
        ax.set_title(names[j])
    # hide empty subplots
    for k in range(d, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle(f"{title} – Histograms (n={n})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sampling_hist.png"), dpi=150)
    plt.close(fig)

    # 2) Pairwise scatter (first up to 4 vars)
    m = min(4, d)
    if m >= 2:
        fig2, axes2 = plt.subplots(m, m, figsize=(3.2 * m + 2, 3.2 * m), squeeze=False)
        for i in range(m):
            for j in range(m):
                ax = axes2[i][j]
                if i == j:
                    ax.hist(X[:, j], bins=12, color="#70ad47", alpha=0.85)
                else:
                    ax.scatter(X[:, j], X[:, i], s=14, alpha=0.7, color="#2f5597")
                if i == m - 1:
                    ax.set_xlabel(names[j])
                if j == 0:
                    ax.set_ylabel(names[i])
        fig2.suptitle(f"{title} – Pairwise (first {m} vars)", y=0.98)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "sampling_scatter.png"), dpi=150)
        plt.close(fig2)
