#!/usr/bin/env python3
"""One-shot extraction of benchmark metadata into surge/benchmarks/metadata.yaml.

Sources:
  1. The retired editor canvas leaderboard (surge/viz/leaderboard.canvas.tsx,
     read from git history) — structured fields: name, citation, url, shape,
     n, capability, tier, primary metric, threshold, IO feature descriptions.
  2. docs/BENCHMARK_VERIFICATION_BRIEF.md — verified one-line problem
     statements per benchmark key.

The YAML becomes the single source of truth for leaderboard reports; the
snapshot result numbers embedded in the canvas are NOT extracted (results
are regenerated from benchmark_reports/**/result.json).

Usage (from repo root):
    python scripts/extract_benchmark_metadata.py \
        [--ref model-bench] [--out surge/benchmarks/metadata.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_canvas(ref: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:surge/viz/leaderboard.canvas.tsx"],
        capture_output=True, text=True, check=True, cwd=REPO).stdout


def slice_data_block(tsx: str) -> str:
    start = tsx.index("const DATA: Benchmark[] = [") + len("const DATA: Benchmark[] = ")
    depth = 0
    for i in range(start, len(tsx)):
        c = tsx[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return tsx[start:i + 1]
    raise ValueError("unterminated DATA array")


def js_literal_to_json(src: str) -> str:
    out, in_str, i = [], False, 0
    while i < len(src):
        c = src[i]
        if in_str:
            out.append(c)
            if c == "\\":
                out.append(src[i + 1]); i += 2; continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        if c == "/" and i + 1 < len(src) and src[i + 1] == "/":
            i = src.index("\n", i)
            continue
        out.append(c); i += 1
    text = "".join(out)
    # JS string concatenation across lines: "..." + "..." -> single literal
    text = re.sub(r'"\s*\+\s*"', "", text)
    text = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing commas
    return text


def parse_brief_problems(brief_path: Path) -> dict:
    """Map registry key -> verified 'Problem' sentence from the brief."""
    problems: dict[str, str] = {}
    if not brief_path.exists():
        return problems
    text = brief_path.read_text()
    for match in re.finditer(
            r"###\s+`([\w.]+)`[^\n]*\n(.*?)(?=\n###\s|\n## |\Z)",
            text, re.DOTALL):
        key, body = match.group(1), match.group(2)
        prob = re.search(r"\|\s*Problem\s*\|\s*(.+?)\s*\|", body)
        if prob:
            problems[key] = re.sub(r"\*\*", "", prob.group(1)).strip()
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="model-bench")
    ap.add_argument("--out", default=str(REPO / "surge" / "benchmarks" / "metadata.yaml"))
    args = ap.parse_args()

    data = json.loads(js_literal_to_json(slice_data_block(read_canvas(args.ref))))
    problems = parse_brief_problems(REPO / "docs" / "BENCHMARK_VERIFICATION_BRIEF.md")

    # The brief sometimes uses a different key than the canvas; map aliases.
    key_aliases = {
        "classification.covertype": "tabular.covertype",
        "classification.plasma_stability": "tabular.plasma_stability",
    }

    entries = []
    for b in data:
        key = b["key"]
        entry = {
            "key": key,
            "name": b.get("name"),
            "capability": b.get("capability"),
            "tier": b.get("tier"),
            "citation": b.get("citation"),
            "url": b.get("url"),
            "shape": b.get("shape"),
            "n": b.get("n"),
            "primary_metric": b.get("primaryMetric"),
            "threshold": b.get("threshold"),
        }
        if b.get("thresholdNote"):
            entry["threshold_note"] = b["thresholdNote"]
        desc = problems.get(key) or problems.get(key_aliases.get(key, ""))
        if desc:
            entry["description"] = desc
        if b.get("ioNote"):
            entry["io_note"] = b["ioNote"]
        if b.get("inputs"):
            entry["inputs"] = [
                {"name": f["name"], "desc": f["desc"]} for f in b["inputs"]]
        outs = b.get("outputs") or ([b["output"]] if b.get("output") else [])
        if outs:
            entry["outputs"] = [
                {"name": f["name"], "desc": f["desc"]} for f in outs]
        entries.append(entry)

    import yaml
    header = (
        "# SURGE benchmark metadata — single source of truth for leaderboard\n"
        "# reports (names, citations, tiers, thresholds, IO descriptions).\n"
        "# Extracted 2026-07 from the retired canvas leaderboard and the\n"
        "# benchmark verification brief by scripts/extract_benchmark_metadata.py.\n"
        "# Edit THIS file to change benchmark descriptions; result numbers\n"
        "# always come from benchmark_reports/**/result.json at render time.\n")
    out_path = Path(args.out)
    out_path.write_text(header + yaml.safe_dump(
        {"benchmarks": entries}, sort_keys=False, allow_unicode=True, width=88))
    print(f"wrote {out_path} ({len(entries)} benchmarks, "
          f"{sum(1 for e in entries if 'description' in e)} with descriptions)")


if __name__ == "__main__":
    main()
