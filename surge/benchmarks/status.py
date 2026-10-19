"""Render the benchmark status ledger (docs/benchmarks/README.md).

Generated from ``metadata.yaml`` (status, status_note, shape, threshold, ...)
plus the newest ``result.json`` per benchmark under a results root.
Command: ``surge-benchmark --status-md [PATH]``.
"""
from __future__ import annotations

import json
from pathlib import Path

from surge.report.leaderboard import load_metadata

STATUS_ORDER = ("ready", "in_progress", "needs_rerun", "tutorial", "sanity")
STATUS_TITLES = {
    "ready": "Ready",
    "in_progress": "In progress",
    "needs_rerun": "Needs rerun",
    "tutorial": "Tutorial / plumbing",
    "sanity": "Sanity checks (not surrogate results)",
}
STATUS_DOC = {
    "ready": "real data, real runs on file, gate meaningful.",
    "in_progress": "the benchmark or its data pipeline is still being built.",
    "needs_rerun": "a result may exist but it is invalid, stale, synthetic or not produced by `surge bench`.",
    "tutorial": "small or self-referential dataset; checks plumbing, not scientific capability.",
    "sanity": "too easy as posed or measuring a proxy; not a surrogate result.",
}
_PRIMARY = (
    "test_r2_equilibrium_mean", "test_r2", "test_r2_mean", "test_accuracy",
    "test_rel_l2", "test_nrmse", "test_nrmse_mean")
_LOWER_IS_BETTER = ("test_rel_l2", "test_nrmse", "test_nrmse_mean", "test_rmse")


def _is_lower_better(metric_key: str) -> bool:
    return metric_key in _LOWER_IS_BETTER
DEFAULT_OUT = Path("docs/benchmarks/README.md")
_SEL_FIELDS = ("visible", "focus", "focus_note")


def is_visible(meta: dict) -> bool:
    return meta.get("visible", True) is not False


def is_focus(meta: dict) -> bool:
    return meta.get("focus", False) is True


def focus_lines(meta: dict[str, dict]) -> list[str]:
    """Markdown bullets for visible focus benchmarks (empty if none)."""
    out = []
    for k in sorted(meta):
        m = meta[k]
        if is_focus(m) and is_visible(m):
            note = m.get("focus_note")
            out.append(f"- `{k}`" + (f": {note}" if note else ""))
    return out


def apply_selection(selection_path: Path, metadata_path: Path | None = None) -> dict:
    """Set ``visible`` / ``focus`` / ``focus_note`` in metadata.yaml from a picker JSON.

    Input: ``{"benchmarks": [{"key", "show", "push", "note"}]}``. The YAML is
    edited textually (only those three keys inside matching entries), so
    comments, order and formatting elsewhere are preserved. Unknown keys warn.
    Returns ``{"visible": n, "focus": n, "unknown": [...]}``.
    """
    import re
    import sys

    path = Path(metadata_path) if metadata_path else (
        Path(__file__).resolve().parent / "metadata.yaml")
    sel = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    rows = {b["key"]: b for b in sel.get("benchmarks", []) if "key" in b}
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(r"- key:\s*", ln)]
    ends = starts[1:] + [len(lines)]
    known = {lines[i].split(":", 1)[1].strip().strip("'\""): (i, e)
             for i, e in zip(starts, ends)}
    unknown = [k for k in rows if k not in known]
    for k in unknown:
        print(f"warning: unknown benchmark key in selection: {k}", file=sys.stderr)
    # edit bottom-up so indices stay valid
    for k, (a, b) in sorted(known.items(), key=lambda kv: -kv[1][0]):
        if k not in rows:
            continue
        row = rows[k]
        body = [ln for ln in lines[a:b]
                if not re.match(r"  (visible|focus|focus_note):", ln)]
        while body and not body[-1].strip():
            body.pop()
        new = [f"  visible: {'true' if row.get('show', True) else 'false'}",
               f"  focus: {'true' if row.get('push', False) else 'false'}"]
        note = (row.get("note") or "").strip()
        if note:
            new.append(f"  focus_note: {json.dumps(note, ensure_ascii=False)}")
        lines[a:b] = body + new
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = load_metadata(path)
    return {"visible": sum(is_visible(m) for m in meta.values()),
            "focus": sum(is_focus(m) for m in meta.values()),
            "unknown": unknown}


def _ts(d: dict, path: Path) -> str:
    return str(d.get("timestamp") or path.parent.name)


SWEEP_WINDOW_HOURS = 3.0


def _parse_ts(ts: str):
    """Parse the result timestamp (ISO or ``YYYY-MM-DDTHH-MM-SSZ`` dir names)."""
    from datetime import datetime

    core = ts[:19].replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H-%M-%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(core[: len(fmt) + 1].strip(), fmt)
        except ValueError:
            continue
    return None


def _newest_results(reports_dir: Path) -> dict[str, dict]:
    """benchmark_key -> best model of the NEWEST SWEEP.

    A sweep is the newest result plus every result within
    ``SWEEP_WINDOW_HOURS`` before it (``surge bench -m all`` writes one
    ``result.json`` per model, seconds apart). Within that sweep the best
    score wins; runs older than the window are ignored even if better, so
    a benchmark whose definition changed (e.g. a re-scope of its targets)
    is reported by its current definition only.
    """
    from datetime import timedelta

    runs: dict[str, list[dict]] = {}
    for rj in sorted(reports_dir.glob("*/*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        key = d.get("benchmark_key")
        if not key:
            continue
        d["_ts"] = _ts(d, rj)
        d["_dt"] = _parse_ts(d["_ts"])
        runs.setdefault(key, []).append(d)

    def score(r):
        m = r.get("metrics") or {}
        for k in _PRIMARY:
            if isinstance(m.get(k), (int, float)):
                v = float(m[k])
                return -v if _is_lower_better(k) else v
        return float("-inf")

    out: dict[str, dict] = {}
    for key, rs in runs.items():
        newest = max(rs, key=lambda r: (r["_dt"] or _parse_ts("1970-01-01"), r["_ts"]))
        if newest["_dt"] is None:
            cand = [r for r in rs if r["_ts"][:10] == newest["_ts"][:10]]
        else:
            lo = newest["_dt"] - timedelta(hours=SWEEP_WINDOW_HOURS)
            cand = [r for r in rs if r["_dt"] is not None and lo <= r["_dt"] <= newest["_dt"]]
        out[key] = max(cand, key=lambda r: (score(r), r["_ts"]))
    return out


def _fmt_result(r: dict | None) -> str:
    if not r:
        return "no result on file"
    m = r.get("metrics") or {}
    for k in _PRIMARY:
        if isinstance(m.get(k), (int, float)):
            val = f"{k.removeprefix('test_')} {float(m[k]):.3f}"
            break
    else:
        val = "n/a"
    syn = " (synthetic fixture)" if (r.get("extra") or {}).get("synthetic") else ""
    ok = "" if r.get("passed") is None else (" ✓" if r["passed"] else " ✗")
    return f"{val}{ok}{syn} ({r.get('model_key')}, {r['_ts'][:10]})"


def _source(meta: dict, r: dict | None) -> str:
    bm = ((r or {}).get("extra") or {}).get("benchmark_metadata") or {}
    src = bm.get("dataset_source") or meta.get("citation") or "n/a"
    lic = bm.get("license")
    return f"{src}; {lic}" if lic else str(src)


def _cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_status_md(reports_dir: Path, metadata_path: Path | None = None) -> str:
    meta = load_metadata(metadata_path)
    results = _newest_results(Path(reports_dir))
    lines = [
        "# SURGE benchmark ledger",
        "",
        "Per-benchmark status, generated from `surge/benchmarks/metadata.yaml` and the newest "
        "`result.json` per benchmark. Do not edit by hand; regenerate with:",
        "",
        "```bash",
        "surge-benchmark --status-md docs/benchmarks/README.md",
        "```",
        "",
        "Statuses:",
        "",
    ]
    lines += [f"- **{STATUS_TITLES[s]}** (`{s}`): {STATUS_DOC[s]}" for s in STATUS_ORDER]
    lines += ["", "The on-file result is the best model of the newest sweep (runs within 3 h of the newest one); older runs are ignored even if better; "
              "a green check/cross reflects the benchmark gate at that run."]
    focus = focus_lines(meta)
    if focus:
        lines += ["", "## Focus", ""] + focus
    for st in STATUS_ORDER:
        keys = sorted(k for k, m in meta.items() if (m.get("status") or "tutorial") == st)
        lines += ["", f"## {STATUS_TITLES[st]}", ""]
        if not keys:
            lines.append("_None._")
            continue
        lines += ["| benchmark | task shape | data source + license | best on-file result (model, date) | gate | note | visible | focus |",
                  "|---|---|---|---|---|---|---|---|"]
        for k in keys:
            m, r = meta[k], results.get(k)
            row = [f"`{k}`", m.get("shape", ""), _source(m, r), _fmt_result(r),
                   m.get("threshold", ""), m.get("status_note", ""),
                   "✓" if is_visible(m) else "—", "✓" if is_focus(m) else "—"]
            lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def write_status_md(out: Path | None, reports_dir: Path,
                    metadata_path: Path | None = None) -> Path:
    out = Path(out) if out else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_status_md(reports_dir, metadata_path), encoding="utf-8")
    return out
