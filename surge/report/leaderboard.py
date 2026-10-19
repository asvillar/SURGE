"""SURGE benchmark leaderboard report — one self-contained HTML file.

Everything on the page is generated from machine-readable sources:

* results        benchmark_reports/**/result.json   (aggregated mean ± std)
* descriptions   surge/benchmarks/metadata.yaml     (names, citations, tiers,
                 thresholds, IO documentation — extracted from the retired
                 canvas leaderboard and the verification brief)

The page needs no server and no network: spider charts are inline SVG
rendered with the SURGE matplotlib theme, styling is inline CSS on the
dark palette, and the only JavaScript is a ~20-line table sorter.

Usage (from repo root):
    python -m surge.report.leaderboard \
        [--reports benchmark_reports] [--out surge_leaderboard.html]
"""
from __future__ import annotations

import argparse
import html
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from surge.report.dataset_previews import preview_svg
from surge.viz.theme import PALETTES, fmt_metric, surge_theme

# Metrics where lower is better (everything else: higher is better).
_LOWER_BETTER = ("rmse", "nrmse", "rel_l2", "mae", "loss")

# Display order and labels for metric columns, first match wins per benchmark.
_PRIMARY_CANDIDATES = (
    "test_r2", "test_r2_mean", "test_accuracy", "test_rel_l2", "test_nrmse",
    "test_nrmse_mean",
)


def _is_lower_better(metric: str) -> bool:
    return any(tag in metric for tag in _LOWER_BETTER)


# --------------------------------------------------------------------- data

def load_metadata(path: Path | None = None) -> dict[str, dict]:
    import yaml
    path = path or Path(__file__).resolve().parents[1] / "benchmarks" / "metadata.yaml"
    doc = yaml.safe_load(path.read_text())
    return {b["key"]: b for b in doc.get("benchmarks", [])}


def _aggregate(runs: list[dict]) -> dict:
    """Mean/std aggregate over raw result dicts (each carries ``_path``)."""
    metrics: dict[str, list] = defaultdict(list)
    for r in runs:
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                metrics[k].append(float(v))
    agg = {}
    for k, v in metrics.items():
        mean = statistics.fmean(v)
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        if std < 1e-9 * max(1.0, abs(mean)):  # float noise, not spread
            std = 0.0
        agg[k] = {"mean": mean, "std": std, "n": len(v)}
    passed = [bool(r.get("passed")) for r in runs if r.get("passed") is not None]
    return {
        "metrics": agg,
        "n_runs": len(runs),
        "passed": (sum(passed) / len(passed) >= 0.5) if passed else None,
        "last": max(r.get("timestamp", "") for r in runs),
        "synthetic": bool(runs) and all(
            bool((r.get("extra") or {}).get("synthetic")) for r in runs),
        "runs": runs,
    }


def load_results(reports_dir: Path) -> dict[str, dict[str, dict]]:
    """benchmark_key -> model_key -> aggregate over every result.json.

    Each aggregate keeps its raw ``runs`` (with ``_path`` = result.json path
    relative to the reports directory's parent) so the newest sweep can be
    re-aggregated by :func:`newest_sweep`.
    """
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rj in sorted(reports_dir.glob("*/*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        bench, model = d.get("benchmark_key"), d.get("model_key")
        if not bench or not model:
            continue
        d["_path"] = str(rj.relative_to(reports_dir.parent))
        d["_ts"] = str(d.get("timestamp") or rj.parent.name)
        raw[bench][model].append(d)
    return {bench: {m: _aggregate(runs) for m, runs in models.items()}
            for bench, models in raw.items()}


def newest_sweep(bench_results: dict[str, dict]) -> dict[str, dict]:
    """Models of the newest sweep only: every run within
    ``SWEEP_WINDOW_HOURS`` of the newest run of the benchmark (same rule as
    ``surge.benchmarks.status._newest_results``)."""
    from datetime import timedelta

    from surge.benchmarks.status import SWEEP_WINDOW_HOURS, _parse_ts

    allruns = [r for m in bench_results.values() for r in m["runs"]]
    for r in allruns:
        r.setdefault("_dt", _parse_ts(r.get("_ts", "")))
    dated = [r for r in allruns if r["_dt"] is not None]
    if not dated:
        return bench_results
    newest = max(r["_dt"] for r in dated)
    lo = newest - timedelta(hours=SWEEP_WINDOW_HOURS)
    out = {}
    for model, agg in bench_results.items():
        keep = [r for r in agg["runs"] if r["_dt"] is not None and lo <= r["_dt"] <= newest]
        if keep:
            out[model] = _aggregate(keep)
    return out


def primary_metric_key(bench_results: dict[str, dict]) -> str | None:
    present: set = set()
    for model in bench_results.values():
        present.update(model["metrics"].keys())
    for cand in _PRIMARY_CANDIDATES:
        if cand in present:
            return cand
    scored = [k for k in present if k.startswith("test_")]
    return min(scored) if scored else None


def rank_models(bench_results: dict[str, dict], metric: str) -> list[str]:
    def score(model: str) -> float:
        m = bench_results[model]["metrics"].get(metric)
        if m is None:
            return float("-inf")
        return -m["mean"] if _is_lower_better(metric) else m["mean"]
    return sorted((m for m in bench_results), key=score, reverse=True)


# ------------------------------------------------------------------- spider

def spider_svg(category: str, benches: list[str], results: dict[str, dict[str, dict]],
               meta: dict[str, dict], mode: str = "dark", max_models: int = 4) -> str:
    """Radar chart: axes = benchmarks in a category, series = top models.

    Values are normalised per axis to the best model (1.0 = best on that
    benchmark), inverting lower-is-better metrics, so shape = breadth of
    capability rather than raw metric scale.
    """
    import matplotlib.pyplot as plt

    axes_keys = [b for b in benches if results.get(b)]
    if len(axes_keys) < 3:
        return ""

    per_axis: dict[str, dict[str, float]] = {}
    for b in axes_keys:
        pm = primary_metric_key(results[b])
        if pm is None:
            continue
        vals = {m: r["metrics"][pm]["mean"]
                for m, r in results[b].items() if pm in r["metrics"]}
        if not vals:
            continue
        if _is_lower_better(pm):
            best = min(vals.values())
            per_axis[b] = {m: (best / v if v > 0 else 1.0) for m, v in vals.items()}
        else:
            best = max(vals.values())
            per_axis[b] = {m: (max(v, 0.0) / best if best > 0 else 0.0)
                           for m, v in vals.items()}
    axes_keys = [b for b in axes_keys if b in per_axis]
    if len(axes_keys) < 3:
        return ""

    coverage: dict[str, list[float]] = defaultdict(list)
    for b in axes_keys:
        for m, v in per_axis[b].items():
            coverage[m].append(v)
    candidates = [m for m, v in coverage.items() if len(v) >= max(2, len(axes_keys) // 2)]
    candidates.sort(key=lambda m: -statistics.fmean(coverage[m]))
    models = candidates[:max_models]
    if not models:
        return ""

    labels = [meta.get(b, {}).get("name", b).replace(" (", "\n(") for b in axes_keys]
    n = len(axes_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    linestyles = ["-", "-", "-", (0, (4, 2))]

    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(4.9, 4.4))
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor(p["surface"])
        ax.spines["polar"].set_color(p["grid"])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7, color=p["muted"])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=7.5, color=p["ink2"])
        ax.grid(color=p["grid"], linewidth=0.6)

        for i, model in enumerate(models):
            vals = [per_axis[b].get(model, np.nan) for b in axes_keys]
            vv = np.array(vals + vals[:1], dtype=float)
            color = p["series"][i]
            ax.plot(closed, vv, color=color, linewidth=1.8,
                    linestyle=linestyles[i], label=model)
            if i == 0:  # fill only the leader; stacked fills go muddy
                ax.fill(closed, np.nan_to_num(vv), color=color, alpha=0.12)
        ax.set_title(f"{category} — relative to best per benchmark",
                     fontsize=10, pad=22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
                  ncol=2, fontsize=8, frameon=False,
                  columnspacing=1.4, handlelength=1.6)

        buf = io.StringIO()
        fig.savefig(buf, format="svg", metadata={"Date": None},
                    bbox_inches="tight")
        plt.close(fig)
    svg = buf.getvalue()
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else ""


# --------------------------------------------------------------------- html

_CSS_TEMPLATE = """
.badge {{ font-size: 10.5px; padding: 1.5px 8px; border-radius: 99px; font-weight: 600;
  margin-left: 8px; border: 1px solid var(--line); color: var(--muted); }}
.badge.good {{ color: var(--good); }} .badge.warn {{ color: var(--warn); border-color: var(--warn); }}
.chip.syn {{ color: var(--warn); border: 1px solid var(--warn); }}
.receipts {{ margin-top: 10px; color: var(--muted); font-size: 11.5px; }}
.receipts a {{ color: var(--muted); }}
code.path {{ font-family: ui-monospace, Menlo, monospace; font-size: 10.5px; }}
.tilenote {{ color: var(--muted); font-size: 12px; margin: -4px 0 20px; }}
details.tail {{ margin-top: 40px; border-top: 1px solid var(--line); padding-top: 14px; }}
details.tail > summary {{ cursor: pointer; font-size: 16px; font-weight: 600; color: var(--ink2); }}
:root {{
  --page: {page}; --surface: {surface}; --card: {surface};
  --ink: {ink}; --ink2: {ink2}; --muted: {muted};
  --line: {grid}; --accent: {accent};
  --good: {good}; --bad: {critical}; --warn: {warning};
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background:
    radial-gradient(1100px 500px at 15% -10%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%),
    var(--page);
  color: var(--ink); font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 40px 28px 80px; }}
header h1 {{ font-size: 26px; letter-spacing: .4px; margin: 0; }}
header .sub {{ color: var(--ink2); font-size: 13px; margin-top: 4px; }}
.accentbar {{ height: 3px; width: 88px; border-radius: 2px; margin: 14px 0 26px;
  background: linear-gradient(90deg, var(--accent), transparent); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
  gap: 12px; margin-bottom: 34px; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; box-shadow: 0 0 0 1px transparent; }}
.tile b {{ display: block; font-size: 24px; font-weight: 650; }}
.tile span {{ color: var(--muted); font-size: 12px; }}
h2.cat {{ font-size: 17px; margin: 42px 0 4px; }}
.cat-sub {{ color: var(--muted); font-size: 12.5px; margin: 0 0 14px; }}
.catgrid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
.spider {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px; display: flex; justify-content: center; }}
.spider svg {{ max-width: 640px; height: auto; }}
.bench {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 20px; }}
.bench:hover {{ border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }}
.bench h3 {{ margin: 0; font-size: 15px; }}
.pills {{ display: inline-flex; gap: 6px; margin-left: 10px; vertical-align: 2px; }}
.pill {{ font-size: 10.5px; padding: 1.5px 8px; border-radius: 99px;
  border: 1px solid var(--line); color: var(--ink2); }}
.pill.tier {{ color: var(--accent); border-color: color-mix(in srgb, var(--accent) 55%, transparent); }}
.bench-top {{ display: flex; gap: 22px; align-items: flex-start;
  flex-wrap: wrap; }}
.bench-info {{ flex: 1 1 320px; }}
figure.preview {{ margin: 4px 0 8px; flex: 0 1 330px; }}
figure.preview svg {{ max-width: 330px; height: auto; border: 1px solid
  var(--line); border-radius: 8px; background: var(--surface); }}
figure.preview figcaption {{ color: var(--muted); font-size: 10.5px;
  text-align: center; margin-top: 2px; }}
.desc {{ color: var(--ink2); font-size: 13px; margin: 8px 0 4px; max-width: 72ch; }}
.meta {{ color: var(--muted); font-size: 12px; margin-bottom: 10px; }}
.meta a {{ color: var(--accent); text-decoration: none; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 5px 14px 5px 0;
  border-bottom: 1px solid var(--line); cursor: pointer; white-space: nowrap; }}
td {{ padding: 6px 14px 6px 0; border-bottom: 1px solid
  color-mix(in srgb, var(--line) 55%, transparent);
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
tr.best td {{ background: color-mix(in srgb, var(--accent) 9%, transparent); }}
td.model {{ font-weight: 550; }}
.chip {{ font-size: 10.5px; padding: 1.5px 8px; border-radius: 99px; font-weight: 600; }}
.chip.pass {{ color: var(--good); border: 1px solid color-mix(in srgb, var(--good) 55%, transparent); }}
.chip.fail {{ color: var(--bad);  border: 1px solid color-mix(in srgb, var(--bad) 55%, transparent); }}
.bar {{ display: inline-block; height: 5px; border-radius: 3px;
  background: var(--accent); vertical-align: middle; margin-right: 7px; }}
details.io {{ margin-top: 10px; font-size: 12.5px; color: var(--ink2); }}
details.io summary {{ cursor: pointer; color: var(--muted); }}
footer {{ margin-top: 48px; color: var(--muted); font-size: 12px; }}
"""

_SORT_JS = """
document.querySelectorAll("table.sortable th").forEach((th, i) => {
  th.addEventListener("click", () => {
    const tb = th.closest("table").tBodies[0];
    const dir = th.dataset.dir = th.dataset.dir === "a" ? "d" : "a";
    [...tb.rows].sort((r1, r2) => {
      const a = r1.cells[i].dataset.v ?? r1.cells[i].textContent;
      const b = r2.cells[i].dataset.v ?? r2.cells[i].textContent;
      const na = parseFloat(a), nb = parseFloat(b);
      const cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb : String(a).localeCompare(b);
      return dir === "a" ? cmp : -cmp;
    }).forEach(r => tb.appendChild(r));
  });
});
"""


def _metric_columns(bench_results: dict[str, dict]) -> list[str]:
    keys: set = set()
    for r in bench_results.values():
        keys.update(r["metrics"].keys())
    order = ["test_r2", "test_r2_mean", "test_accuracy", "test_f1", "test_f1_macro",
             "test_auroc", "test_rmse", "test_rmse_mean", "test_nrmse",
             "test_nrmse_mean", "test_rel_l2", "runtime_s", "peak_memory_mb"]
    return [k for k in order if k in keys]


_LABELS = {
    "test_r2": "R²", "test_r2_mean": "R̄²", "test_accuracy": "Acc",
    "test_f1": "F1", "test_f1_macro": "F1", "test_auroc": "AUROC",
    "test_rmse": "RMSE", "test_rmse_mean": "RMSE", "test_nrmse": "NRMSE",
    "test_nrmse_mean": "NRMSE", "test_rel_l2": "RelL2",
    "runtime_s": "Runtime", "peak_memory_mb": "Peak mem",
}


def _fmt_cell(metric: str, agg: dict | None) -> str:
    if agg is None:
        return "<td data-v=''>—</td>"
    kind = ("runtime" if metric == "runtime_s"
            else "rmse" if _is_lower_better(metric) or "memory" in metric
            else "r2")
    txt = fmt_metric(agg["mean"], kind)
    if metric == "peak_memory_mb":
        txt = f"{agg['mean']:.0f} MB"
    if agg["std"] > 0:
        std = (fmt_metric(agg["std"], "runtime") if metric == "runtime_s"
               else f"{agg['std']:.3g}")
        txt += f" <span style='color:var(--muted)'>±{std}</span>"
    return f"<td data-v='{agg['mean']:.6g}'>{txt}</td>"


_STATUS_COLOR = {"ready": "good", "tutorial": "muted", "in_progress": "warn",
                 "needs_rerun": "warn", "sanity": "muted"}
_TAIL = ("in_progress", "needs_rerun", "sanity")


def _status_of(info: dict) -> str:
    return info.get("status") or "ready"


def _table(bench_results: dict[str, dict], pm: str | None, cols: list[str]) -> str:
    ranked = rank_models(bench_results, pm) if pm else sorted(bench_results)
    best_pm = None
    if pm and ranked:
        top = bench_results[ranked[0]]["metrics"].get(pm)
        best_pm = top["mean"] if top else None
    rows = []
    for i, model in enumerate(ranked):
        r = bench_results[model]
        pm_agg = r["metrics"].get(pm) if pm else None
        if pm_agg is not None and best_pm:
            rel = max(0.0, min(1.0, (pm_agg["mean"] / best_pm)
                               if not _is_lower_better(pm)
                               else (best_pm / pm_agg["mean"] if pm_agg["mean"] else 0)))
            bar = f"<span class='bar' style='width:{6 + 54 * rel:.0f}px'></span>"
        else:
            bar = ""
        cells = [(f"<td class='model' data-v='{html.escape(model)}'>"
                  f"{bar}{html.escape(model)}</td>")]
        for c in cols:
            cells.append(_fmt_cell(c, r["metrics"].get(c)))
        if r.get("synthetic"):
            chip = "<span class='chip syn'>synthetic</span>"
        else:
            chip = ("<span class='chip pass'>PASS</span>" if r["passed"]
                    else "<span class='chip fail'>FAIL</span>" if r["passed"] is not None
                    else "—")
        cells.append(f"<td data-v='{int(bool(r['passed']))}'>{chip}</td>")
        cells.append(f"<td data-v='{r['n_runs']}'>{r['n_runs']}</td>")
        rows.append(f"<tr class='{'best' if i == 0 else ''}'>{''.join(cells)}</tr>")
    header = "".join(f"<th>{_LABELS.get(c, c)}</th>" for c in cols)
    return (f"<table class='sortable'><thead><tr><th>model</th>{header}"
            f"<th>gate</th><th>runs</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _best_result_path(bench_results: dict[str, dict], pm: str | None) -> str:
    if not bench_results:
        return ""
    ranked = rank_models(bench_results, pm) if pm else sorted(bench_results)
    runs = bench_results[ranked[0]].get("runs") or []
    if not runs:
        return ""

    def sc(r):
        v = (r.get("metrics") or {}).get(pm)
        if not isinstance(v, (int, float)):
            return float("-inf")
        return -v if _is_lower_better(pm) else v
    return max(runs, key=lambda r: (sc(r), r.get("_ts", ""))).get("_path", "")


def _bench_card(key: str, meta: dict[str, dict],
                bench_results: dict[str, dict], preview: str = "",
                history: dict[str, dict] | None = None) -> str:
    """``bench_results`` = headline (newest sweep); ``history`` = every run."""
    info = meta.get(key, {})
    status = _status_of(info)
    history = history if history is not None else bench_results
    pm = primary_metric_key(bench_results)
    cols = _metric_columns(bench_results)
    synthetic = bool(bench_results) and all(r.get("synthetic") for r in bench_results.values())

    badges = ""
    if status != "ready":
        badges += (f"<span class='badge {_STATUS_COLOR.get(status, 'muted')}'>"
                   f"{html.escape(status.replace('_', ' '))}</span>")
    if synthetic:
        badges += "<span class='badge warn'>synthetic fixture</span>"

    cite = ""
    if info.get("citation"):
        url = html.escape(info.get("url") or "#")
        cite = f"<a href='{url}'>{html.escape(info['citation'])}</a>"
    io_block = ""
    feats = (info.get("inputs") or []) + (info.get("outputs") or [])
    if feats or info.get("io_note"):
        items = "".join(
            f"<li><code>{html.escape(f['name'])}</code> — {html.escape(f['desc'])}</li>"
            for f in feats[:60])
        note = (f"<p>{html.escape(info['io_note'])}</p>" if info.get("io_note") else "")
        io_block = (f"<details class='io'><summary>Inputs & outputs "
                    f"({len(feats)} documented)</summary>{note}<ul>{items}</ul></details>")

    n_hist = sum(r["n_runs"] for r in history.values())
    hist_block = ""
    if n_hist > sum(r["n_runs"] for r in bench_results.values()):
        hist_block = (f"<details class='io'><summary>run history (all {n_hist} results)"
                      f"</summary>{_table(history, primary_metric_key(history), _metric_columns(history))}"
                      "</details>")

    rec = []
    src = info.get("source") or info.get("data_source")
    lic = info.get("license")
    if src or lic:
        rec.append(f"data: {html.escape(str(src or '—'))} · license: {html.escape(str(lic or '—'))}")
    if info.get("threshold") or cite:
        rec.append(f"gate: {html.escape(info.get('threshold') or '—')}"
                   + (f" ({cite})" if cite else ""))
    if info.get("status_note"):
        rec.append(f"status: {html.escape(str(info['status_note']))}")
    path = _best_result_path(bench_results, pm)
    if path:
        rec.append(f"<code class='path'>{html.escape(path)}</code>")
    receipts = (f"<div class='receipts'>{''.join(f'<div>{x}</div>' for x in rec)}</div>"
                if rec else "")

    tier = info.get("tier")
    return f"""
<section class="bench" id="{html.escape(key)}">
  <h3>{html.escape(info.get('name', key))} {badges}
    <span class="pills">
      {f"<span class='pill tier'>tier {tier}</span>" if tier is not None else ""}
      <span class="pill">{html.escape(info.get('shape', ''))}</span>
      <span class="pill">n = {html.escape(str(info.get('n', '?')))}</span>
      {f"<span class='pill'>{html.escape(info.get('threshold', ''))}</span>" if info.get('threshold') else ""}
    </span>
  </h3>
  <div class="bench-top">
    <div class="bench-info">
      <p class="desc">{html.escape(info.get('description', ''))}</p>
      <p class="meta"><code>{html.escape(key)}</code></p>
    </div>
    {f'<figure class="preview">{preview}<figcaption>sample of the data</figcaption></figure>' if preview else ''}
  </div>
  {_table(bench_results, pm, cols)}
  {hist_block}
  {io_block}
  {receipts}
</section>"""


def _group(keys: list[str], meta: dict, results: dict) -> dict[str, list[str]]:
    cats: dict[str, list[str]] = defaultdict(list)
    for key in sorted(keys):
        cap = meta.get(key, {}).get("capability") or key.split(".")[0].title()
        cats[cap].append(key)
    return cats


def _category_blocks(cats: dict[str, list[str]], heads: dict, results: dict, meta: dict,
                     mode: str, first: str | None = None) -> str:
    order = sorted(cats, key=lambda c: (c != first, c))
    out = []
    for cap in order:
        keys = cats[cap]
        try:
            svg = spider_svg(cap, keys, heads, meta, mode=mode)
        except Exception:  # noqa: BLE001 - a chart bug must not kill the report
            svg = ""
        cards = "\n".join(
            _bench_card(k, meta, heads[k], preview=preview_svg(k, mode), history=results[k])
            for k in keys)
        out.append(f"""
<h2 class="cat">{html.escape(cap)}</h2>
<p class="cat-sub">{len(keys)} benchmark{'s' if len(keys) != 1 else ''}</p>
<div class="catgrid">
  {f"<div class='spider'>{svg}</div>" if svg else ""}
  {cards}
</div>""")
    return "".join(out)


def build_report(reports_dir: Path, out_path: Path, mode: str = "dark",
                 video: Path | None = None, demo_runs: dict[str, str] | None = None,
                 anim_dir: Path | None = None, budget_mb: float = 12.0) -> Path:
    from surge.report import demo

    meta = load_metadata()
    results = load_results(reports_dir)
    if not results:
        raise SystemExit(f"no result.json files under {reports_dir}")
    heads = {k: newest_sweep(v) for k, v in results.items()}

    status = {k: _status_of(meta.get(k, {})) for k in results}
    ready = [k for k in results if status[k] == "ready"]
    tutorial = [k for k in results if status[k] == "tutorial"]
    tail = [k for k in results if status[k] not in ("ready", "tutorial")]

    def models(keys):
        return {m for k in keys for m in heads[k]}

    n_models = len(models(ready))
    n_runs = sum(r["n_runs"] for k in ready for r in heads[k].values())
    real = [r for k in ready for r in heads[k].values() if not r.get("synthetic")]
    n_pass = sum(1 for r in real if r["passed"])
    n_gated = sum(1 for r in real if r["passed"] is not None)
    n_total_runs = sum(r["n_runs"] for b in results.values() for r in b.values())
    snapshot = max((r["last"] for b in results.values() for r in b.values()),
                   default="")[:10]

    p = PALETTES[mode]
    css = _CSS_TEMPLATE.format(accent=p["series"][0], **p) + demo.CSS

    ready_cats = _group(ready, meta, results)
    first = "Scientific Domain" if "Scientific Domain" in ready_cats else None
    ready_html = _category_blocks(ready_cats, heads, results, meta, mode, first=first)
    tut_html = ""
    if tutorial:
        tut_html = ("<h2 class='cat'>Tutorial / plumbing</h2>"
                    "<p class='cat-sub'>Small public datasets that exercise the pipeline.</p>"
                    + _category_blocks(_group(tutorial, meta, results), heads, results,
                                       meta, mode))
    tail_html = ""
    if tail:
        tail_html = (f"<details class='tail'><summary>In progress · needs rerun · sanity "
                     f"checks ({len(tail)})</summary>"
                     + _category_blocks(_group(tail, meta, results), heads, results,
                                        meta, mode) + "</details>")

    names = {k: meta.get(k, {}).get("name", k) for k in results}
    demo_runs = (demo.DEFAULT_DEMO_RUNS if demo_runs is None else demo_runs)
    anim_dir = Path(anim_dir) if anim_dir else demo.DEFAULT_ANIM_DIR
    video = demo.DEFAULT_VIDEO if video is None else Path(video)
    notes = []

    def assemble(include_mission: bool) -> str:
        demo_html = demo.video_section(video) + demo.replay_panels(
            demo_runs, mode, anim_dir, names, include_mission=include_mission)
        extra = (" ".join(notes))
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SURGE benchmark leaderboard</title><style>{css}</style></head>
<body><main>
<header>
  <h1>SURGE Benchmark Leaderboard</h1>
  <div class="sub">Scientific surrogate evaluation · generated from
    {n_total_runs} result artifacts · latest run {html.escape(snapshot)}</div>
  <div class="accentbar"></div>
</header>
{demo_html}
<div class="tiles">
  <div class="tile"><b>{len(ready)}</b><span>ready benchmarks</span></div>
  <div class="tile"><b>{n_models}</b><span>models evaluated</span></div>
  <div class="tile"><b>{n_pass} / {n_gated}</b><span>pass published threshold</span></div>
  <div class="tile"><b>{len(ready_cats)}</b><span>capability domains</span></div>
</div>
<p class="tilenote">+{len(tutorial)} tutorial, +{len(tail)} in progress/rerun/sanity
 · {n_runs} newest-sweep results behind the tiles; synthetic fixtures excluded from pass counts.</p>
{ready_html}
{tut_html}
{tail_html}
<footer>Generated by <code>surge.report.leaderboard</code> from
<code>benchmark_reports/**/result.json</code> +
<code>surge/benchmarks/metadata.yaml</code>. Click column headers to sort.
Headline tables show the newest sweep only (runs within
{_sweep_hours():g} h of the newest run of each benchmark); the full
mean ± std over every result is under "run history". {extra}</footer>
</main><script>{_SORT_JS}{demo.JS}</script></body></html>"""

    doc = assemble(True)
    if len(doc.encode()) > budget_mb * 1024 * 1024:
        notes.append("Mission-control clip dropped to stay under the "
                     f"{budget_mb:g} MB size budget.")
        doc = assemble(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path


def _sweep_hours() -> float:
    from surge.benchmarks.status import SWEEP_WINDOW_HOURS
    return SWEEP_WINDOW_HOURS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="benchmark_reports")
    ap.add_argument("--out", default="surge_leaderboard.html")
    ap.add_argument("--mode", choices=("dark", "light"), default="dark")
    ap.add_argument("--video", default=None, help="mp4 embedded under the header")
    ap.add_argument("--demo-run", action="append", default=[], metavar="KEY=RUN_DIR",
                    help="HPO campaign replay for a benchmark (repeatable)")
    ap.add_argument("--anim-dir", default=None, help="pre-rendered *_<mode>.webm clips")
    args = ap.parse_args()
    demo_runs = dict(kv.split("=", 1) for kv in args.demo_run) or None
    out = build_report(Path(args.reports), Path(args.out), mode=args.mode,
                       video=Path(args.video) if args.video else None,
                       demo_runs=demo_runs, anim_dir=args.anim_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
