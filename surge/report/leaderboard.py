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


def load_results(reports_dir: Path) -> dict[str, dict[str, dict]]:
    """benchmark_key -> model_key -> aggregate over every result.json."""
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rj in sorted(reports_dir.glob("*/*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        bench, model = d.get("benchmark_key"), d.get("model_key")
        if not bench or not model:
            continue
        raw[bench][model].append(d)

    out: dict[str, dict[str, dict]] = {}
    for bench, models in raw.items():
        out[bench] = {}
        for model, runs in models.items():
            metrics: dict[str, list] = defaultdict(list)
            for r in runs:
                for k, v in (r.get("metrics") or {}).items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        metrics[k].append(float(v))
            agg = {}
            for k, v in metrics.items():
                if not v:
                    continue
                mean = statistics.fmean(v)
                std = statistics.stdev(v) if len(v) > 1 else 0.0
                if std < 1e-9 * max(1.0, abs(mean)):  # float noise, not spread
                    std = 0.0
                agg[k] = {"mean": mean, "std": std, "n": len(v)}
            passed = [bool(r.get("passed")) for r in runs if r.get("passed") is not None]
            out[bench][model] = {
                "metrics": agg,
                "n_runs": len(runs),
                "passed": (sum(passed) / len(passed) >= 0.5) if passed else None,
                "last": max(r.get("timestamp", "") for r in runs),
            }
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
    return svg[svg.index("<svg"):]


# --------------------------------------------------------------------- html

_CSS_TEMPLATE = """
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


def _bench_card(key: str, meta: dict[str, dict],
                bench_results: dict[str, dict]) -> str:
    info = meta.get(key, {})
    pm = primary_metric_key(bench_results)
    ranked = rank_models(bench_results, pm) if pm else sorted(bench_results)
    cols = _metric_columns(bench_results)
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
        chip = ("<span class='chip pass'>PASS</span>" if r["passed"]
                else "<span class='chip fail'>FAIL</span>" if r["passed"] is not None
                else "—")
        cells.append(f"<td data-v='{int(bool(r['passed']))}'>{chip}</td>")
        cells.append(f"<td data-v='{r['n_runs']}'>{r['n_runs']}</td>")
        rows.append(f"<tr class='{'best' if i == 0 else ''}'>{''.join(cells)}</tr>")

    header = "".join(
        f"<th>{_LABELS.get(c, c)}</th>" for c in cols)
    cite = ""
    if info.get("citation"):
        url = html.escape(info.get("url") or "#")
        cite = f" · <a href='{url}'>{html.escape(info['citation'])}</a>"
    io_block = ""
    feats = (info.get("inputs") or []) + (info.get("outputs") or [])
    if feats or info.get("io_note"):
        items = "".join(
            f"<li><code>{html.escape(f['name'])}</code> — {html.escape(f['desc'])}</li>"
            for f in feats[:60])
        note = (f"<p>{html.escape(info['io_note'])}</p>" if info.get("io_note") else "")
        io_block = (f"<details class='io'><summary>Inputs & outputs "
                    f"({len(feats)} documented)</summary>{note}<ul>{items}</ul></details>")

    tier = info.get("tier")
    return f"""
<section class="bench" id="{html.escape(key)}">
  <h3>{html.escape(info.get('name', key))}
    <span class="pills">
      {f"<span class='pill tier'>tier {tier}</span>" if tier is not None else ""}
      <span class="pill">{html.escape(info.get('shape', ''))}</span>
      <span class="pill">n = {html.escape(str(info.get('n', '?')))}</span>
      {f"<span class='pill'>{html.escape(info.get('threshold', ''))}</span>" if info.get('threshold') else ""}
    </span>
  </h3>
  <p class="desc">{html.escape(info.get('description', ''))}</p>
  <p class="meta"><code>{html.escape(key)}</code>{cite}</p>
  <table class="sortable">
    <thead><tr><th>model</th>{header}<th>gate</th><th>runs</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {io_block}
</section>"""


def build_report(reports_dir: Path, out_path: Path, mode: str = "dark") -> Path:
    meta = load_metadata()
    results = load_results(reports_dir)
    if not results:
        raise SystemExit(f"no result.json files under {reports_dir}")

    categories: dict[str, list[str]] = defaultdict(list)
    for key in sorted(results):
        cap = meta.get(key, {}).get("capability") or key.split(".")[0].title()
        categories[cap].append(key)

    n_bench = len(results)
    n_models = len({m for b in results.values() for m in b})
    n_runs = sum(r["n_runs"] for b in results.values() for r in b.values())
    n_pass = sum(1 for b in results.values() for r in b.values() if r["passed"])
    n_gated = sum(1 for b in results.values() for r in b.values()
                  if r["passed"] is not None)
    snapshot = max((r["last"] for b in results.values() for r in b.values()),
                   default="")[:10]

    p = PALETTES[mode]
    css = _CSS_TEMPLATE.format(accent=p["series"][0], **p)

    sections = []
    for cap in sorted(categories):
        keys = categories[cap]
        svg = spider_svg(cap, keys, results, meta, mode=mode)
        cards = "\n".join(_bench_card(k, meta, results[k]) for k in keys)
        sections.append(f"""
<h2 class="cat">{html.escape(cap)}</h2>
<p class="cat-sub">{len(keys)} benchmark{'s' if len(keys) != 1 else ''}</p>
<div class="catgrid">
  {f"<div class='spider'>{svg}</div>" if svg else ""}
  {cards}
</div>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SURGE benchmark leaderboard</title><style>{css}</style></head>
<body><main>
<header>
  <h1>SURGE Benchmark Leaderboard</h1>
  <div class="sub">Scientific surrogate evaluation · generated from
    {n_runs} result artifacts · latest run {html.escape(snapshot)}</div>
  <div class="accentbar"></div>
</header>
<div class="tiles">
  <div class="tile"><b>{n_bench}</b><span>benchmarks with results</span></div>
  <div class="tile"><b>{n_models}</b><span>models evaluated</span></div>
  <div class="tile"><b>{n_pass} / {n_gated}</b><span>pass published threshold</span></div>
  <div class="tile"><b>{len(categories)}</b><span>capability domains</span></div>
</div>
{''.join(sections)}
<footer>Generated by <code>surge.report.leaderboard</code> from
<code>benchmark_reports/**/result.json</code> +
<code>surge/benchmarks/metadata.yaml</code>. Click column headers to sort.
Values are mean ± std over repeated runs where available.</footer>
</main><script>{_SORT_JS}</script></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="benchmark_reports")
    ap.add_argument("--out", default="surge_leaderboard.html")
    ap.add_argument("--mode", choices=("dark", "light"), default="dark")
    args = ap.parse_args()
    out = build_report(Path(args.reports), Path(args.out), mode=args.mode)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
