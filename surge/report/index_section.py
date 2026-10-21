"""Benchmark index: a searchable, filterable, sortable table for the top of
the leaderboard HTML, right under the header tiles.

One row per *visible* benchmark (maintainer-hidden benchmarks are excluded
upstream, in :func:`surge.report.leaderboard.build_report`, before this
module ever sees them). Row data is embedded as a small JS array
(``BENCH_INDEX``) so the live search/filter/"cards follow table" behaviour
needs no server and no framework — vanilla JS only, per project policy.
"""
from __future__ import annotations

import html
import json


def build_index_rows(results: dict[str, dict], heads: dict[str, dict],
                     meta: dict[str, dict]) -> list[dict]:
    """One row dict per key in ``results`` (already filtered to visible)."""
    # Lazy import: leaderboard imports this module, so importing leaderboard
    # at module scope here would be circular.
    from surge.report.leaderboard import (
        _category_of, _fmt_value, _gate_chip, _status_of, primary_metric_key,
        rank_models,
    )

    rows = []
    for key in sorted(results):
        info = meta.get(key, {})
        bench_heads = heads.get(key, {})
        cat = _category_of(key, meta)
        status = _status_of(info)
        pm = primary_metric_key(bench_heads)
        ranked = rank_models(bench_heads, pm) if pm else sorted(bench_heads)
        best_model = ranked[0] if ranked else None
        best_agg = bench_heads.get(best_model, {}).get("metrics", {}).get(pm) \
            if (best_model and pm) else None
        value_text = _fmt_value(pm, best_agg) if best_agg else "—"
        gate_html = _gate_chip(bench_heads[best_model]) if best_model else "—"
        model_keys = sorted(bench_heads)
        haystack = " ".join([
            info.get("name", key), key, cat, str(info.get("task_explainer") or ""),
            " ".join(model_keys),
        ]).lower()
        rows.append({
            "key": key, "name": info.get("name", key), "category": cat,
            "status": status, "tier": info.get("tier"), "shape": info.get("shape", ""),
            "focus": bool(info.get("focus") is True), "best_model": best_model,
            "value_text": value_text, "gate_html": gate_html, "haystack": haystack,
        })
    return rows


_CSS = """
.bench-index { margin: 28px 0 40px; }
.idx-controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; margin-bottom: 8px; }
#idx-search { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  color: var(--ink); padding: 7px 12px; font-size: 13px; min-width: 240px; }
.idx-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip-toggle { font-size: 11px; padding: 3px 10px; border-radius: 99px; cursor: pointer;
  background: transparent; border: 1px solid var(--line); color: var(--muted); }
.chip-toggle.active { color: var(--accent); border-color: var(--accent); }
.idx-follow { font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; gap: 5px; }
.idx-count { color: var(--muted); font-size: 12px; margin: 4px 0 10px; }
.idx-table td, .idx-table th { white-space: nowrap; }
.idx-star { color: var(--warn); font-size: 11px; }
.idx-link { color: var(--ink); text-decoration: none; }
.idx-link:hover { color: var(--accent); }
.bench { position: relative; scroll-margin-top: 18px; }
h2.cat { scroll-margin-top: 18px; }
.idx-back { position: absolute; top: 16px; right: 20px; font-size: 11px;
  color: var(--muted); text-decoration: none; }
.idx-back:hover { color: var(--accent); }
html { scroll-behavior: smooth; }
"""

_JS = """
(function () {
  const idx = Object.fromEntries(BENCH_INDEX.map(r => [r.key, r]));
  const table = document.getElementById("idx-table");
  if (!table) return;
  const rows = [...table.tBodies[0].rows];
  const searchEl = document.getElementById("idx-search");
  const followEl = document.getElementById("idx-follow");
  const countEl = document.getElementById("idx-count");
  const selected = { status: new Set(), cat: new Set() };
  document.querySelectorAll(".chip-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
      const set = selected[btn.dataset.kind];
      if (set.has(btn.dataset.val)) { set.delete(btn.dataset.val); btn.classList.remove("active"); }
      else { set.add(btn.dataset.val); btn.classList.add("active"); }
      apply();
    });
  });
  function apply() {
    const q = (searchEl.value || "").toLowerCase().trim();
    const follow = followEl.checked;
    const catVisible = {};
    let shown = 0;
    rows.forEach(tr => {
      const key = tr.dataset.key, r = idx[key] || {};
      const ok = (!q || (r.haystack || "").includes(q))
        && (selected.status.size === 0 || selected.status.has(tr.dataset.status))
        && (selected.cat.size === 0 || selected.cat.has(tr.dataset.cat));
      tr.style.display = ok ? "" : "none";
      if (ok) shown++;
      catVisible[tr.dataset.cat] = catVisible[tr.dataset.cat] || ok;
      const card = document.getElementById(key);
      if (card) card.style.display = (follow && !ok) ? "none" : "";
    });
    document.querySelectorAll("section[data-cat]").forEach(sec => {
      const cat = sec.getAttribute("data-cat");
      sec.style.display = (follow && !catVisible[cat]) ? "none" : "";
    });
    countEl.textContent = `showing ${shown} of ${rows.length} benchmarks`;
  }
  searchEl.addEventListener("input", apply);
  followEl.addEventListener("change", apply);
  rows.forEach(tr => tr.addEventListener("click", e => {
    if (e.target.closest("a")) return;
    location.hash = "#" + tr.dataset.key;
  }));
  const q = new URLSearchParams(location.search).get("q");
  if (q) searchEl.value = q;
  // the index sits above the cards, so on first load the card sections do
  // not exist yet; defer the initial apply until the DOM is complete.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
"""


def index_section_html(rows: list[dict]) -> str:
    if not rows:
        return ""
    from surge.report.leaderboard import _STATUS_COLOR

    statuses = sorted({r["status"] for r in rows})
    cats = sorted({r["category"] for r in rows})

    def row_html(r: dict) -> str:
        star = " <span class='idx-star' title='focus'>&#9733;</span>" if r["focus"] else ""
        tier = (f"<span class='pill tier'>tier {html.escape(str(r['tier']))}</span>"
                if r.get("tier") is not None else "—")
        status_badge = (f"<span class='badge {_STATUS_COLOR.get(r['status'], 'muted')}'>"
                        f"{html.escape(r['status'].replace('_', ' '))}</span>")
        return (f"<tr data-key='{html.escape(r['key'])}' data-status='{html.escape(r['status'])}' "
                f"data-cat='{html.escape(r['category'])}'>"
                f"<td class='model' data-v='{html.escape(r['name'])}'>"
                f"<a href='#{html.escape(r['key'])}' class='idx-link'>{html.escape(r['name'])}</a>"
                f"{star}</td>"
                f"<td><code>{html.escape(r['key'])}</code></td>"
                f"<td data-v='{html.escape(r['category'])}'>{html.escape(r['category'])}</td>"
                f"<td data-v='{html.escape(r['status'])}'>{status_badge}</td>"
                f"<td>{tier}</td>"
                f"<td>{html.escape(r.get('shape') or '—')}</td>"
                f"<td data-v='{html.escape(r['value_text'])}'>{html.escape(r['value_text'])} "
                f"{r['gate_html']}</td></tr>")

    body = "".join(row_html(r) for r in rows)
    status_chips = "".join(
        f"<button type='button' class='chip-toggle' data-kind='status' data-val='{html.escape(s)}'>"
        f"{html.escape(s.replace('_', ' '))}</button>" for s in statuses)
    cat_chips = "".join(
        f"<button type='button' class='chip-toggle' data-kind='cat' data-val='{html.escape(c)}'>"
        f"{html.escape(c)}</button>" for c in cats)
    data_json = json.dumps([
        {"key": r["key"], "haystack": r["haystack"]} for r in rows
    ])

    return f"""
<section id="bench-index" class="bench-index">
  <h2 class="cat">Benchmark index</h2>
  <p class="cat-sub">search, filter and jump to any benchmark card</p>
  <div class="idx-controls">
    <input id="idx-search" type="search" placeholder="Search name, key, category, model…" autocomplete="off">
    <div class="idx-chips">{status_chips}</div>
    <div class="idx-chips">{cat_chips}</div>
    <label class="idx-follow"><input type="checkbox" id="idx-follow" checked> cards follow table</label>
  </div>
  <p class="idx-count" id="idx-count"></p>
  <table class="sortable idx-table" id="idx-table">
    <thead><tr><th>name</th><th>key</th><th>category</th><th>status</th><th>tier</th>
      <th>shape</th><th>best</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</section>
<script>const BENCH_INDEX = {data_json};</script>
<script>{_JS}</script>
"""


CSS = _CSS
