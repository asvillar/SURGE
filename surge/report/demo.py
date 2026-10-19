"""Embedded media for the HTML leaderboard: the end-to-end demo video and
per-benchmark HPO "campaign replay" panels (single-file, base64 payloads)."""
from __future__ import annotations

import base64
import html
import tempfile
from pathlib import Path

DEFAULT_VIDEO = Path("docs/assets/readme/surge_end_to_end.mp4")
DEFAULT_ANIM_DIR = Path("audit/presentation/assets/anim")
DEFAULT_DEMO_RUNS = {"plasma.qlknn_transport": "runs/qlknn_hpo_200_best"}

CSS = """
.demo { margin: 0 0 30px; }
.demo video { width: 100%; max-height: 520px; border-radius: 10px; background: #000;
  border: 1px solid var(--line); }
.demo .cap { color: var(--muted); font-size: 12.5px; margin: 6px 0 0; }
.replay { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; margin: 16px 0; }
.replay h4 { margin: 0 0 8px; font-size: 14px; }
.replay .clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); gap: 10px; }
.replay video { width: 100%; border-radius: 6px; }
.replay input[type=range] { width: 100%; margin-top: 8px; }
"""

JS = """
document.querySelectorAll('.replay').forEach(p => {
  const vids = p.querySelectorAll('video'), s = p.querySelector('input[type=range]');
  if (!vids.length || !s) return;
  const dur = () => Math.max(...[...vids].map(v => v.duration || 0));
  s.addEventListener('input', () => vids.forEach(v => {
    v.pause(); v.currentTime = (s.value / 1000) * (v.duration || dur()); }));
});
"""


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def video_section(video: Path | None, poster: Path | None = None) -> str:
    """Embed the end-to-end demo video (base64, single-file page).

    ``poster`` defaults to a sibling ``<video stem>_poster.jpg`` when present,
    so the player shows a real frame instead of the animation's dark fade-in.
    """
    if not video or not Path(video).is_file():
        return ""
    video = Path(video)
    if poster is None:
        cand = video.with_name(video.stem + "_poster.jpg")
        poster = cand if cand.is_file() else None
    poster_attr = ""
    if poster and Path(poster).is_file():
        poster_attr = f" poster='data:image/jpeg;base64,{_b64(Path(poster))}'"
    frag = "" if poster_attr else "#t=0.1"
    src = f"data:video/mp4;base64,{_b64(video)}{frag}"
    return (f"<section class='demo'><video controls muted loop playsinline "
            f"preload='none'{poster_attr} src='{src}'></video>"
            "<p class='cap'>See it run: dataset to surrogate to HPO to benchmark "
            "leaderboard, end to end.</p></section>")


def _clip(run_dir: Path, kind: str, mode: str, anim_dir: Path) -> Path | None:
    """Reuse a pre-rendered ``<kind>_<mode>.webm`` or render it from the run."""
    pre = Path(anim_dir) / f"{kind}_{mode}.webm"
    if pre.is_file():
        return pre
    try:
        from surge.viz import animate
        tmp = Path(tempfile.mkdtemp(prefix=f"surge_{kind}_"))
        fn = {"hpo_evolution": animate.hpo_evolution_frames,
              "mission_control": animate.mission_control_frames}[kind]
        fn(Path(run_dir), tmp / "frames", mode)
        animate.frames_to_video(tmp / "frames", tmp / kind)
        out = tmp / f"{kind}.webm"
        return out if out.is_file() else None
    except Exception:  # noqa: BLE001 - media must never break the report
        return None


def replay_panels(demo_runs: dict[str, str], mode: str, anim_dir: Path,
                  names: dict[str, str] | None = None,
                  include_mission: bool = True) -> str:
    out = []
    for key, run in demo_runs.items():
        rd = Path(run)
        if not rd.is_dir() and not (Path(anim_dir) / f"hpo_evolution_{mode}.webm").is_file():
            continue
        kinds = ["hpo_evolution"] + (["mission_control"] if include_mission else [])
        vids = []
        for kind in kinds:
            p = _clip(rd, kind, mode, anim_dir)
            if p:
                vids.append(f"<video muted loop playsinline preload='metadata' "
                            f"src='data:video/webm;base64,{_b64(p)}'></video>")
        if vids:
            title = (names or {}).get(key, key)
            out.append(f"<div class='replay'><h4>Campaign replay · {html.escape(title)}"
                       f" <span style='color:var(--muted);font-weight:400'>"
                       f"({html.escape(str(run))})</span></h4>"
                       f"<div class='clips'>{''.join(vids)}</div>"
                       "<input type='range' min='0' max='1000' value='0' "
                       "aria-label='scrub campaign'></div>")
    return "".join(out)
