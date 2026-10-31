"""Embedded media for the HTML leaderboard: the end-to-end demo video and
per-benchmark HPO "campaign replay" panels (single-file, base64 payloads)."""
from __future__ import annotations

import base64
import html
import tempfile
from pathlib import Path

from surge.benchmarks import paths

DEFAULT_VIDEO = None
DEFAULT_ANIM_DIR = None
DEFAULT_DEMO_RUNS: dict[str, str] = {}


def default_video() -> Path | None:
    """``<repo>/docs/assets/readme/surge_end_to_end.mp4`` when running from a
    checkout and the asset is present; ``None`` otherwise (installed wheel,
    or the file hasn't been fetched) so the report never assumes it exists."""
    root = paths.repo_root()
    if root is None:
        return None
    video = root / "docs" / "assets" / "readme" / "surge_end_to_end.mp4"
    return video if video.is_file() else None


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


def _clip(run_dir: Path, kind: str, mode: str, anim_dir: Path | None,
          allow_flat: bool = False) -> bytes | None:
    """Bytes of a pre-rendered clip, or one rendered from the run.

    Looks up ``<anim_dir>/<run_dir.name>/<kind>_<mode>.webm`` first; the flat
    ``<anim_dir>/<kind>_<mode>.webm`` is accepted only when ``allow_flat`` (a
    single demo run), so several campaigns never share one clip.  Rendering
    happens in a temporary directory that is always removed, so no
    frames/webm are left behind (also on failure).
    """
    if anim_dir is not None:
        cands = [Path(anim_dir) / Path(run_dir).name / f"{kind}_{mode}.webm"]
        if allow_flat:
            cands.append(Path(anim_dir) / f"{kind}_{mode}.webm")
        for pre in cands:
            if pre.is_file():
                return pre.read_bytes()
    try:
        from surge.viz import animate
        with tempfile.TemporaryDirectory(prefix=f"surge_{kind}_") as tmp_s:
            tmp = Path(tmp_s)
            fn = {"hpo_evolution": animate.hpo_evolution_frames,
                  "mission_control": animate.mission_control_frames}[kind]
            fn(Path(run_dir), tmp / "frames", mode)
            animate.frames_to_video(tmp / "frames", tmp / kind)
            out = tmp / f"{kind}.webm"
            return out.read_bytes() if out.is_file() else None
    except Exception:  # noqa: BLE001 - media must never break the report
        return None


def replay_panels(demo_runs: dict[str, str], mode: str, anim_dir: Path | None,
                  names: dict[str, str] | None = None,
                  include_mission: bool = True) -> str:
    out = []
    for key, run in demo_runs.items():
        rd = Path(run)
        allow_flat = len(demo_runs) == 1
        pre_rendered = (anim_dir is not None and any(
            f.is_file() for f in (
                Path(anim_dir) / rd.name / f"hpo_evolution_{mode}.webm",
                *([Path(anim_dir) / f"hpo_evolution_{mode}.webm"] if allow_flat else []))))
        if not rd.is_dir() and not pre_rendered:
            continue
        kinds = ["hpo_evolution"] + (["mission_control"] if include_mission else [])
        vids = []
        for kind in kinds:
            p = _clip(rd, kind, mode, anim_dir, allow_flat=allow_flat)
            if p:
                vids.append(f"<video muted loop playsinline preload='metadata' "
                            f"src='data:video/webm;base64,"
                            f"{base64.b64encode(p).decode()}'></video>")
        if vids:
            title = (names or {}).get(key, key)
            out.append(f"<div class='replay'><h4>Campaign replay · {html.escape(title)}"
                       f" <span style='color:var(--muted);font-weight:400'>"
                       f"({html.escape(str(run))})</span></h4>"
                       f"<div class='clips'>{''.join(vids)}</div>"
                       "<input type='range' min='0' max='1000' value='0' "
                       "aria-label='scrub campaign'></div>")
    return "".join(out)
