# pip install streamlit pillow numpy pillow-jxl-plugin
# streamlit run app.py
#
# ---------------------------------------------------------------------------
# ImageBench - Hardware-Native Image Compression Benchmarking Sandbox
# Copyright (c) 2026 skunkworxdark - MIT License (see LICENSE)
# ---------------------------------------------------------------------------
# Quick start (Windows):     setup.bat   then   run.bat
# Quick start (macOS/Linux): ./setup.sh  then   ./run.sh
# Manual:
#   python -m venv .venv
#   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate (POSIX)
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Measurement policy: NOTHING on screen is simulated or estimated.
#   * File sizes  -> os.path.getsize() of the real file written to disk.
#   * Save time   -> time.perf_counter() wrapped tightly around the real
#                    Pillow .save() call (encoder + file write).
#   * Load time   -> time.perf_counter() wrapped around opening the saved file
#                    and fully decoding every pixel into RAM (Pillow .load()).
#   All codecs run through Pillow: PNG/WebP natively, JPEG XL through the
#   pillow-jxl-plugin.
#
# Architecture (state decoupled from file handling):
#   sidebar config widgets -> build_profiles(cfg, caps)  [pure]
#   measurement cache       -> results[file_id]["records"][profile.key]
#   file loop               -> encodes ONLY (file, profile) pairs missing from
#                              the cache; weight-slider / toggle changes never
#                              re-encode anything.
#   analytics               -> pure functions over cached measurements.
# ---------------------------------------------------------------------------

import csv
import html
import importlib.metadata
import io
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from statistics import fmean

import numpy as np
import PIL
import streamlit as st
from PIL import Image, features

# Local desktop tool working on the user's own files: allow very large images.
Image.MAX_IMAGE_PIXELS = None

MIB = 1024.0 * 1024.0
SPIKE_THRESHOLD = 2.0  # 200 % of the running average
DEFAULT_STORAGE_WEIGHT = 0.70
JXL_DEFAULT_EFFORT = 7  # libjxl default; omitted from kwargs when unchanged
JXL_DEFAULT_DSPEED = 0  # libjxl default decoding-speed tier

PIL_NATIVE = "PIL NATIVE"

UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif", "tga", "ppm", "pgm", "jxl"]

# Codec engine families that can be toggled as whole units.
FAMILIES: dict[str, tuple[str, str]] = {
    "png": ("PNG · Lossless", PIL_NATIVE),
    "webp_ll": ("WebP · Lossless", PIL_NATIVE),
    "webp_lossy": ("WebP · Lossy", PIL_NATIVE),
    "jxl_ll": ("JPEG XL · Lossless · pillow-jxl", PIL_NATIVE),
    "jxl_lossy": ("JPEG XL · Lossy · pillow-jxl", PIL_NATIVE),
}

PNG_NAMES = {0: "Store", 1: "Fast", 3: "Balanced", 6: "Default", 9: "Maximum"}
WEBP_METHOD_NAMES = {0: "Fastest", 1: "Fast", 3: "Balanced", 4: "Pillow Default", 6: "Max Effort"}
# Official libjxl effort preset names (cjxl -e 1..9); 7 is the libjxl default.
JXL_EFFORT_NAMES = {1: "Lightning", 2: "Thunder", 3: "Falcon", 4: "Cheetah", 5: "Hare",
                    6: "Wombat", 7: "Squirrel · Default", 8: "Kitten", 9: "Tortoise"}
QUALITY_OPTIONS = [50, 60, 70, 75, 80, 85, 90, 92, 95, 98, 100]

DEFAULT_CFG = {
    "families": list(FAMILIES),
    "png_levels": [1, 3, 6],
    "webp_methods": [1, 3, 6],
    "webp_qualities": [95],
    "webp_lossy_method": 4,
    "jxl_ll_efforts": [JXL_DEFAULT_EFFORT],
    "jxl_ll_dspeed": JXL_DEFAULT_DSPEED,
    "jxl_lossy_qualities": [95],
    "jxl_lossy_effort": JXL_DEFAULT_EFFORT,
    "jxl_lossy_dspeed": JXL_DEFAULT_DSPEED,
}


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    key: str  # stable identity: family + exact encoder kwargs (cache key)
    label: str
    short: str
    engine: str  # PIL_NATIVE
    family: str
    mode: str  # lossless | lossy
    ext: str
    fmt: str  # Pillow format id passed to Image.save(format=...)
    params: tuple  # (name, value) pairs passed verbatim to the encoder
    param_desc: str
    available: bool = True
    reason: str = ""

    @property
    def kwargs(self) -> dict:
        return dict(self.params)


def _make(family, label, short, mode, ext, fmt, params, desc, caps) -> Profile:
    ok, reason = caps["family_ok"][family]
    key = family + "|" + ",".join(f"{k}={v}" for k, v in params)
    return Profile(key, label, short, FAMILIES[family][1], family, mode, ext, fmt, tuple(params), desc, ok, reason)


def _jxl_speed_params(effort: int, dspeed: int) -> tuple[list, list]:
    """pillow-jxl kwargs for effort / decoding_speed (omitted when at libjxl defaults)."""
    params, desc = [], []
    if effort != JXL_DEFAULT_EFFORT:
        params.append(("effort", int(effort)))
        desc.append(f"effort={effort}")
    else:
        desc.append(f"effort={effort} (default)")
    if dspeed != JXL_DEFAULT_DSPEED:
        params.append(("decoding_speed", int(dspeed)))
        desc.append(f"decoding_speed={dspeed}")
    return params, desc


def build_profiles(cfg: dict, caps: dict) -> tuple[Profile, ...]:
    """Pure function: (sidebar configuration, codec capabilities) -> PROFILES."""
    out: list[Profile] = []
    fams = set(cfg["families"])

    if "png" in fams:
        for lvl in sorted(set(cfg["png_levels"])):
            name = PNG_NAMES.get(lvl)
            label = f"PNG (Lossless - {name + ' ' if name else ''}Level {lvl})"
            out.append(_make("png", label, f"PNG L{lvl}", "lossless", "png", "PNG",
                             [("compress_level", int(lvl))], f"compress_level={lvl}", caps))

    if "webp_ll" in fams:
        for m in sorted(set(cfg["webp_methods"])):
            name = WEBP_METHOD_NAMES.get(m)
            label = f"WebP (True Lossless Mode - {name + ' ' if name else ''}Method {m})"
            out.append(_make("webp_ll", label, f"WebP-LL m{m}", "lossless", "webp", "WEBP",
                             [("lossless", True), ("method", int(m))], f"lossless=True, method={m}", caps))

    if "webp_lossy" in fams:
        m = int(cfg["webp_lossy_method"])
        for q in sorted(set(cfg["webp_qualities"]), reverse=True):
            label = f"WebP (Lossy Mode - {q}% Quality, Method {m})"
            out.append(_make("webp_lossy", label, f"WebP q{q} m{m}", "lossy", "webp", "WEBP",
                             [("quality", int(q)), ("method", m)], f"quality={q}, method={m}", caps))

    # ---- JPEG XL · pillow-jxl (lossless and lossy are independent engines) ----
    if "jxl_ll" in fams:
        dspeed = int(cfg["jxl_ll_dspeed"])
        for e in sorted(set(cfg["jxl_ll_efforts"])):
            sp, sd = _jxl_speed_params(e, dspeed)
            out.append(_make("jxl_ll", f"JPEG XL - PIL Lossless (Effort {e} {_effort_name(e)})",
                             f"JXL LL e{e}", "lossless", "jxl", "JPEGXL",
                             [("lossless", True)] + sp,
                             ", ".join(['format="JPEGXL", lossless=True'] + sd), caps))
    if "jxl_lossy" in fams:
        e, dspeed = int(cfg["jxl_lossy_effort"]), int(cfg["jxl_lossy_dspeed"])
        sp, sd = _jxl_speed_params(e, dspeed)
        for q in sorted(set(cfg["jxl_lossy_qualities"]), reverse=True):
            out.append(_make("jxl_lossy", f"JPEG XL - PIL Lossy (Quality {q}, Effort {e})",
                             f"JXL q{q} e{e}", "lossy", "jxl", "JPEGXL",
                             [("lossless", False), ("quality", int(q))] + sp,
                             ", ".join([f'format="JPEGXL", lossless=False, quality={q}'] + sd), caps))
    return tuple(out)


def _effort_name(e: int) -> str:
    name = JXL_EFFORT_NAMES.get(int(e))
    return f"· {name}" if name else ""


# ---------------------------------------------------------------------------
# Codec environment probing (real capability checks, cached per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def probe_pillow_jxl():
    """Dynamically import + register the pillow-jxl plugin.
    Returns (pillow_format|None, version_info|error_message)."""
    try:
        import pillow_jxl  # noqa: F401  (import registers the plugin with Pillow)
    except Exception as exc:
        return None, f"pillow_jxl import failed ({type(exc).__name__}: {exc}) — pip install pillow-jxl-plugin"
    Image.init()
    handler = Image.SAVE.get("JPEGXL") or Image.SAVE.get("JXL")
    if handler is None:
        return None, "pillow_jxl imported but registered no JPEG XL save handler"
    if "JPEGXL" not in Image.SAVE:
        # The plugin registers itself as "JXL"; expose the canonical "JPEGXL" id too.
        Image.register_save("JPEGXL", handler)
    try:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEGXL", lossless=True)
        buf.seek(0)
        with Image.open(buf) as im:
            im.load()
    except Exception as exc:
        return None, f"pillow_jxl round-trip failed: {type(exc).__name__}: {exc}"
    try:
        ver = importlib.metadata.version("pillow-jxl-plugin")
    except Exception:
        ver = "?"
    return "JPEGXL", f"pillow-jxl-plugin {ver}"


def capabilities() -> dict:
    pjxl_fmt, pjxl_info = probe_pillow_jxl()
    webp_ok = bool(features.check("webp"))
    webp_reason = "" if webp_ok else "Pillow was built without libwebp"
    return {
        "pjxl_fmt": pjxl_fmt, "pjxl_info": pjxl_info,
        "webp_ok": webp_ok,
        "family_ok": {
            "png": (True, ""),
            "webp_ll": (webp_ok, webp_reason),
            "webp_lossy": (webp_ok, webp_reason),
            "jxl_ll": (pjxl_fmt is not None, "" if pjxl_fmt else pjxl_info),
            "jxl_lossy": (pjxl_fmt is not None, "" if pjxl_fmt else pjxl_info),
        },
    }


# ---------------------------------------------------------------------------
# Core measurement engine (unchanged capture mechanics)
# ---------------------------------------------------------------------------
def encode_to_disk(prof: Profile, img: Image.Image, path: str) -> float:
    """Encode + write to disk. Returns elapsed milliseconds measured with
    time.perf_counter() tightly around the real compression write."""
    fmt, kwargs = prof.fmt, prof.kwargs
    t0 = time.perf_counter()
    img.save(path, format=fmt, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def decode_from_disk(path: str) -> tuple[float, tuple]:
    """Open the saved file and force a full pixel decode into RAM.
    Returns (elapsed milliseconds, decoded shape)."""
    t0 = time.perf_counter()
    with Image.open(path) as im:
        im.load()  # forces every pixel block to be unpacked into memory
        t1 = time.perf_counter()
        shape = (im.height, im.width, len(im.getbands()))
    return (t1 - t0) * 1000.0, shape


def prepare_source(raw: bytes) -> tuple[Image.Image, dict]:
    """Decode the uploaded file once (untimed) into a canonical 8-bit RGB/RGBA
    raw pixel matrix so that every profile compresses the identical payload."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        src_mode, src_format = im.mode, (im.format or "?")
        has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
            im.mode == "P" and "transparency" in im.info
        )
        if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
            a = np.asarray(im, dtype=np.float64)
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
            work = Image.fromarray(np.round(a).astype(np.uint8)).convert("RGB")
        else:
            work = im.convert("RGBA" if has_alpha else "RGB")
    work.load()
    meta = {
        "width": work.width,
        "height": work.height,
        "src_mode": src_mode,
        "src_format": src_format,
        "work_mode": work.mode,
    }
    return work, meta


def measure_profile(prof: Profile, img: Image.Image, meta: dict, workdir: str) -> dict:
    """One isolated hardware run: encode -> getsize -> full decode -> delete."""
    rec = {"ok": False, "bytes": 0, "save_ms": float("nan"), "load_ms": float("nan"), "error": None}
    path = os.path.join(workdir, f"{uuid.uuid4().hex}.{prof.ext}")
    try:
        rec["save_ms"] = encode_to_disk(prof, img, path)
        rec["bytes"] = os.path.getsize(path)
        rec["load_ms"], shape = decode_from_disk(path)
        if shape[0] != meta["height"] or shape[1] != meta["width"]:
            raise RuntimeError(f"decoded geometry {shape} != source {meta['height']}x{meta['width']}")
        rec["ok"] = True
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return rec


@st.cache_resource(show_spinner=False)
def warm_up_codecs() -> float:
    """Run each codec family once on a tiny image so one-off library
    initialisation is never billed to a user measurement. Results discarded."""
    t0 = time.perf_counter()
    caps = capabilities()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    meta = {"width": 32, "height": 32}
    tmp = tempfile.mkdtemp(prefix="imagebench_warmup_")
    try:
        for prof in build_profiles(DEFAULT_CFG, caps):
            if prof.available:
                measure_profile(prof, img, meta, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Aggregation / analytics (pure functions over cached measurements)
# ---------------------------------------------------------------------------
def aggregate(active: list[dict], profiles) -> dict:
    agg = {p.key: {"bytes": [], "save": [], "load": [], "items": [], "errors": 0} for p in profiles}
    for res in active:
        for prof in profiles:
            rec = res["records"].get(prof.key)
            if rec is None:
                continue
            bucket = agg[prof.key]
            if rec["ok"]:
                bucket["bytes"].append(rec["bytes"])
                bucket["save"].append(rec["save_ms"])
                bucket["load"].append(rec["load_ms"])
                bucket["items"].append((res, rec))
            else:
                bucket["errors"] += 1
    return agg


def triple(values: list[float]):
    if not values:
        return None
    return fmean(values), min(values), max(values)


def detect_spikes(agg: dict, profiles) -> list[dict]:
    alerts = []
    for prof in profiles:
        bucket = agg[prof.key]
        if len(bucket["items"]) < 2:
            continue
        avg_bytes = fmean(bucket["bytes"])
        avg_save = fmean(bucket["save"])
        for res, rec in bucket["items"]:
            if avg_save > 0 and rec["save_ms"] > SPIKE_THRESHOLD * avg_save:
                alerts.append({
                    "file": res["name"], "profile": prof, "metric": "Save latency",
                    "observed": f"{rec['save_ms']:.2f} ms", "average": f"{avg_save:.2f} ms",
                    "ratio": rec["save_ms"] / avg_save * 100.0,
                })
            if avg_bytes > 0 and rec["bytes"] > SPIKE_THRESHOLD * avg_bytes:
                alerts.append({
                    "file": res["name"], "profile": prof, "metric": "File size",
                    "observed": f"{rec['bytes'] / MIB:.4f} MiB", "average": f"{avg_bytes / MIB:.4f} MiB",
                    "ratio": rec["bytes"] / avg_bytes * 100.0,
                })
    alerts.sort(key=lambda a: a["ratio"], reverse=True)
    return alerts


def _normalise(value: float, best: float, worst: float) -> float:
    """0-100 scale anchored to the worst baseline (0) and the best tier (100)."""
    span = worst - best
    if span <= 1e-12:
        return 100.0
    return max(0.0, min(100.0, (worst - value) / span * 100.0))


def lossless_leaderboard(agg: dict, profiles, w_storage: float, w_latency: float):
    rows = []
    for prof in profiles:
        if prof.mode != "lossless" or not prof.available:
            continue
        b = agg[prof.key]
        if not b["items"]:
            continue
        rows.append({
            "profile": prof, "n": len(b["items"]),
            "size": fmean(b["bytes"]) / MIB, "save": fmean(b["save"]), "load": fmean(b["load"]),
        })
    if not rows:
        return [], None
    worst_size, best_size = max(r["size"] for r in rows), min(r["size"] for r in rows)
    worst_save, best_save = max(r["save"] for r in rows), min(r["save"] for r in rows)
    for r in rows:
        r["storage_score"] = _normalise(r["size"], best_size, worst_size)
        r["latency_score"] = _normalise(r["save"], best_save, worst_save)
        r["composite"] = w_storage * r["storage_score"] + w_latency * r["latency_score"]
        r["vs_worst_size"] = (1.0 - r["size"] / worst_size) * 100.0 if worst_size > 0 else 0.0
    rows.sort(key=lambda r: (r["composite"], r["storage_score"]), reverse=True)
    baselines = {
        "worst_size": worst_size, "worst_save": worst_save,
        "worst_size_prof": max(rows, key=lambda r: r["size"])["profile"],
        "worst_save_prof": max(rows, key=lambda r: r["save"])["profile"],
    }
    return rows, baselines


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -12%, rgba(56,189,248,.10) 0%, transparent 60%),
    radial-gradient(900px 520px at 108% 4%, rgba(167,139,250,.12) 0%, transparent 55%),
    #070a12;
}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1220 0%, #0a0e18 100%); border-right: 1px solid #1c2438; }
[data-testid="stHeader"] { background: transparent; }
.ib-hero { padding: 22px 26px; border-radius: 18px; border: 1px solid #1f2a44;
  background: linear-gradient(135deg, rgba(30,41,74,.65), rgba(17,22,38,.65)); margin-bottom: 18px;
  box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.ib-hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #e0f2fe, #a5b4fc 55%, #f0abfc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ib-hero p { margin: 6px 0 0; color: #94a3b8; font-size: 14px; }
.ib-chips { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip { font: 600 11px/1 'JetBrains Mono', monospace; padding: 6px 10px; border-radius: 999px; border: 1px solid #27324d; color: #cbd5e1; background: #0f1526; }
.chip.ok { border-color: rgba(52,211,153,.45); color: #6ee7b7; }
.chip.bad { border-color: rgba(248,113,113,.5); color: #fca5a5; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font: 700 10.5px/1.5 'JetBrains Mono', monospace;
  letter-spacing: .04em; white-space: nowrap; }
.badge.pil { background: rgba(45,212,191,.12); color: #5eead4; border: 1px solid rgba(45,212,191,.5); }
.badge.na  { background: rgba(248,113,113,.10); color: #fca5a5; border: 1px solid rgba(248,113,113,.5); margin-left: 6px; }
.kpi { border: 1px solid #1f2a44; border-radius: 14px; padding: 14px 16px; background: rgba(15,21,38,.75); }
.kpi .l { color: #8190a9; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.kpi .v { color: #f1f5f9; font-size: 24px; font-weight: 800; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi .s { color: #64748b; font-size: 12px; margin-top: 2px; }
.tbl-wrap { overflow-x: auto; border: 1px solid #1f2a44; border-radius: 14px; background: rgba(12,17,30,.8); margin: 6px 0 14px; }
table.ib { width: 100%; border-collapse: collapse; font-size: 13px; color: #d7deea; }
table.ib th { background: #10172a; color: #93a3bd; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  padding: 10px 12px; border-bottom: 1px solid #1f2a44; text-align: left; white-space: nowrap; }
table.ib th.grp { text-align: center; color: #c7d2fe; border-left: 1px solid #1f2a44; }
table.ib th.num, table.ib td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
table.ib td { padding: 9px 12px; border-bottom: 1px solid #161f35; vertical-align: middle; }
table.ib td.sep, table.ib th.sep { border-left: 1px solid #1f2a44; }
table.ib tr:hover td { background: rgba(99,102,241,.06); }
table.ib tr.top td { background: rgba(52,211,153,.07); }
table.ib tr.na td { opacity: .6; }
td .pname { font-weight: 600; color: #e2e8f0; }
td .pdesc { display: block; color: #64748b; font: 500 11px 'JetBrains Mono', monospace; margin-top: 2px; }
.muted { color: #64748b; }
.err { color: #fca5a5; font-size: 12px; }
.bar { position: relative; height: 6px; background: #1a2238; border-radius: 99px; width: 110px; display: inline-block; vertical-align: middle; margin-right: 8px; overflow: hidden; }
.bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: linear-gradient(90deg, #38bdf8, #a78bfa); }
.bar.comp > span { background: linear-gradient(90deg, #34d399, #22d3ee); }
.callout { border-radius: 14px; padding: 16px 18px; border: 1px solid; margin: 8px 0 14px; }
.callout.win { border-color: rgba(52,211,153,.45); background: rgba(6,78,59,.25); color: #d1fae5; }
.callout.warn { border-color: rgba(251,191,36,.45); background: rgba(120,53,15,.22); color: #fde68a; }
.callout.alert { border-color: rgba(248,113,113,.5); background: rgba(127,29,29,.22); color: #fecaca; }
.callout.info { border-color: rgba(56,189,248,.4); background: rgba(12,74,110,.2); color: #bae6fd; }
.callout h4 { margin: 0 0 6px; font-size: 15px; }
.lossy-card { border: 1px solid #1f2a44; border-radius: 14px; padding: 16px; background: rgba(15,21,38,.8); }
.lossy-card h4 { margin: 0 0 8px; color: #e2e8f0; font-size: 15px; }
.lossy-card .row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dashed #1c2540; font-size: 13px; }
.lossy-card .row b { font-family: 'JetBrains Mono', monospace; color: #f1f5f9; }
.sev { font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.sev.hi { color: #f87171; } .sev.md { color: #fbbf24; }
.side-note { color: #64748b; font-size: 12px; }
.wsplit { display: flex; height: 10px; border-radius: 99px; overflow: hidden; border: 1px solid #1f2a44; margin: 2px 0 6px; }
.wsplit .s { background: linear-gradient(90deg, #38bdf8, #6366f1); }
.wsplit .l { background: linear-gradient(90deg, #f472b6, #fb923c); }
.wlegend { display: flex; justify-content: space-between; font: 600 11.5px 'JetBrains Mono', monospace; color: #cbd5e1; }
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def badge(engine: str) -> str:
    return f'<span class="badge pil">[{esc(engine)}]</span>'


def engine_cell(prof: Profile) -> str:
    out = badge(prof.engine)
    if not prof.available:
        out += '<span class="badge na">UNAVAILABLE</span>'
    return out


def profile_cell(prof: Profile) -> str:
    return f'<span class="pname">{esc(prof.label)}</span><span class="pdesc">{esc(prof.param_desc)}</span>'


def fmt(v, spec: str, dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return dash
    return format(v, spec)


def render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def table(headers: list[tuple[str, str]], rows: list[tuple[str, list[str]]]) -> str:
    """headers: [(label_html, css_class)], rows: [(tr_class, [cell_html...])]."""
    th = "".join(f'<th class="{c}">{h}</th>' for h, c in headers)
    body = []
    for tr_cls, cells in rows:
        if len(cells) == 1:  # full-width row
            body.append(f'<tr class="{tr_cls}"><td colspan="{len(headers)}">{cells[0]}</td></tr>')
            continue
        tds = "".join(f'<td class="{headers[i][1]}">{cell}</td>' for i, cell in enumerate(cells))
        body.append(f'<tr class="{tr_cls}">{tds}</tr>')
    return f'<div class="tbl-wrap"><table class="ib"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'


def bar(value: float, comp: bool = False) -> str:
    cls = "bar comp" if comp else "bar"
    return f'<span class="{cls}"><span style="width:{max(0.0, min(100.0, value)):.1f}%"></span></span><b>{value:.1f}</b>'


def pct(w: float) -> str:
    return f"{w * 100:.0f}%"


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def chk_key(fid: str) -> str:
    return f"chk::{fid}"


def get_workdir() -> str:
    wd = st.session_state.get("workdir")
    if not wd or not os.path.isdir(wd):
        wd = tempfile.mkdtemp(prefix="imagebench_")
        st.session_state["workdir"] = wd
    return wd


def reset_workspace() -> None:
    nonce = st.session_state.get("uploader_nonce", 0) + 1
    wd = st.session_state.get("workdir")
    if wd:
        shutil.rmtree(wd, ignore_errors=True)
    st.session_state.clear()  # also restores weight slider + codec matrix defaults
    st.cache_data.clear()
    st.session_state["uploader_nonce"] = nonce  # new key -> uploader is emptied
    st.session_state["flash"] = "Workspace reset: all cached results, settings, session state and files were flushed."


def set_all_checks(value: bool) -> None:
    for fid in st.session_state.get("results", {}):
        st.session_state[chk_key(fid)] = value


# ---------------------------------------------------------------------------
# Sidebar configuration widgets (pure UI -> cfg dict; no file handling here)
# ---------------------------------------------------------------------------
def render_weight_controls() -> tuple[float, float]:
    st.markdown("#### ⚖️ Scoring Weights")
    w_storage = st.slider(
        "Storage Size Weight", min_value=0.0, max_value=1.0, value=DEFAULT_STORAGE_WEIGHT, step=0.05,
        key="w_storage",
        help="Save Latency Weight is automatically set to 1 − Storage Size Weight. "
             "Changing this only re-scores cached measurements; nothing is re-encoded.",
    )
    w_storage = round(float(w_storage), 4)
    w_latency = round(1.0 - w_storage, 4)
    render_html(
        f'<div class="wsplit"><div class="s" style="width:{w_storage * 100:.2f}%"></div>'
        f'<div class="l" style="width:{w_latency * 100:.2f}%"></div></div>'
        f'<div class="wlegend"><span>Storage {pct(w_storage)}</span><span>Latency {pct(w_latency)}</span></div>'
    )
    return w_storage, w_latency


def render_codec_matrix(caps: dict) -> dict:
    st.markdown("#### 🧬 Codec Matrix")
    d = DEFAULT_CFG
    cfg = dict(d)
    cfg["families"] = st.multiselect(
        "Enabled codec engines", options=list(FAMILIES), default=d["families"], key="cfg_families",
        format_func=lambda f: f"{FAMILIES[f][0]}  [{FAMILIES[f][1]}]"
                              + ("" if caps["family_ok"][f][0] else "  (unavailable)"),
        help="Disabled engines are stripped from every table. Their cached measurements are kept, "
             "so re-enabling them is instant.",
    )
    fams = set(cfg["families"])

    if "png" in fams:
        with st.expander("PNG tiers  ·  [PIL NATIVE]"):
            cfg["png_levels"] = st.multiselect("compress_level values (0-9)", list(range(10)),
                                               default=d["png_levels"], key="cfg_png_levels")
    if "webp_ll" in fams:
        with st.expander("WebP Lossless tiers  ·  [PIL NATIVE]"):
            cfg["webp_methods"] = st.multiselect("method values (0-6)", list(range(7)),
                                                 default=d["webp_methods"], key="cfg_webp_methods")
    if "webp_lossy" in fams:
        with st.expander("WebP Lossy tiers  ·  [PIL NATIVE]"):
            cfg["webp_qualities"] = st.multiselect("Target quality values", QUALITY_OPTIONS,
                                                   default=d["webp_qualities"], key="cfg_webp_q")
            cfg["webp_lossy_method"] = st.slider("Encoder method (0 fast - 6 slow)", 0, 6,
                                                 d["webp_lossy_method"], key="cfg_webp_lossy_method")
    def _na_note(fam: str) -> None:
        ok, reason = caps["family_ok"][fam]
        if not ok:
            st.caption(f"⚠ {reason}")

    def _effort_fmt(e: int) -> str:
        return f"{e} · {JXL_EFFORT_NAMES[e]}" if e in JXL_EFFORT_NAMES else str(e)

    def _dspeed_slider(label_key: str) -> int:
        return st.slider("Decoding speed tier (0 best density - 4 fastest decode)", 0, 4,
                         d[label_key], key=f"cfg_{label_key}")

    # JPEG XL · pillow-jxl — lossless: multiple effort tiers (like PNG levels / WebP methods)
    if "jxl_ll" in fams:
        with st.expander("JPEG XL Lossless tiers  ·  [PIL NATIVE] pillow-jxl"):
            _na_note("jxl_ll")
            cfg["jxl_ll_efforts"] = st.multiselect("effort values (1-9)", list(range(1, 10)),
                                                   default=d["jxl_ll_efforts"], format_func=_effort_fmt,
                                                   key="cfg_jxl_ll_efforts")
            cfg["jxl_ll_dspeed"] = _dspeed_slider("jxl_ll_dspeed")
    # JPEG XL · pillow-jxl — lossy: multiple quality targets + one effort (like WebP lossy)
    if "jxl_lossy" in fams:
        with st.expander("JPEG XL Lossy tiers  ·  [PIL NATIVE] pillow-jxl"):
            _na_note("jxl_lossy")
            cfg["jxl_lossy_qualities"] = st.multiselect("Target quality values", QUALITY_OPTIONS,
                                                        default=d["jxl_lossy_qualities"],
                                                        key="cfg_jxl_lossy_qualities")
            cfg["jxl_lossy_effort"] = st.slider("Encoder effort (1 fast - 9 slow)", 1, 9,
                                                d["jxl_lossy_effort"], key="cfg_jxl_lossy_effort")
            cfg["jxl_lossy_dspeed"] = _dspeed_slider("jxl_lossy_dspeed")
    return cfg


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ImageBench · Native Compression Sandbox",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_html(CSS)

st.session_state.setdefault("uploader_nonce", 0)
st.session_state.setdefault("results", {})
st.session_state.setdefault("failed", {})
results: dict[str, dict] = st.session_state["results"]
failed: dict[str, str] = st.session_state["failed"]

caps = capabilities()

# ---- Sidebar: reset, weights, codec matrix (configuration only) --------------
with st.sidebar:
    st.markdown("### 🗜️ ImageBench Sandbox")
    st.button("🧹 Reset Workspace", type="primary", on_click=reset_workspace,
              help="Flush every cached result, setting, session state entry and uploaded file.")
    st.markdown("---")
    W_STORAGE, W_LATENCY = render_weight_controls()
    st.markdown("---")
    CFG = render_codec_matrix(caps)
    sidebar_list = st.container()

PROFILES: tuple[Profile, ...] = build_profiles(CFG, caps)
RUNNABLE = [p for p in PROFILES if p.available]

# ---- Hero -------------------------------------------------------------------
chips = [
    f'<span class="chip">Python {esc(platform.python_version())}</span>',
    f'<span class="chip ok">Pillow {esc(PIL.__version__)}</span>',
    f'<span class="chip {"ok" if caps["webp_ok"] else "bad"}">WebP {"available" if caps["webp_ok"] else "MISSING"}</span>',
    f'<span class="chip {"ok" if caps["pjxl_fmt"] else "bad"}">{esc(caps["pjxl_info"] if caps["pjxl_fmt"] else "pillow-jxl unavailable")}</span>',
    f'<span class="chip">{esc(platform.machine() or "cpu")} · {os.cpu_count() or "?"} logical cores</span>',
    f'<span class="chip">{len(RUNNABLE)}/{len(PROFILES)} active profiles</span>',
]
render_html(
    '<div class="ib-hero"><h1>ImageBench · Hardware-Native Compression Sandbox</h1>'
    '<p>Every number below is measured live on this machine: real encoders, real files on disk, '
    'real <code>time.perf_counter()</code> timings, full pixel decodes. No models, no estimates. '
    'Re-weighting and toggling engines re-scores cached measurements instantly.</p>'
    f'<div class="ib-chips">{"".join(chips)}</div></div>'
)
for fam in CFG["families"]:
    ok, reason = caps["family_ok"][fam]
    if not ok:
        render_html(f'<div class="callout warn"><h4>⚠ {esc(FAMILIES[fam][0])} engine unavailable</h4>'
                    f'{esc(reason)}<br>Its rows are labelled UNAVAILABLE and excluded from scoring.</div>')

flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash, icon="🧹")

# ---- Upload ------------------------------------------------------------------
uploaded = st.file_uploader(
    "Drop a batch of images (PNG, JPEG, TIFF, BMP, WebP, GIF, …) — each file runs through every active profile",
    type=UPLOAD_TYPES,
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_nonce']}",
)

# ---- File loop: encode only missing (file, profile) pairs ----------------------
current: dict[str, object] = {}
for uf in uploaded or []:
    current[f"{uf.name}|{uf.size}|{getattr(uf, 'file_id', '')}"] = uf

for fid in list(results):  # files removed from the uploader leave the dataset
    if fid not in current:
        results.pop(fid, None)
        st.session_state.pop(chk_key(fid), None)
for fid in list(failed):
    if fid not in current:
        failed.pop(fid, None)

jobs = []
for fid, uf in current.items():
    if fid in failed:
        continue
    have = results.get(fid, {}).get("records", {})
    missing = [p for p in RUNNABLE if p.key not in have]
    if missing:
        jobs.append((fid, uf, missing))

if jobs:
    with st.spinner("Warming up codec libraries (untimed)…"):
        warm_up_codecs()
    workdir = get_workdir()
    total_runs = sum(len(m) for _, _, m in jobs)
    done = 0
    progress = st.progress(0.0)
    status = st.empty()
    t_batch = time.perf_counter()
    for fid, uf, missing in jobs:
        try:
            img, meta = prepare_source(uf.getvalue())  # untimed canonical raster
        except Exception as exc:
            failed[fid] = f"{uf.name}: {type(exc).__name__}: {exc}"
            done += len(missing)
            continue
        entry = results.setdefault(fid, {
            "name": uf.name, "source_bytes": uf.size,
            "megapixels": meta["width"] * meta["height"] / 1e6, "records": {}, **meta,
        })
        st.session_state.setdefault(chk_key(fid), True)
        for prof in missing:
            status.markdown(f"⏱️ **{esc(uf.name)}** → {esc(prof.short)}  ({done + 1}/{total_runs} hardware runs)")
            entry["records"][prof.key] = measure_profile(prof, img, meta, workdir)
            done += 1
            progress.progress(done / total_runs)
        del img
    progress.empty()
    status.empty()
    st.toast(f"Executed {total_runs} new encode+decode run(s) in {time.perf_counter() - t_batch:.2f} s", icon="✅")

for msg in failed.values():
    st.warning(f"Could not decode source image — {msg}")

# ---- Sidebar: dataset toggles --------------------------------------------------
with sidebar_list:
    st.markdown("---")
    st.markdown(f"#### 🗂️ Active dataset · {len(results)} file(s)")
    if results:
        c1, c2 = st.columns(2)
        c1.button("Check all", on_click=set_all_checks, args=(True,))
        c2.button("Uncheck all", on_click=set_all_checks, args=(False,))
        for fid, res in results.items():
            st.session_state.setdefault(chk_key(fid), True)
            st.checkbox(
                f"{res['name']}",
                key=chk_key(fid),
                help=f"{res['width']}×{res['height']} · {res['src_format']} {res['src_mode']} · "
                     f"{res['source_bytes'] / MIB:.3f} MiB source",
            )
    else:
        render_html('<span class="side-note">Upload images to populate the dataset.</span>')
    cached_runs = sum(len(r["records"]) for r in results.values())
    render_html(f'<br><span class="side-note">{cached_runs} measured run(s) cached. Unchecking files, '
                'moving the weight slider or toggling engines never re-encodes — only newly requested '
                '(file, profile) pairs are measured.</span>')

active = [res for fid, res in results.items() if st.session_state.get(chk_key(fid), True)]

# ---- Empty state ----------------------------------------------------------------
if not active or not PROFILES:
    if not PROFILES:
        msg = "No profiles are enabled — select at least one codec engine and tier in the sidebar."
    elif not results:
        msg = "Upload images to begin."
    else:
        msg = "All files are unchecked — tick at least one file in the sidebar."
    render_html(f'<div class="callout info"><h4>ℹ Nothing to show yet</h4>{esc(msg)}</div>')
    if PROFILES:
        st.markdown("#### Active benchmark profile catalogue")
        render_html(table(
            [("#", "num"), ("Profile", ""), ("Engine", ""), ("Mode", "")],
            [("na" if not p.available else "", [str(i + 1), profile_cell(p), engine_cell(p), esc(p.mode.title())])
             for i, p in enumerate(PROFILES)],
        ))
    st.stop()

# ---- Analytics (pure, recomputed every rerun from cache) --------------------------
agg = aggregate(active, RUNNABLE)
alerts = detect_spikes(agg, RUNNABLE)
board, baselines = lossless_leaderboard(agg, RUNNABLE, W_STORAGE, W_LATENCY)

total_src = sum(r["source_bytes"] for r in active)
total_mp = sum(r["megapixels"] for r in active)
measured_ms = sum(sum(b["save"]) + sum(b["load"]) for b in agg.values())
n_runs = sum(len(b["items"]) for b in agg.values())
n_err = sum(b["errors"] for b in agg.values())

k = st.columns(5)
k[0].markdown(kpi("Active images", f"{len(active)} / {len(results)}", "checked in sidebar"), unsafe_allow_html=True)
k[1].markdown(kpi("Source payload", f"{total_src / MIB:.2f} MiB", "uploaded bytes"), unsafe_allow_html=True)
k[2].markdown(kpi("Pixels per profile", f"{total_mp:.2f} MP", f"{len(RUNNABLE)} active profile(s)"),
              unsafe_allow_html=True)
k[3].markdown(kpi("Measured codec time", f"{measured_ms / 1000:.2f} s", f"{n_runs} encode+decode runs"),
              unsafe_allow_html=True)
k[4].markdown(kpi("Spike alerts", str(len(alerts)), f"{n_err} failed run(s)"), unsafe_allow_html=True)
st.write("")

tabs = st.tabs([
    "📊 Aggregated Metrics Ledger",
    f"🚨 Performance Spike Log ({len(alerts)})",
    "🏆 Lossless Architecture Leaderboard",
    "🗜️ Lossy Archive Transformation Index",
    "🔬 Per-File Results",
    "🖥️ Environment",
])

# ---- 1. Ledger --------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Running Average · Minimum · Maximum across all checked images")
    head = (
        '<tr><th rowspan="2">Profile</th><th rowspan="2">Engine</th><th rowspan="2" class="num">N</th>'
        '<th colspan="3" class="grp">File Size (MiB)</th><th colspan="3" class="grp">Save / Encode (ms)</th>'
        '<th colspan="3" class="grp">Load / Decode (ms)</th></tr><tr>'
        + ('<th class="num sep">Avg</th><th class="num">Min</th><th class="num">Max</th>' * 3) + "</tr>"
    )
    body = []
    for prof in PROFILES:
        if not prof.available:
            body.append(f'<tr class="na"><td>{profile_cell(prof)}</td><td>{engine_cell(prof)}</td>'
                        f'<td colspan="10" class="err">Unavailable — {esc(prof.reason)}</td></tr>')
            continue
        b = agg[prof.key]
        s, sv, ld = triple([x / MIB for x in b["bytes"]]), triple(b["save"]), triple(b["load"])
        row = f'<td>{profile_cell(prof)}</td><td>{engine_cell(prof)}</td><td class="num">{len(b["items"])}</td>'
        for t, spec in ((s, ".4f"), (sv, ".2f"), (ld, ".2f")):
            if t is None:
                row += '<td class="num sep muted">—</td><td class="num muted">—</td><td class="num muted">—</td>'
            else:
                row += (f'<td class="num sep"><b>{t[0]:{spec}}</b></td>'
                        f'<td class="num">{t[1]:{spec}}</td><td class="num">{t[2]:{spec}}</td>')
        body.append(f"<tr>{row}</tr>")
        if b["errors"]:
            body.append(f'<tr><td colspan="12" class="err">⚠ {b["errors"]} run(s) failed for {esc(prof.short)} '
                        f'— see Per-File Results.</td></tr>')
    render_html(f'<div class="tbl-wrap"><table class="ib"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>')
    st.caption("Size = os.path.getsize() of the encoded file · Save = perf_counter() around encoder + disk write · "
               "Load = perf_counter() around file open + full pixel decode into RAM.")

# ---- 2. Spike log -------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"#### Flags any image whose save latency or file size exceeds {SPIKE_THRESHOLD * 100:.0f}% "
                "of the active running average for that profile tier")
    if len(active) < 3:
        render_html('<div class="callout info"><h4>ℹ Small sample</h4>With fewer than 3 active images no value can '
                    'mathematically exceed 200% of the mean that includes it. Add more images for spike detection.</div>')
    if not alerts:
        render_html('<div class="callout win"><h4>✅ No performance spikes</h4>All measured save latencies and file '
                    'sizes are within 200% of their tier\'s running average.</div>')
    else:
        render_html(f'<div class="callout alert"><h4>🚨 {len(alerts)} spike(s) detected</h4>'
                    'Sorted by severity (observed value as % of the tier running average).</div>')
        rows = []
        for a in alerts:
            sev = "hi" if a["ratio"] >= 300 else "md"
            rows.append(("", [
                f'<span class="sev {sev}">{"▲▲" if sev == "hi" else "▲"}</span>',
                esc(a["file"]), profile_cell(a["profile"]), engine_cell(a["profile"]),
                esc(a["metric"]), esc(a["observed"]), esc(a["average"]),
                f'<span class="sev {sev}">{a["ratio"]:.0f}%</span>',
            ]))
        render_html(table(
            [("Sev", ""), ("File", ""), ("Profile", ""), ("Engine", ""), ("Metric", ""),
             ("Observed", "num"), ("Running Avg", "num"), ("% of Avg", "num")],
            rows,
        ))

# ---- 3. Lossless leaderboard ------------------------------------------------------
with tabs[2]:
    st.markdown("#### Lossless Architecture Leaderboard — unit-normalised, live-weighted composite score")
    render_html(
        '<div class="callout info"><h4>Scoring method</h4>'
        'Raw MiB and ms are never mixed. For each active lossless tier the average size and average save latency '
        'are mapped onto 0–100 relative to the <b>worst-performing baseline</b> (largest average size → 0, slowest '
        'average save → 0) with the best tier at 100:<br>'
        '<code>score = (worst − tier) / (worst − best) × 100</code><br>'
        f'<b>Composite = {pct(W_STORAGE)} Storage Weight + {pct(W_LATENCY)} Latency Weight</b> '
        '<span class="muted">(adjust live with the sidebar slider)</span></div>'
    )
    na_lossless = [p for p in PROFILES if p.mode == "lossless" and not p.available]
    if not board:
        st.info("No successful lossless runs among the active images and enabled profiles.")
    else:
        bl = baselines
        c1, c2 = st.columns(2)
        c1.markdown(kpi("Worst size baseline", f"{bl['worst_size']:.4f} MiB", bl["worst_size_prof"].label),
                    unsafe_allow_html=True)
        c2.markdown(kpi("Worst save baseline", f"{bl['worst_save']:.2f} ms", bl["worst_save_prof"].label),
                    unsafe_allow_html=True)
        st.write("")
        rows = []
        for i, r in enumerate(board):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
            rows.append(("top" if i == 0 else "", [
                medal, profile_cell(r["profile"]), engine_cell(r["profile"]), str(r["n"]),
                f'{r["size"]:.4f}', f'{r["vs_worst_size"]:.1f}%', f'{r["save"]:.2f}', f'{r["load"]:.2f}',
                bar(r["storage_score"]), bar(r["latency_score"]), bar(r["composite"], comp=True),
            ]))
        for p in na_lossless:
            rows.append(("na", ["—", profile_cell(p), engine_cell(p), "0", "—", "—", "—", "—",
                                f'<span class="err">{esc(p.reason)}</span>', "—", "—"]))
        render_html(table(
            [("Rank", ""), ("Profile", ""), ("Engine", ""), ("N", "num"), ("Avg Size MiB", "num"),
             ("Saved vs Worst", "num"), ("Avg Save ms", "num"), ("Avg Load ms", "num"),
             (f"Storage ({pct(W_STORAGE)})", ""), (f"Latency ({pct(W_LATENCY)})", ""), ("Composite", "")],
            rows,
        ))
        top = board[0]
        render_html(
            f'<div class="callout win"><h4>🏆 Project recommendation: {esc(top["profile"].label)} {badge(top["profile"].engine)}</h4>'
            f'Composite <b>{top["composite"]:.1f}</b>/100 at {pct(W_STORAGE)} storage / {pct(W_LATENCY)} latency · '
            f'average <b>{top["size"]:.4f} MiB</b> ({top["vs_worst_size"]:.1f}% smaller than the worst baseline) · '
            f'average save <b>{top["save"]:.2f} ms</b> · average load <b>{top["load"]:.2f} ms</b>, '
            f'measured over {top["n"]} active image(s).</div>'
        )
        st.caption("Load latency is shown for context only and is not part of the composite score. "
                   "Re-weighting re-sorts cached measurements; no image is re-encoded.")

# ---- 4. Lossy archive index ------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Lossy Archive Transformation Index")
    lossy = [p for p in RUNNABLE if p.mode == "lossy"]
    na_lossy = [p for p in PROFILES if p.mode == "lossy" and not p.available]
    for p in na_lossy:
        render_html(f'<div class="callout warn">{esc(p.label)} {engine_cell(p)} — {esc(p.reason)}</div>')

    if len(lossy) < 2:
        st.info("Enable at least two available lossy profiles (e.g. WebP Lossy and a JPEG XL lossy tier) "
                "to run the side-by-side transformation index.")
    else:
        def _default(pred, fallback):
            for i, p in enumerate(lossy):
                if pred(p):
                    return i
            return fallback

        ia = _default(lambda p: p.family == "webp_lossy", 0)
        ib = _default(lambda p: p.family == "jxl_lossy", 1 if ia == 0 else 0)
        sc1, sc2 = st.columns(2)
        prof_a = sc1.selectbox("Contender A", lossy, index=ia, format_func=lambda p: p.label, key="lossy_a")
        prof_b = sc2.selectbox("Contender B", lossy, index=ib, format_func=lambda p: p.label, key="lossy_b")

        if prof_a.key == prof_b.key:
            st.warning("Pick two different lossy profiles to compare.")
        else:
            # Only compare images where BOTH lossy encoders succeeded (apples to apples).
            paired = [r for r in active
                      if r["records"].get(prof_a.key, {}).get("ok") and r["records"].get(prof_b.key, {}).get("ok")]

            def lossy_stats(prof: Profile) -> dict:
                recs = [r["records"][prof.key] for r in paired]
                enc_total = sum(x["bytes"] for x in recs)
                src_total = sum(r["source_bytes"] for r in paired)
                return {
                    "size": fmean(x["bytes"] for x in recs) / MIB,
                    "save": fmean(x["save_ms"] for x in recs),
                    "load": fmean(x["load_ms"] for x in recs),
                    "enc_total": enc_total, "src_total": src_total,
                    "reclaim": src_total - enc_total,
                    "reclaim_pct": (1 - enc_total / src_total) * 100 if src_total else 0.0,
                    "bpp": enc_total * 8 / max(1, sum(r["width"] * r["height"] for r in paired)),
                }

            if not paired:
                st.info("No active image has successful runs for both selected lossy profiles.")
            else:
                sa, sb = lossy_stats(prof_a), lossy_stats(prof_b)

                def card(prof, s):
                    return (f'<div class="lossy-card"><h4>{esc(prof.label)}</h4>{engine_cell(prof)}'
                            f'<span class="pdesc">{esc(prof.param_desc)}</span>'
                            f'<div class="row"><span>Average file size</span><b>{s["size"]:.4f} MiB</b></div>'
                            f'<div class="row"><span>Average save / encode</span><b>{s["save"]:.2f} ms</b></div>'
                            f'<div class="row"><span>Average load / decode</span><b>{s["load"]:.2f} ms</b></div>'
                            f'<div class="row"><span>Bits per pixel</span><b>{s["bpp"]:.3f}</b></div>'
                            f'<div class="row"><span>Space reclaimed vs originals</span><b>{s["reclaim"] / MIB:+.3f} MiB '
                            f'({s["reclaim_pct"]:+.1f}%)</b></div></div>')

                c1, c2 = st.columns(2)
                c1.markdown(card(prof_a, sa), unsafe_allow_html=True)
                c2.markdown(card(prof_b, sb), unsafe_allow_html=True)
                st.write("")

                def win(a, b, lower_better=True):
                    if abs(a - b) <= 1e-12:
                        return "Tie"
                    return prof_a.short if (a < b) == lower_better else prof_b.short

                def ratio(a, b):
                    return f"{b / a:.3f}×" if a else "—"

                render_html(table(
                    [("Metric", ""), (f"{esc(prof_a.short)} {badge(prof_a.engine)}", "num"),
                     (f"{esc(prof_b.short)} {badge(prof_b.engine)}", "num"), ("B ÷ A", "num"), ("Winner", "")],
                    [
                        ("", ["Average File Size (MiB)", f'{sa["size"]:.4f}', f'{sb["size"]:.4f}',
                              ratio(sa["size"], sb["size"]), esc(win(sa["size"], sb["size"]))]),
                        ("", ["Average Save Time (ms)", f'{sa["save"]:.2f}', f'{sb["save"]:.2f}',
                              ratio(sa["save"], sb["save"]), esc(win(sa["save"], sb["save"]))]),
                        ("", ["Average Load Time (ms)", f'{sa["load"]:.2f}', f'{sb["load"]:.2f}',
                              ratio(sa["load"], sb["load"]), esc(win(sa["load"], sb["load"]))]),
                        ("", ["Total Encoded (MiB)", f'{sa["enc_total"] / MIB:.3f}', f'{sb["enc_total"] / MIB:.3f}',
                              ratio(sa["enc_total"], sb["enc_total"]), esc(win(sa["enc_total"], sb["enc_total"]))]),
                        ("", ["Reclaimed vs Originals (MiB)", f'{sa["reclaim"] / MIB:+.3f}',
                              f'{sb["reclaim"] / MIB:+.3f}', "—",
                              esc(win(sa["reclaim"], sb["reclaim"], lower_better=False))]),
                    ],
                ))

                # Reclaiming disk space is the objective -> size decides, speed is the caveat.
                if sb["size"] < sa["size"]:
                    wp, ws_, lp, ls_ = prof_b, sb, prof_a, sa
                else:
                    wp, ws_, lp, ls_ = prof_a, sa, prof_b, sb
                size_gain = (1 - ws_["size"] / ls_["size"]) * 100 if ls_["size"] else 0.0
                save_ratio = ws_["save"] / ls_["save"] if ls_["save"] else float("nan")
                load_ratio = ws_["load"] / ls_["load"] if ls_["load"] else float("nan")

                def speed_word(r):
                    if r != r or r <= 0:
                        return "n/a"
                    return f"{r:.2f}× slower" if r > 1 else f"{1 / r:.2f}× faster"

                if ws_["reclaim"] > 0:
                    render_html(
                        f'<div class="callout win"><h4>♻️ Legacy disk-space recommendation: {esc(wp.label)} {badge(wp.engine)}</h4>'
                        f'Averages <b>{size_gain:.1f}% smaller</b> files than {esc(lp.short)} on the {len(paired)} paired '
                        f'image(s). Converting them would reclaim <b>{ws_["reclaim"] / MIB:.3f} MiB</b> '
                        f'(<b>{ws_["reclaim_pct"]:.1f}%</b> of the original {ws_["src_total"] / MIB:.3f} MiB). '
                        f'Trade-off: encoding is {speed_word(save_ratio)} and decoding is {speed_word(load_ratio)} than '
                        f'{esc(lp.short)} on this hardware.</div>'
                    )
                else:
                    render_html(
                        f'<div class="callout warn"><h4>⚠ No space to reclaim</h4>Even the smaller lossy output '
                        f'({esc(wp.short)}, {ws_["enc_total"] / MIB:.3f} MiB) is larger than your original files '
                        f'({ws_["src_total"] / MIB:.3f} MiB) — the sources are already compactly encoded (e.g. JPEG). '
                        f'Re-encoding them would grow the archive by {-ws_["reclaim"] / MIB:.3f} MiB. Between the two, '
                        f'{esc(wp.short)} is {size_gain:.1f}% smaller than {esc(lp.short)}.</div>'
                    )
                st.caption("Only images where both selected lossy encoders succeeded are compared. "
                           "'Originals' = the uploaded file bytes.")

        st.markdown("##### All active lossy tiers")
        rows = []
        for p in lossy:
            b = agg[p.key]
            if not b["items"]:
                continue
            rows.append(("", [profile_cell(p), engine_cell(p), str(len(b["items"])),
                              f'{fmean(b["bytes"]) / MIB:.4f}', f'{fmean(b["save"]):.2f}', f'{fmean(b["load"]):.2f}']))
        render_html(table([("Profile", ""), ("Engine", ""), ("N", "num"), ("Avg Size MiB", "num"),
                           ("Avg Save ms", "num"), ("Avg Load ms", "num")], rows))

# ---- 5. Per-file results ------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Raw measurements per file")
    ids = [fid for fid, res in results.items() if st.session_state.get(chk_key(fid), True)]
    sel = st.selectbox("Inspect file", ids, format_func=lambda f: results[f]["name"])
    res = results[sel]
    st.markdown(
        f"`{res['width']}×{res['height']}` · source `{res['src_format']} {res['src_mode']}` → benchmark raster "
        f"`{res['work_mode']}` · original `{res['source_bytes'] / MIB:.4f} MiB`"
    )
    rows = []
    for prof in PROFILES:
        rec = res["records"].get(prof.key)
        if not prof.available:
            rows.append(("na", [profile_cell(prof), engine_cell(prof), "—", "—", "—", "—", "—",
                                f'<span class="err">Unavailable — {esc(prof.reason)}</span>']))
        elif rec is None:
            rows.append(("", [profile_cell(prof), engine_cell(prof), "—", "—", "—", "—", "—", "⏳ pending"]))
        elif rec["ok"]:
            ratio_v = res["source_bytes"] / rec["bytes"] if rec["bytes"] else float("nan")
            rows.append(("", [profile_cell(prof), engine_cell(prof), f'{rec["bytes"]:,}', f'{rec["bytes"] / MIB:.4f}',
                              fmt(ratio_v, ".3f") + "×", f'{rec["save_ms"]:.2f}', f'{rec["load_ms"]:.2f}', "✅"]))
        else:
            rows.append(("", [profile_cell(prof), engine_cell(prof), "—", "—", "—", "—", "—",
                              f'<span class="err">❌ {esc(rec["error"])}</span>']))
    render_html(table(
        [("Profile", ""), ("Engine", ""), ("Bytes", "num"), ("MiB", "num"), ("Src ÷ Out", "num"),
         ("Save ms", "num"), ("Load ms", "num"), ("Status", "")],
        rows,
    ))

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["file", "width", "height", "source_bytes", "profile", "engine", "params",
                 "ok", "bytes", "size_mib", "save_ms", "load_ms", "error"])
    for r in active:
        for prof in RUNNABLE:
            rec = r["records"].get(prof.key)
            if rec is None:
                continue
            wr.writerow([r["name"], r["width"], r["height"], r["source_bytes"], prof.label, f"[{prof.engine}]",
                         prof.param_desc, rec["ok"], rec["bytes"], f'{rec["bytes"] / MIB:.6f}',
                         f'{rec["save_ms"]:.4f}', f'{rec["load_ms"]:.4f}', rec["error"] or ""])
    st.download_button("⬇️ Export all active measurements (CSV)", buf.getvalue().encode("utf-8"),
                       file_name="imagebench_results.csv", mime="text/csv")

# ---- 6. Environment -----------------------------------------------------------------------
with tabs[5]:
    st.markdown("#### Measurement environment")
    env_rows = [
        ("Python", sys.version.split()[0] + f" ({platform.python_implementation()})"),
        ("Platform", platform.platform()),
        ("Processor", platform.processor() or platform.machine()),
        ("Logical cores", str(os.cpu_count())),
        ("Pillow", PIL.__version__),
        ("Pillow WebP support", "yes" if caps["webp_ok"] else "no"),
        ("libwebp", str(features.version("webp") or "n/a")),
        ("zlib (PNG)", str(features.version("zlib") or "n/a")),
        ("libjxl (via pillow-jxl-plugin)", caps["pjxl_info"]),
        ("NumPy", np.__version__),
        ("Streamlit", st.__version__),
        ("Timer", f"time.perf_counter() · resolution {time.get_clock_info('perf_counter').resolution:.1e} s"),
        ("Scoring weights", f"Storage {pct(W_STORAGE)} · Latency {pct(W_LATENCY)}"),
        ("Scratch directory", st.session_state.get("workdir", "—")),
    ]
    render_html(table([("Component", ""), ("Value", "")], [("", [esc(a), esc(b)]) for a, b in env_rows]))
    st.caption("Encoded files are written to the scratch directory on disk, sized, decoded, then deleted. "
               "A one-time untimed warm-up primes each codec so library initialisation is never billed to a "
               "measurement. Threading uses each library's defaults (pillow-jxl encodes with "
               "num_threads=-1, i.e. all logical cores; PNG and WebP encode single-threaded).")
