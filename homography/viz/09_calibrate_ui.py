#!/usr/bin/env python3
"""
viz/09_calibrate_ui.py -- Interactive browser tool to hand-tune camera
intrinsics (and pose, as a visual aid) per view until the known field
wireframe locks onto the real broadcast frame.

Why this exists
-----------------
pipeline/02_search_focal.py now fits intrinsics automatically and well,
but only once a view has enough AprilTag correspondence points to
constrain the fit (see docs/pose_calibration_research.md) -- a view with
just 2-4 tags is fundamentally underdetermined for that, no search
algorithm fixes too little data. This tool is the fallback for exactly
that case: closes the loop by eye instead, the same approach this
project's original (now-retired) manual-fit tool used, rebuilt as real
browser sliders instead of OpenCV trackbars (which can't show a converted
physical value, only a raw tick position -- not a problem here since this
is plain HTML/JS controlling what's drawn).

What it draws, per view, on one grabbed video frame:
  - the field boundary + center line
  - all 32 field AprilTags (green = used in this view's currently solved
    pose, orange = projection only)
  - cyan quads = a fresh single-frame AT3 detection (ground truth,
    baked in at generation time, independent of the sliders)
projected live with a plain pinhole + radial-distortion (k1, k2) camera
model as you drag position/orientation/intrinsics sliders, with a live
RMS reprojection error against the cyan ground-truth quads so you have a
number to minimize, not just an eyeball fit.

Workflow
--------
1. Generate + open this page.
2. Per view, drag sliders until the wireframe locks onto the real tags
   and field edges, using the live RMS reprojection number as your guide.
3. Copy the exported K/dist JSON (bottom of the sidebar) into
   data/calibration/<stem>_intrinsics.json, per view.
4. Re-run pipeline/03_solve_pose.py -- it re-solves the actual pose via
   solvePnP from the (now-correct) intrinsics, which is more rigorous
   than trusting the hand-tuned position/orientation sliders directly.
   Then regenerate viz/03_overlay.py's images / viz/field3d.html to
   confirm it actually improved.

Usage
-----
  python viz/09_calibrate_ui.py --video match.mp4
  python viz/09_calibrate_ui.py --video match.mp4 --frame 2580 --open

Install: pip install opencv-python numpy pupil-apriltags
"""

import argparse, base64, json, math, pathlib, sys, webbrowser
import numpy as np
import cv2
from pupil_apriltags import Detector as AT3Detector

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR      = DATA_DIR / "field"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"
OUT_DIR        = pathlib.Path(__file__).parent

TAG_SIZE_M = 0.1651
TAG_HALF_M = TAG_SIZE_M / 2.0


# ---------------------------------------------------------------------------
# Geometry (matches pipeline/03_solve_pose.py's conventions)
# ---------------------------------------------------------------------------

def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-9:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


def tag_corners_field(tag: dict, half: float = TAG_HALF_M) -> np.ndarray:
    # Order is BL,BR,TR,TL -- matches AT3's real det.corners winding, not
    # the "obvious" TL,TR,BR,BL reading. See pipeline/03_solve_pose.py's
    # tag_corners_field for how this was confirmed (~20px off vs ~1-2px).
    local = np.array([[0, -half, -half], [0,  half, -half],
                      [0,  half,  half], [0, -half,  half]])
    R = quat_to_rot(tag["qw"], tag["qx"], tag["qy"], tag["qz"])
    t = np.array([tag["x"], tag["y"], tag["z"]])
    return (R @ local.T).T + t


def seed_ypr_from_rvec(rvec) -> tuple[float, float, float]:
    """
    Extract (yaw, pitch, roll) in THIS tool's own convention (see the JS
    buildBasis()) from a solved rvec -- just a seed for the sliders, this
    tool's math never needs to match pipeline/03_solve_pose.py's
    rvec_to_ypr exactly since intrinsics (not pose) is the thing actually
    fed back into the pipeline; see module docstring.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    fwd   = R[2, :]   # camera local +Z in world frame
    right = R[0, :]   # camera local +X in world frame
    pitch = math.degrees(math.asin(np.clip(fwd[2], -1.0, 1.0)))
    yaw   = math.degrees(math.atan2(fwd[1], fwd[0]))
    world_up = np.array([0.0, 0.0, 1.0])
    right0_raw = np.cross(fwd, world_up)
    n = np.linalg.norm(right0_raw)
    if n < 1e-6:
        roll = 0.0
    else:
        right0 = right0_raw / n
        up0    = np.cross(right0, fwd)
        roll   = math.degrees(math.atan2(np.dot(right, up0), np.dot(right, right0)))
    return round(yaw, 2), round(pitch, 2), round(roll, 2)


# ---------------------------------------------------------------------------
# Fresh single-frame detection (ground truth, baked in at generation time)
# ---------------------------------------------------------------------------

def _make_detector() -> AT3Detector:
    return AT3Detector(families="tag36h11", nthreads=4, quad_decimate=1.0,
                       quad_sigma=0.0, refine_edges=1, decode_sharpening=1.25)


def detect_in_crop(crop: np.ndarray, detector: AT3Detector) -> list:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return [{"id": int(d.tag_id), "corners": d.corners.round(2).tolist()}
            for d in detector.detect(gray)]


# ---------------------------------------------------------------------------
# Per-view data assembly
# ---------------------------------------------------------------------------

def build_view_data(view_name: str, box: list, frame_bgr: np.ndarray,
                    pose: dict | None, K: np.ndarray, dist: np.ndarray,
                    detector: AT3Detector) -> dict:
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1, x0:x1]
    h, w = crop.shape[:2]

    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        sys.exit(f"[error] could not encode frame crop for {view_name}")
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    if pose is not None:
        pos = pose["camera_position_field_m"]
        yaw, pitch, roll = seed_ypr_from_rvec(pose["rvec"])
        used_ids = sorted(int(t) for t in pose.get("tag_residuals", {}))
    else:
        pos = [K[0, 2] / K[0, 0] * 0 + 8.0, 4.0, 3.0]  # arbitrary, no seed available
        yaw, pitch, roll = 90.0, 0.0, 0.0
        used_ids = []

    return {
        "name": view_name, "image": image_b64, "w": w, "h": h,
        "detected": detect_in_crop(crop, detector),
        "used_ids": used_ids,
        "seed": {
            "x": round(pos[0], 3), "y": round(pos[1], 3), "z": round(pos[2], 3),
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "fx": round(float(K[0, 0]), 1),
            "cx": round(float(K[0, 2]), 1), "cy": round(float(K[1, 2]), 1),
            "k1": round(float(dist[0]), 4) if len(dist) > 0 else 0.0,
            "k2": round(float(dist[1]), 4) if len(dist) > 1 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Intrinsics loader -- same fallback logic as pipeline/03_solve_pose.py
# ---------------------------------------------------------------------------

def load_or_estimate_K(intrinsics_path: pathlib.Path, view_name: str,
                       view_box: list, fov_deg: float):
    if intrinsics_path.exists():
        intr = json.loads(intrinsics_path.read_text())
        if view_name in intr:
            return (np.array(intr[view_name]["K"], dtype=np.float64),
                    np.array(intr[view_name]["dist"], dtype=np.float64))
    x0, y0, x1, y1 = view_box
    w, h = x1 - x0, y1 - y0
    f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render_html(field_data: dict, views_data: list, video_name: str, frame_idx: int) -> str:
    tags_js = []
    for tid_str, t in sorted(field_data["tags"].items(), key=lambda kv: int(kv[0])):
        corners = tag_corners_field(t).round(4).tolist()
        tags_js.append({"id": int(tid_str), "corners": corners})

    data_blob = {
        "video": video_name, "frame": frame_idx,
        "field": {"length": field_data.get("field_length_m", 16.541),
                  "width": field_data.get("field_width_m", 8.069),
                  "tags": tags_js},
        "views": views_data,
    }
    data_json = json.dumps(data_blob)

    return HTML_SHELL.replace("__DATA_JSON__", data_json)


HTML_SHELL = r"""<title>Lens Calibration</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #0a0f0c;
  --panel:   #101815;
  --border:  #203026;
  --field:   #0c1712;
  --text:    #a9c4af;
  --dim:     #4c6b55;
  --hl:      #eef7ef;
  --accent:  #e0a83f;
  --accent-dim: #7a5f28;
  --good:    #39e588;
  --bad:     #ff5f5f;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  --sans: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="light"]) {
    --bg:#eef3ee; --panel:#dfe9e0; --border:#b4c9b8; --field:#cfe1d2;
    --text:#233a29; --dim:#5c7a63; --hl:#0d1a10; --accent:#a9701a; --accent-dim:#c99b52;
    --good:#1c8a4e; --bad:#c23b3b;
  }
}
:root[data-theme="light"] {
  --bg:#eef3ee; --panel:#dfe9e0; --border:#b4c9b8; --field:#cfe1d2;
  --text:#233a29; --dim:#5c7a63; --hl:#0d1a10; --accent:#a9701a; --accent-dim:#c99b52;
  --good:#1c8a4e; --bad:#c23b3b;
}
:root[data-theme="dark"] {
  --bg:#0a0f0c; --panel:#101815; --border:#203026; --field:#0c1712;
  --text:#a9c4af; --dim:#4c6b55; --hl:#eef7ef; --accent:#e0a83f; --accent-dim:#7a5f28;
  --good:#39e588; --bad:#ff5f5f;
}

html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); }
body { display: flex; overflow: hidden; }

#sidebar {
  width: 340px; min-width: 340px; height: 100%; overflow-y: auto;
  background: var(--panel); border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
}
#viewport { flex: 1; display: flex; flex-direction: column; min-width: 0; }

.tabs { display: flex; border-bottom: 1px solid var(--border); }
.tab {
  flex: 1; padding: 12px 6px; text-align: center; cursor: pointer;
  font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--dim); border-bottom: 2px solid transparent; user-select: none;
}
.tab.active { color: var(--hl); border-bottom-color: var(--accent); }
.tab:hover { color: var(--text); }

.group { border-bottom: 1px solid var(--border); padding: 14px 16px; }
.group h3 {
  font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--dim); margin-bottom: 10px; font-weight: 600;
}
.row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.row:last-child { margin-bottom: 0; }
.row label { width: 30px; font-size: 11px; color: var(--dim); font-family: var(--mono); flex-shrink: 0; }
.row input[type="range"] {
  flex: 1; accent-color: var(--accent); height: 4px; cursor: pointer;
}
.row .val {
  width: 64px; text-align: right; font-family: var(--mono); font-size: 11px;
  color: var(--hl); font-variant-numeric: tabular-nums; flex-shrink: 0;
}

.status {
  display: flex; align-items: center; gap: 14px; padding: 10px 16px;
  background: var(--panel); border-bottom: 1px solid var(--border);
  font-family: var(--mono); font-size: 12px; flex-shrink: 0;
}
.status .metric { display: flex; align-items: baseline; gap: 6px; }
.status .metric .n { font-size: 15px; font-variant-numeric: tabular-nums; }
.status .metric .u { color: var(--dim); font-size: 10px; }
.status .spacer { flex: 1; }
.legend { display: flex; gap: 12px; font-size: 10px; color: var(--dim); }
.legend span { display: inline-flex; align-items: center; gap: 4px; }
.legend i { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }

#canvasWrap { flex: 1; position: relative; background: var(--field); overflow: hidden; }
canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

.actions { padding: 12px 16px; display: flex; gap: 8px; border-bottom: 1px solid var(--border); }
button {
  flex: 1; background: var(--field); color: var(--text); border: 1px solid var(--border);
  border-radius: 4px; padding: 7px 10px; font-size: 11px; font-family: var(--sans);
  cursor: pointer; letter-spacing: 0.02em;
}
button:hover { border-color: var(--accent); color: var(--hl); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
button.primary { background: var(--accent-dim); border-color: var(--accent); color: var(--hl); }

.export { padding: 14px 16px; }
.export h3 {
  font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--dim); margin-bottom: 8px;
}
.export textarea {
  width: 100%; height: 150px; resize: vertical; background: var(--field);
  color: var(--good); border: 1px solid var(--border); border-radius: 4px;
  font-family: var(--mono); font-size: 10.5px; padding: 8px; line-height: 1.5;
}
.hint { font-size: 10.5px; color: var(--dim); line-height: 1.5; padding: 0 16px 14px; }
</style>

<div id="sidebar">
  <div class="tabs" id="tabs"></div>

  <div class="actions">
    <button id="resetBtn">Reset to seed</button>
    <button id="copyBtn" class="primary">Copy K/dist JSON</button>
  </div>

  <div class="group">
    <h3>Position (m)</h3>
    <div class="row"><label>x</label><input type="range" id="s_x"><span class="val" id="v_x"></span></div>
    <div class="row"><label>y</label><input type="range" id="s_y"><span class="val" id="v_y"></span></div>
    <div class="row"><label>z</label><input type="range" id="s_z"><span class="val" id="v_z"></span></div>
  </div>

  <div class="group">
    <h3>Orientation (deg)</h3>
    <div class="row"><label>yaw</label><input type="range" id="s_yaw"><span class="val" id="v_yaw"></span></div>
    <div class="row"><label>pitch</label><input type="range" id="s_pitch"><span class="val" id="v_pitch"></span></div>
    <div class="row"><label>roll</label><input type="range" id="s_roll"><span class="val" id="v_roll"></span></div>
  </div>

  <div class="group">
    <h3>Intrinsics (px)</h3>
    <div class="row"><label>fx</label><input type="range" id="s_fx"><span class="val" id="v_fx"></span></div>
    <div class="row"><label>cx</label><input type="range" id="s_cx"><span class="val" id="v_cx"></span></div>
    <div class="row"><label>cy</label><input type="range" id="s_cy"><span class="val" id="v_cy"></span></div>
  </div>

  <div class="group">
    <h3>Radial distortion</h3>
    <div class="row"><label>k1</label><input type="range" id="s_k1"><span class="val" id="v_k1"></span></div>
    <div class="row"><label>k2</label><input type="range" id="s_k2"><span class="val" id="v_k2"></span></div>
  </div>

  <div class="export">
    <h3>Export (this view)</h3>
    <textarea id="exportBox" readonly></textarea>
  </div>
  <p class="hint">
    Drag until the green/orange wireframe locks onto the real tags and the
    field edge, using the reprojection number (top bar) as your guide --
    it's measured against the cyan quads, a fresh detection on this exact
    frame. Then copy the JSON above into
    data/calibration/&lt;stem&gt;_intrinsics.json for this view and
    re-run pipeline/03_solve_pose.py to get a properly solved pose from
    the corrected intrinsics.
  </p>
</div>

<div id="viewport">
  <div class="status">
    <div class="metric"><span class="n" id="rmsVal">--</span><span class="u">px rms (vs fresh detection)</span></div>
    <div class="spacer"></div>
    <div class="legend">
      <span><i style="background:#ff8a3d"></i>boundary</span>
      <span><i style="background:#39e588"></i>used in pose</span>
      <span><i style="background:#ffb347"></i>projection only</span>
      <span><i style="background:#4dd8ff"></i>fresh detection</span>
    </div>
  </div>
  <div id="canvasWrap"><canvas id="c"></canvas></div>
</div>

<script>
const DATA = __DATA_JSON__;

const SLIDER_SPECS = {
  x:     { range: 15,  step: 0.02 },
  y:     { range: 15,  step: 0.02 },
  z:     { range: 8,   step: 0.01 },
  yaw:   { min: 0, max: 360, step: 0.05 },
  pitch: { min: -89, max: 89, step: 0.05 },
  roll:  { min: -45, max: 45, step: 0.05 },
  fx:    { min: 200, max: 3000, step: 0.5 },
  cx:    { rangeFrac: 0.6, step: 0.5 },   // seed +/- 60% of seed value
  cy:    { rangeFrac: 0.6, step: 0.5 },
  k1:    { min: -0.6, max: 0.6, step: 0.001 },
  k2:    { min: -0.6, max: 0.6, step: 0.001 },
};
const PARAM_KEYS = Object.keys(SLIDER_SPECS);

let currentView = null;
let params = {};
const images = {};   // name -> HTMLImageElement (loaded once)

function sliderBounds(key, seedVal) {
  const spec = SLIDER_SPECS[key];
  if ('min' in spec) return [spec.min, spec.max];
  if ('range' in spec) return [seedVal - spec.range, seedVal + spec.range];
  if ('rangeFrac' in spec) {
    const pad = Math.max(Math.abs(seedVal) * spec.rangeFrac, 50);
    return [seedVal - pad, seedVal + pad];
  }
  return [0, 1];
}

function setupSlidersForView(view) {
  params = { ...view.seed };
  for (const key of PARAM_KEYS) {
    const [lo, hi] = sliderBounds(key, view.seed[key]);
    const el = document.getElementById('s_' + key);
    el.min = lo; el.max = hi; el.step = SLIDER_SPECS[key].step;
    el.value = view.seed[key];
  }
  updateReadouts();
}

function updateReadouts() {
  for (const key of PARAM_KEYS) {
    const decimals = (key === 'k1' || key === 'k2') ? 4 : (key === 'x' || key === 'y' || key === 'z') ? 3 : 1;
    document.getElementById('v_' + key).textContent = params[key].toFixed(decimals);
  }
}

// ---------------------------------------------------------------------------
// Camera model -- plain pinhole + 2-term radial distortion. Self-consistent;
// does not need to match pipeline/03_solve_pose.py's rvec convention (see
// module docstring -- only fx/cx/cy/k1/k2 get fed back into the pipeline).
// ---------------------------------------------------------------------------

function toRad(d) { return d * Math.PI / 180; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm(a) { const l = Math.hypot(...a); return l < 1e-9 ? [0,0,1] : a.map(x => x/l); }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }

function buildBasis(p) {
  const yaw = toRad(p.yaw), pitch = toRad(p.pitch), roll = toRad(p.roll);
  const fwd = [Math.cos(yaw)*Math.cos(pitch), Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch)];
  const worldUp = [0, 0, 1];
  let right0 = cross(fwd, worldUp);
  if (Math.hypot(...right0) < 1e-6) right0 = [1, 0, 0];
  right0 = norm(right0);
  const up0 = cross(right0, fwd);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const right = [0,1,2].map(i => right0[i]*cr + up0[i]*sr);
  const up    = [0,1,2].map(i => -right0[i]*sr + up0[i]*cr);
  return { fwd, right, up, pos: [p.x, p.y, p.z] };
}

function project(pointField, p, basis) {
  const d = sub(pointField, basis.pos);
  const camZ = dot(d, basis.fwd);
  if (camZ <= 0.05) return null;
  const camX = dot(d, basis.right);
  const camY = -dot(d, basis.up);
  let xn = camX / camZ, yn = camY / camZ;
  const r2 = xn*xn + yn*yn;
  const rad = 1 + p.k1*r2 + p.k2*r2*r2;
  xn *= rad; yn *= rad;
  return [p.fx*xn + p.cx, p.fx*yn + p.cy];
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvasWrap');

function fitCanvas(view) {
  const rect = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const scale = Math.min(rect.width / view.w, rect.height / view.h);
  return { dpr, scale,
    ox: (rect.width - view.w * scale) / 2, oy: (rect.height - view.h * scale) / 2 };
}

function drawPoly(pts, layout, color, lw) {
  ctx.beginPath();
  pts.forEach((pt, i) => {
    const x = (pt[0] * layout.scale + layout.ox) * layout.dpr;
    const y = (pt[1] * layout.scale + layout.oy) * layout.dpr;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.strokeStyle = color; ctx.lineWidth = lw * layout.dpr; ctx.stroke();
}

function render() {
  const view = DATA.views.find(v => v.name === currentView);
  if (!view) return;
  const img = images[view.name];
  const layout = fitCanvas(view);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img) {
    ctx.drawImage(img, layout.ox * layout.dpr, layout.oy * layout.dpr,
                  view.w * layout.scale * layout.dpr, view.h * layout.scale * layout.dpr);
  }

  const basis = buildBasis(params);
  const fl = DATA.field.length, fw = DATA.field.width;

  // field boundary + center line
  const boundary = [[0,0,0],[fl,0,0],[fl,fw,0],[0,fw,0]]
    .map(pt => project(pt, params, basis));
  if (boundary.every(Boolean)) drawPoly(boundary, layout, '#ff8a3d', 2);
  const centerA = project([fl/2, 0, 0], params, basis);
  const centerB = project([fl/2, fw, 0], params, basis);
  if (centerA && centerB) drawPoly([centerA, centerB], layout, '#ff8a3d99', 1);

  // field tags
  const usedSet = new Set(view.used_ids);
  let sumSq = 0, nErr = 0;
  const detectedById = {};
  view.detected.forEach(d => { detectedById[d.id] = d.corners; });

  DATA.field.tags.forEach(tag => {
    const uv = tag.corners.map(c => project(c, params, basis));
    if (!uv.every(Boolean)) return;
    const seen = usedSet.has(tag.id);
    drawPoly(uv, layout, seen ? '#39e588' : '#ffb347', seen ? 2 : 1);

    const fresh = detectedById[tag.id];
    if (fresh) {
      fresh.forEach((fp, i) => {
        const dx = uv[i][0] - fp[0], dy = uv[i][1] - fp[1];
        sumSq += dx*dx + dy*dy; nErr++;
      });
    }
  });

  // fresh detections (cyan ground truth)
  view.detected.forEach(d => drawPoly(d.corners, layout, '#4dd8ff', 1));

  document.getElementById('rmsVal').textContent = nErr > 0 ? Math.sqrt(sumSq / nErr).toFixed(1) : '--';
  updateExport(view);
}

function updateExport(view) {
  const K = [[params.fx, 0, params.cx], [0, params.fx, params.cy], [0, 0, 1]];
  const dist = [params.k1, params.k2, 0, 0, 0];
  const snippet = {};
  snippet[view.name] = { K, dist };
  document.getElementById('exportBox').value = JSON.stringify(snippet, null, 2);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function loadImage(view) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => { images[view.name] = img; resolve(); };
    img.src = 'data:image/jpeg;base64,' + view.image;
  });
}

async function selectView(name) {
  currentView = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.name === name));
  const view = DATA.views.find(v => v.name === name);
  if (!images[name]) await loadImage(view);
  setupSlidersForView(view);
  render();
}

function init() {
  const tabsEl = document.getElementById('tabs');
  DATA.views.forEach((v, i) => {
    const tab = document.createElement('div');
    tab.className = 'tab'; tab.textContent = v.name; tab.dataset.name = v.name;
    tab.addEventListener('click', () => selectView(v.name));
    tabsEl.appendChild(tab);
  });

  PARAM_KEYS.forEach(key => {
    document.getElementById('s_' + key).addEventListener('input', e => {
      params[key] = parseFloat(e.target.value);
      updateReadouts();
      render();
    });
  });

  document.getElementById('resetBtn').addEventListener('click', () => {
    const view = DATA.views.find(v => v.name === currentView);
    setupSlidersForView(view);
    render();
  });

  document.getElementById('copyBtn').addEventListener('click', () => {
    const box = document.getElementById('exportBox');
    box.select();
    navigator.clipboard?.writeText(box.value).catch(() => {});
  });

  window.addEventListener('resize', render);

  selectView(DATA.views[0].name);
}
init();
</script>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--frame", type=int, metavar="N",
                    help="video frame index to grab (default: middle of the video)")
    ap.add_argument("--tags",  metavar="PATH")
    ap.add_argument("--poses", metavar="PATH")
    ap.add_argument("--fov-deg", type=float, default=70.0,
                    help="fallback horizontal FOV if no intrinsics/pose found "
                         "(default: %(default)s)")
    ap.add_argument("--out", metavar="PATH",
                    help="output path (default: viz/<stem>_calibrate.html)")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    field_path = FIELD_DIR / f"{args.year}_tags.json"
    if not field_path.exists():
        sys.exit(f"[error] field layout not found: {field_path}")
    field_data = json.loads(field_path.read_text())

    stem = pathlib.Path(args.video).stem
    tags_path = pathlib.Path(args.tags) if args.tags else DETECTIONS_DIR / f"{stem}_tags.json"
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    tags_data = json.loads(tags_path.read_text())

    poses_path = pathlib.Path(args.poses) if args.poses else DETECTIONS_DIR / f"{stem}_poses.json"
    poses = json.loads(poses_path.read_text()) if poses_path.exists() else {}

    intrinsics_path = CALIB_DIR / f"{stem}_intrinsics.json"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.frame if args.frame is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[error] could not read frame {frame_idx} from {args.video}")
    print(f"[video] using frame {frame_idx}/{total}", file=sys.stderr)

    detector = _make_detector()
    views_out = []
    for vname, vinfo in tags_data.get("views", {}).items():
        K, dist = load_or_estimate_K(intrinsics_path, vname, vinfo["box"], args.fov_deg)
        vdata = build_view_data(vname, vinfo["box"], frame, poses.get(vname), K, dist, detector)
        views_out.append(vdata)
        print(f"  [{vname}] seed fx={vdata['seed']['fx']:.0f}px  "
              f"{len(vdata['detected'])} fresh detection(s) baked in", file=sys.stderr)

    if not views_out:
        sys.exit("[error] no views found in tags JSON")

    html = render_html(field_data, views_out, pathlib.Path(args.video).name, frame_idx)
    out_path = pathlib.Path(args.out) if args.out else OUT_DIR / f"{stem}_calibrate.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[out] {out_path}", file=sys.stderr)

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
