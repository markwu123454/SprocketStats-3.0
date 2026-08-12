#!/usr/bin/env python3
"""
Step 6 (interactive) — calibrate a video's camera-to-field mapping and
preview inference detections mapped onto a flat, undistorted field.

Real field builds and camera lenses are never perfectly planar/pinhole, so a
single 4-corner homography systematically misplaces points away from the
corners. This tool fits a thin-plate spline (TPS) over an arbitrary set of
control points instead of a fixed 4-point homography, so you can correct
that residual bend by adding more known points:

  - CORNERS (4, required): the field boundary, dragged on the LEFT
    (frame) panel only — their field-side position is fixed to the canvas
    corners.
  - FIELD ELEMENTS: known landmarks (line intersections, tag corners, hub
    corners, ...) loaded ONLY from data/frc/field_elements.json, digitized
    ahead of time with 07_field_elements_editor.py -- there is no in-app
    way to add one. Their field-side position is pre-placed on the RIGHT
    panel; drag each one's LEFT-panel half onto where it appears in the
    frame. A --year with nothing saved in that file is a hard error at
    startup (run the editor first) rather than silently running with zero
    elements.
  - LOCKED DOTS (any number, optional): right-click a live detection dot
    on the left panel to pin it — it freezes as a control point at its
    current frame position and current projected field position. Right-
    click it again to unlock.

All control points feed one TPS fit. Dragging any of them re-solves it,
which reshapes where the *other*, still-unlocked ("loose") detections land
— locked points and elements are hit exactly, everything else is smoothly
interpolated between them. More points = tighter local correction.

Corners + elements + locked dots + field image path are cached per-video
to a JSON sidecar next to the video file, so re-running on the same file
resumes the last calibration.

Requires: tkinterdnd2 (pip) for real OS drag-and-drop of the field image;
without it, use the "Load field image" button instead. yt-dlp/ffmpeg only
needed if you pass a YouTube URL instead of a local file.

Controls:
  Left / Right           step one frame
  slider                 scrub anywhere in the clip
  drag a marker          adjust calibration (corners: left panel only;
                          elements/locked dots: whichever panel they're on)
  right-click a dot      lock/unlock it as a control point (left panel)
  right-click a marker   remove that element instance / unlock that locked dot
  b                      cycle right-panel background: field / warped video / blend
  s                      save calibration now (also autosaves on release)
  q / Esc                quit

Usage:
  python 06_field_calibrate.py video.mp4 --year 2026
  python 06_field_calibrate.py "https://www.youtube.com/watch?v=XXXX" --year 2026
  python 06_field_calibrate.py video.mp4 --field-width-in 317.7 --field-length-in 651.2
"""
import argparse, json, pathlib
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from model import HeatmapNet, decode_peaks, CLASSES

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAVE_DND = True
except ImportError:
    HAVE_DND = False

ROOT = pathlib.Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = ROOT / "data/frc/youtube_cache"

COLOR_BGR = {"blue": (255, 120, 0), "red": (0, 0, 255)}
CH_TO_CLS = {i: c for i, c in enumerate(CLASSES)}
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
CORNER_COLOR = {"TL": (255, 255, 0), "TR": (0, 255, 255), "BR": (255, 0, 255), "BL": (0, 255, 0)}
ELEMENT_COLOR = (0, 165, 255)     # orange
LOCKED_COLOR = (255, 255, 255)    # white

# Real-world FRC field footprint per year, inches, (width, length).
# Only fill in years actually verified against the game manual / field
# dimension drawings -- everything else falls back to STANDARD_FALLBACK_IN
# with a printed warning, rather than silently guessing.
FIELD_DIMS_IN = {
    2026: (317.7, 651.2),   # REBUILT, official field dimension drawings
}
STANDARD_FALLBACK_IN = (324.0, 648.0)  # ~27x54ft "standard" FRC footprint used pre-2026

# Field elements are digitized with 07_field_elements_editor.py and read
# ONLY from here: {"<year>": [{"name": str, "points": [[x_frac,y_frac], ...]}]},
# x_frac/y_frac as fractions of field LENGTH/WIDTH. There is no in-app way
# to add elements -- a year with nothing saved here is a hard error (run
# the editor first), rather than silently continuing with zero elements.
FIELD_ELEMENTS_PATH = ROOT / "data/frc/field_elements.json"


def elements_for_year(year):
    """(name, x_frac, y_frac) tuples, one per polygon vertex -- each vertex
    becomes its own independently-draggable calibration element. Raises if
    this year has nothing digitized yet."""
    if not FIELD_ELEMENTS_PATH.exists():
        raise SystemExit(f"[abort] {FIELD_ELEMENTS_PATH} does not exist. "
                         f"Run 07_field_elements_editor.py --year {year} first.")
    data = json.loads(FIELD_ELEMENTS_PATH.read_text())
    entries = data.get(str(year))
    if not entries:
        raise SystemExit(f"[abort] no field elements saved for year={year} in {FIELD_ELEMENTS_PATH}. "
                         f"Run 07_field_elements_editor.py --year {year} first.")
    out = []
    for e in entries:
        for i, (xf, yf) in enumerate(e["points"]):
            out.append((f"{e['name']}-P{i}", xf, yf))
    return out


# ---------------------------------------------------------------------
# Download (same cache convention as 05_youtube_view.py, so a video
# already pulled for playback is reused here without re-downloading)
# ---------------------------------------------------------------------
def is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def download_video(url, cookies=None):
    import yt_dlp
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/best",
        "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    if cookies:
        ydl_opts["cookiefile"] = str(cookies)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid = info["id"]
    candidates = [p for p in CACHE_DIR.glob(f"{vid}.*")
                 if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not candidates:
        raise SystemExit(f"[abort] download finished but no video file found for id={vid}")
    path = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"[download] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# ---------------------------------------------------------------------
# Model (cfg trusted from the checkpoint, same convention as 04/05)
# ---------------------------------------------------------------------
def load_model_and_cfg(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["cfg"]
    hm = cfg["heatmap"]
    model = HeatmapNet(cfg["train"]["backbone"], len(CLASSES), hm["output_stride"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[load] {ckpt_path.name} epoch={ck.get('epoch')} backbone={cfg['train']['backbone']}")
    return model, hm


@torch.no_grad()
def predict(model, hm, img_bgr, threshold):
    H0, W0 = img_bgr.shape[:2]
    in_h, in_w = hm["input_size"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (in_w, in_h))
    t = torch.from_numpy(img_r).permute(2, 0, 1).float().to(DEVICE) / 255.0
    t = (t - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)) \
        / torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
    logits = model(t.unsqueeze(0))
    dets = decode_peaks(logits, threshold, hm["nms_kernel"], hm["max_instances"])[0]
    out = []
    sx, sy = W0 / in_w, H0 / in_h
    for x_hm, y_hm, c, score in dets:
        x_in = x_hm * hm["output_stride"]
        y_in = y_hm * hm["output_stride"]
        out.append((x_in * sx, y_in * sy, CH_TO_CLS[c], score))
    return out


def resolve_field_dims(year, w_in, l_in):
    if w_in is not None and l_in is not None:
        return w_in, l_in
    if year in FIELD_DIMS_IN:
        return FIELD_DIMS_IN[year]
    print(f"[field] no verified dimensions for year={year}; falling back to "
          f"{STANDARD_FALLBACK_IN[0]:.0f}x{STANDARD_FALLBACK_IN[1]:.0f}in. "
          f"Pass --field-width-in/--field-length-in to override.")
    return STANDARD_FALLBACK_IN


# ---------------------------------------------------------------------
# Thin-plate spline: smooth 2D->2D interpolant that passes exactly through
# every control point and bends gently elsewhere. Used instead of a rigid
# 4-point homography so extra control points can locally soak up lens/
# field-build distortion instead of only ever fitting one flat rectangle.
# ---------------------------------------------------------------------
class TPS:
    def __init__(self, src, dst, reg=1e-4):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        n = src.shape[0]
        r = np.sqrt(((src[:, None, :] - src[None, :, :]) ** 2).sum(-1))
        K = np.where(r > 1e-9, r ** 2 * np.log(r, out=np.zeros_like(r), where=r > 1e-9), 0.0)
        K += reg * np.eye(n)
        P = np.hstack([np.ones((n, 1)), src])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        Y = np.zeros((n + 3, 2))
        Y[:n] = dst
        self.params = np.linalg.solve(L, Y)
        self.src = src

    def apply(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        r = np.sqrt(((pts[:, None, :] - self.src[None, :, :]) ** 2).sum(-1))
        U = np.where(r > 1e-9, r ** 2 * np.log(r, out=np.zeros_like(r), where=r > 1e-9), 0.0)
        basis = np.hstack([np.ones((pts.shape[0], 1)), pts])
        return U @ self.params[:len(self.src)] + basis @ self.params[len(self.src):]


def fit_tps(src, dst, reg=1e-4):
    """None if there aren't enough (non-degenerate) points to fit."""
    if len(src) < 4:
        return None
    try:
        return TPS(src, dst, reg=reg)
    except np.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------
# Calibration sidecar
# ---------------------------------------------------------------------
def calib_path_for(video_path):
    return video_path.with_suffix(".field_calib.json")


class Calibrator:
    MARKER_RADIUS = 8       # px, in displayed (scaled) coordinates
    HIT_RADIUS = 14
    LOCK_RADIUS = 16        # px, distance to snap a right-click onto a live detection
    MAX_PANEL_W = 720

    def __init__(self, video_path, model, hm, threshold, year, field_w_in, field_l_in,
                 builtin_elements, field_image_path=None):
        self.video_path = video_path
        self.model, self.hm, self.threshold = model, hm, threshold
        self.year = year
        self.field_w_in, self.field_l_in = field_w_in, field_l_in
        self.calib_path = calib_path_for(video_path)

        long_in, short_in = max(field_w_in, field_l_in), min(field_w_in, field_l_in)
        self.canvas_w = 700
        self.canvas_h = max(1, round(self.canvas_w * short_in / long_in))
        self.corner_field_xy = [[0, 0], [self.canvas_w - 1, 0],
                                 [self.canvas_w - 1, self.canvas_h - 1],
                                 [0, self.canvas_h - 1]]

        self.cap = cv2.VideoCapture(str(video_path))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.idx = 0
        self.slider_drag = False

        self.corners = [None, None, None, None]   # original-frame px coords, TL/TR/BR/BL
        self.elements = []      # [{"name","frame_xy":[x,y]|None,"field_xy":[x,y]|None}]
        self.locked = []        # [{"cls","frame_xy":[x,y],"field_xy":[x,y]}]
        self.field_img_bgr = None
        self.field_img_path = None
        self.bg_mode = "field"     # field | warp | blend
        self.drag = None           # (kind, index) currently being dragged
        self.left_scale = 1.0
        self.right_scale = 1.0
        self.dets_cache = {}       # idx -> dets

        self._load_calib()
        if not self.elements:
            for name, x_frac, y_frac in builtin_elements:
                self.elements.append({"name": name, "frame_xy": None,
                                      "field_xy": [x_frac * self.canvas_w, y_frac * self.canvas_h]})
            print(f"[field] pre-loaded {len(self.elements)} element corners for year={year} "
                  f"-- drag each one's LEFT-panel half onto the frame")
        if field_image_path:
            self._load_field_image(field_image_path)

        self.root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
        self.root.title(f"Field calibration - {video_path.name} ({year})")

        panels = ttk.Frame(self.root)
        panels.pack()
        self.left_lbl = tk.Label(panels)
        self.left_lbl.pack(side="left", padx=2, pady=2)
        self.right_lbl = tk.Label(panels)
        self.right_lbl.pack(side="left", padx=2, pady=2)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="< frame", command=lambda: self.seek_relative(-1)).pack(side="left")
        ttk.Button(ctrl, text="frame >", command=lambda: self.seek_relative(1)).pack(side="left")
        ttk.Button(ctrl, text="Load field image...", command=self.pick_field_image).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Reset corners", command=self.reset_corners).pack(side="left", padx=8)
        self.bg_btn = ttk.Button(ctrl, text="bg: field", command=self.cycle_bg)
        self.bg_btn.pack(side="left")
        self.time_lbl = ttk.Label(ctrl, text="")
        self.time_lbl.pack(side="right", padx=8)

        self.slider = ttk.Scale(self.root, from_=0, to=max(0, self.n_frames - 1), orient="horizontal")
        self.slider.pack(fill="x", padx=4)
        self.slider.bind("<ButtonPress-1>", lambda e: setattr(self, "slider_drag", True))
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.status = ttk.Label(self.root, text="")
        self.status.pack(fill="x", padx=4, pady=(0, 4))

        self.left_lbl.bind("<ButtonPress-1>", self.on_left_press)
        self.left_lbl.bind("<B1-Motion>", self.on_left_drag)
        self.left_lbl.bind("<ButtonRelease-1>", self.on_release)
        self.left_lbl.bind("<ButtonPress-3>", self.on_left_right_click)

        self.right_lbl.bind("<ButtonPress-1>", self.on_right_press)
        self.right_lbl.bind("<B1-Motion>", self.on_right_drag)
        self.right_lbl.bind("<ButtonRelease-1>", self.on_release)
        self.right_lbl.bind("<ButtonPress-3>", self.on_right_right_click)

        if HAVE_DND:
            self.left_lbl.drop_target_register(DND_FILES)
            self.left_lbl.dnd_bind("<<Drop>>", self.on_drop)
            self.right_lbl.drop_target_register(DND_FILES)
            self.right_lbl.dnd_bind("<<Drop>>", self.on_drop)

        self.root.bind("<Left>", lambda e: self.seek_relative(-1))
        self.root.bind("<Right>", lambda e: self.seek_relative(1))
        self.root.bind("b", lambda e: self.cycle_bg())
        self.root.bind("s", lambda e: self._save_calib())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.render()

    # -- calibration persistence -----------------------------------
    def _load_calib(self):
        if not self.calib_path.exists():
            return
        data = json.loads(self.calib_path.read_text())
        if data.get("corners"):
            self.corners = [list(c) if c else None for c in data["corners"]]
        self.elements = data.get("elements", [])
        self.locked = data.get("locked", [])
        if data.get("field_image"):
            self._load_field_image(data["field_image"], quiet=True)
        print(f"[calib] loaded {self.calib_path.name} "
              f"({len(self.elements)} elements, {len(self.locked)} locked)")

    def _save_calib(self):
        data = {
            "year": self.year,
            "field_width_in": self.field_w_in,
            "field_length_in": self.field_l_in,
            "field_image": str(self.field_img_path) if self.field_img_path else None,
            "corners": self.corners,
            "elements": self.elements,
            "locked": self.locked,
        }
        self.calib_path.write_text(json.dumps(data, indent=2))
        self.status.config(text=f"saved calibration -> {self.calib_path.name}")

    # -- field image --------------------------------------------------
    def _load_field_image(self, path, quiet=False):
        path = pathlib.Path(path)
        img = cv2.imread(str(path))
        if img is None:
            self.status.config(text=f"could not read image: {path}")
            return
        self.field_img_bgr = img
        self.field_img_path = path
        if not quiet:
            self.status.config(text=f"loaded field image: {path.name}")
        if hasattr(self, "root"):
            self.render()

    def pick_field_image(self):
        path = filedialog.askopenfilename(
            title="Choose field image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path:
            self._load_field_image(path)
            self._save_calib()

    def on_drop(self, event):
        # tkinterdnd2 wraps multi-file paths in {}; take the first path.
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0] if "} {" in raw else raw
        self._load_field_image(path)
        self._save_calib()

    def cycle_bg(self):
        order = ["field", "warp", "blend"]
        self.bg_mode = order[(order.index(self.bg_mode) + 1) % len(order)]
        self.bg_btn.config(text=f"bg: {self.bg_mode}")
        self.render()

    def reset_corners(self):
        self.corners = [None, None, None, None]
        self._save_calib()
        self.render()

    # -- frame navigation ----------------------------------------------
    def seek_relative(self, n):
        self.show_frame(max(0, min(self.n_frames - 1, self.idx + n)))

    def on_slider_release(self, event):
        self.slider_drag = False
        self.show_frame(int(float(self.slider.get())))

    def show_frame(self, idx):
        self.idx = max(0, min(self.n_frames - 1, idx))
        if not self.slider_drag:
            self.slider.set(self.idx)
        self.time_lbl.config(text=f"frame {self.idx}/{self.n_frames}  t={self.idx / self.fps:.1f}s")
        self.render()

    def current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def current_dets(self, frame):
        if self.idx not in self.dets_cache:
            self.dets_cache[self.idx] = predict(self.model, self.hm, frame, self.threshold)
        return self.dets_cache[self.idx]

    # -- control points & TPS fit ---------------------------------------
    def control_points(self):
        """All (frame_xy, field_xy) pairs feeding the warp: corners + fully-
        placed elements + locked dots."""
        frame_pts, field_pts = [], []
        for i, c in enumerate(self.corners):
            if c is not None:
                frame_pts.append(c)
                field_pts.append(self.corner_field_xy[i])
        for e in self.elements:
            if e["frame_xy"] is not None and e["field_xy"] is not None:
                frame_pts.append(e["frame_xy"])
                field_pts.append(e["field_xy"])
        for l in self.locked:
            frame_pts.append(l["frame_xy"])
            field_pts.append(l["field_xy"])
        return frame_pts, field_pts

    def fwd_tps(self):
        """frame px -> field canvas px, for plotting detections/markers."""
        frame_pts, field_pts = self.control_points()
        return fit_tps(frame_pts, field_pts)

    def inv_tps(self):
        """field canvas px -> frame px, for warping the whole frame image."""
        frame_pts, field_pts = self.control_points()
        return fit_tps(field_pts, frame_pts)

    # -- generic hit-testing (works in DISPLAYED pixel space) -----------
    def _hits(self, ax, ay, ex, ey):
        return (ax - ex) ** 2 + (ay - ey) ** 2 <= self.HIT_RADIUS ** 2

    def find_hit_left(self, ex, ey):
        s = self.left_scale
        for i, c in enumerate(self.corners):
            if c is not None and self._hits(c[0] * s, c[1] * s, ex, ey):
                return ("corner", i)
        for i, e in enumerate(self.elements):
            if e["frame_xy"] is not None and self._hits(e["frame_xy"][0] * s, e["frame_xy"][1] * s, ex, ey):
                return ("element_frame", i)
        for i, l in enumerate(self.locked):
            if self._hits(l["frame_xy"][0] * s, l["frame_xy"][1] * s, ex, ey):
                return ("locked_frame", i)
        return None

    def find_hit_right(self, ex, ey):
        s = self.right_scale
        for i, e in enumerate(self.elements):
            if e["field_xy"] is not None and self._hits(e["field_xy"][0] * s, e["field_xy"][1] * s, ex, ey):
                return ("element_field", i)
        for i, l in enumerate(self.locked):
            if self._hits(l["field_xy"][0] * s, l["field_xy"][1] * s, ex, ey):
                return ("locked_field", i)
        return None

    # -- left panel: corners (fill+drag), elements (fill frame side+drag) --
    def on_left_press(self, event):
        hit = self.find_hit_left(event.x, event.y)
        if hit:
            self.drag = hit
            return
        pt = [event.x / self.left_scale, event.y / self.left_scale]
        for i, c in enumerate(self.corners):
            if c is None:
                self.corners[i] = pt
                self.drag = ("corner", i)
                self.render()
                return
        for i, e in enumerate(self.elements):
            if e["frame_xy"] is None:
                e["frame_xy"] = pt
                self.drag = ("element_frame", i)
                self.render()
                return

    def on_left_drag(self, event):
        if self.drag is None:
            return
        pt = [event.x / self.left_scale, event.y / self.left_scale]
        self._set_point(self.drag, pt)
        self.render()

    def on_left_right_click(self, event):
        hit = self.find_hit_left(event.x, event.y)
        if hit and hit[0] == "corner":
            self.corners[hit[1]] = None
        elif hit and hit[0] == "element_frame":
            del self.elements[hit[1]]
        elif hit and hit[0] == "locked_frame":
            del self.locked[hit[1]]
        else:
            self._try_lock_dot(event.x, event.y)
        self._save_calib()
        self.render()

    def _try_lock_dot(self, ex, ey):
        frame = self.current_frame()
        if frame is None:
            return
        dets = self.current_dets(frame)
        if not dets:
            return
        s = self.left_scale
        best, best_d = None, (self.LOCK_RADIUS) ** 2
        for x, y, cls, score in dets:
            d = (x * s - ex) ** 2 + (y * s - ey) ** 2
            if d <= best_d:
                best, best_d = (x, y, cls), d
        if best is None:
            return
        x, y, cls = best
        tps = self.fwd_tps()
        if tps is not None:
            field_xy = tps.apply([[x, y]])[0].tolist()
        else:
            field_xy = [self.canvas_w / 2, self.canvas_h / 2]
        self.locked.append({"cls": cls, "frame_xy": [x, y], "field_xy": field_xy})
        self.status.config(text=f"locked a {cls} dot at frame ({x:.0f},{y:.0f})")

    # -- right panel: elements (drag field-side position), locked (drag field) --
    def on_right_press(self, event):
        hit = self.find_hit_right(event.x, event.y)
        if hit:
            self.drag = hit

    def on_right_drag(self, event):
        if self.drag is None:
            return
        pt = [event.x / self.right_scale, event.y / self.right_scale]
        self._set_point(self.drag, pt)
        self.render()

    def on_right_right_click(self, event):
        hit = self.find_hit_right(event.x, event.y)
        if hit and hit[0] == "element_field":
            del self.elements[hit[1]]
        elif hit and hit[0] == "locked_field":
            del self.locked[hit[1]]
        self._save_calib()
        self.render()

    def _set_point(self, hit, pt):
        kind, i = hit
        if kind == "corner":
            self.corners[i] = pt
        elif kind == "element_frame":
            self.elements[i]["frame_xy"] = pt
        elif kind == "element_field":
            self.elements[i]["field_xy"] = pt
        elif kind == "locked_frame":
            self.locked[i]["frame_xy"] = pt
        elif kind == "locked_field":
            self.locked[i]["field_xy"] = pt

    def on_release(self, event):
        if self.drag is not None:
            self.drag = None
            self._save_calib()

    # -- rendering ---------------------------------------------------------
    def render(self):
        frame = self.current_frame()
        if frame is None:
            return
        dets = self.current_dets(frame)
        fwd = self.fwd_tps()

        left = self._render_left(frame, dets)
        right = self._render_right(frame, dets, fwd)
        self._show(self.left_lbl, left, self.MAX_PANEL_W, panel="left")
        self._show(self.right_lbl, right, self.MAX_PANEL_W, panel="right")

        n_corners = sum(c is not None for c in self.corners)
        n_elem_full = sum(e["frame_xy"] is not None and e["field_xy"] is not None for e in self.elements)
        msg = f"corners {n_corners}/4"
        if n_corners < 4:
            msg += f" (click left panel to place {CORNER_NAMES[n_corners]})"
        msg += f"  |  elements {n_elem_full}/{len(self.elements)} placed"
        next_elem = next((e for e in self.elements if e["frame_xy"] is None), None)
        if next_elem is not None:
            msg += f" (click left panel to place {next_elem['name']})"
        msg += f"  |  locked dots {len(self.locked)}"
        if fwd is None:
            msg += "  |  need >=4 control points total to warp"
        if self.field_img_bgr is None:
            msg += "  |  drag a field image onto either panel, or use Load field image"
        self.status.config(text=msg)

    def _render_left(self, frame, dets):
        img = frame.copy()
        for x, y, cls, score in dets:
            color = COLOR_BGR.get(cls, (0, 255, 0))
            cv2.circle(img, (int(x), int(y)), 6, color, -1)

        if all(c is not None for c in self.corners):
            cv2.polylines(img, np.int32([self.corners]), True, (0, 255, 0), 2)
        for name, c in zip(CORNER_NAMES, self.corners):
            if c is None:
                continue
            self._draw_marker(img, c, CORNER_COLOR[name], name)

        for e in self.elements:
            if e["frame_xy"] is not None:
                self._draw_marker(img, e["frame_xy"], ELEMENT_COLOR, e["name"], shape="square")
        for l in self.locked:
            self._draw_marker(img, l["frame_xy"], LOCKED_COLOR, "L", shape="lock")
        return img

    def _render_right(self, frame, dets, fwd):
        cw, ch = self.canvas_w, self.canvas_h
        inv = self.inv_tps() if self.bg_mode in ("warp", "blend") else None

        if fwd is None:
            canvas = np.full((ch, cw, 3), 40, np.uint8)
            cv2.putText(canvas, "need >=4 control points", (20, ch // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            field = cv2.resize(self.field_img_bgr, (cw, ch)) if self.field_img_bgr is not None else None
            warped = self._warp_frame(frame, inv, cw, ch) if inv is not None else None

            if self.bg_mode == "warp" and warped is not None:
                canvas = warped
            elif self.bg_mode == "blend" and warped is not None and field is not None:
                canvas = cv2.addWeighted(field, 0.5, warped, 0.5, 0)
            elif field is not None:
                canvas = field
            elif warped is not None:
                canvas = warped
            else:
                canvas = np.full((ch, cw, 3), 40, np.uint8)
            canvas = canvas.copy()

            if dets:
                pts = fwd.apply([[x, y] for x, y, _, _ in dets])
                for (x, y), (_, _, cls, score) in zip(pts, dets):
                    if 0 <= x < cw and 0 <= y < ch:
                        color = COLOR_BGR.get(cls, (0, 255, 0))
                        cv2.circle(canvas, (int(x), int(y)), 7, color, -1)
                        cv2.circle(canvas, (int(x), int(y)), 7, (0, 0, 0), 1)

        for e in self.elements:
            if e["field_xy"] is not None:
                self._draw_marker(canvas, e["field_xy"], ELEMENT_COLOR, e["name"], shape="square")
        for l in self.locked:
            self._draw_marker(canvas, l["field_xy"], LOCKED_COLOR, "L", shape="lock")
        return canvas

    def _warp_frame(self, frame, inv_tps, cw, ch):
        """Perspective/TPS-correct the whole frame onto the field canvas by
        evaluating the field->frame TPS at every output pixel (i.e. an
        inverse warp) and remapping."""
        xs, ys = np.meshgrid(np.arange(cw), np.arange(ch))
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
        src = inv_tps.apply(grid)
        map_x = src[:, 0].reshape(ch, cw).astype(np.float32)
        map_y = src[:, 1].reshape(ch, cw).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    def _draw_marker(self, img, xy, color, label, shape="circle"):
        p = (int(xy[0]), int(xy[1]))
        if shape == "square":
            r = self.MARKER_RADIUS
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, -1)
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), (0, 0, 0), 1)
        elif shape == "lock":
            cv2.circle(img, p, self.MARKER_RADIUS, color, 2)
            cv2.circle(img, p, 2, color, -1)
        else:
            cv2.circle(img, p, self.MARKER_RADIUS, color, -1)
            cv2.circle(img, p, self.MARKER_RADIUS, (0, 0, 0), 1)
        cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)

    def _show(self, label, img_bgr, max_w, panel):
        h, w = img_bgr.shape[:2]
        scale = min(1.0, max_w / w)
        if panel == "left":
            self.left_scale = scale
        else:
            self.right_scale = scale
        disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.image = photo   # keep a reference, tkinter drops it otherwise
        label.configure(image=photo)

    def run(self):
        self.root.mainloop()
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="local video file, or a YouTube URL to download/cache")
    ap.add_argument("--year", type=int, required=True, help="FRC season, selects field proportions")
    ap.add_argument("--field-image", default=None, help="top-down field image to start with")
    ap.add_argument("--field-width-in", type=float, default=None)
    ap.add_argument("--field-length-in", type=float, default=None)
    ap.add_argument("--cookies", default=None, help="cookies.txt for age/region-gated videos")
    ap.add_argument("--ckpt", default=str(ROOT / "data/frc/checkpoints/best.pt"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's peak_threshold")
    args = ap.parse_args()

    if not HAVE_DND:
        print("[warn] tkinterdnd2 not installed -- drag-and-drop disabled, "
              "use the 'Load field image' button instead (pip install tkinterdnd2)")

    # fail fast, before touching the video or loading the model, if this
    # year has no digitized elements
    builtin_elements = elements_for_year(args.year)

    video_path = download_video(args.video, args.cookies) if is_url(args.video) else pathlib.Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"[abort] video not found: {video_path}")

    model, hm = load_model_and_cfg(pathlib.Path(args.ckpt))
    threshold = args.threshold if args.threshold is not None else hm["peak_threshold"]
    field_w_in, field_l_in = resolve_field_dims(args.year, args.field_width_in, args.field_length_in)
    print(f"[field] year={args.year} dims={field_w_in:.1f}x{field_l_in:.1f}in")

    Calibrator(video_path, model, hm, threshold, args.year, field_w_in, field_l_in,
              builtin_elements, field_image_path=args.field_image).run()


if __name__ == "__main__":
    main()
