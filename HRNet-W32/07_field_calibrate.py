#!/usr/bin/env python3
"""
Step 7 (interactive) — calibrate a video's camera-to-field mapping and
preview inference detections mapped onto a flat, undistorted field.

Broadcast/sideline camera footage of an FRC field is close enough to a
pinhole projection of a flat floor that a single 4-point homography (a
"trapezoid" mapping) is enough to undistort it -- there's no need for a
free-form per-point warp. So the ENTIRE calibration is 4 points:

  - CORNERS (4): the real field boundary, dragged on the LEFT (frame)
    panel — their field-side position is fixed to the canvas corners.
    As soon as a frame is available they're auto-placed at the frame's
    own 4 corners (a reasonable first guess for a wide-ish shot), so you
    normally only need to nudge each one onto the true corner rather than
    click 4 times from nothing.

Once all 4 corners are placed, the homography they define is used to
project every digitized FIELD ELEMENT (line intersections, tag corners,
hub corners, ... loaded from data/frc/field_elements.json, digitized ahead
of time with 06_field_elements_editor.py) onto the LEFT panel at its
predicted frame position. There's nothing to place for these -- they're
purely a visual check: if a projected element lands on top of the actual
matching line/marking in the video, your 4 corners are accurate; if it's
off, drag a corner until it lines up. Because it's one shared homography,
dragging any single corner reshapes the WHOLE trapezoid and moves every
projected element at once, not just the corner you touched.

Corners + field image path are cached per-video to a JSON sidecar next to
the video file, so re-running on the same file resumes the last
calibration.

Requires: tkinterdnd2 (pip) for real OS drag-and-drop of the field image;
without it, use the "Load field image" button instead. yt-dlp/ffmpeg only
needed if you pass a YouTube URL instead of a local file.

Controls:
  Left / Right           step one frame
  slider                 scrub anywhere in the clip
  drag a corner (left)   adjust calibration by moving the frame-side corner
                          directly
  drag a corner (right)  adjust calibration by grabbing a misaligned point
                          in the undistorted view and dropping it onto the
                          true (fixed) field corner -- the marker snaps
                          back on release, the underlying frame corner
                          updates instead
  right-click a corner   clear it (click empty-handed on the left panel to
                          re-place it)
  b                      cycle right-panel background: field / warped video / blend
  s                      save calibration now (also autosaves on release)
  q / Esc                quit

Usage:
  python 07_field_calibrate.py video.mp4 --year 2026
  python 07_field_calibrate.py "https://www.youtube.com/watch?v=XXXX" --year 2026
  python 07_field_calibrate.py video.mp4 --field-width-in 317.7 --field-length-in 651.2
"""
import argparse
import json
import pathlib

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from model import CLASSES, HeatmapNet, decode_peaks

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
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

# Real-world FRC field footprint per year, inches, (width, length).
# Only fill in years actually verified against the game manual / field
# dimension drawings -- everything else falls back to STANDARD_FALLBACK_IN
# with a printed warning, rather than silently guessing.
FIELD_DIMS_IN = {
    2026: (317.7, 651.2),   # REBUILT, official field dimension drawings
}
STANDARD_FALLBACK_IN = (324.0, 648.0)  # ~27x54ft "standard" FRC footprint used pre-2026

# Field elements are digitized with 06_field_elements_editor.py and read
# from here: {"<year>": [{"name": str, "points": [[x_frac,y_frac], ...]}]},
# x_frac/y_frac as fractions of field LENGTH/WIDTH. There is no in-app way
# to add elements, and they're purely a visual aid here -- not required.
FIELD_ELEMENTS_PATH = ROOT / "data/frc/field_elements.json"


def elements_for_year(year):
    """(name, x_frac, y_frac) tuples, one per polygon vertex. [] if nothing
    has been digitized for this year yet -- the corner homography works
    fine without them, they just lose their visual-alignment check."""
    if not FIELD_ELEMENTS_PATH.exists():
        return []
    data = json.loads(FIELD_ELEMENTS_PATH.read_text())
    entries = data.get(str(year))
    if not entries:
        return []
    out = []
    for e in entries:
        for i, (xf, yf) in enumerate(e["points"]):
            out.append((f"{e['name']}-P{i}", xf, yf))
    return out


# ---------------------------------------------------------------------
# Download (same cache convention as 04_view.py/05_youtube_view.py, so a
# video already pulled for playback is reused here without re-downloading)
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
    if (w_in is None) != (l_in is None):
        raise SystemExit("[abort] pass both --field-width-in and --field-length-in, or neither")
    if w_in is not None and l_in is not None:
        return w_in, l_in
    if year in FIELD_DIMS_IN:
        return FIELD_DIMS_IN[year]
    print(f"[field] no verified dimensions for year={year}; falling back to "
          f"{STANDARD_FALLBACK_IN[0]:.0f}x{STANDARD_FALLBACK_IN[1]:.0f}in. "
          f"Pass --field-width-in/--field-length-in to override.")
    return STANDARD_FALLBACK_IN


def perspective_transform(H, pts):
    """cv2.perspectiveTransform wrapper for a plain Nx2 list/array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ---------------------------------------------------------------------
# Calibration sidecar
# ---------------------------------------------------------------------
def calib_path_for(video_path):
    return video_path.with_suffix(".field_calib.json")


class Calibrator:
    MARKER_RADIUS = 8       # px, in displayed (scaled) coordinates
    HIT_RADIUS = 14
    MAX_PANEL_W = 720
    PAD_FRAC = 0.15          # extra margin shown around the field in the undistorted view

    def __init__(self, video_path, model, hm, threshold, year, field_w_in, field_l_in,
                 builtin_elements, field_image_path=None, youtube_url=None):
        self.video_path = video_path
        self.model, self.hm, self.threshold = model, hm, threshold
        self.year = year
        self.field_w_in, self.field_l_in = field_w_in, field_l_in
        self.calib_path = calib_path_for(video_path)
        self.youtube_url = youtube_url

        long_in, short_in = max(field_w_in, field_l_in), min(field_w_in, field_l_in)
        self.canvas_w = 700
        self.canvas_h = max(1, round(self.canvas_w * short_in / long_in))
        self.corner_field_xy = [[0, 0], [self.canvas_w - 1, 0],
                                 [self.canvas_w - 1, self.canvas_h - 1],
                                 [0, self.canvas_h - 1]]
        # the undistorted (right) panel is rendered a bit larger than the
        # field itself, so you can see what's just outside the boundary
        # (e.g. to check a corner isn't clipped) -- corner_field_xy above
        # stays anchored to the un-padded field, only the display canvas grows.
        self.pad_x = round(self.canvas_w * self.PAD_FRAC)
        self.pad_y = round(self.canvas_h * self.PAD_FRAC)
        self.disp_w = self.canvas_w + 2 * self.pad_x
        self.disp_h = self.canvas_h + 2 * self.pad_y

        self.cap = cv2.VideoCapture(str(video_path))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
        self.frame_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
        self.idx = 0
        self.slider_drag = False
        self.playing = False
        self.play_after_id = None
        self._frame_cache_idx = None
        self._frame_cache_img = None

        self.corners = [None, None, None, None]   # original-frame px coords, TL/TR/BR/BL
        self.elements = [{"name": name, "field_xy": [xf * self.canvas_w, yf * self.canvas_h]}
                          for name, xf, yf in builtin_elements]
        self.field_img_bgr = None
        self.field_img_path = None
        self.bg_mode = "field"     # field | warp | blend
        self.drag = None           # ("corner", i) currently being dragged
        self.right_drag = None     # ("corner", i) being dragged in the undistorted panel
        self.right_drag_pt = None  # live mouse pos (unscaled display px) while right_drag is set
        self.left_scale = 1.0
        self.right_scale = 1.0
        self.dets_cache = {}       # idx -> dets

        self._load_calib()
        if all(c is None for c in self.corners):
            # first-ever run on this video: seed a starting trapezoid at the
            # frame's own corners so there's immediately something to drag
            # into place, instead of an empty panel requiring 4 clicks.
            self.corners = [[0, 0], [self.frame_w - 1, 0],
                             [self.frame_w - 1, self.frame_h - 1], [0, self.frame_h - 1]]

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
        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=(8, 0))
        ttk.Button(ctrl, text="Load field image...", command=self.pick_field_image).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Reset corners", command=self.reset_corners).pack(side="left", padx=8)
        self.bg_btn = ttk.Button(ctrl, text="bg: field", command=self.cycle_bg)
        self.bg_btn.pack(side="left")
        self.time_lbl = ttk.Label(ctrl, text="")
        self.time_lbl.pack(side="right", padx=8)

        self.slider = ttk.Scale(self.root, from_=0, to=max(0, self.n_frames - 1), orient="horizontal")
        self.slider.pack(fill="x", padx=4)
        self.slider.bind("<ButtonPress-1>", lambda e: (self.stop_play(), setattr(self, "slider_drag", True)))
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.status = ttk.Label(self.root, text="")
        self.status.pack(fill="x", padx=4, pady=(0, 4))

        self.left_lbl.bind("<ButtonPress-1>", self.on_left_press)
        self.left_lbl.bind("<B1-Motion>", self.on_left_drag)
        self.left_lbl.bind("<ButtonRelease-1>", self.on_release)
        self.left_lbl.bind("<ButtonPress-3>", self.on_left_right_click)

        self.right_lbl.bind("<ButtonPress-1>", self.on_right_press)
        self.right_lbl.bind("<B1-Motion>", self.on_right_drag)
        self.right_lbl.bind("<ButtonRelease-1>", self.on_right_release)

        if HAVE_DND:
            self.left_lbl.drop_target_register(DND_FILES)
            self.left_lbl.dnd_bind("<<Drop>>", self.on_drop)
            self.right_lbl.drop_target_register(DND_FILES)
            self.right_lbl.dnd_bind("<<Drop>>", self.on_drop)

        self.root.bind("<Left>", lambda e: self.seek_relative(-1))
        self.root.bind("<Right>", lambda e: self.seek_relative(1))
        self.root.bind("<space>", lambda e: self.toggle_play())
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
        if data.get("field_image"):
            self._load_field_image(data["field_image"], quiet=True)
        print(f"[calib] loaded {self.calib_path.name}")

    def _save_calib(self):
        data = {
            "year": self.year,
            "field_width_in": self.field_w_in,
            "field_length_in": self.field_l_in,
            "field_image": str(self.field_img_path) if self.field_img_path else None,
            "corners": self.corners,
            "youtube_url": self.youtube_url,
        }
        self.calib_path.write_text(json.dumps(data, indent=2))
        self.status.config(text=f"saved calibration -> {self.calib_path.name}")

    # -- field image --------------------------------------------------
    def _load_field_image(self, path, quiet=False):
        path = pathlib.Path(path)
        img = cv2.imread(str(path))
        if img is None:
            msg = f"could not read image: {path}"
            self.status.config(text=msg) if hasattr(self, "status") else print(f"[field] {msg}")
            return
        self.field_img_bgr = img
        self.field_img_path = path
        if not quiet and hasattr(self, "status"):
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
        self.corners = [[0, 0], [self.frame_w - 1, 0],
                         [self.frame_w - 1, self.frame_h - 1], [0, self.frame_h - 1]]
        self._save_calib()
        self.render()

    # -- playback ------------------------------------------------------
    def toggle_play(self):
        if self.playing:
            self.stop_play()
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self._play_tick()

    def stop_play(self):
        self.playing = False
        self.play_btn.config(text="Play")
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def _play_tick(self):
        if not self.playing:
            return
        if self.idx >= self.n_frames - 1:
            self.stop_play()
            return
        self.show_frame(self.idx + 1)
        self.play_after_id = self.root.after(max(1, int(1000 / self.fps)), self._play_tick)

    # -- frame navigation ----------------------------------------------
    def seek_relative(self, n):
        self.stop_play()
        self.show_frame(max(0, min(self.n_frames - 1, self.idx + n)))

    def on_slider_release(self, event):
        self.stop_play()
        self.slider_drag = False
        self.show_frame(int(float(self.slider.get())))

    def show_frame(self, idx):
        self.idx = max(0, min(self.n_frames - 1, idx))
        if not self.slider_drag:
            self.slider.set(self.idx)
        self.time_lbl.config(text=f"frame {self.idx}/{self.n_frames}  t={self.idx / self.fps:.1f}s")
        self.render()

    def current_frame(self):
        # dragging a corner re-renders on every mouse-move without changing
        # self.idx -- caching the decoded frame is what makes that fast,
        # since cap.set()+read() (a seek) is the expensive part, not the
        # homography/warp math. Sequential playback hits the elif: advancing
        # one frame via read() alone is far cheaper than a seek, since a
        # seek has to walk back to the nearest keyframe and decode forward.
        if self._frame_cache_idx == self.idx:
            return self._frame_cache_img
        if self._frame_cache_idx is not None and self.idx == self._frame_cache_idx + 1:
            ok, frame = self.cap.read()
            self._frame_cache_img = frame if ok else None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
            ok, frame = self.cap.read()
            self._frame_cache_img = frame if ok else None
            self._frame_cache_idx = self.idx
        return self._frame_cache_img

    def current_dets(self, frame):
        if self.idx not in self.dets_cache:
            self.dets_cache[self.idx] = predict(self.model, self.hm, frame, self.threshold)
        return self.dets_cache[self.idx]

    # -- the one homography, defined entirely by the 4 corners -----------
    def homography(self):
        """frame px -> field canvas px, or None until all 4 corners are set."""
        if any(c is None for c in self.corners):
            return None
        return cv2.getPerspectiveTransform(np.float32(self.corners), np.float32(self.corner_field_xy))

    def inv_homography(self, H):
        return np.linalg.inv(H)

    # -- hit-testing (works in DISPLAYED pixel space) -----------
    def _hits(self, ax, ay, ex, ey):
        return (ax - ex) ** 2 + (ay - ey) ** 2 <= self.HIT_RADIUS ** 2

    def find_hit_left(self, ex, ey):
        s = self.left_scale
        for i, c in enumerate(self.corners):
            if c is not None and self._hits(c[0] * s, c[1] * s, ex, ey):
                return ("corner", i)
        return None

    # -- left panel: only the 4 corners are interactive -----------------
    def on_left_press(self, event):
        self.stop_play()
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

    def on_left_drag(self, event):
        if self.drag is None:
            return
        kind, i = self.drag
        self.corners[i] = [event.x / self.left_scale, event.y / self.left_scale]
        self.render()

    def on_left_right_click(self, event):
        hit = self.find_hit_left(event.x, event.y)
        if hit:
            self.corners[hit[1]] = None
            self._save_calib()
            self.render()

    def on_release(self, event):
        if self.drag is not None:
            self.drag = None
            self._save_calib()

    # -- right panel: drag a field-boundary corner to nudge calibration --
    # The field-side corner position is fixed by definition (it's the true
    # field rectangle), so dragging it can't move it permanently. Instead,
    # the marker follows the mouse only as a live preview; on release it
    # snaps back to its fixed spot, and the FRAME-side corner is updated so
    # that whatever content was under the drop point is now what maps to
    # that fixed corner -- i.e. "grab the misaligned feature and drop it
    # onto the true corner" rather than "move the corner".
    def _right_corner_xy(self, i):
        fx, fy = self.corner_field_xy[i]
        return [self.pad_x + fx, self.pad_y + fy]

    def find_hit_right(self, ex, ey):
        if self.homography() is None:
            return None
        s = self.right_scale
        for i in range(4):
            cx, cy = self._right_corner_xy(i)
            if self._hits(cx * s, cy * s, ex, ey):
                return ("corner", i)
        return None

    def on_right_press(self, event):
        self.stop_play()
        hit = self.find_hit_right(event.x, event.y)
        if hit:
            self.right_drag = hit
            self.right_drag_pt = (event.x, event.y)
            self.render()

    def on_right_drag(self, event):
        if self.right_drag is None:
            return
        self.right_drag_pt = (event.x, event.y)
        self.render()

    def on_right_release(self, event):
        if self.right_drag is None:
            return
        i = self.right_drag[1]
        H = self.homography()
        if H is not None:
            s = self.right_scale or 1.0
            field_pt = [event.x / s - self.pad_x, event.y / s - self.pad_y]
            frame_pt = perspective_transform(self.inv_homography(H), [field_pt])[0]
            self.corners[i] = [float(frame_pt[0]), float(frame_pt[1])]
            self._save_calib()
        self.right_drag = None
        self.right_drag_pt = None
        self.render()

    # -- rendering ---------------------------------------------------------
    def render(self):
        frame = self.current_frame()
        if frame is None:
            return
        dets = self.current_dets(frame)
        H = self.homography()

        left = self._render_left(frame, H)
        right = self._render_right(frame, dets, H)
        self._show(self.left_lbl, left, self.MAX_PANEL_W, panel="left")
        self._show(self.right_lbl, right, self.MAX_PANEL_W, panel="right")

        n_corners = sum(c is not None for c in self.corners)
        msg = f"corners {n_corners}/4"
        if n_corners < 4:
            msg += f" (right-clicked/cleared -- click left panel to place {CORNER_NAMES[n_corners]})"
        elif self.elements:
            msg += f"  |  {len(self.elements)} field elements projected -- drag a corner until they line up"
        else:
            msg += "  |  no digitized field elements for this year (calibration still works from corners alone)"
        if self.field_img_bgr is None:
            msg += "  |  drag a field image onto either panel, or use Load field image"
        self.status.config(text=msg)

    def _render_left(self, frame, H):
        img = frame.copy()

        if all(c is not None for c in self.corners):
            cv2.polylines(img, np.int32([self.corners]), True, (0, 255, 0), 2)
        for name, c in zip(CORNER_NAMES, self.corners):
            if c is None:
                continue
            self._draw_marker(img, c, CORNER_COLOR[name], name)

        if H is not None and self.elements:
            H_inv = self.inv_homography(H)
            field_pts = [e["field_xy"] for e in self.elements]
            frame_pts = perspective_transform(H_inv, field_pts)
            for e, p in zip(self.elements, frame_pts):
                self._draw_marker(img, p, ELEMENT_COLOR, e["name"], shape="square")
        return img

    def _render_right(self, frame, dets, H):
        cw, ch = self.canvas_w, self.canvas_h
        dw, dh = self.disp_w, self.disp_h
        px, py = self.pad_x, self.pad_y

        if H is None:
            canvas = np.full((dh, dw, 3), 40, np.uint8)
            cv2.putText(canvas, "place all 4 corners", (20, dh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
            return canvas

        # translate the frame->field homography so field-space (0,0) lands
        # at (px, py) in the padded display canvas, instead of the corner.
        T = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float64)
        H_disp = T @ H

        field = None
        if self.field_img_bgr is not None:
            field_resized = cv2.resize(self.field_img_bgr, (cw, ch))
            field = cv2.copyMakeBorder(field_resized, py, dh - ch - py, px, dw - cw - px,
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))
        warped = cv2.warpPerspective(frame, H_disp, (dw, dh), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(40, 40, 40)) if self.bg_mode in ("warp", "blend") else None

        if self.bg_mode == "warp" and warped is not None:
            canvas = warped
        elif self.bg_mode == "blend" and warped is not None and field is not None:
            canvas = cv2.addWeighted(field, 0.5, warped, 0.5, 0)
        elif field is not None:
            canvas = field
        elif warped is not None:
            canvas = warped
        else:
            canvas = np.full((dh, dw, 3), 40, np.uint8)
        canvas = canvas.copy()

        # the actual field boundary -- everything outside this rectangle is
        # the extra margin, shown for context only (e.g. to see if a corner
        # is clipped just off the true edge).
        cv2.rectangle(canvas, (px, py), (px + cw - 1, py + ch - 1), (0, 255, 0), 2)

        for e in self.elements:
            x, y = e["field_xy"]
            self._draw_marker(canvas, (x + px, y + py), ELEMENT_COLOR, e["name"], shape="square")

        if dets:
            pts = perspective_transform(H, [[x, y] for x, y, _, _ in dets])
            for (x, y), (_, _, cls, score) in zip(pts, dets):
                dx, dy = x + px, y + py
                if 0 <= dx < dw and 0 <= dy < dh:
                    color = COLOR_BGR.get(cls, (0, 255, 0))
                    cv2.circle(canvas, (int(dx), int(dy)), 7, color, -1)
                    cv2.circle(canvas, (int(dx), int(dy)), 7, (0, 0, 0), 1)

        if self.right_drag is not None and self.right_drag_pt is not None:
            i = self.right_drag[1]
            s = self.right_scale or 1.0
            ux, uy = self.right_drag_pt[0] / s, self.right_drag_pt[1] / s
            self._draw_marker(canvas, (ux, uy), CORNER_COLOR[CORNER_NAMES[i]], CORNER_NAMES[i])
        return canvas

    def _draw_marker(self, img, xy, color, label, shape="circle"):
        p = (int(xy[0]), int(xy[1]))
        if shape == "square":
            r = self.MARKER_RADIUS
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, -1)
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), (0, 0, 0), 1)
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

    youtube_url = args.video if is_url(args.video) else None
    video_path = download_video(args.video, args.cookies) if is_url(args.video) else pathlib.Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"[abort] video not found: {video_path}")

    builtin_elements = elements_for_year(args.year)
    if not builtin_elements:
        print(f"[field] no digitized field elements for year={args.year} in {FIELD_ELEMENTS_PATH} -- "
              f"calibration still works from the 4 corners alone, you just won't get a visual alignment "
              f"check. Run 06_field_elements_editor.py --year {args.year} to add some.")

    model, hm = load_model_and_cfg(pathlib.Path(args.ckpt))
    threshold = args.threshold if args.threshold is not None else hm["peak_threshold"]
    field_w_in, field_l_in = resolve_field_dims(args.year, args.field_width_in, args.field_length_in)
    print(f"[field] year={args.year} dims={field_w_in:.1f}x{field_l_in:.1f}in")

    Calibrator(video_path, model, hm, threshold, args.year, field_w_in, field_l_in,
               builtin_elements, field_image_path=args.field_image,
               youtube_url=youtube_url).run()


if __name__ == "__main__":
    main()
