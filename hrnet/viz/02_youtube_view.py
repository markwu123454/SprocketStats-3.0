#!/usr/bin/env python3
"""
Step 2 viz (interactive) — pull a YouTube video, run best.pt over it, and scrub
through it with predicted keypoints overlaid.

Design: inference runs ONCE, up front, over the whole (downloaded) clip and
is cached to a .json sidecar next to the video file. Playback then only
decodes frames and draws the cached dots (cv2.circle/putText — cheap), so
scrubbing or hitting Play never blocks on the model. Re-running on the same
video/threshold/stride reuses both the downloaded file and the inference
cache, so the second run opens almost instantly.

Requires: yt-dlp (pip), ffmpeg on PATH (used by yt-dlp to merge/cut).

Controls:
  Space               play / pause
  Left / Right        step one frame (also works while paused)
  <<1s / 1s>> buttons seek by one second
  g                   toggle the detection overlay
  q / Esc             quit
  slider              drag to scrub anywhere in the clip

Usage:
  python 02_youtube_view.py "https://www.youtube.com/watch?v=XXXX"
  python 02_youtube_view.py URL --start 60 --end 120   # only download/infer that clip
  python 02_youtube_view.py URL --stride 2             # infer every 2nd frame (faster prep)
"""
import argparse, json, pathlib, time
import torch
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import sys as _sys
ROOT = pathlib.Path(__file__).parent.parent
_sys.path.insert(0, str(ROOT))

from model import HeatmapNet, decode_peaks, CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # input size is fixed (heatmap.input_size), so
                                       # cuDNN can pick a fixed-shape conv algo once and reuse it
CACHE_DIR = ROOT / "data/frc/youtube_cache"

COLOR_BGR = {"blue": (255, 120, 0), "red": (0, 0, 255)}
CH_TO_CLS = {i: c for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------
# Download (cached: re-running with the same url/start/end reuses the file)
# ---------------------------------------------------------------------
def download_video(url, start, end, cookies=None):
    import yt_dlp
    import yt_dlp.utils as ydl_utils
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"_{int(start or 0)}-{int(end)}" if (start is not None or end is not None) else ""
    ydl_opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/best",
        "outtmpl": str(CACHE_DIR / f"%(id)s{tag}.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    if cookies:
        ydl_opts["cookiefile"] = str(cookies)
    if start is not None or end is not None:
        ydl_opts["download_ranges"] = ydl_utils.download_range_func(
            None, [(start or 0, end if end is not None else 10 ** 9)])
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid = info["id"]

    candidates = [p for p in CACHE_DIR.glob(f"{vid}{tag}.*")
                 if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not candidates:
        raise SystemExit(f"[abort] download finished but no video file found for id={vid}")
    path = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"[download] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# ---------------------------------------------------------------------
# Model (same loading convention as 04_view.py: trust the checkpoint's own
# saved cfg over the live config.yaml, since they can drift after retrains)
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


_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)


@torch.no_grad()
def predict_batch(model, hm, imgs_bgr, threshold):
    """One forward pass over a list of BGR frames (can differ in native size).
    Batching keeps the GPU fed instead of idling between single-frame kernel
    launches, which is what actually saturates it vs. calling this per-frame."""
    in_h, in_w = hm["input_size"]
    tensors, scales = [], []
    for img_bgr in imgs_bgr:
        H0, W0 = img_bgr.shape[:2]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img, (in_w, in_h))
        tensors.append(torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0)
        scales.append((W0 / in_w, H0 / in_h))

    x = torch.stack(tensors).to(DEVICE)
    x = (x - _MEAN) / _STD
    logits = model(x)
    dets_batch = decode_peaks(logits, threshold, hm["nms_kernel"], hm["max_instances"])

    out = []
    for dets, (sx, sy) in zip(dets_batch, scales):
        frame_out = []
        for x_hm, y_hm, c, score in dets:
            x_in = x_hm * hm["output_stride"]
            y_in = y_hm * hm["output_stride"]
            frame_out.append([x_in * sx, y_in * sy, CH_TO_CLS[c], score])
        out.append(frame_out)
    return out


# ---------------------------------------------------------------------
# Inference cache: {"n_frames": N, "dets": [[[x,y,cls,score],...], ...]}
# ---------------------------------------------------------------------
def cache_path_for(video_path, threshold, stride):
    return video_path.with_suffix(f".dets_t{threshold:.2f}_s{stride}.json")


def build_or_load_cache(video_path, model, hm, threshold, stride, batch_size=8):
    cpath = cache_path_for(video_path, threshold, stride)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if cpath.exists():
        cached = json.loads(cpath.read_text())
        if cached.get("n_frames") == n_frames:
            cap.release()
            print(f"[cache] reusing {cpath.name} ({n_frames} frames)")
            return cached["dets"], fps, n_frames
        print(f"[cache] stale ({cached.get('n_frames')} != {n_frames} frames), recomputing")

    print(f"[infer] {n_frames} frames @ stride={stride}, batch={batch_size} -> {cpath.name}")
    computed = {}          # idx -> dets, only for frames actually run through the model
    batch_frames, batch_idxs = [], []
    i = 0
    t0 = time.time()

    def flush():
        if not batch_frames:
            return
        results = predict_batch(model, hm, batch_frames, threshold)
        for idx, dets in zip(batch_idxs, results):
            computed[idx] = dets
        batch_frames.clear()
        batch_idxs.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            batch_frames.append(frame)
            batch_idxs.append(i)
            if len(batch_frames) >= batch_size:
                flush()
        i += 1
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = len(computed) / elapsed if elapsed > 0 else 0
            eta = (n_frames // stride - len(computed)) / rate if rate > 0 else 0
            print(f"[infer] {i}/{n_frames} scanned, {len(computed)} inferred "
                  f"({rate:.1f} fps, eta {eta:.0f}s)")
    flush()
    cap.release()

    # forward-fill: frames skipped by --stride hold the last computed detection
    dets_per_frame = [[] for _ in range(n_frames)]
    last = []
    for idx in range(n_frames):
        if idx in computed:
            last = computed[idx]
        dets_per_frame[idx] = last

    cpath.write_text(json.dumps({"n_frames": n_frames, "dets": dets_per_frame}))
    print(f"[infer] done: {len(computed)} inferred / {n_frames} frames in "
          f"{time.time() - t0:.1f}s -> cached to {cpath.name}")
    return dets_per_frame, fps, n_frames


# ---------------------------------------------------------------------
# Spatial smoothing: greedy nearest-neighbor tracking per class + EMA on
# each track's position. Applied after the (cached) inference pass so
# smoothing params can be tuned without re-running the model.
# ---------------------------------------------------------------------
def smooth_dets(dets_per_frame, alpha=0.3, max_dist=120, max_misses=5):
    tracks = {}  # cls -> list of {"pos": [x, y], "misses": int}
    out = []
    for dets in dets_per_frame:
        by_cls = {}
        for d in dets:
            by_cls.setdefault(d[2], []).append(d)

        frame_out = []
        for cls in set(tracks) | set(by_cls):
            cls_tracks = tracks.get(cls, [])
            used = set()
            for x, y, _, score in by_cls.get(cls, []):
                best_i, best_d = None, max_dist
                for i, t in enumerate(cls_tracks):
                    if i in used:
                        continue
                    d2 = ((t["pos"][0] - x) ** 2 + (t["pos"][1] - y) ** 2) ** 0.5
                    if d2 < best_d:
                        best_d, best_i = d2, i
                if best_i is not None:
                    t = cls_tracks[best_i]
                    t["pos"][0] = alpha * x + (1 - alpha) * t["pos"][0]
                    t["pos"][1] = alpha * y + (1 - alpha) * t["pos"][1]
                    t["misses"] = 0
                    used.add(best_i)
                else:
                    t = {"pos": [x, y], "misses": 0}
                    used.add(len(cls_tracks))
                    cls_tracks.append(t)
                frame_out.append([t["pos"][0], t["pos"][1], cls, score])

            survivors = []
            for i, t in enumerate(cls_tracks):
                if i not in used:
                    t["misses"] += 1
                if i in used or t["misses"] <= max_misses:
                    survivors.append(t)
            tracks[cls] = survivors
        out.append(frame_out)
    return out


# ---------------------------------------------------------------------
# Player: decode + draw only. No model calls happen on this path, so
# playback speed is bounded by video decode, not GPU inference.
# ---------------------------------------------------------------------
class Player:
    def __init__(self, video_path, dets_per_frame, fps, n_frames):
        self.cap = cv2.VideoCapture(str(video_path))
        self.dets = dets_per_frame
        self.fps = fps
        self.n_frames = n_frames
        self.idx = 0
        self._next_read_idx = 0
        self.playing = False
        self.show_dets = True
        self.slider_drag = False
        self.delay_ms = max(1, int(1000 / fps))

        self.root = tk.Tk()
        self.root.title(f"YouTube viewer - {video_path.name}")

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x")
        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4, pady=4)
        ttk.Button(ctrl, text="<< 1s", command=lambda: self.seek_relative(-int(fps))).pack(side="left")
        ttk.Button(ctrl, text="< frame", command=lambda: self.seek_relative(-1)).pack(side="left")
        ttk.Button(ctrl, text="frame >", command=lambda: self.seek_relative(1)).pack(side="left")
        ttk.Button(ctrl, text="1s >>", command=lambda: self.seek_relative(int(fps))).pack(side="left")
        self.dets_btn = ttk.Button(ctrl, text="Dets: on", command=self.toggle_dets)
        self.dets_btn.pack(side="left", padx=8)
        self.time_lbl = ttk.Label(ctrl, text="")
        self.time_lbl.pack(side="right", padx=8)

        self.slider = ttk.Scale(self.root, from_=0, to=max(0, n_frames - 1), orient="horizontal")
        self.slider.pack(fill="x", padx=4)
        self.slider.bind("<ButtonPress-1>", lambda e: setattr(self, "slider_drag", True))
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.seek_relative(-1))
        self.root.bind("<Right>", lambda e: self.seek_relative(1))
        self.root.bind("g", lambda e: self.toggle_dets())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.show_frame(0)
        self.root.after(self.delay_ms, self.tick)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")

    def toggle_dets(self):
        self.show_dets = not self.show_dets
        self.dets_btn.config(text=f"Dets: {'on' if self.show_dets else 'off'}")
        self.show_frame(self.idx)

    def seek_relative(self, n):
        self.show_frame(max(0, min(self.n_frames - 1, self.idx + n)))

    def on_slider_release(self, event):
        self.slider_drag = False
        self.show_frame(int(float(self.slider.get())))

    def show_frame(self, idx):
        idx = max(0, min(self.n_frames - 1, idx))
        if idx != self._next_read_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._next_read_idx = idx + 1
        self.idx = idx

        if self.show_dets and idx < len(self.dets):
            for x, y, cls, score in self.dets[idx]:
                color = COLOR_BGR.get(cls, (0, 255, 0))
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                cv2.putText(frame, f"{score:.2f}", (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w = 1200
        if w > max_w:
            scale = max_w / w
            rgb = cv2.resize(rgb, (max_w, int(h * scale)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self.photo)

        if not self.slider_drag:
            self.slider.set(idx)
        self.time_lbl.config(text=f"frame {idx}/{self.n_frames}  t={idx / self.fps:.1f}s")

    def tick(self):
        if self.playing:
            nxt = self.idx + 1
            if nxt >= self.n_frames:
                self.playing = False
                self.play_btn.config(text="Play")
            else:
                self.show_frame(nxt)
        self.root.after(self.delay_ms, self.tick)

    def run(self):
        self.root.mainloop()
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube video URL")
    ap.add_argument("--start", type=float, default=None, help="clip start, seconds")
    ap.add_argument("--end", type=float, default=None, help="clip end, seconds")
    ap.add_argument("--cookies", default=None,
                    help="path to a cookies.txt (Netscape format) for age/region-gated or private videos")
    ap.add_argument("--ckpt", default=str(ROOT / "data/frc/checkpoints/best.pt"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's peak_threshold")
    ap.add_argument("--stride", type=int, default=1,
                    help="run inference every Nth frame; detections are held "
                         "between sampled frames (bigger = faster prep, laggier dots)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="frames per GPU forward pass during the inference-caching pass")
    ap.add_argument("--no-smooth", action="store_true",
                    help="disable spatial smoothing and show raw per-frame detections")
    ap.add_argument("--smooth-alpha", type=float, default=0.3,
                    help="EMA weight on each new detection (lower = smoother/laggier, default 0.3)")
    ap.add_argument("--smooth-max-dist", type=float, default=120,
                    help="max pixel distance to associate a detection with an existing track")
    args = ap.parse_args()

    video_path = download_video(args.url, args.start, args.end, args.cookies)
    model, hm = load_model_and_cfg(pathlib.Path(args.ckpt))
    threshold = args.threshold if args.threshold is not None else hm["peak_threshold"]

    dets, fps, n_frames = build_or_load_cache(video_path, model, hm, threshold,
                                               args.stride, args.batch_size)
    if not args.no_smooth:
        dets = smooth_dets(dets, alpha=args.smooth_alpha, max_dist=args.smooth_max_dist)

    Player(video_path, dets, fps, n_frames).run()


if __name__ == "__main__":
    main()
