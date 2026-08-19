#!/usr/bin/env python3
"""
Step 2 viz (interactive) — pull a YouTube video, run YOLO over it, and scrub
through it with predicted bounding boxes overlaid.

Shares the download cache with hrnet so re-running on a URL that HRNet
already downloaded skips the download entirely.

Inference runs once up front and is cached to a .yolo_dets_t{thresh}_s{stride}.json
sidecar. Scrubbing then only decodes frames and draws cached boxes — no GPU
during playback.

Requires: ultralytics, yt-dlp, ffmpeg on PATH, tkinter, Pillow

Controls:
  Space               play / pause
  Left / Right        step one frame
  << 1s / 1s >>       seek by one second
  g                   toggle the detection overlay
  q / Esc             quit
  slider              scrub anywhere

Usage:
  python 02_youtube_view.py "https://www.youtube.com/watch?v=XXXX"
  python 02_youtube_view.py URL --start 60 --end 120
  python 02_youtube_view.py URL --stride 2 --threshold 0.4
"""
import argparse, json, pathlib, sys, time
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

ROOT      = pathlib.Path(__file__).parent.parent
# Shared video cache with hrnet so downloads aren't duplicated
CACHE_DIR = ROOT.parent / "hrnet" / "data" / "frc" / "youtube_cache"

# load credentials for any yt-dlp that needs them (cookies path only, no secrets here)
from dotenv import load_dotenv
_env = ROOT / ".env"
if not _env.exists():
    _env = ROOT.parent / "hrnet" / ".env"
load_dotenv(_env)

CLASSES   = ["blue", "red"]
COLOR_BGR = {"blue": (255, 120, 0), "red": (0, 0, 255)}


# ── download ──────────────────────────────────────────────────────────────────

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
        vid  = info["id"]

    candidates = [p for p in CACHE_DIR.glob(f"{vid}{tag}.*")
                  if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not candidates:
        raise SystemExit(f"[abort] download finished but no video file found for id={vid}")
    path = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"[download] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# ── inference cache ───────────────────────────────────────────────────────────

def cache_path_for(video_path, threshold, stride):
    return video_path.with_suffix(f".yolo_dets_t{threshold:.2f}_s{stride}.json")


def build_or_load_cache(video_path, model, threshold, stride):
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
        print(f"[cache] stale ({cached.get('n_frames')} != {n_frames}), recomputing")

    print(f"[infer] {n_frames} frames @ stride={stride} threshold={threshold} -> {cpath.name}")
    computed = {}
    i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            results = model(frame, conf=threshold, verbose=False)[0]
            boxes = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([cls_id, x1, y1, x2, y2, conf])
            computed[i] = boxes
        i += 1
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = len(computed) / elapsed if elapsed > 0 else 0
            eta  = (n_frames // stride - len(computed)) / rate if rate > 0 else 0
            print(f"[infer] {i}/{n_frames} scanned  ({rate:.1f} fps inferred, eta {eta:.0f}s)")
    cap.release()

    # forward-fill skipped frames
    dets_per_frame = [[] for _ in range(n_frames)]
    last = []
    for idx in range(n_frames):
        if idx in computed:
            last = computed[idx]
        dets_per_frame[idx] = last

    cpath.write_text(json.dumps({"n_frames": n_frames, "dets": dets_per_frame}))
    print(f"[infer] done: {len(computed)} inferred / {n_frames} total in "
          f"{time.time() - t0:.1f}s -> {cpath.name}")
    return dets_per_frame, fps, n_frames


# ── player ────────────────────────────────────────────────────────────────────

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
        self.root.title(f"YOLO viewer — {video_path.name}")

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x")
        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4, pady=4)
        ttk.Button(ctrl, text="<< 1s",   command=lambda: self.seek_relative(-int(fps))).pack(side="left")
        ttk.Button(ctrl, text="< frame", command=lambda: self.seek_relative(-1)).pack(side="left")
        ttk.Button(ctrl, text="frame >", command=lambda: self.seek_relative(1)).pack(side="left")
        ttk.Button(ctrl, text="1s >>",   command=lambda: self.seek_relative(int(fps))).pack(side="left")
        self.dets_btn = ttk.Button(ctrl, text="Dets: on", command=self.toggle_dets)
        self.dets_btn.pack(side="left", padx=8)
        self.time_lbl = ttk.Label(ctrl, text="")
        self.time_lbl.pack(side="right", padx=8)

        self.slider = ttk.Scale(self.root, from_=0, to=max(0, n_frames - 1), orient="horizontal")
        self.slider.pack(fill="x", padx=4)
        self.slider.bind("<ButtonPress-1>",  lambda e: setattr(self, "slider_drag", True))
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>",  lambda e: self.seek_relative(-1))
        self.root.bind("<Right>", lambda e: self.seek_relative(1))
        self.root.bind("g",       lambda e: self.toggle_dets())
        self.root.bind("q",       lambda e: self.root.destroy())
        self.root.bind("<Escape>",lambda e: self.root.destroy())

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
            for cls_id, x1, y1, x2, y2, conf in self.dets[idx]:
                cls   = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                color = COLOR_BGR.get(cls, (0, 255, 0))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}",
                            (int(x1), int(y1) - 6),
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
        self.time_lbl.config(text=f"frame {idx}/{self.n_frames}  t={idx/self.fps:.1f}s")

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url",        help="YouTube video URL")
    ap.add_argument("--start",    type=float, default=None)
    ap.add_argument("--end",      type=float, default=None)
    ap.add_argument("--cookies",  default=None,
                    help="path to cookies.txt for age/region-gated videos")
    ap.add_argument("--ckpt",     default=None,
                    help="path to best.pt (default: data/runs/robot_detection/weights/best.pt)")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="confidence threshold (default 0.25)")
    ap.add_argument("--stride",   type=int, default=1,
                    help="run inference every Nth frame (default 1)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[error] ultralytics not installed — pip install ultralytics")

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else \
           ROOT / "data" / "runs" / "robot_detection" / "weights" / "best.pt"
    if not ckpt.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt}\n"
                 f"  Run 02_train.py first, or pass --ckpt <path>")

    model = YOLO(str(ckpt))
    video_path = download_video(args.url, args.start, args.end, args.cookies)
    dets, fps, n_frames = build_or_load_cache(video_path, model, args.threshold, args.stride)
    Player(video_path, dets, fps, n_frames).run()


if __name__ == "__main__":
    main()
