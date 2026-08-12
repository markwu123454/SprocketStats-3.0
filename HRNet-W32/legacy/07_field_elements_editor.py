#!/usr/bin/env python3
"""
Step 7 (interactive) — manually digitize field elements from a reference
field image, save them keyed by year for 06_field_calibrate.py to load as
built-in draggable elements (so a fresh calibration starts with them
pre-placed on the field side instead of needing "+ Element" each time).

Workflow:
  1. Drag a top-down field image onto the window (or "Load image...").
  2. Click the field's 4 real corners, in TL -> TR -> BR -> BL order (TL/TR
     share the long/length edge, TL/BL share the short/width edge) — this
     defines the image-pixel -> field-fraction mapping. Drag to adjust.
  3. "+ Element", name it, then click points on the image to trace its
     outline (any number of points, any shape). Press Finish (f) to close
     the polygon; "+ Element" again for the next one.
  4. "Save as year" writes every finished element's points (converted to
     field-length/width fractions via the boundary) into a shared JSON at
     data/frc/field_elements.json, keyed by year. 06_field_calibrate.py
     reads this file first and falls back to its own hardcoded defaults
     (currently just the 2026 HUB bars) if a year has nothing saved.

Work-in-progress state autosaves to a per-year draft file so a closed
window doesn't lose your clicking; "Load existing" pulls in whatever was
last saved (or drafted) for the chosen year, mapped onto the CURRENT
boundary so you can touch up an element set on a fresh image.

Requires: tkinterdnd2 (pip) for real OS drag-and-drop; falls back to the
"Load image..." button otherwise.

Controls:
  left-click             place next boundary corner / add a vertex to the
                          active (in-progress) element
  drag a marker           adjust any boundary corner or element vertex
  right-click a marker    delete that vertex (boundary corner just clears)
  f / "Finish element"    close the active element's polygon (needs >=3 pts)
  Escape                  discard the active (unfinished) element
  s / "Save as year"      write to field_elements.json
  q                       quit

Usage:
  python 07_field_elements_editor.py --year 2026
  python 07_field_elements_editor.py --year 2026 --image field.png
"""
import argparse, json, pathlib
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAVE_DND = True
except ImportError:
    HAVE_DND = False

ROOT = pathlib.Path(__file__).parent
FIELD_ELEMENTS_PATH = ROOT / "data/frc/field_elements.json"

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
CORNER_COLOR = {"TL": (255, 255, 0), "TR": (0, 255, 255), "BR": (255, 0, 255), "BL": (0, 255, 0)}
PALETTE = [(0, 165, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255),
          (180, 105, 255), (0, 200, 200), (255, 128, 0)]


def load_all():
    if FIELD_ELEMENTS_PATH.exists():
        return json.loads(FIELD_ELEMENTS_PATH.read_text())
    return {}


def save_all(data):
    FIELD_ELEMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIELD_ELEMENTS_PATH.write_text(json.dumps(data, indent=2))


def draft_path_for(year):
    return ROOT / "data/frc" / f"field_elements.draft_{year}.json"


class Editor:
    MAX_W = 1000
    HIT_R = 12
    MARKER_R = 6

    def __init__(self, year, image_path=None):
        self.year = year
        self.img_bgr = None
        self.img_path = None
        self.boundary = [None, None, None, None]   # image px, TL/TR/BR/BL
        self.elements = []      # [{"name","points":[[x,y],...],"finished":bool}]
        self.active_idx = None
        self.drag = None        # ("boundary", i) | ("element", ei, pi)
        self.scale = 1.0

        self._load_draft()
        if image_path:
            self._load_image(image_path)

        self.root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
        self.root.title(f"Field element editor - year {year}")

        self.canvas_lbl = tk.Label(self.root)
        self.canvas_lbl.pack(padx=2, pady=2)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Load image...", command=self.pick_image).pack(side="left")
        ttk.Button(ctrl, text="+ Element", command=self.add_element).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Finish element", command=self.finish_active).pack(side="left")
        ttk.Button(ctrl, text="Delete last", command=self.delete_last).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Reset boundary", command=self.reset_boundary).pack(side="left")
        ttk.Button(ctrl, text="Load existing", command=self.load_existing).pack(side="left", padx=8)
        ttk.Button(ctrl, text=f"Save as {year}", command=self.save).pack(side="right", padx=4)

        self.status = ttk.Label(self.root, text="")
        self.status.pack(fill="x", padx=4, pady=(0, 4))

        self.canvas_lbl.bind("<ButtonPress-1>", self.on_press)
        self.canvas_lbl.bind("<B1-Motion>", self.on_drag)
        self.canvas_lbl.bind("<ButtonRelease-1>", self.on_release)
        self.canvas_lbl.bind("<ButtonPress-3>", self.on_right_click)

        if HAVE_DND:
            self.canvas_lbl.drop_target_register(DND_FILES)
            self.canvas_lbl.dnd_bind("<<Drop>>", self.on_drop)

        self.root.bind("f", lambda e: self.finish_active())
        self.root.bind("<Escape>", lambda e: self.cancel_active())
        self.root.bind("s", lambda e: self.save())
        self.root.bind("q", lambda e: self.root.destroy())

        self.render()

    # -- image loading ---------------------------------------------------
    def _load_image(self, path):
        path = pathlib.Path(path)
        img = cv2.imread(str(path))
        if img is None:
            self.status.config(text=f"could not read image: {path}")
            return
        self.img_bgr = img
        self.img_path = path
        if hasattr(self, "root"):
            self.status.config(text=f"loaded {path.name} ({img.shape[1]}x{img.shape[0]})")
            self.render()

    def pick_image(self):
        path = filedialog.askopenfilename(
            title="Choose reference field image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path:
            self._load_image(path)
            self._save_draft()

    def on_drop(self, event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0] if "} {" in raw else raw
        self._load_image(path)
        self._save_draft()

    # -- draft persistence (per year, raw pixel-space working state) -----
    def _draft_data(self):
        return {
            "image": str(self.img_path) if self.img_path else None,
            "boundary": self.boundary,
            "elements": self.elements,
        }

    def _save_draft(self):
        draft_path_for(self.year).write_text(json.dumps(self._draft_data(), indent=2))

    def _load_draft(self):
        p = draft_path_for(self.year)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        if data.get("image"):
            img = cv2.imread(data["image"])
            if img is not None:
                self.img_bgr = img
                self.img_path = pathlib.Path(data["image"])
        if data.get("boundary"):
            self.boundary = [list(c) if c else None for c in data["boundary"]]
        self.elements = data.get("elements", [])
        print(f"[draft] resumed year={self.year} from {p.name}")

    # -- boundary / homography -------------------------------------------
    def reset_boundary(self):
        self.boundary = [None, None, None, None]
        self._save_draft()
        self.render()

    def _fwd_h(self):
        """image px -> unit square (x=length frac, y=width frac), or None."""
        if any(c is None for c in self.boundary):
            return None
        src = np.float32(self.boundary)
        dst = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        return cv2.getPerspectiveTransform(src, dst)

    # -- element editing ----------------------------------------------------
    def add_element(self):
        self._auto_finish_active(warn=True)
        name = simpledialog.askstring("Element name", "Name for this field element:", parent=self.root)
        if not name:
            return
        self.elements.append({"name": name, "points": [], "finished": False})
        self.active_idx = len(self.elements) - 1
        self.status.config(text=f"click points to trace '{name}', then press Finish (f)")
        self.render()

    def _auto_finish_active(self, warn=False):
        if self.active_idx is None:
            return
        el = self.elements[self.active_idx]
        if len(el["points"]) >= 3:
            el["finished"] = True
        else:
            if warn:
                self.status.config(text=f"discarded '{el['name']}' (needs >=3 points)")
            del self.elements[self.active_idx]
        self.active_idx = None

    def finish_active(self):
        if self.active_idx is None:
            return
        el = self.elements[self.active_idx]
        if len(el["points"]) < 3:
            self.status.config(text=f"'{el['name']}' needs >=3 points before it can be finished")
            return
        self._auto_finish_active()
        self._save_draft()
        self.render()

    def cancel_active(self):
        if self.active_idx is None:
            return
        name = self.elements[self.active_idx]["name"]
        del self.elements[self.active_idx]
        self.active_idx = None
        self.status.config(text=f"discarded '{name}'")
        self._save_draft()
        self.render()

    def delete_last(self):
        if not self.elements:
            return
        was_active = (self.active_idx == len(self.elements) - 1)
        self.elements.pop()
        if was_active:
            self.active_idx = None
        self._save_draft()
        self.render()

    def load_existing(self):
        h = self._fwd_h()
        if h is None:
            self.status.config(text="set all 4 boundary corners first")
            return
        data = load_all()
        entries = data.get(str(self.year), [])
        if not entries:
            self.status.config(text=f"no saved elements for year {self.year} in {FIELD_ELEMENTS_PATH.name}")
            return
        h_inv = np.linalg.inv(h)
        for e in entries:
            frac_pts = np.float32(e["points"]).reshape(-1, 1, 2)
            img_pts = cv2.perspectiveTransform(frac_pts, h_inv).reshape(-1, 2).tolist()
            self.elements.append({"name": e["name"], "points": img_pts, "finished": True})
        self.status.config(text=f"loaded {len(entries)} saved elements for year {self.year}")
        self._save_draft()
        self.render()

    def save(self):
        self._auto_finish_active(warn=True)
        h = self._fwd_h()
        if h is None:
            self.status.config(text="set all 4 boundary corners before saving")
            return
        out = []
        skipped = 0
        for e in self.elements:
            if len(e["points"]) < 3:
                skipped += 1
                continue
            pts = cv2.perspectiveTransform(np.float32(e["points"]).reshape(-1, 1, 2), h).reshape(-1, 2)
            out.append({"name": e["name"], "points": [[round(float(x), 5), round(float(y), 5)] for x, y in pts]})
        data = load_all()
        data[str(self.year)] = out
        save_all(data)
        msg = f"saved {len(out)} elements for year {self.year} -> {FIELD_ELEMENTS_PATH.name}"
        if skipped:
            msg += f" ({skipped} skipped, <3 points)"
        self.status.config(text=msg)

    # -- hit-testing / mouse -------------------------------------------------
    def find_hit(self, ex, ey):
        s = self.scale
        for i, c in enumerate(self.boundary):
            if c is not None and (c[0] * s - ex) ** 2 + (c[1] * s - ey) ** 2 <= self.HIT_R ** 2:
                return ("boundary", i)
        for ei, el in enumerate(self.elements):
            for pi, p in enumerate(el["points"]):
                if (p[0] * s - ex) ** 2 + (p[1] * s - ey) ** 2 <= self.HIT_R ** 2:
                    return ("element", ei, pi)
        return None

    def on_press(self, event):
        if self.img_bgr is None:
            return
        hit = self.find_hit(event.x, event.y)
        if hit:
            self.drag = hit
            return
        pt = [event.x / self.scale, event.y / self.scale]
        for i, c in enumerate(self.boundary):
            if c is None:
                self.boundary[i] = pt
                self.drag = ("boundary", i)
                self._save_draft()
                self.render()
                return
        if self.active_idx is not None:
            self.elements[self.active_idx]["points"].append(pt)
            self._save_draft()
            self.render()

    def on_drag(self, event):
        if self.drag is None:
            return
        pt = [event.x / self.scale, event.y / self.scale]
        kind = self.drag
        if kind[0] == "boundary":
            self.boundary[kind[1]] = pt
        else:
            self.elements[kind[1]]["points"][kind[2]] = pt
        self.render()

    def on_release(self, event):
        if self.drag is not None:
            self.drag = None
            self._save_draft()

    def on_right_click(self, event):
        hit = self.find_hit(event.x, event.y)
        if hit is None:
            return
        if hit[0] == "boundary":
            self.boundary[hit[1]] = None
        else:
            _, ei, pi = hit
            del self.elements[ei]["points"][pi]
            if not self.elements[ei]["points"]:
                if self.active_idx == ei:
                    self.active_idx = None
                del self.elements[ei]
        self._save_draft()
        self.render()

    # -- rendering -------------------------------------------------------
    def render(self):
        if self.img_bgr is None:
            canvas = np.full((500, 900, 3), 40, np.uint8)
            cv2.putText(canvas, "drag a field image in, or use Load image...", (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            self._show(canvas)
            self.status.config(text="no image loaded")
            return

        img = self.img_bgr.copy()
        if all(c is not None for c in self.boundary):
            cv2.polylines(img, np.int32([self.boundary]), True, (0, 255, 0), 2)
        for name, c in zip(CORNER_NAMES, self.boundary):
            if c is not None:
                self._marker(img, c, CORNER_COLOR[name], name, shape="circle")

        for ei, el in enumerate(self.elements):
            color = PALETTE[ei % len(PALETTE)]
            pts = np.int32(el["points"])
            if el.get("finished") and len(pts) >= 3:
                cv2.polylines(img, [pts], True, color, 2)
            elif len(pts) >= 2:
                cv2.polylines(img, [pts], False, color, 2)
            for p in el["points"]:
                self._marker(img, p, color, "", shape="square", small=True)
            if len(pts):
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(img, el["name"], (int(cx) + 6, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

        self._show(img)

        n_corners = sum(c is not None for c in self.boundary)
        n_finished = sum(1 for e in self.elements if e.get("finished"))
        msg = f"boundary {n_corners}/4"
        if n_corners < 4:
            msg += f" (click to place {CORNER_NAMES[n_corners]})"
        msg += f"  |  elements: {n_finished} finished"
        if self.active_idx is not None:
            el = self.elements[self.active_idx]
            msg += f", '{el['name']}' active ({len(el['points'])} pts, f to finish)"
        self.status.config(text=msg)

    def _marker(self, img, xy, color, label, shape="circle", small=False):
        p = (int(xy[0]), int(xy[1]))
        r = self.MARKER_R if not small else 4
        if shape == "square":
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, -1)
        else:
            cv2.circle(img, p, r + 2, color, -1)
            cv2.circle(img, p, r + 2, (0, 0, 0), 1)
        if label:
            cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

    def _show(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.scale = min(1.0, self.MAX_W / w)
        disp = cv2.resize(img_bgr, (int(w * self.scale), int(h * self.scale))) if self.scale < 1.0 else img_bgr
        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas_lbl.image = photo
        self.canvas_lbl.configure(image=photo)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="FRC season to save these elements under")
    ap.add_argument("--image", default=None, help="reference field image to start with")
    args = ap.parse_args()

    if not HAVE_DND:
        print("[warn] tkinterdnd2 not installed -- drag-and-drop disabled, "
              "use the 'Load image...' button instead (pip install tkinterdnd2)")

    Editor(args.year, image_path=args.image).run()


if __name__ == "__main__":
    main()
