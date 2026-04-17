# FRC Robot Detection Pipeline

End-to-end tooling to go from FRC match video → labeled dataset → YOLO training.

## Pipeline

```
video(s) ──► sample_frames.py ──► sampled_frames/*.jpg
                                        │
                                        ▼
                                  Label Studio
                                (labeling_config.xml)
                                        │
                                        ▼
                                  export.json
                                        │
                                        ▼
                                  ls_to_yolo.py
                                        │
                                        ▼
                                  yolo_dataset/  ──► Ultralytics training
```

## 1. Sample frames

```bash
pip install opencv-python yt-dlp
python sample_frames.py path/to/match.mp4 --out sampled_frames --step 12
```

The input can be any of:

- a local video file: `python sample_frames.py match.mp4`
- a folder of videos: `python sample_frames.py videos/`
- a YouTube URL: `python sample_frames.py "https://youtu.be/..."`

YouTube URLs are downloaded via `yt-dlp` (must be on PATH; `ffmpeg` too, which yt-dlp uses for merging and trimming). The downloaded video is cleaned up after sampling — pass `--keep-download` to keep it.

`--step 12` keeps one frame every 12, which at 30 fps gives 2.5 frames/second — dense enough to catch most robot positions without flooding you with near-duplicates. Bump it up (e.g. `--step 30`) if you're labeling solo and want fewer frames.

### Time ranges

Use `--start` and `--end` to limit sampling to a window of the video. All three time formats are accepted:

```bash
# From 2:30 to 5:00 of the match
python sample_frames.py match.mp4 --start 2:30 --end 5:00

# Just a plain seconds value works too
python sample_frames.py match.mp4 --start 150 --end 300

# Full HH:MM:SS for long recordings (e.g. a whole event stream)
python sample_frames.py "https://youtu.be/..." --start 1:23:45 --end 1:26:15 --step 12
```

When both `--start` and `--end` are given with a YouTube URL, `yt-dlp` downloads only that slice instead of the full video — a big win for 6-hour event livestreams.

## 2. Set up Label Studio

```bash
pip install label-studio
label-studio start
```

Then, in the web UI (opens at http://localhost:8080):

1. Create a new project.
2. **Settings → Labeling Interface → Code tab** — paste the contents of `label_studio_project/labeling_config.xml`.
3. **Import** the contents of `sampled_frames/` (drag-and-drop works; for large sets use the [local storage method](https://labelstud.io/guide/storage.html#Local-storage) so LS reads from disk instead of copying).
4. Label away. Hotkeys `1` = Red, `2` = Blue.

## 3. Export and convert to YOLO format

In Label Studio: **Export → JSON-MIN** (this format is the cleanest). Save it as `export.json`.

```bash
python ls_to_yolo.py export.json --images sampled_frames --out yolo_dataset
```

This produces:

```
yolo_dataset/
    data.yaml
    images/train/  images/val/
    labels/train/  labels/val/
```

with a default 80/20 train/val split (override with `--val-split 0.15`).

## 4. Train YOLO

Using Ultralytics (works for v8, v11, and the newer releases):

```bash
pip install ultralytics
yolo detect train data=yolo_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

Swap `yolo11n.pt` for whichever checkpoint you're targeting — the `data.yaml` is agnostic to the model version. For FRC footage, starting from a nano or small pretrained checkpoint and fine-tuning is usually enough; a few hundred labeled frames can already get you a workable detector.

## Tips specific to FRC footage

- **Label bumpers, not the whole robot silhouette.** Bumpers are the most reliable alliance-color signal and they generalize across seasons.
- **Field broadcasts vary.** If you're training for your scouting app, grab frames from several events and camera angles — single-event datasets overfit hard.
- **Watch out for pre-match and post-match frames** where robots are on carts or partially off-field. Either skip them or label them consistently; mixed treatment confuses the model.
- **Bumpers change color** during elimination matches when an alliance swaps sides. Make sure your labels reflect what's *currently* shown, not the team's usual alliance.
