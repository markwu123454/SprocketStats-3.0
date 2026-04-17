# FRC Robot Detection Pipeline

End-to-end tooling to go from FRC match video → labeled dataset → YOLO training.

## Pipeline

```
video(s) ──► SampleFrames.py ──► sampled_frames/*.jpg
                                        │
                                        ▼
                                  Label Studio
                                (labeling_config.xml)
                                        │
                                  ┌─────┴──────┐
                                  ▼            ▼
                          (local images)   (GCS images)
                                  │            │
                                  ▼            ▼
                          ExtractData.py   TurntoImage.py
                                  │            │
                                  └─────┬──────┘
                                        ▼
                                  split_data.py ──► yolo_data/
                                        │
                                        ▼
                                    train.py  ──► trained model
```

## 1. Sample frames

```bash
pip install opencv-python yt-dlp
python SampleFrames.py path/to/match.mp4 --out sampled_frames --step 12
```

The input can be any of:

- a local video file: `python SampleFrames.py match.mp4`
- a folder of videos: `python SampleFrames.py videos/`
- a YouTube URL: `python SampleFrames.py "https://youtu.be/..."`

YouTube URLs are downloaded via `yt-dlp` (must be on PATH; `ffmpeg` too, which yt-dlp uses for merging and trimming). The downloaded video is cleaned up after sampling — pass `--keep-download` to keep it.

`--step 12` keeps one frame every 12, which at 30 fps gives 2.5 frames/second — dense enough to catch most robot positions without flooding you with near-duplicates. Bump it up (e.g. `--step 30`) if you're labeling solo and want fewer frames.

### Time ranges

Use `--start` and `--end` to limit sampling to a window of the video. All three time formats are accepted:

```bash
# From 2:30 to 5:00 of the match
python SampleFrames.py match.mp4 --start 2:30 --end 5:00

# Just a plain seconds value works too
python SampleFrames.py match.mp4 --start 150 --end 300

# Full HH:MM:SS for long recordings (e.g. a whole event stream)
python SampleFrames.py "https://youtu.be/..." --start 1:23:45 --end 1:26:15 --step 12
```

When both `--start` and `--end` are given with a YouTube URL, `yt-dlp` downloads only that slice instead of the full video — a big win for 6-hour event livestreams.

## 2. Set up Label Studio

```bash
pip install label-studio
label-studio start
```

Then, in the web UI (opens at http://localhost:8080):

1. Create a new project.
2. **Settings → Labeling Interface → Code tab** — paste the contents of `labeling_config.xml`.
3. **Import** the contents of `sampled_frames/` (drag-and-drop works; for large sets use the [local storage method](https://labelstud.io/guide/storage.html#Local-storage) so LS reads from disk instead of copying, or use GCS storage).
4. Label away. Hotkeys `1` = Red, `2` = Blue.

## 3. Export and convert to YOLO format

In Label Studio: **Export → JSON** (full format). Save the export file.

There are two paths depending on where your images live:

### Option A: Images stored locally

If the source images are already on disk (e.g. in `sampled_frames/`), use `ExtractData.py` to generate YOLO label files from the export:

```bash
python ExtractData.py
```

Edit the `JSON_FILE` variable at the top of the script to point to your export filename. Labels are written to `labels/`.

### Option B: Images stored in Google Cloud Storage

If your Label Studio project references images in a GCS bucket (e.g. `gs://sprocket3/...`), use `TurntoImage.py` to download the images and create labels in one step:

```bash
pip install google-cloud-storage
gcloud auth login
python TurntoImage.py
```

Edit the `JSON_FILE` variable at the top of the script to point to your export filename. Images are downloaded to `images/` and labels are written to `labels/`.

## 4. Split into train/val

Once you have `images/` and `labels/` directories ready, run `split_data.py` to create the YOLO folder structure:

```bash
python split_data.py
```

This copies images and labels into an 80/20 train/val split:

```
yolo_data/
    train/images/  train/labels/
    val/images/    val/labels/
```

The split ratio is configured via `split_ratio` at the top of the script.

## 5. Train YOLO

The `train.py` script handles model loading and training using Ultralytics:

```bash
pip install ultralytics torch
python train.py
```

This loads `yolo11x.pt` (extra-large checkpoint) and trains for 100 epochs at 640px on GPU. It will auto-detect your GPU and print a hardware check at startup. Results are saved to `my_yolo_project/version_1/`.

To adjust settings (model size, batch size, epochs, etc.), edit the parameters in `train.py`. For FRC footage, starting from a nano or small pretrained checkpoint (`yolo11n.pt`, `yolo11s.pt`) and fine-tuning is usually enough; a few hundred labeled frames can already get you a workable detector.

### Alternative: LS2YOLO.py (all-in-one)

`LS2YOLO.py` is a self-contained alternative that handles export conversion, image copying, and train/val splitting in one command:

```bash
python LS2YOLO.py export.json --images sampled_frames --out yolo_dataset
```

It accepts both JSON and JSON-MIN exports and produces a ready-to-train dataset with `data.yaml` included. Override the default 80/20 split with `--val-split 0.15`.

## Tips specific to FRC footage

- **Label bumpers, not the whole robot silhouette.** Bumpers are the most reliable alliance-color signal and they generalize across seasons.
- **Field broadcasts vary.** If you're training for your scouting app, grab frames from several events and camera angles — single-event datasets overfit hard.
- **Watch out for pre-match and post-match frames** where robots are on carts or partially off-field. Either skip them or label them consistently; mixed treatment confuses the model.
- **Bumpers change color** during elimination matches when an alliance swaps sides. Make sure your labels reflect what's *currently* shown, not the team's usual alliance.
