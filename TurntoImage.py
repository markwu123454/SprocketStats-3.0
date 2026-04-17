import json
import os
from google.cloud import storage

# --- CONFIGURATION ---
JSON_FILE = 'project-2-at-2026-04-17-06-43-6bc53548.json'
OUTPUT_DIR = '.'
CLASSES = ["Red Alliance Robot", "Blue Alliance Robot"]  # MUST match Label Studio exactly

# Initialize GCS Client (Will use your 'gcloud auth login' credentials)
client = storage.Client()

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Processing {len(data)} tasks...")

for task in data:
    # 1. Handle Filenames
    gcs_path = task['data']['image']  # e.g. gs://sprocket3/frame_01.jpg
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1]
    filename = os.path.basename(blob_name)
    basename = os.path.splitext(filename)[0]

    # 2. Download Image
    print(f"Downloading {filename}...")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(f"{OUTPUT_DIR}/images/{filename}")

    # 3. Create YOLO Label
    label_path = f"{OUTPUT_DIR}/labels/{basename}.txt"
    with open(label_path, 'w') as f_out:
        if not task.get('annotations'): continue

        results = task['annotations'][0].get('result', [])
        for res in results:
            if res['type'] != 'rectanglelabels': continue

            val = res['value']
            label_name = val['rectanglelabels'][0]
            if label_name not in CLASSES: continue

            class_id = CLASSES.index(label_name)

            # Convert LS (0-100 top-left) to YOLO (0-1 center)
            w = val['width'] / 100
            h = val['height'] / 100
            x_c = (val['x'] / 100) + (w / 2)
            y_c = (val['y'] / 100) + (h / 2)

            f_out.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

print("\n✅ Success! Your YOLO dataset is ready in the 'yolo_dataset' folder.")