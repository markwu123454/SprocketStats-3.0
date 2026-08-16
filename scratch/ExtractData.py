import json
import os

# 1. SETTINGS
JSON_FILE = 'project-2-at-2026-04-17-17-58-1b35df34.json'
OUTPUT_DIR = 'labels'
# Match these EXACTLY to your Label Studio labels
# The index in this list will be the number in the TXT file (0, 1, 2...)
CLASSES = ["Red Alliance Robot", "Blue Alliance Robot"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

for task in data:
    # Get the real filename from GCS path (e.g. NsI8...jpg)
    gcs_path = task['data']['image']
    base_name = os.path.splitext(os.path.basename(gcs_path))[0]

    label_file = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

    with open(label_file, 'w') as f_out:
        # Check if there are annotations
        if not task.get('annotations'):
            continue

        # Get the first annotation result
        results = task['annotations'][0].get('result', [])

        for res in results:
            if res['type'] != 'rectanglelabels':
                continue

            # Label Studio uses 0-100, YOLO uses 0-1
            val = res['value']
            label = val['rectanglelabels'][0]

            if label not in CLASSES:
                print(f"Warning: Unknown label {label}")
                continue

            class_id = CLASSES.index(label)

            # YOLO Math: Center_X, Center_Y, Width, Height (all 0 to 1)
            # LS gives top-left x/y, we need to convert to center
            w = val['width'] / 100
            h = val['height'] / 100
            x_center = (val['x'] / 100) + (w / 2)
            y_center = (val['y'] / 100) + (h / 2)

            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print(f"Finished! Labels are in {OUTPUT_DIR}")