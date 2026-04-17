import os
import random
import shutil

source_images = "images"
source_labels = "labels"

target_dir = "yolo_data"
split_ratio = 0.8  # 80% for training, 20% for validation

# 1. Create the YOLO folder structure
for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    os.makedirs(os.path.join(target_dir, folder), exist_ok=True)

# 2. Get a list of all your images
# (Assumes they are .jpg, change to .png if needed)
all_images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
random.shuffle(all_images)

split_index = int(len(all_images) * split_ratio)
train_images = all_images[:split_index]
val_images = all_images[split_index:]


def move_files(files, folder_type):
    for filename in files:
        # Get the name without the .jpg (e.g. "image_01")
        name_only = os.path.splitext(filename)[0]

        # Move Image
        shutil.copy(os.path.join(source_images, filename),
                    os.path.join(target_dir, folder_type, 'images', filename))

        # Move Label (check if it exists first)
        label_file = name_only + ".txt"
        if os.path.exists(os.path.join(source_labels, label_file)):
            shutil.copy(os.path.join(source_labels, label_file),
                        os.path.join(target_dir, folder_type, 'labels', label_file))


# 4. Do the actual moving
print(f"Moving {len(train_images)} files to TRAIN")
move_files(train_images, 'train')
print(f"Moving {len(val_images)} files to VAL")
move_files(val_images, 'val')
