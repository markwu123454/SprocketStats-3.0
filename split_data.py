import os
import random
import shutil

source_images = "images"
source_labels = "labels"

target_dir = "yolo_data"
train_ratio = 0.7   # 70% train
val_ratio   = 0.15  # 15% val
test_ratio  = 0.15  # 15% test

# 1. Create the YOLO folder structure
for folder in ['train/images', 'train/labels',
               'val/images',   'val/labels',
               'test/images',  'test/labels']:
    os.makedirs(os.path.join(target_dir, folder), exist_ok=True)

# 2. Get a list of all your images
all_images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
random.shuffle(all_images)

# 3. Compute split indices
train_end = int(len(all_images) * train_ratio)
val_end   = train_end + int(len(all_images) * val_ratio)

train_images = all_images[:train_end]
val_images   = all_images[train_end:val_end]
test_images  = all_images[val_end:]


def move_files(files, folder_type):
    for filename in files:
        name_only = os.path.splitext(filename)[0]

        shutil.copy(os.path.join(source_images, filename),
                    os.path.join(target_dir, folder_type, 'images', filename))

        label_file = name_only + ".txt"
        if os.path.exists(os.path.join(source_labels, label_file)):
            shutil.copy(os.path.join(source_labels, label_file),
                        os.path.join(target_dir, folder_type, 'labels', label_file))


# 4. Copy files
print(f"Moving {len(train_images)} files to TRAIN")
move_files(train_images, 'train')
print(f"Moving {len(val_images)} files to VAL")
move_files(val_images, 'val')
print(f"Moving {len(test_images)} files to TEST")
move_files(test_images, 'test')