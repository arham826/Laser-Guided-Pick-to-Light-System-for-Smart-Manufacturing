import os
import random
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_dir = os.path.join(BASE_DIR, "dataset_raw", "images")
label_dir = os.path.join(BASE_DIR, "dataset_raw", "labels")

output_dir = os.path.join(BASE_DIR, "dataset_split")

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Get image list
images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

train_end = int(train_ratio * len(images))
val_end = int((train_ratio + val_ratio) * len(images))

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]


def copy_files(image_list, split):

    for img in image_list:

        label = img.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(
            os.path.join(image_dir, img),
            os.path.join(output_dir, "images", split, img),
        )

        shutil.copy(
            os.path.join(label_dir, label),
            os.path.join(output_dir, "labels", split, label),
        )


copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

print("Dataset successfully split!")