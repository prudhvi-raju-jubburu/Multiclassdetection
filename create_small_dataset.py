import shutil
from pathlib import Path

# ============================================================
# SOURCE DATASET
# ============================================================

SOURCE = Path(
    r"C:\Users\jubbu\.cache\kagglehub\datasets\a7madmostafa\bdd100k-yolo\versions\4"
)

# ============================================================
# DESTINATION DATASET
# ============================================================

DEST = Path("dataset")

# Number of images
TRAIN_COUNT = 1500
VAL_COUNT = 375

# ============================================================
# SOURCE FOLDERS
# ============================================================

source_train_images = SOURCE / "train" / "images"
source_train_labels = SOURCE / "train" / "labels"

source_val_images = SOURCE / "val" / "images"
source_val_labels = SOURCE / "val" / "labels"

# ============================================================
# DESTINATION FOLDERS
# ============================================================

train_images = DEST / "train" / "images"
train_labels = DEST / "train" / "labels"

val_images = DEST / "val" / "images"
val_labels = DEST / "val" / "labels"

# Create directories
for folder in [
    train_images,
    train_labels,
    val_images,
    val_labels
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCTION TO COPY IMAGES + LABELS
# ============================================================

def copy_dataset(source_images, source_labels,
                 destination_images, destination_labels,
                 count):

    images = sorted(source_images.glob("*.jpg"))

    copied = 0

    for image in images:

        if copied >= count:
            break

        label = source_labels / f"{image.stem}.txt"

        # Make sure corresponding label exists
        if not label.exists():
            continue

        shutil.copy2(
            image,
            destination_images / image.name
        )

        shutil.copy2(
            label,
            destination_labels / label.name
        )

        copied += 1

    print(f"Copied {copied} images")
    print(f"Destination: {destination_images}")


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

print("\nCreating TRAIN dataset...")

copy_dataset(
    source_train_images,
    source_train_labels,
    train_images,
    train_labels,
    TRAIN_COUNT
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

print("\nCreating VALIDATION dataset...")

copy_dataset(
    source_val_images,
    source_val_labels,
    val_images,
    val_labels,
    VAL_COUNT
)


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET CREATION COMPLETED")
print("=" * 60)

print(
    "Train images :",
    len(list(train_images.glob("*.jpg")))
)

print(
    "Train labels :",
    len(list(train_labels.glob("*.txt")))
)

print(
    "Val images   :",
    len(list(val_images.glob("*.jpg")))
)

print(
    "Val labels   :",
    len(list(val_labels.glob("*.txt")))
)

print("=" * 60)