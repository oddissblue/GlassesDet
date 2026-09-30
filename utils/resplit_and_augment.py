from pathlib import Path
from collections import Counter, defaultdict
import random
import shutil

import cv2
import albumentations as A


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\oalyami\CAV-DDE\Repos\External\GlassesDet\data\DST_GlassesDet"
)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# Final training set = originals * this number
# Example: 107 originals -> ~321 total train images
TRAIN_MULTIPLIER = 3

CLASS_NAMES = {
    0: "not_wearing_glasses",
    1: "wearing_glasses",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)


# ============================================================
# Albumentations pipeline
# ============================================================

augment = A.Compose(
    [
        A.HorizontalFlip(p=0.5),

        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=15,
                    val_shift_limit=15,
                    p=1.0,
                ),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=1.0,
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.04),
                    p=1.0,
                ),
            ],
            p=0.15,
        ),

        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-5, 5),
            shear=(-2, 2),
            p=0.4,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.50,
    ),
)


# ============================================================
# Helpers
# ============================================================

def find_image_for_stem(image_dir: Path, stem: str):
    """Find an image matching a label/image stem."""

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    return None


def read_yolo_label(label_path: Path):
    """
    Read a YOLO label.

    Returns:
        boxes:
            [
                [x_center, y_center, width, height],
                ...
            ]

        classes:
            [class_id, ...]
    """

    boxes = []
    classes = []

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return boxes, classes

    for line in text.splitlines():

        parts = line.strip().split()

        if len(parts) < 5:
            raise ValueError(
                f"Invalid YOLO annotation:\n"
                f"{label_path}\n"
                f"Line: {line}"
            )

        class_id = int(float(parts[0]))

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        classes.append(class_id)

        boxes.append(
            [
                x_center,
                y_center,
                width,
                height,
            ]
        )

    return boxes, classes


def write_yolo_label(
    label_path: Path,
    boxes,
    classes,
):
    """Write YOLO annotations."""

    lines = []

    for class_id, box in zip(classes, boxes):

        x_center, y_center, width, height = box

        lines.append(
            f"{int(class_id)} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def primary_class(classes):
    """
    Determine the class used for stratification.

    This dataset is expected to be a glasses detection dataset where
    each image belongs to one semantic class.

    If multiple objects exist, use the most common class.
    """

    if not classes:
        return None

    counts = Counter(classes)

    return counts.most_common(1)[0][0]


def copy_pair(
    image_path,
    label_path,
    destination,
):
    """Copy image + YOLO label."""

    image_dir = destination / "images"
    label_dir = destination / "labels"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        image_path,
        image_dir / image_path.name,
    )

    shutil.copy2(
        label_path,
        label_dir / label_path.name,
    )


# ============================================================
# Collect existing dataset
# ============================================================

print("=" * 70)
print("COLLECTING EXISTING DATASET")
print("=" * 70)

samples = []

seen_images = set()

for split in ["train", "val", "test"]:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists():
        continue

    for image_path in image_dir.iterdir():

        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        # Prevent accidental duplicate collection
        key = image_path.name.lower()

        if key in seen_images:
            raise RuntimeError(
                f"Duplicate image found across splits:\n"
                f"{image_path.name}"
            )

        seen_images.add(key)

        label_path = (
            label_dir / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for:\n"
                f"{image_path}"
            )

        boxes, classes = read_yolo_label(
            label_path
        )

        cls = primary_class(classes)

        samples.append(
            {
                "image": image_path,
                "label": label_path,
                "boxes": boxes,
                "classes": classes,
                "stratify_class": cls,
            }
        )


print(f"Total samples found: {len(samples)}")


# ============================================================
# Separate positive and negative samples
# ============================================================

positive_samples = [
    sample
    for sample in samples
    if sample["stratify_class"] is not None
]

negative_samples = [
    sample
    for sample in samples
    if sample["stratify_class"] is None
]


print(
    f"Positive samples   : {len(positive_samples)}"
)

print(
    f"Negative samples   : {len(negative_samples)}"
)


# ============================================================
# Group positives by class
# ============================================================

by_class = defaultdict(list)

for sample in positive_samples:

    by_class[
        sample["stratify_class"]
    ].append(sample)


print("\nOriginal class distribution:")

for class_id in sorted(by_class):

    name = CLASS_NAMES.get(
        class_id,
        f"class_{class_id}",
    )

    print(
        f"  {class_id} "
        f"({name:<22}): "
        f"{len(by_class[class_id])}"
    )


# ============================================================
# Stratified split
# ============================================================

split_samples = {
    "train": [],
    "val": [],
    "test": [],
}


for class_id, class_samples in sorted(
    by_class.items()
):

    rng = random.Random(
        SEED + class_id
    )

    rng.shuffle(class_samples)

    total = len(class_samples)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    # Remainder goes to test
    test_count = (
        total
        - train_count
        - val_count
    )

    train_items = class_samples[
        :train_count
    ]

    val_items = class_samples[
        train_count:
        train_count + val_count
    ]

    test_items = class_samples[
        train_count + val_count:
    ]

    split_samples["train"].extend(
        train_items
    )

    split_samples["val"].extend(
        val_items
    )

    split_samples["test"].extend(
        test_items
    )

    name = CLASS_NAMES.get(
        class_id,
        f"class_{class_id}",
    )

    print(
        f"\n{name}:"
    )

    print(
        f"  train: {len(train_items)}"
    )

    print(
        f"  val  : {len(val_items)}"
    )

    print(
        f"  test : {len(test_items)}"
    )


# ============================================================
# Force ALL negatives into train
# ============================================================

split_samples["train"].extend(
    negative_samples
)


# Shuffle final split lists
for split_name in split_samples:

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        split_samples[split_name]
    )


# ============================================================
# Safety checks
# ============================================================

train_names = {
    sample["image"].name
    for sample in split_samples["train"]
}

val_names = {
    sample["image"].name
    for sample in split_samples["val"]
}

test_names = {
    sample["image"].name
    for sample in split_samples["test"]
}


assert train_names.isdisjoint(val_names)
assert train_names.isdisjoint(test_names)
assert val_names.isdisjoint(test_names)


for sample in negative_samples:

    assert (
        sample["image"].name
        in train_names
    )


# ============================================================
# IMPORTANT:
# Copy originals to temporary directory BEFORE deleting splits
# ============================================================

TEMP_DIR = DATASET_DIR.parent / (
    DATASET_DIR.name + "_resplit_temp"
)


if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)


TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for split_name, items in split_samples.items():

    destination = (
        TEMP_DIR / split_name
    )

    for sample in items:

        copy_pair(
            sample["image"],
            sample["label"],
            destination,
        )


# ============================================================
# Remove old splits
# ============================================================

for split in ["train", "val", "test"]:

    split_path = DATASET_DIR / split

    if split_path.exists():
        shutil.rmtree(split_path)


# ============================================================
# Move reconstructed splits into dataset
# ============================================================

for split in ["train", "val", "test"]:

    source = TEMP_DIR / split

    destination = (
        DATASET_DIR / split
    )

    if source.exists():

        shutil.move(
            str(source),
            str(destination),
        )


shutil.rmtree(TEMP_DIR)


# ============================================================
# Print post-split distribution
# ============================================================

print("\n" + "=" * 70)
print("STRATIFIED SPLIT COMPLETE")
print("=" * 70)


for split_name in [
    "train",
    "val",
    "test",
]:

    counts = Counter()

    negatives = 0

    for sample in split_samples[split_name]:

        cls = sample["stratify_class"]

        if cls is None:
            negatives += 1
        else:
            counts[cls] += 1

    print(
        f"\n{split_name.upper()} "
        f"({len(split_samples[split_name])} originals)"
    )

    for class_id in sorted(CLASS_NAMES):

        print(
            f"  {CLASS_NAMES[class_id]:<22}: "
            f"{counts[class_id]}"
        )

    print(
        f"  {'negative':<22}: "
        f"{negatives}"
    )


# ============================================================
# Augment TRAIN only
# ============================================================

print("\n" + "=" * 70)
print("AUGMENTING TRAINING SET")
print("=" * 70)


train_originals = list(
    split_samples["train"]
)

original_train_count = len(
    train_originals
)

target_train_count = (
    original_train_count
    * TRAIN_MULTIPLIER
)

augmentations_needed = (
    target_train_count
    - original_train_count
)


print(
    f"Original train size : "
    f"{original_train_count}"
)

print(
    f"Target train size   : "
    f"{target_train_count}"
)

print(
    f"Augmentations needed: "
    f"{augmentations_needed}"
)


train_image_dir = (
    DATASET_DIR
    / "train"
    / "images"
)

train_label_dir = (
    DATASET_DIR
    / "train"
    / "labels"
)


# ============================================================
# Generate augmented samples
# ============================================================

augmentation_index = 0

while augmentation_index < augmentations_needed:

    source_sample = train_originals[
        augmentation_index
        % len(train_originals)
    ]

    source_image_path = (
        train_image_dir
        / source_sample["image"].name
    )

    source_label_path = (
        train_label_dir
        / source_sample["label"].name
    )

    image = cv2.imread(
        str(source_image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n"
            f"{source_image_path}"
        )

    boxes, classes = read_yolo_label(
        source_label_path
    )

    # Albumentations can also handle an empty bbox list
    transformed = augment(
        image=image,
        bboxes=boxes,
        class_labels=classes,
    )

    augmented_image = transformed[
        "image"
    ]

    augmented_boxes = list(
        transformed["bboxes"]
    )

    augmented_classes = list(
        transformed["class_labels"]
    )

    # If original was positive but augmentation removed
    # every box, retry rather than accidentally turning
    # it into a negative sample.
    if classes and not augmented_boxes:
        continue

    augmentation_index += 1

    output_stem = (
        f"{source_image_path.stem}"
        f"_aug_{augmentation_index:04d}"
    )

    output_image_path = (
        train_image_dir
        / f"{output_stem}.jpg"
    )

    output_label_path = (
        train_label_dir
        / f"{output_stem}.txt"
    )

    success = cv2.imwrite(
        str(output_image_path),
        augmented_image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to write:\n"
            f"{output_image_path}"
        )

    write_yolo_label(
        output_label_path,
        augmented_boxes,
        augmented_classes,
    )


# ============================================================
# Final verification
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)


for split_name in [
    "train",
    "val",
    "test",
]:

    image_dir = (
        DATASET_DIR
        / split_name
        / "images"
    )

    label_dir = (
        DATASET_DIR
        / split_name
        / "labels"
    )

    image_count = sum(
        1
        for path in image_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    label_count = sum(
        1
        for path in label_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            == ".txt"
        )
    )

    print(
        f"{split_name:<5}: "
        f"{image_count:>4} images | "
        f"{label_count:>4} labels"
    )


print("\nDone.")

print(
    "\nValidation/test were NOT augmented."
)

print(
    "All augmentation was applied only "
    "to the training set."
)