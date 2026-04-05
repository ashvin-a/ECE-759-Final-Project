"""
prepare_dataset.py — Extract 64x64 crops from the Roboflow rock dataset.

Expected Roboflow export format: YOLOv8 (images + labels in .txt files).
Label format per line: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

Usage:
    python scripts/prepare_dataset.py \
        --data_dir data/roboflow/train/\
        --out_dir  data/crops \
        --neg_ratio 2

Output layout:
    data/crops/
        train/pos/  *.png   64x64 rock patches
        train/neg/  *.png   64x64 background patches
        test/pos/   *.png
        test/neg/   *.png
        train_images.txt    image paths used for training split
        test_images.txt     image paths used for test split
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WIN = 64
TRAIN_FRAC = 0.8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="project/data/roboflow/train/",
                   help="Root of Roboflow dataset (contains images/ and labels/)")
    p.add_argument("--out_dir",   default="project/data/crops",
                   help="Output directory for crops")
    p.add_argument("--neg_ratio", type=int, default=2,
                   help="Negative crops per positive (default 2)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_boxes_yolo(label_path: Path, img_w: int, img_h: int):
    """Return list of [x1,y1,x2,y2] pixel boxes from a YOLO label file.

    Supports both standard YOLO format (class cx cy w h) and polygon/OBB
    format (class x1 y1 x2 y2 ... xn yn). For polygons, the axis-aligned
    bounding box is computed from the min/max of all vertices.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            # Standard YOLO: cx cy w h (normalized)
            cx, cy, w, h = coords
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
        else:
            # Polygon/OBB: x1 y1 x2 y2 ... xn yn (normalized)
            xs = coords[0::2]
            ys = coords[1::2]
            x1 = int(min(xs) * img_w)
            y1 = int(min(ys) * img_h)
            x2 = int(max(xs) * img_w)
            y2 = int(max(ys) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 - x1 >= 24 and y2 - y1 >= 24:
            boxes.append([x1, y1, x2, y2])
    return boxes


def extract_positive(img, box):
    """Extract tight bounding box with context padding, resize to WIN x WIN."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Add 20% context padding so the rock doesn't fill the window edge-to-edge
    pad_x = max(int(bw * 0.2), 4)
    pad_y = max(int(bh * 0.2), 4)
    H, W = img.shape[:2]
    lx = max(0, x1 - pad_x)
    ly = max(0, y1 - pad_y)
    rx = min(W, x2 + pad_x)
    ry = min(H, y2 + pad_y)
    crop = img[ly:ry, lx:rx]
    if crop.size == 0:
        crop = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    return cv2.resize(crop, (WIN, WIN), interpolation=cv2.INTER_AREA)


def sample_negatives(img, boxes, n, rng):
    """Sample n random WIN x WIN crops that don't overlap any rock box."""
    H, W = img.shape[:2]
    if H < WIN or W < WIN:
        return []
    crops = []
    attempts = 0
    while len(crops) < n and attempts < n * 50:
        attempts += 1
        x1 = rng.randint(0, W - WIN)
        y1 = rng.randint(0, H - WIN)
        candidate = [x1, y1, x1 + WIN, y1 + WIN]
        if all(iou(candidate, b) < 0.1 for b in boxes):
            crops.append(img[y1:y1+WIN, x1:x1+WIN])
    return crops


def save_crops(crops, out_dir: Path, prefix: str, start_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(crops):
        cv2.imwrite(str(out_dir / f"{prefix}_{start_idx+i:06d}.png"), crop)
    return start_idx + len(crops)


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # Make the output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all images (support common splits in subfolders)
    image_paths = sorted(data_dir.rglob("*.jpg")) + \
                  sorted(data_dir.rglob("*.jpeg")) + \
                  sorted(data_dir.rglob("*.png")) + \
                  sorted(data_dir.rglob("*.bmp"))
    image_paths = [p for p in image_paths if "labels" not in str(p)]

    if not image_paths:
        sys.exit(f"No images found under {data_dir}")

    print(f"Found {len(image_paths)} images")

    # Image-level train/test split
    rng.shuffle(image_paths)
    split_idx   = int(len(image_paths) * TRAIN_FRAC)
    train_imgs  = image_paths[:split_idx]
    test_imgs   = image_paths[split_idx:]

    (out_dir / "train_images.txt").write_text("\n".join(str(p) for p in train_imgs))
    (out_dir / "test_images.txt").write_text("\n".join(str(p) for p in test_imgs))

    stats = {"train": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}}

    for split, img_list in [("train", train_imgs), ("test", test_imgs)]:
        pos_idx = neg_idx = 0
        for img_path in img_list:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}", file=sys.stderr)
                continue
            H, W = img.shape[:2]

            # Label file: look for same stem in any labels/ sibling directory
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                # Try sibling labels/ folder
                label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name

            boxes = load_boxes_yolo(label_path, W, H)

            # Positive crops
            pos_crops = [extract_positive(img, b) for b in boxes]
            pos_idx = save_crops(pos_crops, out_dir / split / "pos", "pos", pos_idx)

            # Negative crops
            n_neg = max(len(pos_crops) * args.neg_ratio, 2) if pos_crops else args.neg_ratio
            neg_crops = sample_negatives(img, boxes, n_neg, rng)
            neg_idx = save_crops(neg_crops, out_dir / split / "neg", "neg", neg_idx)

        stats[split]["pos"] = pos_idx
        stats[split]["neg"] = neg_idx
        print(f"  {split}: {pos_idx} positives, {neg_idx} negatives")

    print("\nDataset summary:")
    for split in ("train", "test"):
        p, n = stats[split]["pos"], stats[split]["neg"]
        print(f"  {split}: {p} pos  {n} neg  ratio={n/max(p,1):.1f}:1")


if __name__ == "__main__":
    main()
