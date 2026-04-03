"""
train_svm.py — Train a Linear SVM on HOG features extracted from crops.

Stages:
  1. Extract HOG features from train/pos and train/neg crops
  2. Train LinearSVC (base model)
  3. Hard Negative Mining: run base model on negative-only images,
     collect false positives, retrain
  4. Evaluate F1 on test set
  5. Export weights.bin (float32 raw binary) and bias.txt

Usage:
    python scripts/train_svm.py \
        --crops_dir data/crops \
        --neg_images_dir data/roboflow/train/images \
        --out_dir . \
        [--no-hnm]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

# Add scripts/ to path so we can import hog_utils
sys.path.insert(0, str(Path(__file__).parent))
from hog_utils import extract_hog, WIN

SLIDE_STRIDE = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir",      required=True,
                   help="Directory produced by prepare_dataset.py")
    p.add_argument("--neg_images_dir", default=None,
                   help="Directory of full images known to contain NO rocks "
                        "(for hard negative mining). If omitted, HNM is skipped.")
    p.add_argument("--out_dir",        default=".",
                   help="Where to write weights.bin and bias.txt")
    p.add_argument("--no-hnm",         action="store_true",
                   help="Skip hard negative mining")
    p.add_argument("--C",              type=float, default=0.01,
                   help="LinearSVC regularization (default 0.01)")
    p.add_argument("--max_iter",       type=int, default=10000)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def load_crops(directory: Path, label: int):
    """Load all PNGs from directory, extract HOG, return (X, y)."""
    files = sorted(directory.glob("*.png"))
    if not files:
        return np.empty((0, 1764), dtype=np.float32), np.empty(0, dtype=int)
    feats = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (WIN, WIN):
            img = cv2.resize(img, (WIN, WIN))
        feats.append(extract_hog(img))
    X = np.array(feats, dtype=np.float32)
    y = np.full(len(X), label, dtype=int)
    return X, y


def sliding_window_detections(img, clf, threshold=0.0):
    """
    Run a trained LinearSVC over all WIN x WIN windows in img.
    Returns list of (x, y, score) for positive detections.
    """
    H, W = img.shape[:2]
    detections = []
    for y in range(0, H - WIN + 1, SLIDE_STRIDE):
        for x in range(0, W - WIN + 1, SLIDE_STRIDE):
            patch = img[y:y+WIN, x:x+WIN]
            feat  = extract_hog(patch).reshape(1, -1)
            score = clf.decision_function(feat)[0]
            if score > threshold:
                detections.append((x, y, score))
    return detections


def mine_hard_negatives(neg_image_dir: Path, clf, max_per_image: int = 50):
    """
    Collect false-positive crops from negative-only images.
    Returns (X_hn, y_hn).
    """
    image_files = list(neg_image_dir.glob("*.jpg")) + \
                  list(neg_image_dir.glob("*.jpeg")) + \
                  list(neg_image_dir.glob("*.png"))
    if not image_files:
        print("  Warning: no images found for hard negative mining")
        return np.empty((0, 1764), dtype=np.float32), np.empty(0, dtype=int)

    print(f"  Mining hard negatives from {len(image_files)} images...")
    hn_feats = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = sliding_window_detections(img, clf, threshold=0.0)
        # Sort by score descending, keep worst offenders
        dets.sort(key=lambda d: d[2], reverse=True)
        for x, y, _ in dets[:max_per_image]:
            patch = img[y:y+WIN, x:x+WIN]
            hn_feats.append(extract_hog(patch))

    if not hn_feats:
        return np.empty((0, 1764), dtype=np.float32), np.empty(0, dtype=int)

    X_hn = np.array(hn_feats, dtype=np.float32)
    y_hn = np.zeros(len(X_hn), dtype=int)
    print(f"  Found {len(X_hn)} hard negatives")
    return X_hn, y_hn


def main():
    args   = parse_args()
    crops  = Path(args.crops_dir)
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    #  Load training crops 
    print("Loading training crops...")
    X_pos, y_pos = load_crops(crops / "train" / "pos", label=1)
    X_neg, y_neg = load_crops(crops / "train" / "neg", label=0)
    print(f"  Positives: {len(X_pos)}  Negatives: {len(X_neg)}")

    if len(X_pos) == 0 or len(X_neg) == 0:
        sys.exit("ERROR: Need both positive and negative crops. Run prepare_dataset.py first.")

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([y_pos, y_neg])

    #  Base model 
    print("\nTraining base LinearSVC...")
    t0  = time.time()
    clf = LinearSVC(C=args.C, max_iter=args.max_iter, random_state=args.seed)
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    #  Hard Negative Mining 
    if not args.no_hnm and args.neg_images_dir:
        neg_img_dir = Path(args.neg_images_dir)
        print("\nHard Negative Mining...")
        X_hn, y_hn = mine_hard_negatives(neg_img_dir, clf)
        if len(X_hn) > 0:
            X_train = np.vstack([X_train, X_hn])
            y_train = np.concatenate([y_train, y_hn])
            print(f"  Retraining with {len(X_train)} total samples...")
            t0  = time.time()
            clf = LinearSVC(C=args.C, max_iter=args.max_iter, random_state=args.seed)
            clf.fit(X_train, y_train)
            print(f"  Retrained in {time.time()-t0:.1f}s")
    else:
        print("\nSkipping hard negative mining.")

    #  Evaluate on test set 
    print("\nEvaluating on test set...")
    X_test_pos, y_test_pos = load_crops(crops / "test" / "pos", label=1)
    X_test_neg, y_test_neg = load_crops(crops / "test" / "neg", label=0)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([y_test_pos, y_test_neg])

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(classification_report(y_test, y_pred, target_names=["background", "rock"]))
    print(f"F1 (rock): {f1:.4f}")
    if f1 < 0.85:
        print("WARNING: F1 < 0.85 — consider adjusting C, training more data, or tuning HNM.")

    #  Export weights 
    weights = clf.coef_[0].astype(np.float32)
    bias    = float(clf.intercept_[0])

    weights_path = out / "weights.bin"
    bias_path    = out / "bias.txt"

    weights.tofile(str(weights_path))
    bias_path.write_text(f"{bias:.10f}\n")

    print(f"\nExported:")
    print(f"  {weights_path}  ({len(weights)} float32 values, {weights_path.stat().st_size} bytes)")
    print(f"  {bias_path}     bias = {bias:.6f}")
    print(f"\nWeight stats: min={weights.min():.4f}  max={weights.max():.4f}  "
          f"mean={weights.mean():.4f}  std={weights.std():.4f}")


if __name__ == "__main__":
    main()
