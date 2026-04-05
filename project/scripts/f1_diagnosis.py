"""F1 diagnostic: grid search over C and class_weight without HNM."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import cv2
import numpy as np
from pathlib import Path
from hog_utils import extract_hog, WIN_H
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

WIN = WIN_H


def load_crops(directory: Path, label: int):
    files = sorted(directory.glob("*.png"))
    if not files:
        return np.empty((0,), dtype=np.float32).reshape(0, 1764), np.empty(0, dtype=int)
    feats = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (WIN, WIN):
            img = cv2.resize(img, (WIN, WIN))
        feats.append(extract_hog(img))
    return np.array(feats, dtype=np.float32), np.full(len(feats), label, dtype=int)


def main():
    crops = Path("project/data/crops")

    print("Loading training crops...")
    X_pos, y_pos = load_crops(crops / "train" / "pos", 1)
    X_neg, y_neg = load_crops(crops / "train" / "neg", 0)
    print(f"  Train: {len(X_pos)} pos  {len(X_neg)} neg")

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([y_pos, y_neg])

    print("Loading test crops...")
    Xtp, ytp = load_crops(crops / "test" / "pos", 1)
    Xtn, ytn = load_crops(crops / "test" / "neg", 0)
    X_test = np.vstack([Xtp, Xtn])
    y_test = np.concatenate([ytp, ytn])
    print(f"  Test:  {len(Xtp)} pos  {len(Xtn)} neg")

    print("\nGrid search (no HNM):")
    print(f"{'C':>6}  {'class_weight':>14}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print("-" * 50)

    for C in [0.01, 0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]:
            clf = LinearSVC(C=C, max_iter=10000, random_state=42, class_weight=cw)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            pr = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rc = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            cw_str = str(cw) if cw else "None"
            print(f"{C:>6.2f}  {cw_str:>14}  {f1:>6.3f}  {pr:>6.3f}  {rc:>6.3f}")


if __name__ == "__main__":
    main()
