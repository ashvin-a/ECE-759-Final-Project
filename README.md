# Parallelized Rock Detection using HOG+SVM

**ECE 759 — Parallel Programming Final Project**

Parallelized object detection pipeline using Histogram of Oriented Gradients (HOG) and a Linear SVM, targeting Martian rock detection from **Space Rover** imagery. Benchmarks sequential C++, OpenMP, and CUDA implementations across resolutions.

---

## Phase 1: Sequential Baseline

### Overview

Phase 1 establishes the sequential C++ baseline: a complete HOG+SVM sliding window detector with no parallelism. It serves as the correctness reference and profiling target for Phases 2A and 2B.

### HOG Parameters

| Parameter | Value |
|-----------|-------|
| Window size | 64x64 px |
| Cell size | 8x8 px |
| Block size | 16x16 px (2x2 cells) |
| Block stride | 8 px |
| Orientation bins | 9 (unsigned, 0–180°) |
| Blocks per window | 7x7 = 49 |
| Feature vector dim | 49 x 4 cells x 9 bins = **1764** |

Unsigned gradients are used because rock shape is invariant to sun angle — the direction of a dark-to-light transition is irrelevant; the edge itself is what matters.

### SVM

- Linear SVM trained with scikit-learn (`LinearSVC`)
- Decision: `f(x) = w^T * x + b > 0` → rock detected
- Weights exported via `numpy.tofile` to `weights.bin` (raw binary float32)
- Bias exported as a scalar to `bias.txt`
- Hard Negative Mining applied: false positives from negative-only images are added to the training set and the SVM is retrained

### Dataset

Roboflow "Robotic Arm Dataset - Rocks" (pre-labeled bounding boxes).

- Positive crops: 64x64 rock regions from labeled bounding boxes
- Negative crops: random 64x64 regions from images with no rock annotations
- Training ratio: 1:2 positive:negative (to prevent class collapse)
- Train/test split: **image-level** (not crop-level) to prevent data leakage from overlapping crops

Target F1 score on held-out test images: ≥ 85%.

---

## Directory Structure

```
project/
  scripts/
    prepare_dataset.py   # Extract crops from Roboflow annotations
    hog_utils.py         # Python HOG (matches C++ exactly — used for validation)
    train_svm.py         # Train LinearSVC, hard negative mining, export weights
  src/
    hog.cpp / hog.h      # Manual HOG extraction
    svm.cpp / svm.h      # Weight loading and dot product inference
    sliding_window.cpp   # Sliding window loop over full frame
    nms.cpp / nms.h      # Non-maximum suppression (CPU)
    main.cpp             # Entry point, timing, output
  include/
    types.h              # BoundingBox struct and shared types
  test/
    test_hog.cpp         # C++ vs. Python HOG correctness validation
CMakeLists.txt
weights.bin            # Exported SVM weights (generated, not checked in)
bias.txt               # Exported SVM bias scalar (generated, not checked in)
```

---

## Build

### Dependencies

- CMake >= 3.18
- OpenCV >= 4.x (image I/O and video capture only — not HOG)
- GCC with C++17 support
- `gprof` (for profiling)

### Steps

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## Usage

### 1. Prepare the dataset and train the SVM

```bash
# Extract crops and train the SVM
python scripts/prepare_dataset.py --data_dir data/roboflow --out_dir data/crops
python scripts/train_svm.py --crops_dir data/crops --out_dir .
# Outputs: weights.bin, bias.txt
```

### 2. Run the detector

```bash
./build/hog_detector <input_video_or_image> weights.bin bias.txt
```

### 3. Run correctness validation

```bash
./build/test_hog weights.bin bias.txt <test_patch.png>
# Checks max absolute diff between C++ and Python HOG vectors
# Expected tolerance: <= 1e-2 (due to float32 rounding after L2-Hys)
```

### 4. Profile

```bash
# Recompile with profiling enabled
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_PROFILING=ON ..
make -j$(nproc)
./build/hog_detector <input> weights.bin bias.txt
gprof build/hog_detector gmon.out | head -60
```

Expected bottleneck: L2-Hys block normalization and redundant memory reads during overlapping block assembly — not the Sobel gradient step.

---

## Latency Metric

- **What is timed:** wall-clock time per frame, starting after video decode, ending after NMS
- **What is excluded:** `cv::VideoCapture` decode time
- **Reported metrics:** average latency (ms/frame), p99 latency (ms/frame), FPS = 1000 / avg_latency
- **Frame count:** fixed at 300 frames per benchmark run

---

## Correctness Validation

Because parallel reductions in Phases 2A/2B alter floating-point addition order, strict bitwise equality is not expected. The tolerance is empirically determined by comparing the sequential and Python HOG outputs on a single test frame.

| Check | Tolerance |
|-------|-----------|
| Max absolute diff of 1764-dim HOG vector | ≤ 1e-2 |
| Absolute diff of SVM decision scalar `f(x)` | same sign (functionally correct) |

---

## Non-Maximum Suppression

Without NMS, a single rock triggers dozens of overlapping bounding boxes, making precision/recall meaningless. NMS runs on the CPU after the detector loop:

1. Sort detections by SVM score (descending)
2. Greedily suppress any box with IoU > 0.4 against a higher-scoring kept box

NMS is included in the latency benchmark. Because the SVM threshold filters ~99% of windows, NMS typically processes fewer than 100 boxes per frame and contributes negligible latency.
