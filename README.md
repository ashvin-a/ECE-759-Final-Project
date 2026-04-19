# Parallelized Rock Detection using HOG+SVM

**ECE 759 — Parallel Programming Final Project**

Parallelized object detection pipeline using Histogram of Oriented Gradients (HOG) and a Linear SVM, targeting Martian rock detection from **Space Rover** imagery. Benchmarks sequential C++, OpenMP, and CUDA implementations across resolutions.

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


