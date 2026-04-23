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
python project/scripts/prepare_dataset.py --data_dir project/data/roboflow --out_dir project/data/crops
python project/scripts/train_svm.py --crops_dir project/data/crops --out_dir project/models
# Outputs: project/models/weights.bin, project/models/bias.txt
```

### 2. Run the detector

```bash
./build/hog_detector <input_video_or_image> project/models/weights.bin project/models/bias.txt
# Optional flags:
#   [output_path]       annotated video/image to write
#   [threshold]         SVM decision threshold (default 0.7)
#   --mode seq|omp|cuda implementation to use (default seq)
```

### 3. Run correctness validation

Generate a 64×64 reference patch and its Python HOG feature vector:

```bash
# Produce project/scripts/ref_feat.bin from project/scripts/patch.png
python project/scripts/generate_ref_bin.py
```

Then compare against the C++ implementation (run from the repo root):

```bash
# HOG-only check
./build/test_hog project/scripts/patch.png project/scripts/ref_feat.bin

# HOG + SVM decision check
./build/test_hog project/scripts/patch.png project/scripts/ref_feat.bin \
    project/models/weights.bin project/models/bias.txt
```

Expected output: both `max_abs_diff < 1e-3` (tight) and `< 1e-2` (loose) should print `[PASS]`.


