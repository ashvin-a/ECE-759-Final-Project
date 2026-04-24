# Parallelized Rock Detection using HOG+SVM

**ECE 759 — Parallel Programming Final Project**

Parallelized object detection pipeline using Histogram of Oriented Gradients (HOG) and a Linear SVM, targeting Martian rock detection from **Space Rover** imagery. Benchmarks sequential C++, OpenMP, and CUDA implementations across resolutions.

---

## Build

### Dependencies

- CMake >= 3.18
- OpenCV >= 4.x (image I/O and video capture only — not HOG)
- GCC with C++17 support
- CUDA toolkit (for `--mode cuda`)

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
#   [output_path]       annotated video/image to write (e.g. project/results/output.png)
#   [threshold]         SVM decision threshold (default 0.7)
#   --mode seq|omp|cuda implementation to use (default seq)
```

The detector always writes a results CSV alongside the output path:
- `project/results/output_sequential_results.csv`
- `project/results/output_OpenMP_results.csv`
- `project/results/output_CUDA_results.csv`

Each CSV contains per-frame columns (`mode, frame, raw_detections, kept_detections, latency_ms`) followed by a summary block with avg/p50/p99 latency and FPS.

### 3. Generate latency plots

```bash
# Reads all *_results.csv files from project/results/ and writes PNG plots
python project/scripts/plots.py
# Outputs: project/results/<mode>_latency_plot.png
```

### 4. Run correctness validation

Generate a 64×64 reference patch and its Python HOG feature vector:

```bash
# Reads project/results/patch.png, writes project/results/ref_feat.bin
python project/scripts/generate_ref_bin.py
```

Then compare against the C++ implementation (run from the repo root):

```bash
# HOG-only check
./build/test_hog project/results/patch.png project/results/ref_feat.bin

# HOG + SVM decision check
./build/test_hog project/results/patch.png project/results/ref_feat.bin \
    project/models/weights.bin project/models/bias.txt
```

Expected output: both `max_abs_diff < 1e-3` (tight) and `< 1e-2` (loose) should print `[PASS]`.


