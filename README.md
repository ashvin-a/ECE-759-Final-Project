# Parallelized Rock Detection using HOG+SVM

**ECE 759 — Parallel Programming Final Project**

Parallelized object detection pipeline using Histogram of Oriented Gradients (HOG) and a Linear SVM, targeting Martian rock detection from **Space Rover** imagery. Benchmarks sequential C++, OpenMP, and CUDA implementations across resolutions.

---

## Initial Setup

```bash
git clone git@github.com:ashvin-a/FinalProject.git
cd FinalProject/
module load gcc/12.2.0 nvidia/cuda/12.2.0 conda/miniforge/24.3.0
conda create -n hog_env -c conda-forge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hog_env
conda install -c conda-forge libstdcxx-ng cmake opencv -y
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
make -j$(nproc)
cd ../
```

---

## Rock Detection

Run the detector on an input image using CUDA, OpenMP, or Sequential mode:

```bash
sbatch slurm.sh
```

To run a specific mode manually:

```bash
# CUDA
./build/hog_detector input_image.png project/models/weights.bin project/models/bias.txt project/results/output_cuda.png --mode cuda

# OpenMP
./build/hog_detector input_image.png project/models/weights.bin project/models/bias.txt project/results/output_omp.png --mode omp

# Sequential
./build/hog_detector input_image.png project/models/weights.bin project/models/bias.txt project/results/output_seq.png --mode seq
```

The SVM decision threshold defaults to `0.7`. Pass it as a positional argument before `--mode` to override:

```bash
./build/hog_detector input_image.png project/models/weights.bin project/models/bias.txt project/results/output_cuda.png 0.5 --mode cuda
```

Each run writes a results CSV alongside the output path with per-frame columns (`mode, frame, raw_detections, kept_detections, latency_ms`) followed by a summary block with avg/p50/p99 latency and FPS.

---

## Profiling

Rebuild with gprof instrumentation, then submit via sbatch:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
    -DENABLE_PROFILING=ON
make -j$(nproc)
cd ../
sbatch run_prof.sh
```

The job runs the sequential detector to generate `gmon.out`, then writes the gprof report to `profile.txt`.

Note: `-pg` instruments only the C++ host code. For GPU kernel profiling use `ncu` or `nvprof`.

---

## Dataset Preparation

Download and extract the Roboflow rock dataset, then generate 64×64 training crops:

```bash
cd project
mkdir data && cd data
curl -L "https://app.roboflow.com/ds/AVwkFhBSLM?key=CvWn4MvvWE" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
cd ../../
python project/scripts/prepare_dataset.py --data_dir project/data/roboflow --out_dir project/data/crops
```

This produces `project/data/crops/train/{pos,neg}/` and `project/data/crops/test/{pos,neg}/` with an image-level train/test split to prevent data leakage.

---

## Training the SVM

```bash
python project/scripts/train_svm.py --crops_dir project/data/crops --out_dir project/models
```

Outputs `project/models/weights.bin` (1764 raw float32 values) and `project/models/bias.txt`. Training includes Hard Negative Mining by default.

---

## Correctness Validation

Generate a Python HOG reference vector and compare against the C++ implementation:

```bash
# Reads project/results/patch.png, writes project/results/ref_feat.bin
python project/scripts/generate_ref_bin.py

# HOG-only check
./build/test_hog project/results/patch.png project/results/ref_feat.bin

# HOG + SVM decision check
./build/test_hog project/results/patch.png project/results/ref_feat.bin \
    project/models/weights.bin project/models/bias.txt
```

Expected output: both `max_abs_diff < 1e-3` (tight) and `< 1e-2` (loose) should print `[PASS]`.

---

## Generate Latency Plots

```bash
python project/scripts/plots.py
```

Reads all `*_results.csv` files from `project/results/` and writes per-mode latency plots to `project/results/<mode>_latency_plot.png`.
