#!/usr/bin/env bash
#SBATCH --job-name=hog_detector
#SBATCH --partition=instruction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --mem=8G

# ---------------------------------------------------------------------------
# HOG+SVM Rock Detector — Sequential / OpenMP / CUDA
# Usage: sbatch slurm.sh [input_image]   (default: input_image.png)
# ---------------------------------------------------------------------------

# -- Modules ------------------------------------------------------------------
module load gcc/12.2.0
module load cmake
module load nvidia/cuda/12.2.0
module load conda/miniforge/24.3.0

# `conda activate` requires shell functions — source the init script explicitly
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda env (create it first if needed:
conda create -n hog_env -c conda-forge opencv -y
conda activate hog_env
export OpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"

# Ensure nvcc is visible to CMake
export CUDAToolkit_ROOT="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
export PATH="$CUDAToolkit_ROOT/bin:$PATH"

# ---------------------------------------------------------------------------

INPUT="${1:-input_image.png}"
WEIGHTS="project/models/weights.bin"
BIAS="project/models/bias.txt"
RESULTS_DIR="project/results"

mkdir -p logs "$RESULTS_DIR"

# OpenMP thread count matches the allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================================"
echo " Job     : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Input   : $INPUT"
echo " OMP     : $OMP_NUM_THREADS threads"
echo " GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " OpenCV  : $OpenCV_DIR"
echo "========================================================"

# -- Build --------------------------------------------------------------------
echo ""
echo "[0/3] Building with CMake..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOpenCV_DIR="$OpenCV_DIR" \
         -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT"
make -j"$SLURM_CPUS_PER_TASK"
cd ..
echo "Build done."

# -- Sequential ---------------------------------------------------------------
echo ""
echo "[1/3] Running sequential..."
./build/hog_detector \
    "$INPUT" \
    "$WEIGHTS" \
    "$BIAS" \
    "$RESULTS_DIR/output_seq.png" \
    --mode seq
echo "Sequential done."

# -- OpenMP -------------------------------------------------------------------
echo ""
echo "[2/3] Running OpenMP (${OMP_NUM_THREADS} threads)..."
./build/hog_detector \
    "$INPUT" \
    "$WEIGHTS" \
    "$BIAS" \
    "$RESULTS_DIR/output_omp.png" \
    --mode omp
echo "OpenMP done."

# -- CUDA ---------------------------------------------------------------------
echo ""
echo "[3/3] Running CUDA..."
./build/hog_detector \
    "$INPUT" \
    "$WEIGHTS" \
    "$BIAS" \
    "$RESULTS_DIR/output_cuda.png" \
    --mode cuda
echo "CUDA done."

# -- Summary ------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Results written to $RESULTS_DIR/"
echo " CSV files:"
ls -1 "$RESULTS_DIR"/*_results.csv 2>/dev/null | sed 's/^/   /'
echo "========================================================"
