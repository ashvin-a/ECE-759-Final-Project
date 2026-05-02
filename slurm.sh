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
# Usage: sbatch run.slurm [input_image]   (default: input_image.png)
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
echo "========================================================"

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
