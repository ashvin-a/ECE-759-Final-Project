#!/usr/bin/env bash
#SBATCH --job-name=hog_detector_prof
#SBATCH --partition=instruction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --mem=8G

cd $SLURM_SUBMIT_DIR

# Require a pre-built binary compiled with -pg — run build.sh with
# -DENABLE_PROFILING=ON on the login node first.
if [[ ! -x ./build/hog_detector ]]; then
    echo "ERROR: ./build/hog_detector not found. Run 'bash build.sh' on the login node first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hog_env

INPUT="${1:-input_image.png}"
WEIGHTS="project/models/weights.bin"
BIAS="project/models/bias.txt"
RESULTS_DIR="project/results"

mkdir -p logs "$RESULTS_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "========================================================"
echo " Job     : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Input   : $INPUT"
echo " OMP     : $OMP_NUM_THREADS threads"
echo " GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================================"

# Run the instrumented binary — this produces gmon.out in the working directory
./build/hog_detector \
    "$INPUT" \
    "$WEIGHTS" \
    "$BIAS" \
    "$RESULTS_DIR/output_seq.png" \
    --mode seq

# Now generate the profile report
gprof ./build/hog_detector gmon.out > profile.txt

echo "Profile written to profile.txt"
