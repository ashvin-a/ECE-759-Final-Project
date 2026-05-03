#!/usr/bin/env bash
# DO NOT run this script in sbatch. Run each line of command, one by one.
set -euo pipefail

# -- Modules ------------------------------------------------------------------
module load gcc/12.2.0
module load nvidia/cuda/12.2.0
module load conda/miniforge/24.3.0

# -- Conda env ----------------------------------------------------------------
conda create -n hog_env -c conda-forge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hog_env

conda install -c conda-forge libstdcxx-ng cmake opencv -y

# -- Build --------------------------------------------------------------------
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
make -j"$(nproc)"
cd ..

echo ""
echo "Build complete. Submit the job with: sbatch slurm.sh [input_image]"
