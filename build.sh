#!/usr/bin/env bash
# Run this once on the SLURM login node before submitting slurm.sh:
#   bash build.sh
set -euo pipefail

# -- Modules ------------------------------------------------------------------
module load gcc/12.2.0
module load nvidia/cuda/12.2.0
module load conda/miniforge/24.3.0

# -- Conda env ----------------------------------------------------------------
# conda create -n hog_env -c conda-forge opencv -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hog_env

# Install conda-native cmake and a modern libstdc++ into the env so we never
# touch the system cmake (which requires GLIBCXX_3.4.32 / CXXABI_1.3.15,
# i.e. GCC 13+, unavailable on this cluster).
conda install -c conda-forge libstdcxx-ng cmake -y

# -- Build --------------------------------------------------------------------
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
make -j"$(nproc)"
cd ..

echo ""
echo "Build complete. Submit the job with: sbatch slurm.sh [input_image]"
