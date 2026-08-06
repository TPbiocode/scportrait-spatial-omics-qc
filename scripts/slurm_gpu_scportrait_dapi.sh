#!/bin/bash
#SBATCH --job-name=scportrait_workflow      # Job name (can override at submit time with --job-name)
#SBATCH --output=logs/%x_%j.out             # Standard output file (%x=job name, %j=job id)
#SBATCH --error=logs/%x_%j.err              # Error log file (%x=job name, %j=job id)
#SBATCH --time=24:00:00                     # Walltime (can override: sbatch --time=12:00:00 script.sh)
#SBATCH --mem=128gb                         # RAM (can override: sbatch --mem=128gb script.sh)
#SBATCH --cpus-per-task=1                   # CPU cores (earlier 16)
#SBATCH --partition=gpu                     # GPU partition
#SBATCH -G h100:1                           # Request 1 x H100 GPU (can override: sbatch -G h100:2 script.sh)

# ---------------------------------------------
# User Configuration - Edit these for your setup
# ---------------------------------------------
# Set environment defaults for base
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-/scratch/${USER}/conda_envs/env_scportrait}"
BASE_DIR="${BASE_DIR:-/scratch/${USER}/scportrait-spatial-omics-qc}"

# ---------------------------------------------
# Set PyTorch CUDA allocator BEFORE any Python/torch loads
# ---------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------
# Load and activate conda environment
# ---------------------------------------------
source ${CONDA_BASE}/etc/profile.d/conda.sh 
conda activate ${CONDA_ENV}

# Optional: prepend conda libs. Keep disabled by default because this can shadow
# system NVIDIA libraries and make torch report CUDA unavailable.
# Ensure conda lib first to resolve llvmlite/numba symbol mismatches
# Set FORCE_CONDA_LD_LIBRARY_PATH=0 when you need to prefer system libs (e.g., specialized CUDA setups)
if [[ "${FORCE_CONDA_LD_LIBRARY_PATH:-1}" == "1" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------
# Optional: Use scratch for faster I/O
# ---------------------------------------------
SCRATCH_LABEL="${SCRATCH_LABEL:-02aug26_scportrait_dapi_cellfeaturizer}"
SCRATCH_DIR="/scratch/${USER}/${SCRATCH_LABEL}_$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy the Python script and any required files from project folder
cp ${BASE_DIR}/scripts/scportrait_dapi_workflow.py .

# ---------------------------------------------
# Log environment (debug + reproducibility)
# ---------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Workspace: ${BASE_DIR}"
which python
python --version

# ---------------------------------------------
# Log GPU state for debugging
# ---------------------------------------------
echo "=== GPU status at job start ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

# ---------------------------------------------
# Run the Python script
# ---------------------------------------------
echo "Starting pipeline at $(date)"
python scportrait_dapi_workflow.py
echo "Pipeline finished at $(date)"
