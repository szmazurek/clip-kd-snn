#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=26:00:00
#SBATCH --account=plgisoftware-gpu-rtx
#SBATCH --partition=plgrid-gpu-rtx
#SBATCH --output=logs/slurm/out_%j.out
#SBATCH -C memfs

ml GCCcore/14.3.0 Python/3.13.5 CUDA/12.9 
cd $SCRATCH/clip-kd-snn
source .venv-rtx/bin/activate

mkdir -p logs/slurm outputs

export HF_DATASETS_OFFLINE=1
export STORAGE=$PLG_GROUPS_STORAGE/plggwie/plgmazurekagh
export TMPDIR=$MEMFS
export TORCHINDUCTOR_CACHE_DIR=$MEMFS/torchinductor_cache

srun python scripts/train.py \
    +experiment=kd_vit_b16_to_b16_cc3m12m \
    dataset=combined_wds_dali_pretok \
    "hydra.run.dir=outputs/${SLURM_JOB_ID}_kd_vit_b16_to_b16_cc3m12m" 