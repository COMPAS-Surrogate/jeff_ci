#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --output=logs/generate_data_%j.out
#SBATCH --error=logs/generate_data_%j.err
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-5

export PYTHONUNBUFFERED=1


IDX=$((SLURM_ARRAY_TASK_ID + 20))

echo "Starting job"




ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate
generate_random_samples /fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5 -s $IDX -o "out/out_{$IDX}_512M.h5"
