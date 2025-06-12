#!/bin/bash
#SBATCH --job-name=logs/generate_data_%j
#SBATCH --output=logs/generate_data_%j.out
#SBATCH --error=generate_data_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-5

export PYTHONUNBUFFERED=1


echo "Starting job"

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate
generate_random_samples /fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5 -s $SLURM_ARRAY_TASK_ID -o "out_{$SLURM_ARRAY_TASK_ID}_512M.h5"
