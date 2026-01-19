#!/bin/bash
#SBATCH --job-name=compas_ml
#SBATCH --output=compas_ml_%j.out
#SBATCH --error=compas_ml_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

export PYTHONUNBUFFERED=1


echo "Starting job"

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate
python simulation_study.py 
