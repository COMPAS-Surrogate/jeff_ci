#!/bin/bash
#SBATCH --job-name=cosmic_integration
#SBATCH --output=cosmic_integration_%j.out
#SBATCH --error=cosmic_integration_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

echo "Starting job"

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate
run_cosmic_integration out_512M -i h5out_512M.h5 -p /fred/oz101/avajpeyi/COMPAS_DATA/ -n 1 -v
