#!/bin/bash
#SBATCH --job-name=generate_jeff_data
#SBATCH --output=generate_jeff_data%j.out
#SBATCH --error=generate_jeff_data%j.err
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

export PYTHONUNBUFFERED=1


echo "Starting job"

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate
run_cosmic_integration out_512M_FULL -i h5out_512M.h5 -p /fred/oz101/avajpeyi/COMPAS_DATA/ -n 1 -v
