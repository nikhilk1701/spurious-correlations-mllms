#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=11:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch

module purge

singularity exec \
            --nv --overlay /scratch/nk3853/overlay-50G-10M.ext3:ro \
            /scratch/work/public/singularity/ubuntu-20.04.1.sif \
            /bin/bash -c "source /ext3/miniconda3/etc/profile.d/conda.sh; 
            export PATH=/ext3/miniconda3/bin:$PATH;
            export PYTHONPATH=/ext3/miniconda3/bin:$PATH;
            conda activate llava;
            python llava_inference.py"