#!/bin/bash

#SBATCH --job-name=compileOutput
#SBATCH --output=compileOutput.out
#SBATCH --error=compileOutput.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

module load python
python /home/haynes13/code/stat-model-followup/compileOutput.py