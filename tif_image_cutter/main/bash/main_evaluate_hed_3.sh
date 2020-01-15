#!/bin/bash
#PBS -q main
#PBS -l walltime=28:00:00
#PBS -l select=1:ncpus=8:mem=64000mb
#PBS -l software=smallimg
#PBS -N main_evaluate_hed_3
#PBS -m be
echo ". /home/doszke/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate tensorflow_env
cd /home/doszke/model_256
python3 ./main_evaluate_hed_3.py >& main_evaluate_hed_3.txt