#!/bin/bash
#PBS -q main
#PBS -l walltime=40:00:00
#PBS -l select=1:ncpus=8:mem=90000mb
#PBS -l software=smallimg
#PBS -N main_myprep
#PBS -m be
echo ". /home/doszke/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate tensorflow_env
cd /home/doszke/model_256/better_dataset/
python3 ./main_reinhard.py >& main_reinhard.txt