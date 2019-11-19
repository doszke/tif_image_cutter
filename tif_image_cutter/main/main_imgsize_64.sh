#!/bin/bash
#PBS -q main
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=64000mb
#PBS -l software=smallimg
#PBS -N main_imgsize_64
#PBS -m be
echo ". /home/doszke/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate tensorflow_env
cd /home/doszke/model_256
python3 ./main_imgsize_64.py >& main_imgsize_64.txt