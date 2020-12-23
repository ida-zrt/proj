#!/bin/bash
#SBATCH -J VNL_Train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07

source activate VNL
python -u classifier_train.py
echo ------------------------------------------------------------
python -u dump_val_data.py

# python -u modelconvert.py
