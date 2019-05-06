#!/bin/bash

#SBATCH --job-name=tf_v8
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --error=train_20192904_8.err
#SBATCH --partition=gpu_v100
#SBATCH --nodelist=bm1-lxslurm33


python training_flair_vol2.py
