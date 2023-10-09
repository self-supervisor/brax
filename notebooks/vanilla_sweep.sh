#!/bin/bash
#$ -cwd
#$ -j y
#$ -t 1-5:1  # Adjust this range based on the total number of combinations
#$ -o vanilla_sweep_log
#$ -e vanilla_sweep
#$ -N vanilla_sweep
#$ -l gpu=1

seeds=(1 2 3 4 5)

python training.py  --config-name=hopper use_lff=false seed=${seeds[$SGE_TASK_ID-1]}
