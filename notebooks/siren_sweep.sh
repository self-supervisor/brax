#!/bin/bash
#$ -cwd
#$ -j y
#$ -t 1-5:1  # Adjust this range based on the total number of combinations (7 seeds * 9 lff_scales = 63 combinations)
#$ -tc 20
#$ -o scale_sweep_log
#$ -e scale_sweep
#$ -N scale_sweep
#$ -l gpu=1

seed_list=(1 2 3 4 5)
lff_scale_list=(0.000001)

# Calculate the total number of combinations
total_combinations=$(( ${#seed_list[@]} * ${#lff_scale_list[@]} ))


# Calculate indices into seed_list and lff_scale_list based on SGE_TASK_ID
# Assuming SGE_TASK_ID starts at 1
index=$(( $SGE_TASK_ID - 1 ))
seed_index=$(( $index % ${#seed_list[@]} ))
lff_scale_index=$(( $index / ${#seed_list[@]} ))

python training.py --config-name=walker2d use_lff=false seed=${seed_list[$seed_index]} lff_scale=${lff_scale_list[$lff_scale_index]}
