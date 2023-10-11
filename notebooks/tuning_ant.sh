#!/bin/bash
#$ -cwd
#$ -j y
#$ -tc 9
#$ -o scale_sweep_log
#$ -e scale_sweep
#$ -N scale_sweep
#$ -l gpu=1
#$ -t 1-750

seed_list=(1 2 3 4 5)
reward_scaling=(1 5 10 20 30 60)
batch_size=(64 128 256 512 1024)
grad_updates_per_step=(16 32 48 64 128)

# Calculate total number of combinations
total_combinations=$((${#seed_list[@]} * ${#reward_scaling[@]} * ${#batch_size[@]} * ${#grad_updates_per_step[@]}))


# Calculate indices based on SGE_TASK_ID, assuming SGE_TASK_ID starts at 1
index=$(( $SGE_TASK_ID - 1 ))

# Calculate the indices into each hyperparameter array
seed_index=$(( $index % ${#seed_list[@]} ))
index=$(( $index / ${#seed_list[@]} ))

reward_scaling_index=$(( $index % ${#reward_scaling[@]} ))
index=$(( $index / ${#reward_scaling[@]} ))

batch_size_index=$(( $index % ${#batch_size[@]} ))
index=$(( $index / ${#batch_size[@]} ))

grad_updates_per_step_index=$(( $index % ${#grad_updates_per_step[@]} ))

# Call python script with the selected hyperparameters
python training.py \
    --config-name=ant \
    use_lff=false \
    seed=${seed_list[$seed_index]} \
    reward_scaling=${reward_scaling[$reward_scaling_index]} \
    batch_size=${batch_size[$batch_size_index]} \
    grad_updates_per_step=${grad_updates_per_step[$grad_updates_per_step_index]}
