seed_list=(1 2 3 4 5)
lff_scale_list=(0.0001 0.001 0.01 0.1 1.0 10 100)

for lff_scale in ${lff_scale_list[@]}; do
    for seed in ${seed_list[@]}; do
        python training.py --config-name=ant_ppo lff_scale=$lff_scale seed=$seed
    done
done

# seed_list=(1 2 3 4 5)
# lff_scale_list=(0.0002 0.0004 0.0006 0.0008)

# for seed in ${seed_list[@]}; do
#     for lff_scale in ${lff_scale_list[@]}; do
#         python training.py --config-name=ant_ppo lff_scale=$lff_scale seed=$seed
#     done
# done

# seed_list=(1 2 3 4 5)
# lff_scale_list=(0.02 0.04 0.06 0.08)

# for seed in ${seed_list[@]}; do
#     for lff_scale in ${lff_scale_list[@]}; do
#         python training.py --config-name=ant_ppo lff_scale=$lff_scale seed=$seed
#     done
# done
