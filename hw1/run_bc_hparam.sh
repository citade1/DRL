# run_bc_hparam.sh

# candidates that might affect performance # 
# - number of train steps 
# - train data size
# - network depth OR network width

STEPS=(500 1000 2000 5000 10000)

for steps in "${STEPS[@]}"; do
    echo "Running BC with train_steps=${steps}"

    python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Walker2d.pkl \
        --env_name Walker2d-v4 \
        --exp_name bc_walker2d_steps${steps} \
        --n_iter 1 \
        --learning_rate 5e-3 \
        --train_batch_size 100 \
        --n_layers 2 \
        --size 64 \
        --num_agent_train_steps_per_iter ${steps} \
        --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
        --video_log_freq -1
done

