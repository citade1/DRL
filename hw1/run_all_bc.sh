
# Ant
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 \
--exp_name bc_ant \
--n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1

# HalfCheetah
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v4 \
--exp_name bc_halfcheetah \
--n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
--video_log_freq -1

# Hopper
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Hopper.pkl \
--env_name Hopper-v4 \
--exp_name bc_hopper \
--n_iter 1 \
--expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1

# Walker2d
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 \
--exp_name bc_walker2d \
--n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1
