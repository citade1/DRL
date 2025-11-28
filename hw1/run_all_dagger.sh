
# ================================
# DAgger for Ant-v4
# ================================
python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/Ant.pkl \
  --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
  --env_name Ant-v4 \
  --exp_name dagger_ant \
  --n_iter 10 \
  --do_dagger \
  --video_log_freq -1

# ================================
# DAgger for HalfCheetah-v4
# ================================
python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
  --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
  --env_name HalfCheetah-v4 \
  --exp_name dagger_halfcheetah \
  --n_iter 10 \
  --do_dagger \
  --video_log_freq -1

# ================================
# DAgger for Hopper-v4
# ================================
python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/Hopper.pkl \
  --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
  --env_name Hopper-v4 \
  --exp_name dagger_hopper \
  --n_iter 10 \
  --do_dagger \
  --video_log_freq -1

# ================================
# DAgger for Walker2d-v4
# ================================
python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/Walker2d.pkl \
  --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
  --env_name Walker2d-v4 \
  --exp_name dagger_walker2d \
  --n_iter 10 \
  --do_dagger \
  --video_log_freq -1
