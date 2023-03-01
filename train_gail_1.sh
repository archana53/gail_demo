#!/bin/sh -l
#SBATCH -p short
#SBATCH --gres=gpu:4
#SBATCH -J trainingppo_gail_ant
#SBATCH -o train_gail_ant_5.log
#SBATCH -t 10:00:00
hostname
echo $CUDA_AVAILABLE_DEVICES
python GAIL/run_gail.py --multiprocess_gym_envs 16 --expert_data expert/Hopper_more_data.pkl --env_name Hopper-v4 --exp_name gail_Hopper-v4_seed_9_ppo_20_trajs_states_only --gae_lambda 0.99 -n 500 -b 50000 --seed 9 --states_only