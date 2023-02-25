#!/bin/sh -l
#SBATCH -p short
#SBATCH --gres=gpu:4
#SBATCH -J training_gail_ant
#SBATCH -o train_gail_ant.log
#SBATCH -t 08:00:00
hostname
echo $CUDA_AVAILABLE_DEVICES
python GAIL/run_gail.py --expert_data expert/expert_data_Ant-v4.pkl --env_name Ant-v4 --exp_name gail_Antv4_with_baseline_and_gae --gae_lambda 0.99 -n 500 -b 50000 