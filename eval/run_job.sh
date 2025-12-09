#!/bin/bash
#SBATCH --job-name=eureka_vlm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00

# Activate conda environment
source /data/user_data/wenjiel2/miniconda3/bin/activate eureka

# Set library path for IsaacGym
export LD_LIBRARY_PATH=/data/user_data/wenjiel2/miniconda3/envs/eureka/lib:$LD_LIBRARY_PATH

# Change to the eval directory
cd /data/user_data/wenjiel2/Code/roboly/Eureka/eval

# Run the experiment
python run_adverb_experiments_vlm.py --max-concurrent 8
