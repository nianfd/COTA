#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=full-shot     # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error

eval "$($CONDA_EXE shell.bash hook)"

# Set the absolute path to your MVTec AD dataset directory
MVTEC_PATH="datasets/mvtec-ad"

# Set the absolute path to your VisA dataset directory
# (e.g., /path/to/your/datasets/VisA_pytorch/1cls)
VISA_PATH="datasets/VisA_pytorch/1cls"


echo "--- Running Full-Shot (all train images) for MVTec AD ---"
conda run -n subspacead python -u main.py \
    --dataset_name mvtec_ad \
    --dataset_path "$MVTEC_PATH" \
    --image_res 672 \
    --layers="-12,-13,-14,-15,-16,-17,-18" \
    --model_ckpt "facebook/dinov2-with-registers-giant" \
    --pca_ev 0.99 \
    --agg_method "mean" \
    --outdir "results_full_shot/mvtec_dinov2G"

echo "--- Running Full-Shot (all train images) for VisA ---"
conda run -n subspacead python -u main.py \
    --dataset_name visa \
    --dataset_path "$VISA_PATH" \
    --image_res 672 \
    --layers="-12,-13,-14,-15,-16,-17,-18" \
    --model_ckpt "facebook/dinov2-with-registers-giant" \
    --pca_ev 0.99 \
    --agg_method "mean" \
    --outdir "results_full_shot/visa_dinov2G"

echo "--- All experiments complete ---"