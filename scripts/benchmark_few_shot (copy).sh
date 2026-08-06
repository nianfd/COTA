#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=few-shot      # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error


mkdir -p logs

eval "$($CONDA_EXE shell.bash hook)"
conda activate subspacead

# Set the absolute path to your MVTec AD dataset directory
MVTEC_PATH="./datasets/mvtec-ad"

# Set the absolute path to your VisA dataset directory
# (e.g., /path/to/your/datasets/VisA_pytorch/1cls)
VISA_PATH="./datasets/VisA_pytorch/1cls"


echo "--- Starting k-shot experiments for MVTec AD ---"
for k in 1 2 4
do
    echo "--- Running MVTec AD k=$k ---"
    python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path "$MVTEC_PATH" \
        --image_res 672 \
        --k_shot $k \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "./facebook/dinov2-with-registers-giant" \
        --aug_count 30 \
        --pca_ev 0.99 \
        --seed 42 \
        --agg_method "mean" \
        --outdir "./few_shot_results/results_k${k}_mvtec_dinov2G" 
done


echo "--- Starting k-shot experiments for VisA ---"
for k in 1 2 4
do
    echo "--- Running VisA k=$k ---"
    python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        --image_res 672 \
        --k_shot $k \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "./facebook/dinov2-with-registers-giant" \
        --aug_count 30 \
        --pca_ev 0.99 \
        --agg_method "mean" \
        --seed 42 \
        --outdir "./few_shot_results/results_k${k}_visa_dinov2G" 
done

echo "--- All experiments complete ---"
