#!/bin/bash
#SBATCH --partition=gpu_h100          # GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=res-ablation       # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --mem=128G                    # Memory per node
#SBATCH --time=24:00:00               # Walltime
#SBATCH --output=logs/out_%j.txt      # Stdout
#SBATCH --error=logs/err_%j.txt       # Stderr

mkdir -p logs
eval "$($CONDA_EXE shell.bash hook)"
MVTEC_PATH="datasets/mvtec-ad"
VISA_PATH="datasets/VisA_pytorch/1cls"
MODEL="facebook/dinov2-with-registers-giant"
LAYERS="-12,-13,-14,-15,-16,-17,-18"
AGG="mean"
PCA_EV=0.99
RES_LIST=(256 336 448 512 672)

echo "--- Starting resolution ablation (res in ${RES_LIST[*]}) ---"
for RES in "${RES_LIST[@]}"; do
    echo "--- MVTec-AD @ ${RES}px, PCA_EV=${PCA_EV} ---"
    conda run -n subspacead python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path "$MVTEC_PATH" \
        --image_res ${RES} \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev ${PCA_EV} \
        --agg_method "$AGG" \
        --k_shot 4 \
        --aug_count 30 \
        --outdir "results_res_ablation/mvtec_res${RES}_ev${PCA_EV}"

    echo "--- VisA @ ${RES}px, PCA_EV=${PCA_EV} ---"
    conda run -n subspacead python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        --image_res ${RES} \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev ${PCA_EV} \
        --agg_method "$AGG" \
        --k_shot 4 \
        --aug_count 30 \
        --outdir "results_res_ablation/visa_res${RES}_ev${PCA_EV}"
done

echo "--- Resolution ablation complete ---"
