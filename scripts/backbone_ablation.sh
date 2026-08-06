#!/bin/bash
#SBATCH --partition=gpu_a100      # GPU partition
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --job-name=backbone       # job name
#SBATCH --ntasks=1                # number of tasks
#SBATCH --cpus-per-task=4         # number of CPU cores
#SBATCH --mem=128G                # memory per node
#SBATCH --time=24:00:00           # walltime
#SBATCH --output=logs/out_%j.txt  # standard output
#SBATCH --error=logs/err_%j.txt   # standard error

eval "$($CONDA_EXE shell.bash hook)"

# Set the absolute path to your MVTec AD dataset directory
MVTEC_PATH="datasets/mvtec-ad"

# Set the absolute path to your VisA dataset directory
# (e.g., /path/to/your/datasets/VisA_pytorch/1cls)
VISA_PATH="datasets/VisA_pytorch/1cls"

# Model identifiers from Hugging Face
BACKBONES=(
    "facebook/dinov2-small"
    "facebook/dinov2-base"
    "facebook/dinov2-large"
)

# Short names for output directories
NAMES=(
    "dinov2S"
    "dinov2B"
    "dinov2L"
)

# Proportional "equivalent middle layers" based on the
# DINOv2-G selection (layers 22-28, or 55%-70% deep)
LAYERS_STRINGS=(
    "-4,-5"          # S/B (12 layers): 55%-70% deep is ~layers 7-8
    "-4,-5"          # S/B (12 layers): 55%-70% deep is ~layers 7-8
    "-7,-8,-9,-10,-11" # L (24 layers): 55%-70% deep is ~layers 13-17
    "-12,-13,-14,-15,-16,-17,-18" # G (40 layers): 55%-70% deep is layers 22-28
)
for i in ${!BACKBONES[@]}; do
    MODEL_CKPT=${BACKBONES[$i]}
    MODEL_NAME=${NAMES[$i]}
    LAYERS=${LAYERS_STRINGS[$i]}

    echo "========================================================"
    echo "=== STARTING EXPERIMENTS FOR BACKBONE: $MODEL_NAME ($MODEL_CKPT) ==="
    echo "=== Using Layers: $LAYERS ==="
    echo "========================================================"
    
    echo "--- Starting k-shot experiments for MVTec AD ---"
    for k in 1 2 4
    do
        echo "--- Running MVTec AD k=$k for $MODEL_NAME ---"
        conda run -n subspacead python -u main.py \
            --dataset_name mvtec_ad \
            --dataset_path "$MVTEC_PATH" \
            --image_res 448 \
            --k_shot $k \
            --layers="$LAYERS" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count 30 \
            --pca_ev 0.99 \
            --agg_method "mean" \
            --outdir "results_ablations/results_k${k}_mvtec_${MODEL_NAME}" \
            --save_intro_overlays
    done


    echo "--- Starting k-shot experiments for VisA ---"
    for k in 1 2 4
    do
        echo "--- Running VisA k=$k for $MODEL_NAME ---"
        conda run -n subspacead python -u main.py \
            --dataset_name visa \
            --dataset_path "$VISA_PATH" \
            --image_res 448 \
            --k_shot $k \
            --layers="$LAYERS" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count 30 \
            --pca_ev 0.99 \
            --agg_method "mean" \
            --outdir "results_ablations/results_k${k}_visa_${MODEL_NAME}" \
            --save_intro_overlays 
    done

    echo "=== COMPLETED EXPERIMENTS FOR BACKBONE: $MODEL_NAME ==="

done

echo "--- All experiments complete ---"