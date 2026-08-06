#!/bin/bash
#SBATCH --partition=gpu_a100          # GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=agg-ablation       # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --mem=128G                    # Memory per node
#SBATCH --time=24:00:00               # Walltime
#SBATCH --output=logs/out_%j.txt      # Stdout
#SBATCH --error=logs/err_%j.txt       # Stderr

mkdir -p logs

eval "$($CONDA_EXE shell.bash hook)"


MVTEC_PATH="datasets/mvtec-ad"
VISA_PATH="../AnomalyDINO/VisA_pytorch/1cls/"

MODEL="facebook/dinov2-with-registers-giant"
PCA_EV=0.99
RES=448

declare -A LAYERS_LIST
declare -A AGG_LIST

# 1. Mean-pool (Middle-7)
LAYERS_LIST["mean_middle"]="-12,-13,-14,-15,-16,-17,-18"
AGG_LIST["mean_middle"]="mean"

# 2. Mean-pool (Final-7)
LAYERS_LIST["mean_final"]="-1,-2,-3,-4,-5,-6,-7"
AGG_LIST["mean_final"]="mean"

# 3. Concat (Middle-7)
LAYERS_LIST["concat_middle"]="-12,-13,-14,-15,-16,-17,-18"
AGG_LIST["concat_middle"]="concat"

# 4. Last layer only
LAYERS_LIST["last_layer"]="-1"
AGG_LIST["last_layer"]="mean"

for KEY in "${!LAYERS_LIST[@]}"; do
    LAYERS="${LAYERS_LIST[$KEY]}"
    AGG="${AGG_LIST[$KEY]}"
    
    echo "--- Running MVTec-AD (${KEY}) ---"
    conda run -n subspacead python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path "$MVTEC_PATH" \
        --image_res $RES \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev $PCA_EV \
        --agg_method "$AGG" \
        --k_shot 4 \
        --aug_count 30 \
        --outdir "results_ablation/mvtec_${KEY}_res${RES}_ev${PCA_EV}"

    echo "--- Running VisA (${KEY}) ---"
    conda run -n subspacead python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        --image_res $RES \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev $PCA_EV \
        --agg_method "$AGG" \
        --k_shot 4 \
        --aug_count 30 \
        --outdir "results_ablation/visa_${KEY}_res${RES}_ev${PCA_EV}"
done

echo "--- Layer aggregation ablation complete ---"
