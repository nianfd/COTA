#!/bin/bash
#SBATCH --partition=gpu_h100          # GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=ev-ablation-448    # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=128G                    # Memory per node
#SBATCH --time=24:00:00               # Walltime
#SBATCH --output=logs/out_%j.txt      # Standard output
#SBATCH --error=logs/err_%j.txt       # Standard error

mkdir -p logs
eval "$($CONDA_EXE shell.bash hook)"

MVTEC_PATH="datasets/mvtec-ad"
VISA_PATH="datasets/VisA_pytorch/1cls"
MODEL="facebook/dinov2-with-registers-giant"
LAYERS="-12,-13,-14,-15,-16,-17,-18"
AGG="mean"
RES=448
EV_LIST=(0.95 0.96 0.97 0.99 1.0)
K_LIST=(1 2 4)

echo "--- Starting PCA explained variance ablation ---"
echo "EV values: ${EV_LIST[*]}"
echo "K-shot values: ${K_LIST[*]}"
echo "Resolution: ${RES}px"

for EV in "${EV_LIST[@]}"; do
  for K in "${K_LIST[@]}"; do
    echo "------------------------------------------------------------"
    echo "--- MVTec-AD | PCA_EV=${EV} | k=${K} ---"
    conda run -n subspacead python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path "$MVTEC_PATH" \
        --image_res ${RES} \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev ${EV} \
        --agg_method "$AGG" \
        --k_shot ${K} \
        --aug_count 30 \
        --outdir "results_ev_ablation/mvtec_ev${EV}_k${K}"

    echo "--- VisA | PCA_EV=${EV} | k=${K} ---"
    conda run -n subspacead python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        --image_res ${RES} \
        --layers="$LAYERS" \
        --model_ckpt "$MODEL" \
        --pca_ev ${EV} \
        --agg_method "$AGG" \
        --k_shot ${K} \
        --aug_count 30 \
        --outdir "results_ev_ablation/visa_ev${EV}_k${K}"
  done
done
echo "--- PCA EV x K-shot ablation complete ---"
