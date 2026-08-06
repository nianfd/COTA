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

if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
fi

# Defaults preserve the original invocation. Every value may be overridden by
# an environment variable, which is how the one-command ablation script routes
# the four variants without duplicating benchmark settings.
MVTEC_PATH="${MVTEC_PATH:-../datasets/mvtec-ad}"
VISA_PATH="${VISA_PATH:-../datasets/VisA_pytorch/1cls}"
MODEL_CKPT="${MODEL_CKPT:-../facebook/dinov2-with-registers-giant}"
RUN_DATASET="${RUN_DATASET:-all}"
CATEGORIES="${CATEGORIES:-}"
K_VALUES="${K_VALUES:-1 2 4}"
OTM_VARIANT="${OTM_VARIANT:-baseline}"
OTM_BANK_SIZE="${OTM_BANK_SIZE:-8192}"
OTM_BANK_CHUNK="${OTM_BANK_CHUNK:-2048}"
OTM_BANK_SEED="${OTM_BANK_SEED:-42}"
FEW_OUT_ROOT="${FEW_OUT_ROOT:-./few_shot_results}"

CATEGORY_ARGS=()
if [[ -n "${CATEGORIES}" ]]; then
    read -r -a CATEGORY_LIST <<< "${CATEGORIES}"
    CATEGORY_ARGS=(--categories "${CATEGORY_LIST[@]}")
fi

if [[ "${RUN_DATASET}" == "all" || "${RUN_DATASET}" == "mvtec" || "${RUN_DATASET}" == "mvtec_ad" ]]; then
    echo "--- Starting k-shot experiments for MVTec AD | ${OTM_VARIANT} ---"
    for k in ${K_VALUES}
    do
        echo "--- Running MVTec AD k=$k ---"
        python -u main.py \
            --dataset_name mvtec_ad \
            --dataset_path "$MVTEC_PATH" \
            --image_res 672 \
            --k_shot "$k" \
            --layers="-12,-13,-14,-15,-16,-17,-18" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count 30 \
            --pca_ev 0.99 \
            --seed 46 \
            --agg_method "mean" \
            --otm_variant "$OTM_VARIANT" \
            --otm_bank_size "$OTM_BANK_SIZE" \
            --otm_bank_chunk "$OTM_BANK_CHUNK" \
            --otm_bank_seed "$OTM_BANK_SEED" \
            --outdir "$FEW_OUT_ROOT/results_k${k}_mvtec_dinov2G" \
            "${CATEGORY_ARGS[@]}"
    done
fi

if [[ "${RUN_DATASET}" == "all" || "${RUN_DATASET}" == "visa" ]]; then
    echo "--- Starting k-shot experiments for VisA | ${OTM_VARIANT} ---"
    for k in ${K_VALUES}
    do
        echo "--- Running VisA k=$k ---"
        python -u main.py \
            --dataset_name visa \
            --dataset_path "$VISA_PATH" \
            --image_res 672 \
            --k_shot "$k" \
            --layers="-12,-13,-14,-15,-16,-17,-18" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count 30 \
            --pca_ev 0.99 \
            --agg_method "mean" \
            --seed 46 \
            --otm_variant "$OTM_VARIANT" \
            --otm_bank_size "$OTM_BANK_SIZE" \
            --otm_bank_chunk "$OTM_BANK_CHUNK" \
            --otm_bank_seed "$OTM_BANK_SEED" \
            --outdir "$FEW_OUT_ROOT/results_k${k}_visa_dinov2G" \
            "${CATEGORY_ARGS[@]}"
    done
fi

echo "--- All experiments complete ---"
