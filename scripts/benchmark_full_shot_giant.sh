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

mkdir -p logs

if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
fi

# Set the absolute path to your MVTec AD dataset directory
MVTEC_PATH="${MVTEC_PATH:-../datasets/mvtec-ad}"

# Set the absolute path to your VisA dataset directory
# (e.g., /path/to/your/datasets/VisA_pytorch/1cls)
VISA_PATH="${VISA_PATH:-../datasets/VisA_pytorch/1cls}"
MODEL_CKPT="${MODEL_CKPT:-../facebook/dinov2-with-registers-giant}"
RUN_DATASET="${RUN_DATASET:-all}"
CATEGORIES="${CATEGORIES:-}"
OTM_VARIANT="${OTM_VARIANT:-baseline}"
OTM_BANK_SIZE="${OTM_BANK_SIZE:-8192}"
OTM_BANK_CHUNK="${OTM_BANK_CHUNK:-2048}"
OTM_BANK_SEED="${OTM_BANK_SEED:-42}"
FULL_OUT_ROOT="${FULL_OUT_ROOT:-./results_full_shot}"

CATEGORY_ARGS=()
if [[ -n "${CATEGORIES}" ]]; then
    read -r -a CATEGORY_LIST <<< "${CATEGORIES}"
    CATEGORY_ARGS=(--categories "${CATEGORY_LIST[@]}")
fi

if [[ "${RUN_DATASET}" == "all" || "${RUN_DATASET}" == "mvtec" || "${RUN_DATASET}" == "mvtec_ad" ]]; then
    echo "--- Running Full-Shot (all train images) for MVTec AD | ${OTM_VARIANT} ---"
    python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path "$MVTEC_PATH" \
        --image_res 672 \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "$MODEL_CKPT" \
        --pca_ev 0.99 \
        --agg_method "mean" \
        --otm_variant "$OTM_VARIANT" \
        --otm_bank_size "$OTM_BANK_SIZE" \
        --otm_bank_chunk "$OTM_BANK_CHUNK" \
        --otm_bank_seed "$OTM_BANK_SEED" \
        --outdir "$FULL_OUT_ROOT/mvtec_dinov2G" \
        "${CATEGORY_ARGS[@]}"
fi

if [[ "${RUN_DATASET}" == "all" || "${RUN_DATASET}" == "visa" ]]; then
    echo "--- Running Full-Shot (all train images) for VisA | ${OTM_VARIANT} ---"
    python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        --image_res 672 \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "$MODEL_CKPT" \
        --pca_ev 0.99 \
        --agg_method "mean" \
        --otm_variant "$OTM_VARIANT" \
        --otm_bank_size "$OTM_BANK_SIZE" \
        --otm_bank_chunk "$OTM_BANK_CHUNK" \
        --otm_bank_seed "$OTM_BANK_SEED" \
        --outdir "$FULL_OUT_ROOT/visa_dinov2G" \
        "${CATEGORY_ARGS[@]}"
fi

echo "--- All experiments complete ---"
