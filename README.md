# 基于子空间投影正常原型建模与交叉参照一致性的免训练少样本工业图像异常检测

## 数据准备

### MVTec-AD

在线下载数据集 [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad) 并解压到:

```
datasets/mvtec-ad/
```

### VisA

```bash
# 1. 下载和解压
mkdir -p datasets/VisA_20220922
wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
tar -xvf VisA_20220922.tar -C datasets/VisA_20220922

# 2. 格式转换
python tools/prepare_visa.py \
    --data-folder datasets/VisA_20220922 \
    --save-folder datasets/VisA_pytorch
```

## DINOv2视觉特征提取模型准备
从https://huggingface.co/facebook/dinov2-with-registers-giant 下载所有文件放在facebook/dinov2-with-registers-giant文件夹

## 文件结构

```text
cotacodes/
├── datasets/                   # Dataset root
│   ├── mvtec-ad/
│   └── VisA_pytorch/
├── facebook/                   # DINOV2 root
│   └── dinov2-with-registers-giant/
├── logs/                       # Experiment logs
├── scripts/                    # SLURM/Bash benchmark scripts
│   ├── benchmark_cota_ablation_giant.sh
│   ├── benchmark_full_shot.sh
│   └── ...
├── src/
│   └── subspacead/
│       ├── core/               # PCA + feature extraction code
│       ├── data/               # Dataset loaders & transforms
│       ├── post_process/       # Scoring, pixel maps, filters
│       └── utils/              # Visualization, logging
├── tools/
│   └── prepare_visa.py
└── README.md
```

## Usage

### Benchmark Scripts

scripts/ 目录下提供了用于复现所有结果的脚本。请根据需要，在脚本中修改 MVTEC_PATH 和 VISA_PATH 的路径。

**cota少样本实验 (1, 2, 4 Shots)**
每个类别选择*k*张正常样本

注意：默认运行下面的脚本，随机种子是42，若要修改为其它种子，请同时修改benchmark_cota_ablation_giant.sh和benchmark_few_shot_giant.sh中的OTM_BANK_SEED变量。

```bash
PYTHONPATH=./src:$PYTHONPATH K_VALUES="1 2 4" bash scripts/benchmark_cota_ablation_giant.sh few_shot mvtec
```

```bash
PYTHONPATH=./src:$PYTHONPATH K_VALUES="1 2 4" bash scripts/benchmark_cota_ablation_giant.sh few_shot visa
```


**消融实验**

```bash
PYTHONPATH=./src:$PYTHONPATH K_VALUES="1 2 4" COTA_VARIANTS="baseline otm_full otm_decoupled calibrated_q calibrated_sum cota_safe" bash scripts/benchmark_cota_ablation_giant.sh few_shot mvtec
```

```bash
PYTHONPATH=./src:$PYTHONPATH K_VALUES="1 2 4" COTA_VARIANTS="baseline otm_full otm_decoupled calibrated_q calibrated_sum cota_safe" bash scripts/benchmark_cota_ablation_giant.sh few_shot visa
```


## Benchmark Results

### Few-Shot

**表1 MVTec AD 与 VisA 数据集上少样本异常检测与定位结果（%）**

| Shot | 方法 | MVTec-I-AUROC | MVTec-I-AUPR | MVTec-P-AUROC | MVTec-AU-PRO | VisA-I-AUROC | VisA-I-AUPR | VisA-P-AUROC | VisA-AU-PRO |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | COTA | 97.86±0.49 | 99.18±0.16 | 97.10±0.16 | 94.30±0.34 | 93.78±0.46 | 94.34±0.36 | 98.14±0.04 | 94.84±0.15 |
| 2 | COTA | 98.31±0.07 | 99.34±0.08 | 97.54±0.08 | 95.16±0.13 | 94.52±0.25 | 94.94±0.11 | 98.32±0.02 | 95.27±0.10 |
| 4 | COTA | 98.36±0.33 | 99.34±0.15 | 97.82±0.09 | 95.28±0.11 | 94.78±0.21 | 95.16±0.24 | 98.40±0.01 | 95.52±0.09 |
