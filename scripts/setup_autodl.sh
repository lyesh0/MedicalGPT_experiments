#!/bin/bash
#
# AutoDL 新主机一键启动脚本
# 用法: bash scripts/setup_autodl.sh
#

set -euo pipefail

echo "=========================================="
echo "1/6: 配置 conda 环境"
echo "=========================================="
conda init bash
source ~/.bashrc
conda create -n medicalgpt python=3.10 -y
source activate medicalgpt

echo ""
echo "=========================================="
echo "2/6: 安装 PyTorch (CUDA 12.4)"
echo "=========================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=========================================="
echo "3/6: 克隆项目并安装依赖"
echo "=========================================="
cd ~/autodl-tmp
git clone https://github.com/lyesh0/MedicalGPT_experiments.git
cd MedicalGPT_experiments
git checkout exp/sft-ablation

# 先装 datasets 2.x（3.x 不兼容老格式数据集）
pip install "datasets>=2.14.6,<3.0"
# 装其余依赖，用清华源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# flash-attn 可选，编译失败也关系不大
pip install flash-attn --no-build-isolation || echo "flash-attn 编译失败，训练时加 --flash_attn False"

echo ""
echo "=========================================="
echo "4/6: 下载 SFT 数据"
echo "=========================================="
python tools/experiments/download_sft_data.py --use_mirror

echo ""
echo "=========================================="
echo "5/6: 分析 + 清洗 + 构造训练集"
echo "=========================================="
python tools/experiments/analyze_sft_data.py \
    --input_dir data/experiments/sft/medical/ \
    --output_stats data/experiments/sft/medical/analysis.json

python tools/experiments/analyze_sft_data.py \
    --input_dir data/experiments/sft/general/

python tools/experiments/clean_sft_data.py \
    --input_dir data/experiments/sft/medical/ \
    --output data/experiments/sft/cleaned/medical_clean.jsonl \
    --max_samples 5000

python tools/experiments/build_sft_mixture.py \
    --medical_raw data/experiments/sft/medical/ \
    --medical_clean data/experiments/sft/cleaned/medical_clean.jsonl \
    --general data/experiments/sft/general/ \
    --total_samples 5000 \
    --validate

echo ""
echo "=========================================="
echo "6/6: 开始训练 SFT A/B/C"
echo "=========================================="
echo "环境准备完成！执行以下命令开始训练："
echo ""
echo "  # 训练全部三组"
echo "  bash scripts/experiments/run_sft_training.sh"
echo ""
echo "  # 或单独训练一组"
echo "  bash scripts/experiments/run_sft_training.sh --group A"
echo ""
echo "  # 先打印命令确认无误"
echo "  bash scripts/experiments/run_sft_training.sh --dry_run"
