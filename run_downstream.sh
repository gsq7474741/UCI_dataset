#!/bin/bash
set -e
export https_proxy=http://127.0.0.1:33210 http_proxy=http://127.0.0.1:33210 all_proxy=socks5://127.0.0.1:33211

# ============================================================
# 下游任务独立训练脚本
# ============================================================
# 用途: 独立训练下游模型 (不依赖骨干)
# 
# 注意: 
#   - 骨干训练请使用 ./run_pretrain.sh backbone
#   - 评估请使用 ./run_pretrain.sh eval
#   - 此脚本仅用于独立训练下游模型作为 baseline
# ============================================================

echo "=========================================="
echo "Training Downstream Model (Standalone)"
echo "=========================================="

python scripts/downstream_train.py \
  --mode train \
  --data-root .cache \
  --seq-len 1000 \
  --batch-size 32 \
  --downstream-model mlp \
  --hidden-dim 128 \
  --num-layers 3 \
  --dropout 0.1 \
  --max-epochs 200 \
  --early-stopping-patience 30 \
  --learning-rate 1e-3 \
  --log-dir logs/downstream

echo "Done!"
