#!/bin/bash
set -e
export https_proxy=http://127.0.0.1:33210 http_proxy=http://127.0.0.1:33210 all_proxy=socks5://127.0.0.1:33211

# ============================================================
# E-nose 预训练统一入口
# ============================================================
# 用法:
#   ./run_pretrain.sh [MODE] [MODEL] [SIZE]
#
# MODE:    backbone | downstream | eval
# MODEL:   vqvae | mlp | tcn  (backbone模式下有效)
# SIZE:    small | medium | large
#
# 示例:
#   ./run_pretrain.sh backbone tcn small    # 训练 TCN 骨干 (小模型)
#   ./run_pretrain.sh backbone vqvae medium # 训练 VQ-VAE 骨干 (中模型)
#   ./run_pretrain.sh downstream            # 训练下游模型
#   ./run_pretrain.sh eval                  # 评估 (需设置 checkpoint)
# ============================================================

MODE="${1:-backbone}"
MODEL="${2:-tcn}"
SIZE="${3:-small}"

# ============================================================
# 共用参数 (所有模型通用)
# ============================================================
COMMON_ARGS=(
  # 数据
  --datasets twin_gas_sensor_arrays
  --root .cache
  --max-channels 8
  --max-length 1024
  
  # 训练
  --batch-size 32
  --max-epochs 1000
  --early-stopping-patience 200
  --learning-rate 3e-3
  --mask-ratio 0.25
  
  # 损失
  --loss-type mse
  --lambda-visible 1.0
  --lambda-masked 1.0
  
  # 硬件
  --accelerator gpu
  --precision bf16-mixed
  
  # 日志
  --use-mlflow
  --no-export-best
  
  # 探测 (训练时评估下游任务性能)
  --enable-probe
  --probe-every-n-epochs 1
  --probe-mask-channels 0 1 2 3
)

# ============================================================
# VQ-VAE 特有参数
# ============================================================
VQVAE_ARGS=(
  --num-embeddings 1024
  --commitment-cost 0.25
  --disable-vq  # 目前禁用 VQ，作为纯 Transformer AE
  --patch-size 16
)

# VQ-VAE 尺寸配置: d_model, nhead, enc_layers, dec_layers
declare -A VQVAE_SIZES=(
  ["small"]="64 4 2 2"
  ["medium"]="128 8 4 4"
  ["large"]="256 8 6 6"
)

# ============================================================
# MLP 特有参数 (per-channel 架构，无需额外参数)
# ============================================================
# MLP 尺寸配置: d_model, num_layers
declare -A MLP_SIZES=(
  ["small"]="256 4"
  ["medium"]="384 6"
  ["large"]="512 8"
)

# ============================================================
# TCN 特有参数
# ============================================================
TCN_ARGS=(
  --optimizer soap
)

# TCN 尺寸配置: d_model, num_layers
declare -A TCN_SIZES=(
  ["small"]="128 4"
  ["medium"]="160 6"
  ["large"]="192 8"
)

# ============================================================
# 下游模型参数
# ============================================================
DOWNSTREAM_ARGS=(
  --downstream-model mlp
  --hidden-dim 128
  --num-layers 3
  --dropout 0.1
  --max-epochs 500
  --early-stopping-patience 50
  --learning-rate 1e-3
  --optimizer soap
)

# ============================================================
# Checkpoint 路径 (eval 模式需要)
# ============================================================
BACKBONE_CKPT=""
DOWNSTREAM_CKPT=""
# 示例:
# BACKBONE_CKPT="logs/enose_pretrain/enose_tcn_small/checkpoints/last.ckpt"
# DOWNSTREAM_CKPT="logs/downstream/downstream_mlp_concentration/checkpoints/last.ckpt"

# ============================================================
# 训练函数
# ============================================================
train_backbone() {
  local model="$1"
  local size="$2"
  
  echo "=========================================="
  echo "Training Backbone: $model ($size)"
  echo "=========================================="
  
  # 构建模型特定参数
  local model_args=()
  local size_config=""
  
  case "$model" in
    vqvae)
      model_args=("${VQVAE_ARGS[@]}")
      size_config="${VQVAE_SIZES[$size]}"
      if [ -z "$size_config" ]; then
        echo "Error: Unknown size '$size' for vqvae"
        exit 1
      fi
      read -r d_model nhead enc_layers dec_layers <<< "$size_config"
      model_args+=(
        --d-model "$d_model"
        --num-encoder-layers "$enc_layers"
        --num-decoder-layers "$dec_layers"
      )
      ;;
    mlp)
      size_config="${MLP_SIZES[$size]}"
      if [ -z "$size_config" ]; then
        echo "Error: Unknown size '$size' for mlp"
        exit 1
      fi
      read -r d_model num_layers <<< "$size_config"
      model_args+=(
        --d-model "$d_model"
        --num-encoder-layers "$num_layers"
      )
      ;;
    tcn)
      model_args=("${TCN_ARGS[@]}")
      size_config="${TCN_SIZES[$size]}"
      if [ -z "$size_config" ]; then
        echo "Error: Unknown size '$size' for tcn"
        exit 1
      fi
      read -r d_model num_layers <<< "$size_config"
      model_args+=(
        --d-model "$d_model"
        --num-encoder-layers "$num_layers"
      )
      ;;
    *)
      echo "Error: Unknown model '$model'. Use: vqvae, mlp, tcn"
      exit 1
      ;;
  esac
  
  echo "  d_model: $d_model"
  echo "  layers: ${enc_layers:-$num_layers}"
  
  python scripts/pretrain.py \
    "${COMMON_ARGS[@]}" \
    "${model_args[@]}" \
    --model "$model" \
    --run-name "enose_${model}_${size}"
}

train_downstream() {
  echo "=========================================="
  echo "Training Downstream Model"
  echo "=========================================="
  
  python scripts/downstream_train.py \
    --mode train \
    --data-root .cache \
    --seq-len 1000 \
    --batch-size 32 \
    "${DOWNSTREAM_ARGS[@]}" \
    --log-dir logs/downstream \
    --use-mlflow
  
  echo ""
  echo "Best model saved to: logs/downstream/best_downstream.ckpt"
  echo "(Auto-loaded by backbone probe)"
}

run_eval() {
  echo "=========================================="
  echo "Evaluating Models"
  echo "=========================================="
  
  if [ -z "$BACKBONE_CKPT" ] || [ -z "$DOWNSTREAM_CKPT" ]; then
    echo "Error: eval mode requires BACKBONE_CKPT and DOWNSTREAM_CKPT"
    echo ""
    echo "Please set in run_pretrain.sh:"
    echo '  BACKBONE_CKPT="logs/enose_pretrain/.../last.ckpt"'
    echo '  DOWNSTREAM_CKPT="logs/downstream/.../last.ckpt"'
    exit 1
  fi
  
  python scripts/downstream_train.py \
    --mode eval \
    --data-root .cache \
    --seq-len 1000 \
    --batch-size 32 \
    --backbone-model "$MODEL" \
    --backbone-ckpt "$BACKBONE_CKPT" \
    --downstream-ckpt "$DOWNSTREAM_CKPT" \
    --mask-channels 0
}

# ============================================================
# 主入口
# ============================================================
case "$MODE" in
  backbone)
    train_backbone "$MODEL" "$SIZE"
    ;;
  downstream)
    train_downstream
    ;;
  eval)
    run_eval
    ;;
  *)
    echo "============================================================"
    echo "E-nose 预训练统一入口"
    echo "============================================================"
    echo ""
    echo "用法: $0 [MODE] [MODEL] [SIZE]"
    echo ""
    echo "MODE:"
    echo "  backbone   - 训练骨干模型 (默认)"
    echo "  downstream - 训练下游模型"
    echo "  eval       - 评估 (需设置 checkpoint)"
    echo ""
    echo "MODEL (backbone 模式):"
    echo "  vqvae  - Transformer VQ-VAE"
    echo "  mlp    - Per-channel MLP (默认)"
    echo "  tcn    - Temporal Convolutional Network"
    echo ""
    echo "SIZE:"
    echo "  small  - 小模型 (~0.5-1M 参数)"
    echo "  medium - 中模型 (~1-2M 参数)"
    echo "  large  - 大模型 (~2-4M 参数)"
    echo ""
    echo "示例:"
    echo "  $0 backbone tcn small     # TCN 小模型"
    echo "  $0 backbone vqvae medium  # VQ-VAE 中模型"
    echo "  $0 downstream             # 训练下游模型"
    echo ""
    echo "典型流程:"
    echo "  1. $0 downstream          # 先训练下游模型"
    echo "  2. $0 backbone tcn small  # 训练骨干"
    echo "  3. 设置 BACKBONE_CKPT 和 DOWNSTREAM_CKPT"
    echo "  4. $0 eval                # 评估"
    exit 1
    ;;
esac

echo ""
echo "Done!"
