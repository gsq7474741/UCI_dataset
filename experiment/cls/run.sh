#!/bin/bash
set -e

# ============================================================
# E-nose 单数据集分类实验
# ============================================================
# 用法:
#   ./experiment/cls/run.sh [DATASET] [MODE]
#
# DATASET: 数据集名称 (默认: twin_gas_sensor_arrays)
# MODE: train | test | all (默认: train)
#
# 示例:
#   ./experiment/cls/run.sh twin_gas_sensor_arrays train
#   ./experiment/cls/run.sh gas_sensor_array_drift_dataset_at_different_concentrations train
#   ./experiment/cls/run.sh all train  # 训练所有数据集
# ============================================================

DATASET="${1:-twin_gas_sensor_arrays}"
MODE="${2:-train}"

# ============================================================
# 共用参数
# ============================================================
COMMON_ARGS=(
  --root .cache
  --max-length 512
  --num-channels 8
  
  # 训练
  --batch-size 32
  --max-epochs 200
  --learning-rate 3e-3
  --early-stopping-patience 10
  
  # 模型
  --hidden-dim 64
  --num-layers 4
  --kernel-size 7
  --dropout 0.1
  
  # 硬件
  --accelerator gpu
  --precision bf16-mixed
  
  # 日志
  --use-mlflow
  --mlflow-tracking-uri http://localhost:5000
)

# ============================================================
# 数据集列表
# ============================================================
DATASETS=(
  "twin_gas_sensor_arrays"
  "gas_sensor_array_drift_dataset_at_different_concentrations"
  "gas_sensor_array_low_concentration"
  "gas_sensor_array_under_flow_modulation"
  "alcohol_qcm_sensor_dataset"
  "gas_sensor_array_exposed_to_turbulent_gas_mixtures"
  "gas_sensor_array_under_dynamic_gas_mixtures"
  "g919_55"
)

# ============================================================
# 训练单个数据集
# ============================================================
train_single() {
  local dataset="$1"
  
  echo "=========================================="
  echo "Training: $dataset"
  echo "=========================================="
  
  python experiment/cls/train.py \
    "${COMMON_ARGS[@]}" \
    --dataset "$dataset" \
    --download
}

# ============================================================
# 训练所有数据集
# ============================================================
train_all() {
  echo "=========================================="
  echo "Training all datasets"
  echo "=========================================="
  
  for dataset in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Processing: $dataset"
    echo ""
    train_single "$dataset"
  done
  
  echo ""
  echo "=========================================="
  echo "All datasets trained!"
  echo "=========================================="
}

# ============================================================
# 主入口
# ============================================================
case "$DATASET" in
  all)
    train_all
    ;;
  *)
    if [[ " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
      train_single "$DATASET"
    else
      echo "Error: Unknown dataset '$DATASET'"
      echo ""
      echo "Available datasets:"
      for ds in "${DATASETS[@]}"; do
        echo "  - $ds"
      done
      echo ""
      echo "Or use 'all' to train all datasets"
      exit 1
    fi
    ;;
esac

echo ""
echo "Done!"
