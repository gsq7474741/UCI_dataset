# E-nose 单数据集分类实验

本实验在每个数据集上独立训练1D CNN分类器，标签使用全局标签空间。

## 目录结构

```
experiment/cls/
├── __init__.py       # 包初始化
├── dataset.py        # 数据集类，全局标签映射
├── model.py          # 1D CNN 分类模型
├── datamodule.py     # Lightning 数据模块
├── train.py          # 训练脚本
├── run.sh            # 运行脚本
└── README.md         # 本文档
```

## 支持的数据集

| 数据集 | 类别数 | 任务类型 | 描述 |
|--------|--------|----------|------|
| `twin_gas_sensor_arrays` | 4 | 气体分类 | 4种纯气体: Ethanol, CO, Ethylene, Methane |
| `gas_sensor_array_drift_dataset_at_different_concentrations` | 6 | 气体分类 | 6种气体，含漂移 |
| `gas_sensor_array_low_concentration` | 6 | 气体分类 | 6种低浓度VOC |
| `gas_sensor_array_under_flow_modulation` | 4 | 气体分类 | Air, Acetone, Ethanol, Mixture |
| `alcohol_qcm_sensor_dataset` | 5 | 醇类分类 | 5种醇类，QCM传感器 |
| `gas_sensor_array_exposed_to_turbulent_gas_mixtures` | 2 | 混合物分类 | Ethylene+Methane vs Ethylene+CO |
| `gas_sensor_array_under_dynamic_gas_mixtures` | 2 | 混合物分类 | Ethylene+CO vs Ethylene+Methane |

## 模型架构

### 1D CNN

```
Input [B, C, T]
    │
    ├── Input Projection (Conv1d 1x1)
    │
    ├── Conv1d Block × N
    │   ├── Conv1d + BN + GELU + Dropout
    │   ├── Conv1d + BN + GELU + Dropout
    │   └── Residual Connection
    │
    ├── Global Average Pooling
    │
    └── Classification Head (MLP)
        └── Output [B, num_classes]
```

## 使用方法

### 快速开始

```bash
# 激活环境
conda activate enose

# 训练单个数据集
./experiment/cls/run.sh twin_gas_sensor_arrays

# 训练所有数据集
./experiment/cls/run.sh all
```

### 自定义训练

```bash
python experiment/cls/train.py \
    --dataset twin_gas_sensor_arrays \
    --hidden-dim 128 \
    --num-layers 6 \
    --batch-size 64 \
    --max-epochs 300 \
    --use-mlflow
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | twin_gas_sensor_arrays | 数据集名称 |
| `--hidden-dim` | 64 | 隐藏层维度 |
| `--num-layers` | 4 | CNN层数 |
| `--kernel-size` | 7 | 卷积核大小 |
| `--max-length` | 512 | 最大序列长度 |
| `--num-channels` | 8 | 传感器通道数 |
| `--batch-size` | 32 | 批大小 |
| `--learning-rate` | 1e-3 | 学习率 |
| `--use-class-weights` | False | 使用类别权重 |
| `--use-mlflow` | False | 使用MLflow记录 |

## 日志和监控

### TensorBoard

```bash
tensorboard --logdir logs/cls
```

### MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## 输出文件

训练完成后，会在 `logs/cls/<run_name>/<timestamp>/` 下生成：

```
├── checkpoints/
│   ├── epoch=XXX-val_acc=X.XXXX.ckpt  # 最佳模型
│   └── last.ckpt                       # 最后模型
├── csv/
│   └── metrics.csv                     # 训练指标
├── events.out.tfevents.*               # TensorBoard日志
└── hparams.yaml                        # 超参数
```

## 全局标签空间

所有数据集的标签都映射到统一的 `GasLabel` 枚举空间，定义在 `enose_uci_dataset/datasets/_global.py`。

主要标签包括:
- `CO` - 一氧化碳
- `ETHANOL` - 乙醇
- `ETHYLENE` - 乙烯
- `METHANE` - 甲烷
- `ACETONE` - 丙酮
- `AMMONIA` - 氨气
- `TOLUENE` - 甲苯
- ... 等

## 技术栈

- **框架**: PyTorch Lightning
- **日志**: TensorBoard, MLflow, CSV
- **指标**: Accuracy, F1 (macro/micro), Precision, Recall, Confusion Matrix
