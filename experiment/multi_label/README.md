# Multi-label Gas Classification Experiment

## 实验目标

**混合物解耦 (Mixture Decomposition)**：使用纯净气体数据训练模型，测试模型在混合气体数据上能否识别出混合物中包含哪些成分气体。

## 实验设计

### 核心思路

混合数据集中包含 Ethylene+Methane 和 Ethylene+CO 两种混合物，因此需要训练模型识别这3种气体：
- **Ethylene (乙烯)**
- **Methane (甲烷)**
- **CO (一氧化碳)**

### 训练数据（纯净气体，单标签）

| 数据集 | 描述 | 气体类型 |
|--------|------|---------|
| `twin_gas_pure` | TwinGasSensorArrays 纯净气体 | Ethylene, Methane, CO (各10个浓度等级) |

> 注意：TwinGasSensorArrays 还包含 Ethanol，但由于混合数据集中不包含 Ethanol，因此过滤掉。

### 测试数据（混合气体，多标签）

| 数据集 | 描述 | 混合类型 |
|--------|------|---------|
| `gas_sensor_turbulent` | 湍流气体混合物 | Ethylene + Methane 或 Ethylene + CO |
| `gas_sensor_dynamic` | 动态气体混合物 | Ethylene + CO 或 Ethylene + Methane |

## 模型架构

### TCN (Temporal Convolutional Network)
- 适合时间序列数据
- 使用扩张卷积捕获长程依赖
- 默认配置：128 hidden dim, 4 layers

### MLP
- 简单基线模型
- 将时间序列展平后处理
- 默认配置：256 hidden dim, 3 layers

## 使用方法

### 运行训练

```bash
# 激活 conda 环境
conda activate enose

# 使用 TCN 编码器
./experiment/multi_label/run.sh train tcn

# 使用 MLP 编码器
./experiment/multi_label/run.sh train mlp

# 查看实验设计信息
./experiment/multi_label/run.sh info
```

### 直接使用 Python

```bash
python experiment/multi_label/train.py \
    --encoder tcn \
    --hidden-dim 128 \
    --num-layers 4 \
    --max-epochs 200 \
    --batch-size 32
```

## 评估指标

- **Multi-label Accuracy**: 预测正确的标签比例
- **F1-micro/macro**: 多标签 F1 分数
- **Precision/Recall**: 精确率和召回率
- **AUROC**: ROC 曲线下面积

### 特殊指标

- **Mixture Accuracy**: 仅在混合物样本上的准确率，衡量模型识别多组分的能力

## 技术栈

- **框架**: PyTorch Lightning
- **日志**: TensorBoard, CSV
- **优化器**: AdamW + CosineAnnealingWarmRestarts

## 目录结构

```
experiment/multi_label/
├── __init__.py        # 包初始化
├── dataset.py         # 数据集和标签编码器
├── model.py           # 多标签分类模型
├── datamodule.py      # Lightning DataModule
├── train.py           # 训练脚本
├── run.sh             # 运行脚本
└── README.md          # 本文档
```

## 实验结果

训练完成后，结果保存在 `logs/multi_label/` 目录下：
- `checkpoints/`: 模型检查点
- `csv/`: 训练指标 CSV
- TensorBoard 日志

查看 TensorBoard：
```bash
tensorboard --logdir logs/multi_label
```

## 核心思想

1. **特征学习**: 模型在纯净气体上学习每种气体的"指纹"特征
2. **泛化测试**: 在混合气体上测试，看模型能否识别组分
3. **多标签输出**: 使用 BCE loss + sigmoid，输出每个气体类别（3类）的概率
4. **Label Smoothing**: 使用标签平滑提高泛化能力

## 预期挑战

1. **域偏移**: 纯净气体和混合气体的传感器响应模式可能不同
2. **浓度变化**: 不同浓度等级的气体响应幅度不同
3. **传感器交叉敏感**: 某些传感器对多种气体都有响应
