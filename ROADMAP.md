# UCI电子鼻数据集项目重构优化路线图

## 项目现状分析

### 1. 数据集完整性问题

| 数据集 | file_info.sh | extract.sh | metadata.json | note.md | process_data.py | split_data.py |
|--------|:------------:|:----------:|:-------------:|:-------:|:---------------:|:-------------:|
| alcohol_qcm_sensor_dataset | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gas_sensor_array_drift_dataset_at_different_concentrations | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| gas_sensor_array_exposed_to_turbulent_gas_mixtures | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gas_sensor_array_low-concentration | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gas_sensor_array_temperature_modulation | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| gas_sensor_array_under_dynamic_gas_mixtures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gas_sensor_array_under_flow_modulation | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| gas_sensors_for_home_activity_monitoring | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| twin_gas_sensor_arrays | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 2. 核心问题总结

#### 2.1 数据集配置不同构
- **file_info.sh**: shell脚本变量定义，不便于Python程序读取
- **extract.sh**: 每个数据集的解压逻辑分散且不统一
- 缺乏统一的数据集注册中心

#### 2.2 下载机制问题
- 依赖bash脚本和wget，不够优雅
- 缺少进度显示、断点续传功能
- 没有Python API接口
- 无法方便地在代码中按需下载

#### 2.3 元数据不完整
- 多个数据集缺少metadata.json
- 现有metadata缺少关键字段: `dataset_url`, `dataset_license`, `dataset_description`
- `label_columns`普遍为空，标签信息未充分记录

#### 2.4 处理流程不统一
- process_data.py输出格式不完全一致
- 列命名规范不统一 (sensor_0 vs sensor0)
- 缺少统一的数据加载接口

---

## 重构优化路线图

### Phase 1: 基础设施重构 (优先级: 高)

#### 1.1 数据集注册中心 ✨
将分散的file_info.sh合并为统一的Python配置:

```python
# enose_uci_dataset/registry.py
DATASETS = {
    "gas_sensors_for_home_activity_monitoring": {
        "file_name": "gas+sensors+for+home+activity+monitoring.zip",
        "sha1_hash": "34101ca24e556dc14a6ee1e2910111ed49c0e6ce",
        "url": "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        "uci_id": 362,
    },
    # ...
}
```

#### 1.2 下载模块重构 ✨
创建统一的Python下载器:
- 支持进度条显示 (tqdm)
- SHA1完整性校验
- 断点续传支持
- 简洁的Python API

```python
from enose_uci_dataset import download
download("gas_sensors_for_home_activity_monitoring")
download_all()
```

#### 1.3 解压模块重构
将extract.sh逻辑迁移到Python:
- 统一的解压接口
- 自动目录结构整理
- 错误处理和日志

---

### Phase 2: 元数据标准化 (优先级: 高)

#### 2.1 完善metadata.json
为所有数据集补充完整的metadata.json:
- 填充`dataset_url`, `dataset_license`, `dataset_description`
- 完善`label_columns`定义
- 添加`supported_tasks`字段标注适用的下游任务

#### 2.2 元数据验证器
创建JSON Schema验证工具，确保所有metadata符合规范

#### 2.3 统一列命名规范
- 传感器列: `sensor_0`, `sensor_1`, ...
- 时间列: `t_s` (秒为单位)
- 环境列: `temp`, `humidity`
- 标签列: `label_gas`, `label_conc_*`

---

### Phase 3: 数据处理标准化 (优先级: 中)

#### 3.1 统一处理输出格式
所有process_data.py输出到:
```
processed/v1/ssl_samples/  # 自监督学习样本
processed/v1/full/         # 完整时序数据
```

#### 3.2 标准化数据列
输出CSV统一包含:
- 传感器列 (`sensor_0` ~ `sensor_N`)
- 时间列 (`t_s`)
- 环境列 (如有: `temp`, `humidity`)
- 标签列 (`label_*`)

#### 3.3 补充缺失的处理脚本
为以下数据集补充process_data.py:
- [ ] alcohol_qcm_sensor_dataset
- [ ] gas_sensor_array_exposed_to_turbulent_gas_mixtures
- [ ] gas_sensor_array_temperature_modulation
- [ ] gas_sensor_array_under_flow_modulation (需补充metadata)

---

### Phase 4: 下游任务接口 (优先级: 中)

#### 4.1 统一数据加载器
```python
from enose_uci_dataset import load_dataset

# 加载单个数据集
ds = load_dataset("gas_sensors_for_home_activity_monitoring")

# 获取训练/验证/测试集
train, val, test = ds.get_splits()
```

#### 4.2 任务适配器
针对不同下游任务提供数据转换:
- 气体分类: `ds.for_classification()`
- 浓度预测: `ds.for_regression()`
- 漂移补偿: `ds.for_drift_compensation()`
- 时序预测: `ds.for_forecasting()`

---

### Phase 5: 文档与测试 (优先级: 低)

#### 5.1 完善文档
- 更新主README.md
- 补充各数据集note.md
- 添加使用示例和可视化

#### 5.2 单元测试
- 下载模块测试
- 数据处理测试
- 元数据验证测试

---

## 实施优先级

1. **立即执行**: Phase 1.1 + 1.2 (数据集注册中心 + 下载模块)
2. **短期**: Phase 2 (元数据标准化)
3. **中期**: Phase 3 + 4 (处理标准化 + 下游接口)
4. **长期**: Phase 5 (文档测试)

---

## 技术栈

- **包管理**: uv (已配置)
- **依赖**: pandas, numpy, tqdm
- **下载**: requests (新增)
- **测试**: pytest (新增)

---

## 当前进度追踪

### Phase 1 进度
- [ ] 创建 `registry.py` 数据集注册中心
- [ ] 创建 `downloader.py` 下载模块
- [ ] 创建 `extractor.py` 解压模块
- [ ] 迁移所有file_info.sh到registry
- [ ] 测试下载流程

### Phase 2 进度
- [ ] 补充缺失的metadata.json
- [ ] 创建元数据验证器
- [ ] 统一现有metadata格式
