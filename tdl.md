# 电子鼻气体识别数据集任务列表

## enose_uci_dataset

### 0. 下载和校验脚本 (file_info.sh, extract.sh)
- [ ] alcohol_qcm_sensor_dataset
- [ ] gas_sensor_array_drift_dataset_at_different_concentrations
- [ ] gas_sensor_array_exposed_to_turbulent_gas_mixtures
- [ ] gas_sensor_array_low-concentration
- [ ] gas_sensor_array_temperature_modulation
- [ ] gas_sensor_array_under_dynamic_gas_mixtures
- [ ] gas_sensor_array_under_flow_modulation
- [ ] gas_sensors_for_home_activity_monitoring
- [ ] twin_gas_sensor_arrays

### 1. 文档和元数据构建 (note.md, metadata.json)
- [ ] alcohol_qcm_sensor_dataset
- [ ] gas_sensor_array_drift_dataset_at_different_concentrations
- [ ] gas_sensor_array_exposed_to_turbulent_gas_mixtures
- [ ] gas_sensor_array_low-concentration
- [ ] gas_sensor_array_temperature_modulation
- [ ] gas_sensor_array_under_dynamic_gas_mixtures
- [ ] gas_sensor_array_under_flow_modulation
- [ ] gas_sensors_for_home_activity_monitoring
- [ ] twin_gas_sensor_arrays

### 2. 数据处理程序 (process_data.py)
- [ ] alcohol_qcm_sensor_dataset
- [ ] gas_sensor_array_drift_dataset_at_different_concentrations
- [ ] gas_sensor_array_exposed_to_turbulent_gas_mixtures
- [ ] gas_sensor_array_low-concentration
- [ ] gas_sensor_array_temperature_modulation
- [ ] gas_sensor_array_under_dynamic_gas_mixtures
- [ ] gas_sensor_array_under_flow_modulation
- [ ] gas_sensors_for_home_activity_monitoring
- [ ] twin_gas_sensor_arrays

### 3. 数据分割程序 (split_data.py)
- [ ] gas_sensors_for_home_activity_monitoring
- [ ] twin_gas_sensor_arrays
- [ ] 其他需要数据分割的数据集

### 4. 数据集标准化和一致性检查
- [ ] 统一元数据格式 (metadata.json)
- [ ] 验证所有数据集符合 dataset_metadata_schema.json 规范
- [ ] 确保所有传感器数据列命名一致
- [ ] 验证所有数据集的单位转换函数正确

### 5. 下游任务准备
- [ ] 气体分类任务数据准备
- [ ] 浓度预测任务数据准备
- [ ] 传感器漂移补偿任务数据准备
- [ ] 异常检测任务数据准备
- [ ] 多传感器融合任务数据准备
- [ ] 实时监测任务数据准备
- [ ] 气味指纹识别任务数据准备

### 6. 数据集文档完善
- [ ] 完善每个数据集的 note.md 文档
- [ ] 添加数据集使用示例
- [ ] 添加数据集可视化示例
- [ ] 更新主 README.md 文档
