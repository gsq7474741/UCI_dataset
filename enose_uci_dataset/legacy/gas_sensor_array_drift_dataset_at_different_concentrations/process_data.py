import os
import csv
import glob

import pandas as pd
from tqdm import tqdm, trange

# 创建输出目录
output_dir = './ssl_csv_samples'
os.makedirs(output_dir, exist_ok=True)

# 气体标签映射
gas_mapping = {
    '1': 'Ethanol',
    '2': 'Ethylene',
    '3': 'Ammonia',
    '4': 'Acetaldehyde',
    '5': 'Acetone',
    '6': 'Toluene'
}

sample_id = 0

# 处理每个批次文件
for batch_idx in trange(1,11):
    batch_file=f'.raw/batch{batch_idx}.dat'
    with open(batch_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            gas_label, concentration = parts[0].split(';')

            features = parts[1:]

            # 创建16个传感器的8个特征
            sensor_features = [[] for _ in range(16)]
            for i, feature in enumerate(features):
                sensor_idx = i % 16
                sensor_features[sensor_idx].append(feature.split(':')[1])

            # 准备CSV行
            csv_row = []
            for feature_idx in range(8):
                csv_row.append([sensor_features[sensor_idx][feature_idx] for sensor_idx in range(16)]+[gas_mapping[gas_label], concentration, batch_idx])

            df = pd.DataFrame(csv_row)

            df.columns = [f'sensor_{i}' for i in range(16)] + ['label_gas', 'label_ppm', 'label_batch']
            df.to_csv(f'{output_dir}/{sample_id}_{gas_mapping[gas_label]}_{float(concentration):.2f}ppm_batch{batch_idx}.csv', index=False)
            # # 创建CSV文件
            # filename = f"{gas_mapping[gas_label]}_{concentration}_{batch_idx}_{sample_id}.csv"
            # filepath = os.path.join(output_dir, filename)
            #
            # with open(filepath, 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow([f'sensor_{i}' for i in range(16)] + ['label_gas', 'label_ppm', 'label_batch'])
            #     writer.writerow(csv_row)
            sample_id += 1

print("CSV 文件生成完成.")
