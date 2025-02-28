import pandas as pd
import os
from datetime import datetime, timedelta

# 创建输出目录
output_dir = './ssl_csv_samples'
os.makedirs(output_dir, exist_ok=True)

# 读取元数据
metadata = pd.read_csv('raw/HT_Sensor_metadata.dat', sep='\t+')

# 读取传感器数据
sensor_data = pd.read_csv('raw/HT_Sensor_dataset.dat', sep='\s+')


# 定义气体映射字典
gas_mapping = {
    'banana': 0,
    'wine': 1,
    'background': 2,
}

# 处理每个样本
for _, meta_row in metadata.iterrows():
    sample_id = meta_row['id']
    gas_type = meta_row['class']
    date = datetime.strptime(meta_row['date'], '%m-%d-%y')

    # 获取该样本的传感器数据
    sample_data = sensor_data[sensor_data['id'] == sample_id].copy()

    # 添加日期列
    sample_data['date'] = date + pd.to_timedelta(sample_data['time'], unit='h')
    sample_data['date'] = sample_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 重命名列
    sample_data = sample_data.rename(columns={
        'R1': 'sensor_0',
        'R2': 'sensor_1',
        'R3': 'sensor_2',
        'R4': 'sensor_3',
        'R5': 'sensor_4',
        'R6': 'sensor_5',
        'R7': 'sensor_6',
        'R8': 'sensor_7',
        'Temp.': 'temp',
        'Humidity': 'humidity'
    })

    # 添加标签列
    sample_data['label_gas'] = gas_mapping[gas_type]

    # 选择并排序所需的列
    columns = ['sensor_0', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7',
               'temp', 'humidity', 'date', 'label_gas']
    sample_data = sample_data[columns]

    # 保存为CSV文件
    if sample_id == 95:
        # 样本95存在缺失值，跳过
        continue
    output_file = f'{output_dir}/{sample_id}_{gas_type}.csv'
    sample_data.to_csv(output_file, index=False)

print("CSV文件生成完成。")
