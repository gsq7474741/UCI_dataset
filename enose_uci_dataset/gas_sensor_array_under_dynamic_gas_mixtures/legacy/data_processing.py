import numpy as np
import pandas as pd
from pathlib import Path


def load_data(file_path):
    # 读取数据文件
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过表头行
    data = [line.strip().split() for line in lines[1:]]
    
    # 将数据转换为numpy数组
    data = np.array(data, dtype=np.float32)
    
    # 分离特征和标签
    time = data[:, 0]
    gas1_conc = data[:, 1]  # 甲烷或CO浓度
    gas2_conc = data[:, 2]  # 乙烯浓度
    sensor_readings = data[:, 3:]  # 16个传感器的读数
    
    return time, gas1_conc, gas2_conc, sensor_readings


def process_data(file_path, output_dir):
    # 加载数据
    time, gas1_conc, gas2_conc, sensor_readings = load_data(file_path)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存处理后的数据
    np.save(output_dir / 'time.npy', time)
    np.save(output_dir / 'gas1_conc.npy', gas1_conc)
    np.save(output_dir / 'gas2_conc.npy', gas2_conc)
    np.save(output_dir / 'sensor_readings.npy', sensor_readings)


if __name__ == '__main__':
    # 处理ethylene_methane.txt
    process_data(
        file_path='./ethylene_methane.txt',
        output_dir='./processed/ethylene_methane'
    )
    
    # 处理ethylene_CO.txt
    process_data(
        file_path='./ethylene_CO.txt',
        output_dir='./processed/ethylene_CO'
    )
