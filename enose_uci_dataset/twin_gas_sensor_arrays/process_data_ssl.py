import os
import pandas as pd
import re
from tqdm import tqdm

# 定义输入和输出目录
input_dir = './.raw/data1'
output_dir = './ssl_csv_samples'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义气体映射字典
gas_mapping = {
    'Ea': 0,
    'CO': 1,
    'Ey': 2,
    'Me': 3
}

ppm_factor_mapping = {
    'Ea': 1.25,
    'CO': 2.5,
    'Ey': 1.25,
    'Me': 2.5
}


# 处理每个文件
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.txt'):
        # 解析文件名
        match = re.match(r'B(\d+)_G(\w+)_F(\d+)_R(\d+)\.txt', filename)
        if match:
            board, gas, ppm, repeat = match.groups()
            gas_name = gas_mapping.get(gas, gas)
            ppm_value = int(ppm) * ppm_factor_mapping.get(gas, 1)  # 将010-100映射到实际ppm值

            # 读取数据
            df = pd.read_csv(os.path.join(input_dir, filename), sep='\s+', header=None)

            # 重命名列
            df.columns = ['time'] + [f'sensor_{i}' for i in range(8)]

            # 降采样到1Hz (每100行取一行)
            df = df.iloc[::10, :]

            # 添加标签列
            df['label_gas'] = gas_name
            df['label_ppm'] = ppm_value
            df['label_board'] = int(board)

            # 删除time列
            df = df.drop('time', axis=1)

            # 生成输出文件名
            output_filename = f'{gas}_{ppm_value}ppm_board{board}_repeat{repeat}.csv'

            # 保存为CSV
            df.to_csv(os.path.join(output_dir, output_filename), index=False)

print("数据处理完成,CSV文件已生成在./ssl_csv_samples目录下。")
