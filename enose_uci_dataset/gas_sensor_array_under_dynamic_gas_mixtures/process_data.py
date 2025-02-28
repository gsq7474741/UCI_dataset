import os
from typing import Literal

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm


def convert_to_csv(input_file,):
    '''
    将原始数据转换为csv格式，原始数据很乱，不是\t分隔，也不是逗号分隔，而是空格分隔，有的地方有两个空格，有的地方有一个空格，有的地方有三个空格，
    行尾部有的是\n，有的是空格\n，所以需要先处理一下。并且重写表头列名

    :param input_file:
    :return:
    '''
    # 读取原始数据
    count = 0
    with open(input_file, 'r') as f:
        line = f.readline()
        with open(input_file + '.csv', 'w') as f_csv:
            head = ['time', 'gas1_conc', 'ethylene_conc'] + [f'sensor_{i}' for i in range(16)]
            f_csv.write(','.join(head) + '\n')
            while True:
                line = f.readline()  # 读取文件的一行
                if not line:  # 如果行为空，表示文件结束
                    break

                line = line.replace('  ', ' ').replace('  ', ' ').replace(' \n', '').replace('\n', '').split(' ')
                f_csv.write(','.join(line) + '\n')

                count += 1

                # if count == 2000:
                #     d=3
                #     break


def process_file(input_file, output_dir, type:Literal['co', 'methane']):
    # 读取原始数据
    with open(input_file, 'r') as f:
        line = f.readline()
        while line.startswith('Time'):
            line = f.readline()
        line.replace('  ', ' ').replace('  ', ' ').split(' ')

        d = 3
    df = pd.read_csv(input_file)

    # 重命名列
    df.columns = ['time', 'gas1_conc', 'ethylene_conc'] + [f'sensor_{i}' for i in range(16)]

    # 转换sensor读数为电阻值(kOhm)
    for i in range(16):
        df[f'sensor_{i}'] = 40.0 / df[f'sensor_{i}']

    # 确定气体类型
    if 'CO' in input_file:
        gas1_name = 'co'
    else:
        gas1_name = 'methane'

    # 添加第三种气体的浓度列(设为0)
    if gas1_name == 'co':
        df['methane_conc'] = 0.
        df = df.rename(columns={'gas1_conc': 'co_conc'})
    else:
        df['co_conc'] = 0.
        df = df.rename(columns={'gas1_conc': 'methane_conc'})

    df = df.drop('time', axis=1)

    # 重新排列列顺序
    columns = [f'sensor_{i}' for i in range(16)] + ['ethylene_conc', 'co_conc', 'methane_conc']
    df = df[columns]

    # 根据浓度变化分割样本
    sample_id = 0
    prev_conc = df.iloc[0][['ethylene_conc', 'co_conc', 'methane_conc']].values
    sample_start=0

    for i, row in tqdm(df.iterrows()):
        current_conc = row[['ethylene_conc', 'co_conc', 'methane_conc']].values
        if not np.array_equal(current_conc, prev_conc):
            # 保存当前样本
            sample_df = df.iloc[sample_start:i]
            sample_df.to_csv(os.path.join(output_dir, f'sample_{type}_{sample_id}.csv'), index=False)

            # 开始新样本
            sample_id += 1
            sample_start = i
            prev_conc = current_conc

    # 保存最后一个样本
    sample_df = df.iloc[sample_start:]
    sample_df.to_csv(os.path.join(output_dir, f'sample_{type}_{sample_id}.csv'), index=False)


# 创建输出目录
output_dir = './ssl_csv_samples'
os.makedirs(output_dir, exist_ok=True)

# 处理两个输入文件
convert_to_csv('.raw/ethylene_CO.txt', )
process_file('.raw/ethylene_CO.txt.csv', output_dir, type='co')

convert_to_csv('.raw/ethylene_methane.txt',)
process_file('.raw/ethylene_methane.txt.csv', output_dir, type='methane')

print("数据处理完成，CSV文件已生成在 ./ssl_csv_samples 目录下。")
