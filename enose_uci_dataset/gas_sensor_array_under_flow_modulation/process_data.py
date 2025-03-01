# .csv.gz”中，其中每行代表每个传感器的一次测量。因此，需要读取特定的 16 条连续行才能从 16 个传感器获得一次测量。（Ziyatdinov 等人，2014）中提取的特征在第二个文件“features.csv”中提供，其中每行代表从所有 16 个传感器的时间序列中提取的特征（一次测量）。

# 每个样本的原始数据包含 16 个时间序列（每个传感器一个时间序列）。每个时间序列在 5 分钟内以 25 Hz（每秒样本数）的采样率记录，每个时间序列提供 7500 个数据点。原始数据中每个样本的属性总数为 120000。

# 特征数据集包括从每个时间序列中提取的三种类型的特征。每个时间序列（每个传感器一个时间序列）与 1 个最大特征、13 个高频特征和 13 个低频特征相关联（这些特征分别对应于前 13 个呼吸周期）。特征数据集中每个样本的属性总数为 432。

# 原始数据和特征的两个表都有共同的属性：
# 'exp'：整数（范围 100-181）；表示在实验室中注册的实验编号。
# 'batch'：字符串（5 个值）；表示测量的批次标识符；
# 'ace_conc'：浮点数（范围 0-1）；以体积百分比给出的丙酮分析物的浓度；
# 'eth_conc'：浮点数（范围 0-1）；乙醇分析物的体积百分比浓度；
# 'lab'：字符串（12 个值）；气体的类别标签；
# 'gas'：字符串（4 个值）；另一个类别标签，用于编码纯分析物、混合物或空气；
# 'col'：字符串（12 个值）；用于在类别标签之间更好地绘制的颜色代码。

# 原始数据表具有特定属性：
# 'sensor'：整数（范围 1-16）；传感器编号；
# 'sample'：整数（范围 1-58）；样本编号；
# 'dR_t<m>'：浮点数；表示在时刻 <m> 测量的给定传感器和给定样本的时间序列值，其中 <m> 取值范围为 1 到 7500。

# 特征表具有特定属性：
# 'S<j>_max'：浮点数；表示从传感器<j>的时间序列中提取的最大特征值；
# 'S<j>_r<k>_Alf': 浮点数；表示从传感器<j>的时间序列中提取的呼吸次数<k>时的低频特征，其中<j>的取值范围是1到16，<k>的取值范围是1到13；
# 'S<j>_r<k>_Ahf': 浮点数；表示从传感器<j>的时间序列中提取的呼吸次数<k>时的高频特征，其中<j>的取值范围是1到16，<k>的取值范围是1到13。


import pandas as pd
import numpy as np
import os

def read_csv(file):
    return pd.read_csv(file)

def raw_main():
    rawdata = read_csv('raw/rawdata.csv')
    # Reshape the data to store sensor readings in columns
    
    reshaped_data = []
    for i in range(0, len(rawdata), 16):
        sample = rawdata.iloc[i:i+16]
        reshaped_row = {
            'exp': sample['exp'].iloc[0],
            'batch': sample['batch'].iloc[0],
            'ace_conc': sample['ace_conc'].iloc[0],
            'eth_conc': sample['eth_conc'].iloc[0],
            'lab': sample['lab'].iloc[0],
            'gas': sample['gas'].iloc[0],
            'col': sample['col'].iloc[0],
            'sample': sample['sample'].iloc[0]
        }
        for sensor in range(1, 17):
            sensor_data = sample[sample['sensor'] == sensor].filter(regex='^dR_t').values.flatten()
            reshaped_row.update({f'sensor_{sensor}': sensor_data})
        reshaped_data.append(reshaped_row)
    
    # Arrange the reshaped data into multiple CSV files
    os.makedirs('./ssl_csv_samples', exist_ok=True)
    for i, row in enumerate(reshaped_data):
        filename = f"./ssl_csv_samples/{row['gas']}_{row['ace_conc']}_{row['eth_conc']}_{row['batch']}_{row['sample']}.csv"
        sample_df = pd.DataFrame({f'sensor_{j+1}': row[f'sensor_{j+1}'] for j in range(16)})
        sample_df['label_gas'] = row['gas']
        sample_df['label_ace_conc'] = row['ace_conc']
        sample_df['label_eth_conc'] = row['eth_conc']
        sample_df['label_batch'] = row['batch']
        sample_df.to_csv(filename, index=False)

if __name__ == '__main__':
    raw_main()
    