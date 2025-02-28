import os
import random
import csv

# 设置目录和输出文件名
source_dir = 'ssl_csv_samples'
output_file = 'dataset_split.csv'

# 设置训练集和验证集的比例 (这里设置为80%训练集, 20%验证集)
train_ratio = 0.8

# 获取所有csv文件
csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

# 随机打乱文件列表
random.shuffle(csv_files)

# 计算训练集的大小
train_size = int(len(csv_files) * train_ratio)

# 分割训练集和验证集
train_files = csv_files[:train_size]
val_files = csv_files[train_size:]

# 写入输出文件
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'split'])  # 写入表头

    # 写入训练集文件
    for file in train_files:
        writer.writerow([file, 'train'])

    # 写入验证集文件
    for file in val_files:
        writer.writerow([file, 'validation'])

print(f"Split completed. Results saved to {output_file}")
print(f"Training set: {len(train_files)} files")
print(f"Validation set: {len(val_files)} files")
