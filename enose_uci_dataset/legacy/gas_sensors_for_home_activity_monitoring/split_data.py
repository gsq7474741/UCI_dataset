import os
import json
import random
import pandas as pd
from collections import defaultdict


# 数据与输出路径
SOURCE_DIR = 'processed/v1/ssl_samples'
OUTPUT_JSON = os.path.join(SOURCE_DIR, 'splits.json')

# 固定随机种子保证可复现
SEED = 42
random.seed(SEED)

# 目标比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def read_label(filepath: str) -> int:
    """从 CSV 首行读取 label_gas 列，返回整数标签。"""
    df = pd.read_csv(filepath, nrows=1)
    if 'label_gas' not in df.columns:
        raise ValueError(f'label_gas not found in {filepath}')
    return int(df['label_gas'].iloc[0])


def stratified_split(files):
    """按 label_gas 分层，将文件名分成 train/val/test 三个集合。"""
    by_label = defaultdict(list)
    for f in files:
        full = os.path.join(SOURCE_DIR, f)
        label = read_label(full)
        by_label[label].append(f)

    train, val, test = [], [], []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # 将剩余分配给 test，避免四舍五入误差
        n_test = max(0, n - n_train - n_val)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val: n_train + n_val + n_test])

    # 为了整体随机性，打乱每个集合
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f'{SOURCE_DIR} not found. Please run process_data.py first.')

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]
    if not files:
        raise RuntimeError('No CSV files found. Please run process_data.py first.')

    train, val, test = stratified_split(files)

    result = {
        'seed': SEED,
        'ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO,
        },
        'train': train,
        'val': val,
        'test': test,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'Stratified split done. Saved to {OUTPUT_JSON}')
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)} (total {len(files)})')


if __name__ == '__main__':
    main()
