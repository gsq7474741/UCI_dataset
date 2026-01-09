"""
UnifiedEnoseDataset 可视化测试

测试内容:
1. 逐个加载单数据集分类任务，全局标签，全局传感器空间，画出图表
2. 加载组合数据集，画出图表
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from enose_uci_dataset.datasets import (
    UnifiedEnoseDataset,
    TaskType,
    SplitType,
    NormalizeType,
    ChannelAlignMode,
    SENSOR_MODELS,
    get_global_channel_mapping,
    GasLabel,
    TASK_DATASET_COMPATIBILITY,
)


# =============================================================================
# 配置
# =============================================================================

# 分类任务数据集
CLASSIFICATION_DATASETS = [
    "twin_gas_sensor_arrays",
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures",
]

# 气味分类数据集 (需要 local_path)
ODOR_DATASETS = {
    "g919_55": "/root/UCI_dataset/.cache/G919-55",
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_dataset_overview(ds: UnifiedEnoseDataset, name: str, save_path: Path):
    """绘制单个数据集的概览图表"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Dataset: {name}", fontsize=14, fontweight='bold')
    
    # 1. 样本数量分布 (按标签)
    ax = axes[0, 0]
    labels = [ds._sample_labels[i] for i in range(min(len(ds), 1000))]
    label_counts = Counter(labels)
    
    if label_counts:
        sorted_items = sorted(label_counts.items())
        x_labels = [str(k) for k, v in sorted_items]
        counts = [v for k, v in sorted_items]
        
        bars = ax.bar(range(len(counts)), counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title(f'Label Distribution (n={len(ds)})')
        
        if len(x_labels) <= 10:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
        else:
            ax.set_xticks([0, len(x_labels)//2, len(x_labels)-1])
            ax.set_xticklabels([x_labels[0], x_labels[len(x_labels)//2], x_labels[-1]])
    
    # 2. 时序数据示例 (前3个样本)
    ax = axes[0, 1]
    max_display_points = 2000  # 最多显示的点数
    for i in range(min(3, len(ds))):
        data, label = ds[i]
        if data.ndim == 2:
            # 找第一个有数据的通道 (非全零)
            ch_idx = 0
            for c in range(data.shape[0]):
                if np.abs(data[c]).max() > 1e-6:
                    ch_idx = c
                    break
            signal = data[ch_idx]
        else:
            signal = data
        # 降采样以显示完整时序
        if len(signal) > max_display_points:
            step = len(signal) // max_display_points
            signal = signal[::step]
        ax.plot(signal, alpha=0.7, label=f'Sample {i} (label={label})', linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Sample Time Series (first active channel)')
    ax.legend(fontsize=8)
    
    # 3. 通道热力图 (单个样本)
    ax = axes[0, 2]
    if len(ds) > 0:
        data, _ = ds[0]
        if data.ndim == 2:
            # 截断显示
            display_data = data[:, :min(200, data.shape[1])]
            im = ax.imshow(display_data, aspect='auto', cmap='viridis')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Channel')
            ax.set_title(f'Channel Heatmap (Sample 0, shape={data.shape})')
            plt.colorbar(im, ax=ax)
    
    # 4. 全局传感器空间使用情况 (6x6格子阵)
    ax = axes[1, 0]
    if ds.channel_align == ChannelAlignMode.GLOBAL:
        M = len(SENSOR_MODELS)
        grid_size = 6  # 6x6 grid
        usage_grid = np.zeros((grid_size, grid_size))
        used_sensors = set()
        
        for ds_name in ds.dataset_names:
            mapping = get_global_channel_mapping(ds_name)
            for gid in mapping:
                if gid < M:
                    used_sensors.add(gid)
        
        # 填充格子阵
        for gid in range(min(M, grid_size * grid_size)):
            row, col = gid // grid_size, gid % grid_size
            usage_grid[row, col] = 1 if gid in used_sensors else 0
        
        # 绘制格子阵
        im = ax.imshow(usage_grid, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        
        # 在每个格子中显示传感器ID
        sensor_names = [s.name for s in SENSOR_MODELS]
        for i in range(grid_size):
            for j in range(grid_size):
                gid = i * grid_size + j
                if gid < M:
                    short_name = sensor_names[gid][:6]
                    color = 'white' if gid in used_sensors else 'gray'
                    ax.text(j, i, f'{gid}\n{short_name}', ha='center', va='center', 
                           fontsize=6, color=color, fontweight='bold' if gid in used_sensors else 'normal')
        
        ax.set_title(f'Global Sensor Space ({len(used_sensors)}/{M} used)')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        # 显示本地通道信息
        for ds_name, sub_ds in ds._datasets.items():
            channels = sub_ds.channel_models
            ax.barh(range(len(channels)), [1]*len(channels), alpha=0.7)
            ax.set_yticks(range(len(channels)))
            ax.set_yticklabels(channels, fontsize=8)
            ax.set_title(f'Local Channels ({len(channels)})')
            break
    
    # 5. 数据统计
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = [
        f"Dataset: {name}",
        f"Task: {ds.task.value}",
        f"Split: {ds.split.value}",
        f"Total Samples: {len(ds)}",
        f"Num Classes: {ds.num_classes}",
        f"Num Channels: {ds.num_channels}",
        f"Normalize: {ds.normalize.value}",
        f"Channel Align: {ds.channel_align.value}",
        "",
        "Sub-datasets:",
    ]
    for ds_name, sub_ds in ds._datasets.items():
        stats_text.append(f"  - {ds_name}: {len(sub_ds)} samples")
    
    ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. 数据值分布 (只统计非零值，因为全局对齐会有很多padding的0)
    ax = axes[1, 2]
    all_values = []
    for i in range(min(50, len(ds))):
        data, _ = ds[i]
        non_zero = data[data != 0].flatten()[:500]
        all_values.extend(non_zero.tolist())
    
    if all_values:
        ax.hist(all_values, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Value Distribution (mean={np.mean(all_values):.2f}, std={np.std(all_values):.2f})')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_combined_overview(ds: UnifiedEnoseDataset, name: str, save_path: Path):
    """绘制组合数据集的概览图表"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(f"Combined Dataset: {name}", fontsize=14, fontweight='bold')
    
    # 1. 各数据集样本占比
    ax = axes[0, 0]
    ds_counts = Counter(s[0] for s in ds._samples)
    labels = list(ds_counts.keys())
    sizes = list(ds_counts.values())
    
    # 简化标签名
    short_labels = [l.replace('gas_sensor_', '').replace('_', '\n')[:20] for l in labels]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=short_labels, autopct='%1.1f%%',
                                       colors=plt.cm.Set3.colors[:len(labels)])
    ax.set_title(f'Dataset Composition ({len(ds)} total)')
    
    # 2. 标签分布 (全局标签)
    ax = axes[0, 1]
    labels = [ds._sample_labels[i] for i in range(len(ds))]
    label_counts = Counter(l for l in labels if l is not None)
    
    if label_counts:
        sorted_items = sorted(label_counts.items())[:20]  # 最多显示20个
        x_vals = [k for k, v in sorted_items]
        y_vals = [v for k, v in sorted_items]
        
        ax.bar(range(len(y_vals)), y_vals, color='steelblue', alpha=0.7)
        ax.set_xlabel('Global Label ID')
        ax.set_ylabel('Count')
        ax.set_title(f'Global Label Distribution (top 20)')
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(x) for x in x_vals], rotation=45)
    
    # 3. 时序对比 (每个数据集一个样本)
    ax = axes[0, 2]
    colors = plt.cm.tab10.colors
    for i, (ds_name, sub_ds) in enumerate(ds._datasets.items()):
        if len(sub_ds) > 0:
            # 获取第一个样本
            data, meta = sub_ds.get_normalized_sample(0)
            # 找第一个有数据的通道
            ch_idx = 0
            for c in range(data.shape[0]):
                if np.abs(data[c]).max() > 1e-6:
                    ch_idx = c
                    break
            signal = data[ch_idx, :300] if data.shape[1] > 300 else data[ch_idx]
            short_name = ds_name.replace('gas_sensor_', '')[:15]
            ax.plot(signal, alpha=0.7, color=colors[i % len(colors)], label=short_name)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Sample Comparison (first 300 steps)')
    ax.legend(fontsize=7, loc='upper right')
    
    # 4. 全局传感器空间覆盖
    ax = axes[1, 0]
    M = len(SENSOR_MODELS)
    coverage_matrix = np.zeros((len(ds._datasets), M))
    
    for i, ds_name in enumerate(ds._datasets.keys()):
        mapping = get_global_channel_mapping(ds_name)
        for gid in mapping:
            if gid < M:
                coverage_matrix[i, gid] = 1
    
    im = ax.imshow(coverage_matrix, aspect='auto', cmap='Blues')
    ax.set_xlabel('Global Sensor ID')
    ax.set_ylabel('Dataset')
    ax.set_yticks(range(len(ds._datasets)))
    short_names = [n.replace('gas_sensor_', '')[:12] for n in ds._datasets.keys()]
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(f'Global Sensor Coverage Matrix')
    plt.colorbar(im, ax=ax)
    
    # 5. 每个数据集的采样率
    ax = axes[1, 1]
    from enose_uci_dataset.datasets import DATASET_SAMPLE_RATES
    
    ds_names = list(ds._datasets.keys())
    rates = [DATASET_SAMPLE_RATES.get(n, 0) or 0 for n in ds_names]
    short_names = [n.replace('gas_sensor_', '')[:12] for n in ds_names]
    
    bars = ax.barh(range(len(ds_names)), rates, color='teal', alpha=0.7)
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Sample Rate (Hz)')
    ax.set_title('Dataset Sample Rates')
    
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}', va='center', fontsize=8)
    
    # 6. 统计信息
    ax = axes[1, 2]
    ax.axis('off')
    
    stats = [
        f"Combined Dataset Overview",
        f"=" * 30,
        f"Task: {ds.task.value}",
        f"Split: {ds.split.value}",
        f"Total Samples: {len(ds)}",
        f"Num Datasets: {len(ds._datasets)}",
        f"Num Classes: {ds.num_classes}",
        f"Channel Align: {ds.channel_align.value}",
        f"Global Channels: {ds.num_channels}",
        "",
        "Per-dataset breakdown:",
    ]
    for ds_name, sub_ds in ds._datasets.items():
        short = ds_name.replace('gas_sensor_', '')[:25]
        stats.append(f"  {short}: {len(sub_ds)} samples, {sub_ds.num_sensors} ch")
    
    ax.text(0.05, 0.95, '\n'.join(stats), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_all_datasets_overview(save_path: Path):
    """绘制所有数据集的元数据概览图"""
    from enose_uci_dataset.datasets import (
        list_datasets, get_dataset_info, DATASET_SAMPLE_RATES,
        DATASET_CHANNEL_TO_GLOBAL, TASK_DATASET_COMPATIBILITY
    )
    
    # 数据集简称映射
    SHORT_NAMES = {
        "g919_55": "G919",
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "Turbulent",
        "gas_sensor_array_low_concentration": "Low Conc",
        "gas_sensor_array_temperature_modulation": "Temp Mod",
        "gas_sensor_array_under_dynamic_gas_mixtures": "Dyn Mix",
        "gas_sensor_array_under_flow_modulation": "Flow Mod",
        "gas_sensors_for_home_activity_monitoring": "Home Act",
        "twin_gas_sensor_arrays": "Twin",
        "smellnet_pure": "SmellNet P",
        "smellnet_mixture": "SmellNet M",
    }
    
    # 平均样本长度 (时间点数)
    SAMPLE_LENGTHS = {
        "g919_55": 90,
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures": 2970,  # ~59s @ 50Hz
        "gas_sensor_array_low_concentration": 900,
        "gas_sensor_array_temperature_modulation": 13000,  # varies
        "gas_sensor_array_under_dynamic_gas_mixtures": 10000,  # varies
        "gas_sensor_array_under_flow_modulation": 7200,  # varies
        "gas_sensors_for_home_activity_monitoring": 3600,  # 1 hour segments
        "twin_gas_sensor_arrays": 53000,  # ~530s @ 100Hz
        "smellnet_pure": 200,
        "smellnet_mixture": 200,
    }
    
    def get_short_name(ds_name):
        return SHORT_NAMES.get(ds_name, ds_name[:10])
    
    # 获取所有数据集信息
    all_datasets = list_datasets()
    n_datasets = len(all_datasets)
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("E-Nose UCI Dataset Collection Overview", fontsize=16, fontweight='bold')
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. 数据集基本信息表格 (包含样本长度和物理时长)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # 构建表格数据
    table_data = []
    headers = ['Dataset', 'Type', 'Rate(Hz)', 'Ch', 'Avg Len', 'Duration']
    
    for ds_name in all_datasets:
        try:
            info = get_dataset_info(ds_name)
            sample_rate = DATASET_SAMPLE_RATES.get(ds_name, 0) or 1
            n_channels = len(DATASET_CHANNEL_TO_GLOBAL.get(ds_name, []))
            sensor_type = info.sensors.type if info.sensors else 'MOX'
            
            # 获取平均样本长度
            avg_len = SAMPLE_LENGTHS.get(ds_name, None)
            if avg_len:
                duration_s = avg_len / sample_rate
                if duration_s >= 60:
                    duration_str = f"{duration_s/60:.1f}min"
                else:
                    duration_str = f"{duration_s:.1f}s"
            else:
                duration_str = '-'
            
            table_data.append([
                get_short_name(ds_name),
                sensor_type,
                str(sample_rate),
                str(n_channels),
                str(avg_len) if avg_len else '-',
                duration_str
            ])
        except Exception as e:
            table_data.append([get_short_name(ds_name), "-", "-", "-", "-", "-"])
    
    table = ax1.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['lightsteelblue'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax1.set_title('Dataset Metadata Summary', fontsize=12, fontweight='bold', pad=10)
    
    # 2. 全局传感器覆盖矩阵 (显示所有传感器名)
    ax3 = fig.add_subplot(gs[1, :])
    M = len(SENSOR_MODELS)
    coverage = np.zeros((n_datasets, M))
    
    for i, ds_name in enumerate(all_datasets):
        mapping = DATASET_CHANNEL_TO_GLOBAL.get(ds_name, [])
        for gid in mapping:
            if gid < M:
                coverage[i, gid] = 1
    
    im = ax3.imshow(coverage, aspect='auto', cmap='YlOrRd')
    ax3.set_xlabel('Sensor Model', fontsize=10)
    ax3.set_ylabel('Dataset')
    ax3.set_yticks(range(n_datasets))
    ax3.set_yticklabels([get_short_name(ds) for ds in all_datasets], fontsize=9)
    ax3.set_title(f'Global Sensor Coverage Matrix ({M} sensors)', fontsize=11)
    
    # 显示所有传感器名称
    sensor_names = [s.name for s in SENSOR_MODELS]
    ax3.set_xticks(range(M))
    ax3.set_xticklabels(sensor_names, rotation=90, ha='center', fontsize=7)
    plt.colorbar(im, ax=ax3, label='Used', shrink=0.5)
    
    # 3. 任务兼容性矩阵
    ax4 = fig.add_subplot(gs[2, 0])
    task_names = [t.value for t in TaskType if t in TASK_DATASET_COMPATIBILITY]
    task_matrix = np.zeros((len(task_names), n_datasets))
    
    for i, task in enumerate(TaskType):
        if task in TASK_DATASET_COMPATIBILITY:
            compatible = TASK_DATASET_COMPATIBILITY[task]
            if compatible:
                for j, ds_name in enumerate(all_datasets):
                    if ds_name in compatible:
                        task_matrix[i, j] = 1
            else:
                task_matrix[i, :] = 1
    
    im4 = ax4.imshow(task_matrix, aspect='auto', cmap='Greens')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Task')
    ax4.set_xticks(range(n_datasets))
    ax4.set_xticklabels([get_short_name(ds) for ds in all_datasets], rotation=45, ha='right', fontsize=7)
    ax4.set_yticks(range(len(task_names)))
    ax4.set_yticklabels(task_names, fontsize=8)
    ax4.set_title('Task Compatibility', fontsize=11)
    
    # 4. 采样率 & 通道数对比
    ax5 = fig.add_subplot(gs[2, 1])
    rates = [DATASET_SAMPLE_RATES.get(ds, 0) or 0 for ds in all_datasets]
    channel_counts = [len(DATASET_CHANNEL_TO_GLOBAL.get(ds, [])) for ds in all_datasets]
    
    x = np.arange(n_datasets)
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, rates, width, label='Sample Rate (Hz)', color='steelblue', alpha=0.8)
    ax5.set_ylabel('Sample Rate (Hz)', color='steelblue')
    ax5.tick_params(axis='y', labelcolor='steelblue')
    
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, channel_counts, width, label='Channels', color='coral', alpha=0.8)
    ax5_twin.set_ylabel('Channels', color='coral')
    ax5_twin.tick_params(axis='y', labelcolor='coral')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels([get_short_name(ds) for ds in all_datasets], rotation=45, ha='right', fontsize=7)
    ax5.set_title('Sample Rate & Channels', fontsize=11)
    ax5.legend(loc='upper left', fontsize=7)
    ax5_twin.legend(loc='upper right', fontsize=7)
    
    # 5. 统计摘要
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    total_sensors_used = len(set(gid for ds in all_datasets 
                                  for gid in DATASET_CHANNEL_TO_GLOBAL.get(ds, [])))
    
    summary = [
        "COLLECTION STATISTICS",
        "=" * 30,
        f"Total Datasets: {n_datasets}",
        f"Global Sensors: {total_sensors_used}/{M}",
        "",
        "Sample Rate:",
        f"  {min(r for r in rates if r > 0):.0f} - {max(rates):.0f} Hz" if any(r > 0 for r in rates) else "  N/A",
        "",
        "Channels:",
        f"  {min(channel_counts)} - {max(channel_counts)}",
        "",
        "Sensor Types:",
        "  MOX, MEMS, ENV",
    ]
    
    ax6.text(0.1, 0.9, '\n'.join(summary), transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# =============================================================================
# 主测试函数
# =============================================================================

def test_single_datasets():
    """测试1: 逐个加载单数据集"""
    print("=" * 60)
    print("测试1: 逐个加载单数据集 (分类任务, 全局标签, 全局传感器)")
    print("=" * 60)
    
    for ds_name in CLASSIFICATION_DATASETS:
        print(f"\n>>> 加载 {ds_name}")
        try:
            ds = UnifiedEnoseDataset(
                root=".cache",
                datasets=ds_name,
                task=TaskType.GAS_CLASSIFICATION,
                split=SplitType.ALL,
                use_global_labels=True,
                channel_align=ChannelAlignMode.GLOBAL,
                normalize=NormalizeType.ZSCORE,
                target_sample_rate=1.0,  # 重采样到1Hz
            )
            
            print(f"    样本数: {len(ds)}")
            print(f"    类别数: {ds.num_classes}")
            print(f"    通道数: {ds.num_channels}")
            
            # 测试获取样本
            data, label = ds[0]
            print(f"    数据形状: {data.shape}")
            print(f"    标签示例: {label}")
            
            # 画图
            save_path = OUTPUT_DIR / f"single_{ds_name.replace('/', '_')}.png"
            plot_dataset_overview(ds, ds_name, save_path)
            
        except Exception as e:
            print(f"    ❌ 加载失败: {e}")
    
    # 测试 G919 气味分类
    print(f"\n>>> 加载 g919_55 (气味分类)")
    try:
        ds = UnifiedEnoseDataset(
            root=".cache",
            datasets="g919_55",
            task=TaskType.ODOR_CLASSIFICATION,
            split=SplitType.ALL,
            channel_align=ChannelAlignMode.GLOBAL,
            normalize=NormalizeType.ZSCORE,
            local_paths=ODOR_DATASETS,
            target_sample_rate=1.0,  # 重采样到1Hz
        )
        
        print(f"    样本数: {len(ds)}")
        print(f"    类别数: {ds.num_classes}")
        
        data, label = ds[0]
        print(f"    数据形状: {data.shape}")
        print(f"    标签示例: {label}")
        
        save_path = OUTPUT_DIR / "single_g919_55.png"
        plot_dataset_overview(ds, "g919_55", save_path)
        
    except Exception as e:
        print(f"    ❌ 加载失败: {e}")


def test_combined_datasets():
    """测试2: 加载组合数据集"""
    print("\n" + "=" * 60)
    print("测试2: 加载组合数据集")
    print("=" * 60)
    
    # 组合1: 所有气体分类数据集
    print("\n>>> 组合数据集: 气体分类任务")
    try:
        ds = UnifiedEnoseDataset(
            root=".cache",
            datasets=CLASSIFICATION_DATASETS,
            task=TaskType.GAS_CLASSIFICATION,
            split=SplitType.TRAIN,
            use_global_labels=True,
            channel_align=ChannelAlignMode.GLOBAL,
            normalize=NormalizeType.ZSCORE,
        )
        
        print(f"    样本数: {len(ds)}")
        print(f"    数据集: {ds.dataset_names}")
        print(f"    类别数: {ds.num_classes}")
        print(f"    通道数: {ds.num_channels}")
        
        data, label = ds[0]
        print(f"    数据形状: {data.shape}")
        
        save_path = OUTPUT_DIR / "combined_gas_classification.png"
        plot_combined_overview(ds, "Gas Classification (3 datasets)", save_path)
        
    except Exception as e:
        print(f"    ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 组合2: 自监督预训练 (所有数据集)
    print("\n>>> 组合数据集: 自监督预训练")
    try:
        all_datasets = CLASSIFICATION_DATASETS + ["g919_55"]
        
        ds = UnifiedEnoseDataset(
            root=".cache",
            datasets=all_datasets,
            task=TaskType.SELF_SUPERVISED,
            split=SplitType.ALL,
            channel_align=ChannelAlignMode.GLOBAL,
            normalize=NormalizeType.ZSCORE,
            local_paths=ODOR_DATASETS,
        )
        
        print(f"    样本数: {len(ds)}")
        print(f"    数据集: {ds.dataset_names}")
        print(f"    通道数: {ds.num_channels}")
        
        data, label = ds[0]
        print(f"    数据形状: {data.shape}")
        print(f"    标签: {label} (自监督无标签)")
        
        save_path = OUTPUT_DIR / "combined_self_supervised.png"
        plot_combined_overview(ds, "Self-Supervised Pretraining (4 datasets)", save_path)
        
    except Exception as e:
        print(f"    ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("UnifiedEnoseDataset 可视化测试")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 测试0: 数据集集合概览
    print("=" * 60)
    print("测试0: 生成数据集集合概览图")
    print("=" * 60)
    try:
        save_path = OUTPUT_DIR / "dataset_collection_overview.png"
        plot_all_datasets_overview(save_path)
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试1: 单数据集
    test_single_datasets()
    
    # 测试2: 组合数据集
    test_combined_datasets()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"图表已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
