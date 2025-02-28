import os
import h5py
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime
import threading

# 创建一个线程安全的进度条
class ThreadSafeTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def update(self, n=1):
        with self.lock:
            super().update(n)



class CSVtoHDF5Converter:
    def __init__(self, root_dir, output_file):
        self.root_dir = root_dir
        self.output_file = output_file
        self.file_count = 0

    def parse_path(self, path):
        parts = path.split(os.sep)
        gas_part = parts[-3].rsplit('_', 1)
        filename_parts = parts[-1].split('_')
        
        # 使用正则表达式提取详细信息
        pattern = r'(\d{12})_board_setPoint_(\d+)V_fan_setPoint_(\d+)_mfc_setPoint_([\w_]+ppm)_(p\d+)'
        match = re.match(pattern, parts[-1])
        
        if match:
            timestamp, board_voltage, fan_speed, mfc_setting, point_num = match.groups()
        else:
            timestamp = board_voltage = fan_speed = mfc_setting = point_num = 'unknown'
        
        return {
            'gas': gas_part[0],
            'concentration': int(gas_part[1]),
            'level': parts[-2],
            'filename': parts[-1],
            'timestamp': timestamp,
            'board_voltage': int(board_voltage),
            'fan_speed': int(fan_speed),
            'mfc_setting': mfc_setting,
            'point_num': point_num
        }

    def parse_metadata(self, filepath):
        """
        从文件路径中解析元数据
        """
        # 解析路径
        path_parts = filepath.split('/')
        gas_info = path_parts[-3].split('_')
        location = path_parts[-2]
        filename = os.path.basename(filepath).split('.')

        # 解析气体信息
        gas_name = gas_info[0]
        gas_concentration = int(gas_info[1])

        # 解析文件名
        filename_parts = filename[0].split('_')
        timestamp = datetime.strptime(filename_parts[0], '%Y%m%d%H%M')
        board_setpoint = int(filename_parts[3].replace('V', ''))
        fan_setpoint = int(filename_parts[6])
        # mfc_setpoint = filename_parts[6]
        position = int(filename_parts[-1].replace('p', ''))

        return {
            'gas_name': gas_name,
            'gas_concentration': gas_concentration,
            'location': location,
            'timestamp': timestamp.isoformat(),
            'board_setpoint': board_setpoint,
            'fan_setpoint': fan_setpoint,
            # 'mfc_setpoint': mfc_setpoint,
            'position': position
        }

    def process_file(self, filepath, metadata):
        try:
            # 读取CSV并存储
            df = pd.read_csv(filepath, sep='\t')
            with h5py.File(self.output_file, 'a') as hdf:
                # 创建层次结构
                group_path = f"{metadata['gas_name']}/{metadata['gas_concentration']}/{metadata['location']}/"
                group = hdf.require_group(group_path)
                
                # 添加元数据属性
                group.attrs.update({
                    'gas_type': metadata['gas_name'],
                    'concentration_ppm': metadata['gas_concentration'],
                    'line_location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'board_voltage': metadata['board_setpoint'],
                    'fan_speed': metadata['fan_setpoint'],
                    # 'mfc_setting': metadata['mfc_setpoint'],
                    'point_location': metadata['position']
                })
                
                # 存储数据集
                dataset = group.create_dataset(
                    name=os.path.basename(filepath),
                    data=df.values,
                    compression="gzip"
                )
                
                # 添加列名属性
                dataset.attrs['columns'] = list(df.columns)
                self.file_count += 1
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")

    def process_directory(self):
        print(f'Processing directory: {self.root_dir}')
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                filepath = os.path.join(root, file)
                metadata = self.parse_metadata(filepath)
                self.process_file(filepath, metadata)
                pbar.update(1)
        print(f"成功转换 {self.file_count} 个文件")

if __name__ == "__main__":
    converter = CSVtoHDF5Converter(
        root_dir='raw',  # 当前目录
        output_file='gas_data.h5'
    )
    pbar = ThreadSafeTqdm(total=18151)
    converter.process_directory()
