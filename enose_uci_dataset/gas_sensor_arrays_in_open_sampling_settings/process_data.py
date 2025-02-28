import os
import pandas as pd
# from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData
import re

from tqdm import tqdm
from loguru import logger

import concurrent.futures
import threading

logger.add("process_data_sqlite.log", rotation="2 MB")

# 创建SQLite数据库连接
# engine = create_engine('postgresql://odor:Oo7355608@10.133.185.5:5432/mydatabase')
# engine = create_engine('mysql+pymysql://odor:Oo7355608@localhost:33061/open_sample')
# metadata = MetaData()
# fail = MetaData()

# # 创建元数据表
# metadata_table = Table('0_metadata', metadata,
#                        Column('table_name', String(255), primary_key=True),
#                        Column('gas', String(255)),
#                        Column('concentration', Integer),
#                        Column('time', String(255)),
#                        Column('voltage', Float),
#                        Column('fan_set_point', Integer),
#                        Column('mfc_set_point', String(255)),
#                        Column('board_position', Integer)
#                        )

# fail_table = Table('1_fail_files', fail,
#                    Column('id', Integer, autoincrement=True, primary_key=True),
#                    Column('file_path', String(255)),
#                    Column('reason', String(255)),
#                    )

# metadata.create_all(engine)
# fail.create_all(engine)


# 创建一个线程安全的进度条
class ThreadSafeTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def update(self, n=1):
        with self.lock:
            super().update(n)


pbar = ThreadSafeTqdm(total=18151)

# 定义列名
columns = ['reading_time', 'fan_set_point', 'fan_reading', 'mcf1_setpoint', 'mcf2_setpoint', 'mcf3_setpoint',
           'mcf1_read', 'mcf2_read', 'mcf3_read', 'T', 'RH'] + \
          [f'board{i + 1}_sensor{j}' for i in range(9) for j in range(8)]

# 遍历所有文件夹和文件
root_dir = './raw/WTD_upload'
output_dir = './ssl_csv_samples'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


# fail_files = pd.DataFrame(columns=['file_path', 'reason'])

def log_fail(file_path, reason):
    # with engine.connect() as conn:
    #     conn.execute(fail_table.insert().values(
    #         file_path=file_path,
    #         reason=reason
    #     ))
    #     conn.commit()
    ...


@logger.catch
def process_file(file_path, location_folder):
    try:
        file_name = os.path.basename(file_path)
        match = re.match(
            r'(\d+)_board_setPoint_(\d+)V_fan_setPoint_(\d+)_mfc_setPoint_(\w+)_(\d+)ppm_p(\d+)',
            file_name)
        if match:
            time, voltage, fan_set_point, gas, concentration, board_position = match.groups()
            # pbar.set_description(f"Processing {file_path}")

            # 读取数据文件
            df = pd.read_csv(file_path, sep=r'\s+', header=None)

            # if df.shape[1] != 92:
            #     logger.error(f'Column num mismatch, ({df.shape[1]} != 92): ' + file_name)
            #     # fail_files.append({'file_path': file_path, 'reason': 'Column num mismatch'})
            #     log_fail(file_path, 'Column num mismatch')
            #     # continue
            #     return

            if df.shape[0] < 2:
                logger.warning(f'Rows too small, skip... ({df.shape[0]} < 2): ' + file_name)
                # fail_files.append({'file_path': file_path, 'reason': 'Rows too small'})
                log_fail(file_path, 'Rows too small')
                # continue
                return

            # for i in range(9):
            #     col_index = 11 + i * 9
            #     df = df.drop(df.columns[col_index], axis=1)
            #     is_all_ones = df['column_name'].eq(1).all()

            df = df.loc[:, (df != 1).any()]
            if df.shape[1] != 83:
                logger.error(f'Column num mismatch, ({df.shape[1]} != 83): ' + file_name)
                # fail_files.append({'file_path': file_path, 'reason': 'Column num mismatch'})
                log_fail(file_path, 'Column num mismatch')
                # continue
                return

            df.columns = columns

            # 创建表名
            table_name = f"{location_folder}_{gas}_{concentration}_{voltage}_{fan_set_point}_{time}"

            # 获取目录路径
            dir_path = os.path.join(output_dir, location_folder, gas)

            # 如果目录不存在，创建它
            # if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

            # 将数据插入到SQLite表中
            # df.to_sql(table_name, engine, if_exists='replace', index=False)
            df.to_parquet(os.path.join(output_dir, location_folder, gas, table_name) + '.parquet', index=False)

            # 插入元数据
            # with engine.connect() as conn:
            #     conn.execute(metadata_table.insert().values(
            #         table_name=table_name,
            #         gas=gas,
            #         concentration=int(concentration),
            #         time=time,
            #         voltage=float(voltage) / 100,
            #         fan_set_point=int(fan_set_point),
            #         mfc_set_point=f"{gas}_{concentration}ppm",
            #         board_position=(int(board_position) % 2) + 1
            #     ))
            #     conn.commit()

            # print(f"Processed: {file_path}")
            # logger.info(f"Processed: {file_path}")
            pbar.update(1)
            # exit()
        else:
            logger.warning('Not match re: ' + file_name)

    except Exception as e:
        logger.exception(e)
        # info = {'file_path': file_path, 'reason': str(e)}
        # fail_files.add
        log_fail(file_path, str(e))

        # fail_files.to_csv('fail_files.csv', index=False)
        # print("Data processing and insertion completed.")


# 主函数
def main():
    root_dir = './raw'
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for gas_folder in os.listdir(root_dir):
            gas_path = os.path.join(root_dir, gas_folder)
            if os.path.isdir(gas_path):
                for location_folder in os.listdir(gas_path):
                    location_path = os.path.join(gas_path, location_folder)
                    if os.path.isdir(location_path):
                        for file_name in os.listdir(location_path):
                            file_path = os.path.join(location_path, file_name)
                            if os.path.isfile(file_path):
                                futures.append(executor.submit(process_file, file_path, location_folder))

        concurrent.futures.wait(futures)

    print("Data processing and insertion completed.")


if __name__ == '__main__':
    main()
