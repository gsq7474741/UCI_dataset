# Twin gas sensor arrays

## 数据集
### 数据集信息

该数据集包括 8 个传感器阵列的 5 个重复记录。

每个单元都装有 8 个 MOX 传感器，并集成了定制设计的电子设备，用于传感器工作温度控制和信号采集。遵循相同的实验方案来测量
5 个双单元的响应。每天测试一个不同的单元，其中包括以随机顺序呈现的 40 种不同的气体条件。

具体来说，受试单元暴露于 10 种浓度水平的乙醇、甲烷、乙烯和一氧化碳。

每次实验的持续时间为 600 秒，每个传感器的电导率以 100Hz 的频率获取。

通道、传感器类型（来自Figaro）和加热器中的平均电压如下： 
- 0：TGS2611 5.65 V 
- 1：TGS2612 5.65 V 
- 2：TGS2610 5.65 V 
- 3：TGS2602 5.65 V 
- 4：TGS2611 5.00 V
- 5：TGS2612 5.00 V 
- 6：TGS2610 5.00 V 
- 7：TGS2602 5.00 V 

显示的浓度水平如下（单位为 ppm）：
- 乙烯：12.5、25、37.5、50.0、62.5、75.0、87.5、100.0、112.5、125.0 
- 乙醇：12.5、25.0、37.5、50.0、 62.5、75.0、87.5、100.0、112.5、125.0
- 一氧化碳：25.0、50.0、75.0、100.0、125.0、150.0、175.0、200.0、225.0、250.0
- 甲烷：25.0、50.0、75.0、100.0、125.0、150.0、175.0、200.0、225.0、250.0 

测试每个检测平台的天数。
- 单元 1：4,10,15,21 
- 单元 2：1,7,11,16
- 单元 3：2,8,14,17 
- 单元 4：3,9 
- 单元 5：18,22 


### 变量信息

每个实验的传感器响应都以 .txt 文件的形式提供。

文件名格式如`B{i}_G{gas}_F{ppm}_R{n}.txt`，
其中`i`为板子编号、`gas`为气体（Ea：乙醇、CO：一氧化碳、Ey：乙烯、Me：甲烷）、`ppm`为浓度（010-100）、
`n`为重复次数，即统一样本的多次实验。

例如，`B1_GEa_F040_R2.txt` 表示 B1（板 1）、Ea（乙醇）、50 ppm、第2次实验。

每个文件都包含经过的时间（以秒为单位）和每个传感器的电阻（以千欧姆为单位）。
第一列是时间，后面 8 列是8个传感器`sensor_0 - sensor7`。

### 数据样例
`B1_GCO_F010_R1.txt`:
```text
0.00   	38.91  	18.64  	21.60  	5.56   	72.63  	45.67  	55.54  	7.00   
0.01   	38.91  	18.66  	21.60  	5.56   	72.85  	45.67  	55.40  	6.99   
...
```

`B5_GMe_F100_R2.txt`：
```text
0.00   	44.30  	26.25  	28.63  	21.75  	117.86 	64.63  	59.33  	42.86  
0.01   	44.30  	26.29  	28.68  	21.75  	117.86 	64.63  	59.33  	42.95  
0.02   	44.30  	26.29  	28.63  	21.78  	117.86 	64.63  	59.33  	42.95  
0.03   	44.30  	26.25  	28.63  	21.75  	117.86 	64.63  	59.33  	42.95  
0.04   	44.30  	26.29  	28.63  	21.78  	117.86 	64.63  	59.33  	42.95  
0.05   	44.30  	26.29  	28.68  	21.75  	117.86 	64.63  	59.33  	42.95
...  
```

## 数据处理

### 提取metadata

prompt:

```prompt
修改当前文件，增加以下8列
name：sensor0-7，共8列
description：sensor名字,
data_type：fp32
sensor：
| Sensor   | Target Gases                          |
|----------|---------------------------------------|
| TGS2611  | 甲烷                                 |
| TGS2612  | 甲烷、丙烷、丁烷                     |
| TGS2610  | 丙烷                                 |
| TGS2602  | 氨气、硫化氢、挥发性有机化合物（VOC）|
| TGS2611  | 甲烷                                 |
| TGS2612  | 甲烷、丙烷、丁烷                     |
| TGS2610  | 丙烷                                 |
| TGS2602  | 氨气、硫化氢、挥发性有机化合物（VOC）|
unit：KOhm
trans_to_ohm_functor：lambda x: x*1000
column_sample_rate:100
```

### 时间序列SSL预训练任务
元数据：气体、浓度、采样率、电阻单位、每个样本时长

对于每个样本，处理为单个csv文件：

每列为一个传感器的电阻值，最后几列为标签（气体种类、浓度、板子）。
例如：sensor_0,sensor_1,...,sensor_n,label_gas,label_ppm,label_board

prompt:
```text
请帮我根据以上信息处理数据集，生成csv文件。
当前目录结构如下：
.
├── note.md
└── .raw
    └── data1
        ├── B1_GCO_F010_R1.txt
        ├── B1_GCO_F010_R2.txt
        ...
        ├── B5_GMe_F100_R1.txt
        └── B5_GMe_F100_R2.txt
请帮我在./ssl_csv_samples下生成要求的文件，使用python
```

模型输出：
```text
好的,我可以帮你编写一个Python脚本来处理数据集并生成所需的CSV文件。以下是一个可能的实现:

```python
import os
import pandas as pd
import re

# 定义输入和输出目录
input_dir = './.raw/data1'
output_dir = './ssl_csv_samples'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义气体映射字典
gas_mapping = {
    'Ea': 'ethanol',
    'CO': 'carbon_monoxide',
    'Ey': 'ethylene',
    'Me': 'methane'
}

# 处理每个文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # 解析文件名
        match = re.match(r'B(\d+)_G(\w+)_F(\d+)_R(\d+)\.txt', filename)
        if match:
            board, gas, ppm, repeat = match.groups()
            gas_name = gas_mapping.get(gas, gas)
            ppm_value = int(ppm) * 2.5  # 将010-100映射到实际ppm值

            # 读取数据
            df = pd.read_csv(os.path.join(input_dir, filename), sep='\s+', header=None)
            
            # 重命名列
            df.columns = ['time'] + [f'sensor_{i}' for i in range(8)]
            
            # 添加标签列
            df['label_gas'] = gas_name
            df['label_ppm'] = ppm_value
            df['label_board'] = int(board)
            
            # 删除time列
            df = df.drop('time', axis=1)
            
            # 生成输出文件名
            output_filename = f'{gas_name}_{ppm_value}ppm_board{board}_repeat{repeat}.csv'
            
            # 保存为CSV
            df.to_csv(os.path.join(output_dir, output_filename), index=False)

print("数据处理完成,CSV文件已生成在./ssl_csv_samples目录下。")
```

这个脚本会执行以下操作:

1. 遍历`./.raw/data1`目录下的所有.txt文件。
2. 解析文件名以获取板子编号、气体类型、浓度和重复次数。
3. 读取每个.txt文件的内容。
4. 重命名列,添加标签列(气体类型、浓度、板子编号)。
5. 删除时间列,因为它在预训练任务中可能不需要。
6. 将处理后的数据保存为CSV文件在`./ssl_csv_samples`目录下。

生成的CSV文件名格式为`{气体名称}_{浓度}ppm_board{板子编号}_repeat{重复次数}.csv`。

要运行这个脚本,你需要确保你的环境中安装了pandas库。你可以通过运行`pip install pandas`来安装它。

将这个脚本保存为`process_data.py`并在命令行中运行`python process_data.py`即可开始处理数据。处理完成后,你会在`./ssl_csv_samples`目录下看到生成的CSV文件。

```