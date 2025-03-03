# Gas sensor array temperature modulation

## 数据集

### 数据集信息

由 14 个温度调制金属氧化物半导体 (MOX) 气体传感器组成的化学检测平台暴露于气室中一氧化碳 (CO) 和潮湿合成空气的动态混合物中。

提供了传感器获取的时间序列以及气室内 CO 浓度、湿度和温度的测量值。

a) 化学检测平台：
化学检测平台由 14 个 MOX 气体传感器组成，这些传感器对不同的气体刺激产生时间相关的多变量响应。
所用的传感器由 Figaro Engineering（7 个 TGS 3870-A04 单元）和 FIS（7 个 SB-500-12 单元）商业化提供。
传感器的工作温度由内置加热器控制，其电压以 20 和 25 秒为周期在 0.2-0.9 V 范围内调制，遵循制造商的建议（5 秒为 0.9 V，20 秒为 0.2 V，5 秒为 0.9 V，25 秒为 0.2 V，...）。
在开始实验之前，传感器预热一周。
MOX 读出电路由分压器和 1 MOhm 负载电阻组成，供电电压为 5V。
使用 Agilent HP34970A/34901A DAQ 以 3.5 Hz 采样传感器的输出电压，配置为 15 位精度和大于 10 GOhm 的输入阻抗。

b) 动态气体混合物发生器

通过管道系统和质量流量控制器 (MFC)，将 CO 和潮湿合成空气的动态混合物从气瓶中的高纯度气体输送到小型聚四氟乙烯 (PTFE) 测试室 (内部容积为 250 cm3)。

使用质量流量控制器 (MFC) 进行气体混合，该控制器控制三种不同的气流 (CO、湿空气和干空气)。这些气流由气瓶中的高质量加压气体输送。

所选 MFC (EL-FLOW Select，Bronkhorst) 的干湿空气流满量程流速为 1000 mln/min，CO 通道满量程流速为 3 mln/min。

CO 瓶中含有 1600 ppm 的 CO，用 21 Â± 1% O2 稀释在合成空气中。

产生的 CO 浓度的相对不确定度低于 5.5%。
湿空气流和干空气流均由纯度为 99.995% 且 O2 为 21±1% 的合成空气瓶输送。

湿空气流的加湿基于使用玻璃起泡器（Drechsler 瓶）的饱和法。

c) 温度/湿度值

温度/湿度传感器（SHT75，来自 Sensirion）每 5 秒提供测试室内的参考湿度和温度值，公差分别低于 1.8% r.h. 和 0.5 ÂºC。

每次实验中，气室内的温度变化均低于 3 ÂºC。

d) 实验方案：
每个实验包含 100 次测量：10 个实验浓度均匀分布在 0-20 ppm 范围内，每个浓度重复 10 次。

每次重复的相对湿度都是从 15% 到 75% r.h 之间的均匀分布中随机选择的。
每次实验开始时，使用流速为 240 mln/min 的合成空气流清洁气室 15 分钟。
之后，以 240 mln/min 的恒定流速随机释放气体混合物，每次 15 分钟。
单次实验持续 25 小时（100 个样本 x 15 分钟/样本），并在 13 个工作日内重复，自然周期为 17 天。

### 变量信息

数据集以 13 个文本文件的形式呈现，每个文件对应不同的测量日。文件名表示测量开始的时间戳 (yyyymmdd_HHMMSS)。

每个文件包括获取的时间序列，分为 20 列：
时间 (s)、CO 浓度 (ppm)、湿度 (%r.h.)、温度 (ÂºC)、流速 (mL/min)、加热器电压 (V) 以及 14 个气体传感器的电阻：R1 (MOhm)、R2 (MOhm)、R3 (MOhm)、R4 (MOhm)、R5 (MOhm)、R6 (MOhm)、R7 (MOhm)、R8 (MOhm)、R9 (MOhm)、R10 (MOhm)、R11 (MOhm)、R12 (MOhm)、R13 (MOhm)、R14 (MOhm)

电阻值 R1-R7 对应于 FIGARO TGS 3870 A-04 传感器，而 R8-R14 对应于 FIS SB-500-12 单元。
时间序列以 3.5 Hz 采样。

### 数据样例

`20160930_203718.csv`:
```text
Time (s),CO (ppm),Humidity (%r.h.),Temperature (C),Flow rate (mL/min),Heater voltage (V),R1 (MOhm),R2 (MOhm),R3 (MOhm),R4 (MOhm),R5 (MOhm),R6 (MOhm),R7 (MOhm),R8 (MOhm),R9 (MOhm),R10 (MOhm),R11 (MOhm),R12 (MOhm),R13 (MOhm),R14 (MOhm)
0.0000,0.0000,49.7534,23.7184,233.2737,0.8993,0.2231,0.6365,1.1493,0.8483,1.2534,1.4449,1.9906,1.3303,1.4480,1.9148,3.4651,5.2144,6.5806,8.6385
...
```


## 数据处理

### 提取metadata

prompt:

```prompt
修改当前文件，增加以下14列
name：sensor0-13，共14列
description：sensor名字,
data_type：fp32
sensor：
- TGS3870（甲烷和co，下同）
- TGS3870
- TGS3870
- TGS3870
- TGS3870
- TGS3870
- TGS3870
- SB-500-12（co，下同）
- SB-500-12
- SB-500-12
- SB-500-12
- SB-500-12
- SB-500-12
- SB-500-12
unit：MOhm
trans_to_ohm_functor：lambda x: x*1000000
column_sample_rate:3.5
```


### 时间序列SSL预训练任务

元数据：气体、浓度、采样率、电阻单位、每个样本时长

对于每个样本，处理为单个csv文件：

每列为一个传感器的电阻值（单位：千欧姆）,注意此处需要将原始的传感器相应换算为阻值（i.e. 40.0/S_i），最后几列为标签（气体种类浓度）。
例如：sensor_0,sensor_1,...,sensor_n,label_co_ppm,label_ethylene_ppm,label_methane_ppm
这两个文件加一起1.2GB，很大，处理的时候要根据数据的含义进行切分，
具体来说，当原数据中CO conc (ppm)/Methane conc (ppm) 和 Ethylene conc (ppm)任一值发生变化时，认为是一个新的样本。

prompt:

```text
请帮我根据以上信息处理数据集，生成csv文件。
当前目录结构如下：
.
├── note.md
└── .raw
    ├── ethylene_CO.txt
    └── ethylene_methane.txt

请帮我在./ssl_csv_samples下生成要求的文件，使用python
```

模型输出：

见 `process_data.py`