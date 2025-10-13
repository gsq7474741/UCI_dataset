# Gas sensor array temperature modulation

## 数据集

### 数据集信息

该数据集包含由 10 个金属氧化物半导体传感器组成的传感器阵列收集的 6 种气体响应，气体浓度为 ppb 级（低于传感器的最低检测限）

### 变量信息

在gsalc.csv文件中，每一行代表一种气体样本，每行的第一列和第二列分别代表气体标签和浓度标签。从第三列开始，记录的是传感器阵列对该样本的响应。每个样本有9000个响应点，由10个传感器的响应串联而成，顺序为TGS2603、TGS2630、TGS813、TGS822、MQ-135、MQ-137、MQ-138、2M012、VOCS-P、2SH12（即前900个采样点为传感器TGS2603的响应，第901-1800个采样点为传感器TGS2630的响应……）。

类别标签：
我们从阵列中收集了乙醇、丙酮、甲苯、乙酸乙酯、异丙醇和正己烷等六种气体的响应信号。每种气体有三种浓度，分别为 50ppb、100ppb 和 200ppb，每种气体和浓度采集五个样本，总共 90 个样本。采集过程分为三个阶段：基线（5 分钟）、注入（10 分钟）和清洁（15 分钟），采样频率为 1Hz。此数据集提供基线和注入阶段的响应。

### 数据样例

`gsalc.csv`:
```text
ethanol,100ppb,0.3565,0.3345,0.3575,....
ethyl acetate,50ppb,0.3575,0.3415,0.3565....
...
```


## 数据处理

### 提取metadata

prompt:

```prompt
修改当前文件，增加以下10列
name：sensor0-9，共10列
description：sensor名字,
data_type：fp32
sensor：
- TGS2603 Odor
- TGS2630 Methane, Ethanol
- TGS813 Combustible gases
- TGS822 Organic solvent vapors
- MQ-135 Air quality (NH3, NOx, Alcohol, Benzene, Smoke, CO2)
- MQ-137 Ammonia
- MQ-138 Benzene, Toluene, Alcohol
- 2M012 Carbon monoxide
- VOCS-P Volatile organic compounds
- 2SH12 Hydrogen sulfide
unit：MOhm
trans_to_ohm_functor：lambda x: x*1000000
column_sample_rate:1
```

