# Gas sensors for home activity monitoring

## 数据集

### 数据集信息

该数据集包含由 8 个 MOX 气体传感器和一个温度和湿度传感器组成的气体传感器阵列的记录。
该传感器阵列暴露于背景家庭活动，同时受到两种不同的刺激：葡萄酒和香蕉。通过将刺激物放在传感器附近记录对香蕉和葡萄酒刺激的反应。
每次刺激的持续时间从 7 分钟到 2 小时不等，平均持续时间为 42 分钟。该数据集包含来自三种不同条件的一组时间序列：葡萄酒、香蕉和背景活动。
有 36 次葡萄酒诱导、33 次香蕉诱导和 31 次背景活动记录。

一种可能的应用是区分背景、葡萄酒和香蕉。该数据集由两个文件组成：`HT_sensor_dataset.dat`（压缩包），其中存储实际时间序列，以及
`HT_Sensor_metadata.dat`，其中存储每个诱导的元数据。每个诱导在两个文件中都由一个 id 唯一标识。因此，通过匹配每个文件中的列
id，可以轻松找到特定诱导的元数据。我们还提供了 python 脚本来演示如何导入、组织和绘制数据。
脚本可在 GitHub上找到：https://github.com/thmosqueiro/ENose-Decorr_Humdt_Temp
对于每次诱导，我们都包括刺激呈现之前和之后一小时的背景活动。时间序列以**每秒一个样本**的速度记录，由于无线通信问题，某些数据点略有变化。
有关使用了哪些传感器以及如何组织时间序列的详细信息，请参见下面的属性信息。

存储在文件HT_Sensor_metadata.dat 中的元数据分为以下列：
* id：诱导标识，与文件 HT_Sensor_dataset.dat 中的 id 匹配;
* 日期：记录此诱导的日、月、年;
* 类别：用于产生此诱导的东西（葡萄酒、香蕉或背景）;
* t0：诱导开始的时间（以小时为单位）（代表文件HT_Sensor_dataset.dat 中的时间零点））;
* dt：本次诱导持续的间隔。

### 变量信息

数据集由 100 个时间序列片段组成，每个片段都是一个单独的诱导或背景活动。总共有 919438个点。
对于每次诱导，呈现刺激的时间设置为零。有关实际时间，请参阅元数据文件的 t0 列。
在文件 HT_Sensor_dataset.dat中，每列都有一个标题，内容如下 
* id：诱导标识，与文件 HT_Sensor_metadata.dat 中的 id 匹配； 
* 时间：以小时为单位的时间，其中零是诱导的开始时间； 
* R1 - R8：8 个 MOX 传感器中每个传感器当时的电阻值； 
* 温度：当时的摄氏温度测量值； 
* 湿度：当时的湿度百分比测量值。

使用 Sensirion SHT75 测量温度和湿度。 
8 个 MOX 传感器可从 Figaro购买，详细信息如下：
- R1：TGS2611 
- R2：TGS2612 
- R3：TGS2610
- R4：TGS2600 
- R5：TGS2602 
- R6：TGS2602 
- R7：TGS2620 
- R8：TGS2620


### 数据样例

`HT_Sensor_dataset.dat`:

```text
id time       R1         R2         R3         R4         R5         R6         R7        R8        Temp.      Humidity
0  -0.999750  12.862100  10.368300  10.438300  11.669900  13.493100  13.342300  8.041690  8.739010  26.225700  59.052800  
0  -0.999472  12.861700  10.368200  10.437500  11.669700  13.492700  13.341200  8.041330  8.739080  26.230800  59.029900  
0  -0.999194  12.860700  10.368600  10.437000  11.669600  13.492400  13.340500  8.041010  8.739150  26.236500  59.009300 
...
```

`HT_Sensor_metadata.dat`：

```text
id	date		class	t0	dt
0	07-04-15	banana	13.49	1.64 
1	07-05-15	wine	19.61	0.54 
...
99	09-17-15	background	11.93	0.68  
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
- TGS2611，气体：Methane
- TGS2612，气体：Methane, propane, butane
- TGS2610，气体：Propane
- TGS2600，气体：Hydrogen, carbon monoxide
- TGS2602，气体：Ammonia, H2S, VOC
- TGS2602，气体：Ammonia, H2S, VOC
- TGS2620，气体：Carbon monoxide, combustible gases, VOC
- TGS2620，气体：Carbon monoxide, combustible gases, VOC
unit：KOhm
trans_to_ohm_functor：lambda x: x*1000
column_sample_rate:100
```



### 时间序列SSL预训练任务

元数据：气体、采样率、电阻单位、每个样本时长

对于每个样本，处理为单个csv文件：

每列为一个传感器的响应，每个样本应该是8个传感器，然后是温度、湿度和日期，最后一列为标签（气体种类）。
例如：sensor_0,sensor_1,...,sensor_8,temp,humidity,date,label_gas

原数据中的第一列id标志着这行属于哪个样本，与`HT_Sensor_metadata.dat`中的id对应。

处理后的csv文件，文件命名按照 `{样本id}_{气体种类}` 命名，例如 `0_banana.csv`。


prompt:

```text
请帮我根据以上信息处理数据集，生成csv文件。
当前目录结构如下：
.
├── note.md
└── .raw
    ├── HT_Sensor_dataset.dat
    ├── HT_Sensor_dataset.zip
    └── HT_Sensor_metadata.dat

请帮我在./ssl_csv_samples下生成要求的文件，使用python
```

模型输出：

见 `process_data.py`