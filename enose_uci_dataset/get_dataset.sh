#!/bin/bash

# 遍历所有子目录中的file_info.sh
find . -type f -name 'file_info.sh' | while read config_file; do
    dir=$(dirname "$config_file")
    cd "$dir" || exit 1

    if [ "$dir" == "./gas_sensor_arrays_in_open_sampling_settings" ]; then
        echo "Skipping $dir"
        cd - >/dev/null || exit 1
        continue
    fi
    
    # 加载数据集配置
    source ./file_info.sh
    
    # 创建raw目录
    mkdir -p raw
    
    # 检查是否已下载
    if [ ! -f "$FILE_NAME" ]; then
        echo "Downloading $FILE_NAME..."
        wget "$LINK" -O "$FILE_NAME"
    else
        echo "$FILE_NAME already exists, skipping download."
    fi
    
    # 校验文件完整性
    computed_hash=$(sha1sum "$FILE_NAME" | awk '{print $1}')
    if [ "$computed_hash" != "$SHA1_HASH" ]; then
        echo "Error: SHA1 hash mismatch for $FILE_NAME, expected $SHA1_HASH, got $computed_hash"
        exit 1
    fi

    # 解压到raw目录
    bash ./extract.sh "$FILE_NAME"

    cd - >/dev/null || exit 1
done

echo "All datasets processed successfully."