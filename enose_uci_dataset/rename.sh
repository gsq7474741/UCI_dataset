#!/bin/bash

# 遍历当前目录下的所有.zip文件
for file in *.zip; do
    # 提取文件名（不包括路径）
    filename=$(basename "$file")
    # 替换文件名中的加号+为下划线_
    new_filename=${filename//+/_}
    # 如果新文件名与原始文件名不同，则重命名文件
    if [ "$filename" != "$new_filename" ]; then
        # 构造新文件的完整路径（如果文件在子目录中）
        dir=$(dirname "$file")
        new_file="$dir/$new_filename"
        # 去掉.zip后缀
        new_dir=${new_file%.zip}
        # 新建对应目录
        mkdir -p "$new_dir"
        # 重命名文件
        mv "$file" "$new_dir/$new_file"
        echo "Renamed $file to $new_dir/$new_file"
    fi
done
