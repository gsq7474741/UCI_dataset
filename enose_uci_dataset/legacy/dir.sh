#!/bin/bash

# 遍历当前目录下的所有子目录
for dir in */ ; do
    # 移除末尾的斜杠和空格（如果有的话）
    dir_name="${dir%/}"
    dir_name="${dir_name##*/}"

    # 确保我们处理的是目录，而不是文件
    if [[ -d "$dir_name" ]]; then
        # 在子目录中创建.raw目录
        mkdir -p "$dir_name/.raw"

        # 将子目录中的文件和子目录移动到.raw中
        # 注意：这里我们假设没有名为.raw的文件或目录与我们要移动的内容冲突
        mv "$dir_name"/* "$dir_name/.raw"

        # 如果需要，还可以删除空目录（请小心使用，确保不会误删重要内容）
        # find "$dir_name" -type d -empty -delete
    fi
done

echo "处理完成。"
