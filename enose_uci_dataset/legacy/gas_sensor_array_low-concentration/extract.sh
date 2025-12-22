FILE_NAME="$1"

# 解压到raw目录
echo "Extracting $FILE_NAME to raw/..."
unzip -n "$FILE_NAME" -d raw/

# 每个数据集不同的处理方式

