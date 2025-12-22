FILE_NAME="$1"

# 解压到raw目录
echo "Extracting $FILE_NAME to raw/..."
unzip -n "$FILE_NAME" -d raw/

# 每个数据集不同的处理方式
# 把raw/QCM Sensor Alcohol Dataset下的所有文件移动到raw下，并删除QCM Sensor Alcohol Dataset目录
mv raw/QCM\ Sensor\ Alcohol\ Dataset/* raw/
rm -r raw/QCM\ Sensor\ Alcohol\ Dataset
