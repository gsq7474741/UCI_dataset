FILE_NAME="$1"

# 大文件特殊处理
sudo apt install parallel unzip -y
# git clone https://github.com/Routin/ZipTurbo.git 
# cd ZipTurbo

# 解压到raw目录
echo "Extracting $FILE_NAME to raw/... using ZipTurbo (this may take a while)"
../ZipTurbo.sh --num_proc 16 --use_ram -o raw "$FILE_NAME"

# 每个数据集不同的处理方式
mv raw/WTD_upload/* raw/
rm -r raw/WTD_upload

