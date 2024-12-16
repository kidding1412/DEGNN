input_path = r'/home/dj/dataset/jsonfilefiltered'
output_path = r'/home/dj/dataset/x86_naglfar_dataset'

import os
import shutil
from tqdm import tqdm

# 获取input_path下所有文件
files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

# 过滤出包含_x86_的文件
x86_files = [f for f in files if "_x86_" in f]

# 显示进度条并复制文件
for file in tqdm(x86_files, desc="复制进度"):
    src_file_path = os.path.join(input_path, file)
    dst_file_path = os.path.join(output_path, file)
    shutil.copy(src_file_path, dst_file_path)
