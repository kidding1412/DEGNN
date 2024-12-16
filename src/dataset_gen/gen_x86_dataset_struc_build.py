input_path = r'/home/dj/dataset/x86_naglfar_dataset'
output_path = r'/home/dj/dataset/x86_naglfar_pairs_dataset'

import shutil
import os

# 遍历input_path下的所有json文件，并将其文件名存入一个list
print("文件加载ing")
json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]

print("文件清洗ing")
removed_count = 0
for name in json_files[:]:
    #print(name.split(".json")[0].split('_')[-1])
    if not any(char.isalpha() for char in name.split(".json")[0].split('_')[-1]):
        json_files.remove(name)
        #print(name.split(".json")[0].split('_')[-1])
        removed_count += 1
print(f"删除了{removed_count}个文件。")

print("文件结构创建ing")
from tqdm import tqdm

for name in tqdm(json_files[:], desc="复制进度"):
    file_name = name.split('_')[0] + '_' + name.split(".json")[0].split('_')[-1]
    #print(file_name)
    folder_path = os.path.join(output_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建文件夹: {folder_path}")
    src_file_path = os.path.join(input_path, name)
    dst_file_path = os.path.join(folder_path, name)
    shutil.copy(src_file_path, dst_file_path)
    print(f"已将{name}拷贝至{file_name}文件夹")
