import os
import random
import shutil

# 将nocall数据集中的训练集和测试集随机选取十分之一复制到new_train和new_test文件夹中


old_train_path = r'/home/dj/gnn2/dataset/no_call_dataset/train'
new_train_path = r'/home/dj/gnn2/dataset/no_call_dataset/new_train'
old_test_path = r'/home/dj/gnn2/dataset/no_call_dataset/test'
new_test_path = r'/home/dj/gnn2/dataset/no_call_dataset/new_test'

def copy_random_json_files(source_path, destination_path):
    # 获取源路径下所有的.json文件
    json_files = [file for file in os.listdir(source_path) if file.endswith('.json')]
    
    # 计算需要复制的文件数量
    num_files_to_copy = len(json_files) // 10
    
    # 随机选择要复制的文件
    files_to_copy = random.sample(json_files, num_files_to_copy)
    
    # 复制文件到目标路径
    for file in files_to_copy:
        source_file = os.path.join(source_path, file)
        destination_file = os.path.join(destination_path, file)
        shutil.copyfile(source_file, destination_file)
        
    print(f"成功复制 {num_files_to_copy} 个文件到目标路径！")

copy_random_json_files(old_train_path, new_train_path)
copy_random_json_files(old_test_path, new_test_path)