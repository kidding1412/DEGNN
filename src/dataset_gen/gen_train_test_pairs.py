import os
import random
# 随即将所有数据集按照10%/90%的比例分为测试集和训练集，需要配置3个path，分别是sum_pairs、train、test
# 设置文件夹路径
sum_path = r'/home/dj/dataset/nocall_pairs_sum'
train_path = r'/home/dj/gnn2/dataset/no_call_dataset/train'
test_path = r'/home/dj/gnn2/dataset/no_call_dataset/test'

# 获取sum文件夹中的所有文件名
file_names = os.listdir(sum_path)

# 计算10%的数量
test_num = int(len(file_names) * 0.1)

# 随机选取10%的文件名
test_file_names = random.sample(file_names, test_num)

# 将选中的文件移动到test文件夹中
for file_name in test_file_names:
    os.rename(os.path.join(sum_path,file_name), os.path.join(test_path , file_name))
    # 加一个进度条显示进度
    #print('已完成：', test_file_names.index(file_name) / test_num * 100, '%')

# 将剩下的文件移动到train文件夹中
for file_name in os.listdir(sum_path):
    os.rename(os.path.join(sum_path,file_name), os.path.join(train_path , file_name))
    # 加一个进度条显示进度
    #print('已完成：', os.listdir(sum_path).index(file_name) / (len(file_names) - test_num) * 100, '%')