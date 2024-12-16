import os
import json

path_test = r'/home/dj/gnn2/dataset/test'
path_train = r'/home/dj/gnn2/dataset/train'
path_test_big = r'/home/dj/gnn2/dataset/test_big'
path_train_big = r'/home/dj/gnn2/dataset/train_big'
path_middle = r'/home/dj/dataset/jsonfilefiltered'
path_new_test = r'/home/dj/gnn2/dataset/new_test'
path_new_train = r'/home/dj/gnn2/dataset/new_train'
path_new_test_big = r'/home/dj/gnn2/dataset/new_test_big'
path_new_train_big = r'/home/dj/gnn2/dataset/new_train_big'

def get_json_files(path: str) -> list:
    path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                path_list.append(os.path.join(root, file))
    return path_list

wrong_list = []
report_list = []

# 检查特征向量是否连续
def check(path):
    r_count = 0
    f_count = 0
    path_list = get_json_files(path)
    sum = 2*len(path_list)
    for path1 in path_list:
        with open(path1, 'r') as f:
            data = json.load(f)
        len_featrue1 = len(data['feature_1'])
        len_featrue2 = len(data['feature_2'])
        list_for_edge1 = []
        list_for_edge2 = []
        for i in data['edge_index1']:
            for j in i:
                list_for_edge1.append(j)
        for i in data['edge_index2']:
            for j in i:
                list_for_edge2.append(j)

        list_for_edge1 = list(set(list_for_edge1))
        list_for_edge2 = list(set(list_for_edge2))
        list_for_edge1.sort()
        list_for_edge2.sort()
        if len(list_for_edge1) == len_featrue1:
            r_count += 1
        else:
            f_count += 1
            wrong_list.append(path1)
            print(len(list_for_edge1), len_featrue1)
            report_list.append(len(list_for_edge1))
            report_list.append(len_featrue1)
        if len(list_for_edge2) == len_featrue2:
            r_count += 1
        else:
            f_count += 1
            wrong_list.append(path1)
            print(len(list_for_edge2), len_featrue2)
            report_list.append(len(list_for_edge2))
            report_list.append(len_featrue2)
        print(r_count, f_count)
    return wrong_list


# r_count, f_count = check(path_new_train_big)
# print(r_count, f_count)
# print(wrong_list)
# print(report_list)
# print(r_count, f_count)

all_list = [path_new_test, path_new_train, path_new_test_big, path_new_train_big]
big_wrong_list = []

delete_count = 0
error_count = 0

for path_list in all_list:
    big_wrong_list += check(path_list)
for file in big_wrong_list:
    if os.path.exists(file):
    # 如果文件存在，删除它
        os.remove(file)
        print("File deleted successfully.")
        delete_count += 1
    else:
        # 如果文件不存在，输出错误信息
        print("Error: File not found.")
        error_count += 1
print(delete_count, error_count)