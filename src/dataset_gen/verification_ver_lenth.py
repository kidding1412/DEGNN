import json
import os
neg_path = r'/home/dj/dataset/nocall_neg'
pos_path = r'/home/dj/dataset/nocall_pos'

new_train_path = r'/home/dj/gnn2/dataset/new_train'
new_test_path = r'/home/dj/gnn2/dataset/new_test'
new_train_big_path = r'/home/dj/gnn2/dataset/new_train_big'
new_test_big_path = r'/home/dj/gnn2/dataset/new_test_big'
#检查数据集中的json文件中的feature和edge_index的长度是否一致，比如block大于999就会产生不一致


# 将一个path路径下的所有.json文件的绝对路径存入一个list
def get_json_file_list(path):
    json_file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            json_file_list.append(os.path.join(root, file))
    return json_file_list

# 根据一个json文件的绝对路径，读取这个文件返回data文件对象
def get_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

#
def verify(feature,edge_index):
    feature_len = len(feature)
    list1 = edge_index[0]
    list2 = edge_index[1]
    list3 = []
    for i in list1:
        list3.append(i)
    for i in list2:
        list3.append(i)
    # list3去重
    list3 = list(set(list3))
    list_len = len(list3)
    if feature_len == list_len:
        return 1
    else:
        return 0

r1 = 0
w1 = 0
wrong_list = []

big_list = get_json_file_list(new_test_big_path)
for file in big_list:
    data = get_data_from_json(file)
    ver1 = verify(data['feature_1'],data['edge_index1'])
    ver2 = verify(data['feature_2'],data['edge_index2'])
    if ver1 == 1:
        r1 += 1
    else:
        w1 += 1
        wrong_list.append(file)
    if ver2 == 1:
        r1 += 1
    else:
        w1 += 1
        wrong_list.append(file)
    print(r1,'/',w1)

# 删除错误数据集
# r2 = 0
# w2 = 0

# for wrong_file in wrong_list:
#     if os.path.exists(wrong_file):
#         # 删除文件
#         os.remove(wrong_file)
#         print("文件删除成功！")
#         r2 = r2 + 1
#     else:
#         print("文件不存在。")
#         w2 = w2 + 1
# print(r2, '/',w2)