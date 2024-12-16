import json
import os
neg_path = r'/home/dj/dataset/nocall_neg'
pos_path = r'/home/dj/dataset/nocall_pos'

new_train_path = r'/home/dj/gnn2/dataset/new_train'
new_test_path = r'/home/dj/gnn2/dataset/new_test'
new_train_big_path = r'/home/dj/gnn2/dataset/new_train_big'
new_test_big_path = r'/home/dj/gnn2/dataset/new_test_big'
#检查数据集中sim_flag标记是否正确


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

# 根据json文件的绝对路径，获得json文件中两个函数的binname和funcname
def get_binname_funcname_from_json(json_file):
    full_name = json_file.split('/')[-1][:-5]
    binname1 = full_name.split('_')[0]
    funcname1 = full_name.split('_')[5]
    # 如果binname1在full_name中出现了两次返回1
    if full_name.count(binname1) != 2:
        return 0
    else:
        list1 = full_name.split(binname1)
        list1.remove('')
        list2 = list1[0].split('_')
        list2.pop()
        list3 = list1[1].split('_')
        if list2[5:] == list3[5:]:
            return 1
        else:
            test = 9
            return 0
    

count_r = 0
count_w = 0

wrong_list = []

file_list = get_json_file_list(new_test_big_path)
for file in file_list:
    data = get_data_from_json(file)
    flag = get_binname_funcname_from_json(file)
    sim_flat = data['sim_flag']
    if sim_flat == flag:
        count_r += 1
    else:
        count_w += 1
        wrong_list.append(file)
    print(count_r, '/',count_w)
#print(wrong_list)



# 删除错误数据集
r = 0
w = 0

for wrong_file in wrong_list:
    if os.path.exists(wrong_file):
        # 删除文件
        os.remove(wrong_file)
        print("文件删除成功！")
        r = r + 1
    else:
        print("文件不存在。")
        w = w + 1
print(r, '/',w)