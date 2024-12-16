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
path_list = []

def get_json_files(path: str) -> list:
    path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                path_list.append(os.path.join(root, file))
    return path_list

# 检查特征数和节点数
big_list = get_json_files(path_train_big)

def process_list(list1):
    new_list = []
    num_list = []
    first_dict = {}
    # 遍历list，提取每个特征和齐对应的数字，将其作为键值对存入dict，将数字存入list。
    for item in list1:
        n = item[0]*100+item[1]*10+item[2]
        num_list.append(n)
        first_dict[n] = item
    # num_list去重，排序
    num_list = list(set(num_list))
    num_list.sort()
    for i in range(len(num_list)):
        feature = first_dict[num_list[i]]
        new_list.append(feature)
    return new_list

done_count = 0
sum = len(big_list)

for file in big_list:
    with open(file, 'r') as f:
        data = json.load(f)
        new_featrue_1 = process_list(data['feature_1'])
        new_featrue_2 = process_list(data['feature_2'])
        new_edge_index_1 = data['edge_index1']
        new_edge_index_2 = data['edge_index2']
        new_sim_flag = data['sim_flag']
        new_dict = {'feature_1':new_featrue_1,'feature_2':new_featrue_2,'edge_index1':new_edge_index_1,'edge_index2':new_edge_index_2,'sim_flag':new_sim_flag}
        file_name = file.split('/')[-1]
        new_path = os.path.join(path_new_train_big,file_name)
        with open(new_path,'w') as f:
            json.dump(new_dict,f,indent=4)
        done_count += 1
        print(done_count,'/',sum)
