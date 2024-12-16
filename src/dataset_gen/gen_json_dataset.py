from tools import *
import json
import os
import sys
big_list_path = r'/home/dj/dataset/big_list.json'
json_output_path = r'/home/dj/dataset/jsonfile'
pickle_path = r'/home/dj/dataset/pickle_dataset'
filtered_path = r'/home/dj/dataset/jsonfilefiltered'
# 获取所有pickle文件的路径
#pickle_file_list = get_pickle_files(pickle_path)
# 测试用，截断pickle_file_list到前100个
#pickle_file_list = pickle_file_list[:100]

# Pickle转Json
# 根据pickle文件的路径，将pickle文件转换为json文件，并存入json_output_path路径下,实时统计完成数

### 已完成，注释掉
# file_done = 0
# for pickle_file in pickle_file_list:
#     pickle_to_json(pickle_file, json_output_path)
#     file_done += 1
#     print('Pickle to Json',file_done,'/',len(pickle_file_list),'done')
# print('Pickle to Json done')

# 分析Json
# 分析json_output_path中的所有json文件，将json文件中的数据提取出来，存入big_list中
#big_list = []
# 将json_output_path中的所有json文件的绝对路径存入json_file_list中（路径下只有文件，没有文件夹），不使用已有函数方法
json_file_list = []
filtered = 0
for root, dirs, files in os.walk(json_output_path):
    for file in files:
        # 过滤，如果文件名中包含“filtered”，则跳过
        if 'filtered' in file:
            filtered += 1
            print('filtered:',filtered,' done:',len(json_file_list))
            continue
        json_file_list.append(os.path.join(root, file))
        print('filtered:',filtered,' done:',len(json_file_list))
print('json_file_list generated')



# 遍历json_file_list，将json文件中的数据提取出datas对象
# 跳过的空特征计数
nonfeature = 0
# 跳过的空边计数
nonedge = 0
# 已处理的文件计数
done = 0
# 完成函数计数
donefunc = 0
# list中文件数
filelen = len(json_file_list)

# 写一个方法，用'/'分割str取最后一个，再用'.'分割,删除后两个
def get_binname_from_path(path):
    return path.split('/')[-1].split('.elf.')[0]

# print(json_file_list[0])
# print(json_file_list[1])

# new_path = filtered_path + '/' + get_binname_from_path(json_file_list[0]) + '.json'
# new_path2 = filtered_path + '/' + get_binname_from_path(json_file_list[1]) + '.json'
# print(new_path)
# print(new_path2)
nocall_path = r'/home/dj/dataset/nocall_sum_json'

print('filtered json generating...')
# 遍历所有json文件，将json对象提取为data,处理,筛选后存入big_list
for json_file in json_file_list:
    with open(json_file, 'r') as f:
        datas = json.load(f)
    file_name = filtered_path + '/' + get_binname_from_path(json_file) + '.json'
    # 对datas对象进行关键信息补充
    # data_process为创新点方法，data_process_nocall为对比方法
    # datas = data_process(datas)
    datas = data_process_nocall(datas)

    # 建立空list用于存放筛选后的data对象
    filtered_datas = []

    # 将datas对象逐一进行属性筛选，留下后期需要的属性（用于标记的binname、name，和用于对比的features、edge_index）
    for data in datas:
        # 将空特征和空边的数据跳过
        if data['features'] == []:
            nonfeature += 1
            continue
        if data['edge_index'] == []:
            nonedge += 1
            continue
        # 将data对象进行处理，留下后期需要的属性（用于标记的binname、name，和用于对比的features、edge_index）
        data = create_dict(data)
        # 将data对象存入big_list中
        filtered_datas.append(data)
        donefunc += 1
        # 打印实时状态（nonfeature:0  nonedge:0  done: 0/filelen  donefunc: 0）
        #print('nonfeature:',nonfeature,' nonedge:',nonedge,' done:',done,'/',filelen,' donefunc:',donefunc)
    done += 1

    # 将筛选后的data对象存入json文件中
    # 使用对比实验的nocall路径
    nocall_file_name = nocall_path + '/' + get_binname_from_path(json_file) + '.json'
    with open(nocall_file_name, 'w') as f:
        json.dump(filtered_datas, f)
    print('nonfeature:',nonfeature,' nonedge:',nonedge,' done:',done,'/',filelen,' donefunc:',donefunc)

print('filtered json done')