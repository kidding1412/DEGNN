import os
import pickle
import json
import hashlib
import numpy as np


# 待处理的datasets路径,包括子文件夹
# data_set_path = r"D:\work\dataset-test\normal_dataset_pickle"
# 输出的datasets_json路径
# dataset_json_path = r"D:\work\dataset-test\normal_dataset_json"

# 读取一个datasets绝对路径,输出一个list包含该路径下所有子目录的pickle文件的绝对路径
def get_pickle_files(path_tmp):
    pickle_files = []
    for root, dirs, files in os.walk(path_tmp):
        for file in files:
            if file.endswith(".pickle"):
                pickle_files.append(os.path.join(root, file))
    return pickle_files

# pickle文件转换成json文件,输入文件路径和输出路径,在输出路径下生成同名称的json文件,成功则返回1
def pickle_to_json(pickle_file_path, json_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    #在json_file_path路径下生成同名称的json文件
    json_file_path = os.path.join(json_file_path, os.path.basename(pickle_file_path).replace('.pickle', '.json'))
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    return 1

# string转14位int
def string_to_int14(my_string):
    my_hash = int(hashlib.sha256(my_string.encode('utf-8')).hexdigest(), 16)
    output = my_hash % (10 ** 14)
    return output

# cfg_list转edge_index,将二元组列表转换为numpy数组的转置
def list_to_nparray(lst):
    # 如果lst为空，返回空数组
    if not lst:
        return np.array(lst)
    if not all(isinstance(i, list) and len(i) == 2 for i in lst):
        print('list_to_nparray error: lst is not a list of 2-element lists')
        print(lst)
        return 0
    # 如果lst为空,返回空数组
    if not lst:
        return np.array(lst)
    else:
        return np.array(lst).T

# data数据处理函数,生成每个函数的edge_index
def insert_edge_index(data):
    for i in range(len(data)):
        #判断是否有'edge_index'这个key,如果有就置空，重写
        if 'edge_index' in data[i].keys():
            data[i]['edge_index'] = []
        cfg = data[i]['cfg']
        edge_index = list_to_nparray(cfg)
        #print(i,':',edge_index)
        data[i]['edge_index'] = edge_index.tolist()
        # if data[i]['name'] == "usage":
        #     print(data[i]['edge_index'])
    return data

# 输入edge_index和待查询数字,flag为0时返回出度,flag为1时返回入度,报错返回-1
def count_num_in_array(arr, num, flag):
    if not arr:
        return 0
    if flag == 0:
        return arr[0].count(num)
    elif flag == 1:
        return arr[1].count(num)
    else:
        return 0

# data数据处理函数,生成每个函数的x
# 函数的x特征由32位组成其中：
# 0-2位为节点ID,长度位3
# 3-7位为"size",长度为5
# 8-10位为"consts_num",长度为3
# 11-13位为"callees_num",长度为3
# 14-16位为"strings_num",长度为3
# 17-19位为"out_degree",长度为3
# 20-22位为"in_degree",长度为3
# 23位为"compiler_flag",0为gcc,1为clang,长度为1
# 24-25位为"arch_flag",00为x86,01为arm,10为mips,11为mipseb,长度为2
# 26位为架构位数,0为32位,1为64位,长度为1
# 27-28位为"Optimization_flag",00为O0,01为O1,10为O2,11为O3,长度为2
# 29位为"is_ret"flag,0为非ret,1为ret,长度为1
# 30位为"is_first"flag,0为非first,1为first,长度为1
# 31位为"is_block"flag,0为call节点,1为block节点,长度为1
def add_x(data):
    # 遍历data中的每个函数
    for i in range(len(data)):
        # 打印函数名做调试
        #print(data[i]['name'])
        # 获取函数的cfg_size
        tmp = data[i]['cfg_size']
        elf_arc = ''
        elf_opt = ''
        elf_comp = ''
        elf_wei = ''

        if 'gcc' in data[i]['compiler']:
            elf_comp = '0'
        elif 'clang' in data[i]['compiler']:
            elf_comp = '1'

        if 'x86' in data[i]['arch']:
            elf_arc = '00'
        elif 'arm' in data[i]['arch']:
            elf_arc = '01'
        elif 'mipseb' in data[i]['arch']:
            elf_arc = '11'
        elif 'mips' in data[i]['arch']:
            elf_arc = '10'
        
        if '32' in data[i]['arch']:
            elf_wei = '0'
        elif '64' in data[i]['arch']:
            elf_wei = '1'

        if 'O0' in data[i]['opti']:
            elf_opt = '00'
        elif 'O1' in data[i]['opti']:
            elf_opt = '01'
        elif 'O2' in data[i]['opti']:
            elf_opt = '10'
        elif 'O3' in data[i]['opti']:
            elf_opt = '11'

        for j in range(tmp):
            bb_data = data[i]['bb_data'][j]
            x = np.zeros(32, dtype=np.int32)
            # 获取bb的block_id
            block_id = bb_data['block_id']
            # 获取bb的常数项个数
            consts_num = len(bb_data['consts'])
            # 获取bb的调用函数个数
            callee_num = len(bb_data['callees'])
            # 获取bb的字符串个数
            strings_num = len(bb_data['strings'])
            # 获取bb的size
            size = bb_data['size']
            # bb出度计算
            out_degree = count_num_in_array(data[i]['edge_index'], block_id, 0)
            # bb入度计算
            in_degree = count_num_in_array(data[i]['edge_index'], block_id, 1)
            
            # 如果block_id大于999，则block_id的值为999
            if block_id > 999:
                block_id = 999
            # 如果size大于99999，则size的值为99999
            if size > 99999:
                size = 99999
            # 如果consts_num大于999，则consts_num的值为999
            if consts_num > 999:
                consts_num = 999
            # 如果callee_num大于999，则callee_num的值为999
            if callee_num > 999:
                callee_num = 999
            # 如果strings_num大于999，则strings_num的值为999
            if strings_num > 999:
                strings_num = 999
            # 如果out_degree大于999，则out_degree的值为999
            if out_degree > 999:
                out_degree = 999
            # 如果in_degree大于999，则in_degree的值为999
            if in_degree > 999:
                in_degree = 999
            

            #x[0]的值为block_id的百位数
            x[0] = block_id // 100
            #x[1]的值为block_id的十位数
            x[1] = block_id % 100 // 10
            #x[2]的值为block_id的个位数
            x[2] = block_id % 10
            #x[3]的值为size的万位数
            x[3] = size // 10000
            #x[4]的值为size的千位数
            x[4] = size % 10000 // 1000
            #x[5]的值为size的百位数
            x[5] = size % 1000 // 100
            #x[6]的值为size的十位数
            x[6] = size % 100 // 10
            #x[7]的值为size的个位数
            x[7] = size % 10
            #x[8]的值为consts_num的百位数
            x[8] = consts_num // 100
            #x[9]的值为consts_num的十位数
            x[9] = consts_num % 100 // 10
            #x[10]的值为consts_num的个位数
            x[10] = consts_num % 10
            #x[11]的值为callee_num的百位数
            x[11] = callee_num // 100
            #x[12]的值为callee_num的十位数
            x[12] = callee_num % 100 // 10
            #x[13]的值为callee_num的个位数
            x[13] = callee_num % 10
            #x[14]的值为strings_num的百位数
            x[14] = strings_num // 100
            #x[15]的值为strings_num的十位数
            x[15] = strings_num % 100 // 10
            #x[16]的值为strings_num的个位数
            x[16] = strings_num % 10
            #x[17]的值为out_degree的百位数
            x[17] = out_degree // 100
            #x[18]的值为out_degree的十位数
            x[18] = out_degree % 100 // 10
            #x[19]的值为out_degree的个位数
            x[19] = out_degree % 10
            #x[20]的值为in_degree的百位数
            x[20] = in_degree // 100
            #x[21]的值为in_degree的十位数
            x[21] = in_degree % 100 // 10
            #x[22]的值为in_degree的个位数
            x[22] = in_degree % 10
            #x[23]的值为"compiler_flag",0为gcc,1为clang,长度为1
            x[23] = elf_comp
            # 24-25位为"arch_flag",00为x86,01为arm,10为mips,11为mipseb,长度为2
            x[24] = elf_arc[0]
            x[25] = elf_arc[1]
            # 26位为架构位数,0为32位,1为64位,长度为1
            x[26] = elf_wei
            # 27-28位为"Optimization_flag",00为O0,01为O1,10为O2,11为O3,长度为2
            x[27] = elf_opt[0]
            x[28] = elf_opt[1]
            #x[29]的值为1或0,1表示该block_id是ret,0表示不是
            x[29] = 1 if bb_data['is_ret'] else 0
            # 30位为"is_first"flag,0为非first,1为first,长度为1
            x[30] = 1 if block_id == 0 else 0
            # 31位为"is_block"flag,0为call节点,1为block节点,长度为1,这里赋值的都是bb节点所以统一赋值为1
            x[31] = 1
            bb_data["x"] = x.tolist()
            #print(bb_data["x"])
            # if data[i]['name'] == "usage":
            #     print(data[i]['bb_data'][0]['x'])
    return data


# data数据处理函数,将每个data中bb_data的x合并为一个2维数组作为data的一个键值对。
def add_features(data):
    if not data:
        return data
    for i in range(len(data)):
        feature = []
        for j in range(len(data[i]['bb_data'])):
            feature.append(data[i]['bb_data'][j]['x'])
        data[i]['features'] = feature
        # if data[i]['name'] == "usage":
        #     print(data[i]['features'])
    return data

# 将int转换为对应长度的array,如果int数值大于array长度,则返回全9的array。length参数默认为2
def int_to_array(num, length=2):
    num_str = str(num)
    if len(num_str) > length:
        num_str = '9' * length
    arr = np.zeros(length, dtype=int)
    for i in range(len(num_str)):
        arr[-i-1] = int(num_str[-i-1])
    return arr


# 遍历data对象中的所有基本块，清空基本块中的callee值
def clear_callee(data):
    if not data:
        return data
    for i in range(len(data)):
        for j in range(len(data[i]['bb_data'])):
            data[i]['bb_data'][j]['callees'].clear()
    return data

# 根据data对象中函数对象的callers值，构建每个基本块中的callees值。
def add_bb_callees(data):
    if not data:
        return data
    for i in range(len(data)):
        if not data[i]["callers"]:
            # 如果callers为空，则跳过
            continue
        for j in range(len(data[i]["callers"])):
            func_name = data[i]["callers"][j][0]
            func_position = data[i]["callers"][j][1]
            func_self_name = data[i]["name"]
            for k in range(len(data)):
                if data[k]["name"] == func_name:
                    for l in range(len(data[k]["bb_data"])):
                        bb_start = data[k]["bb_data"][l]["startEA"]
                        bb_end = data[k]["bb_data"][l]["endEA"]
                        if bb_start <= func_position < bb_end:
                            data[k]["bb_data"][l]["callees"].append(func_self_name)
                        # 结束循环
                            break
                    break
    return data


# 小工具，二维list去重。输入data中的callers/callees/import_callees对象，返回的list中list[0]为去重后的callee，list[1]为对应的count
def remove_duplicates(lst):
    result = []
    count = []
    for i in range(len(lst)):
        if lst[i][0] not in result:
            result.append(lst[i][0])
            count.append(1)
        else:
            index = result.index(lst[i][0])
            count[index] += 1
    return [result, count]

#小工具，输入参数生成ndarray向量。长度为32
# 参数为callee_index_p，组成0-2位，长度为3，大于999则全为9
# callee_nbame_p，使用sha256_hash14方法将str转14位int，长度为14，组成3-16位
# callee_count_in_func_p，组成17-18位，长度为2，大于99则全为9
# callees_num_p，组成19-20位，长度为2，大于99则全为9
# callees_sum_p，组成21-22位，长度为2，大于99则全为9
# callers_num_p，组成23-24位，长度为2，大于99则全为9
# callers_sum_p，组成25-26位，长度为2，大于99则全为9
# import_callees_num_p，组成27-28位，长度为2，大于99则全为9
# import_callees_sum_p，组成29-30位，长度为2，大于99则全为9
# 第31位为0，表示为call节点
# 返回ndarray
def generate_feature(callee_index_p, callee_name_p, callee_count_in_func_p, callees_num_p, callees_sum_p, callers_num_p, callers_sum_p, import_callees_num_p, import_callees_sum_p):
    feature = np.zeros(32, dtype=int)
    feature[0:3] = int_to_array(callee_index_p if callee_index_p <= 999 else 999, 3)
    inthash = string_to_int14(callee_name_p)
    feature[3:17] = int_to_array(inthash, 14)
    feature[17:19] = int_to_array(callee_count_in_func_p if callee_count_in_func_p <= 99 else 99, 2)
    feature[19:21] = int_to_array(callees_num_p if callees_num_p <= 99 else 99, 2)
    feature[21:23] = int_to_array(callees_sum_p if callees_sum_p <= 99 else 99, 2)
    feature[23:25] = int_to_array(callers_num_p if callers_num_p <= 99 else 99, 2)
    feature[25:27] = int_to_array(callers_sum_p if callers_sum_p <= 99 else 99, 2)
    feature[27:29] = int_to_array(import_callees_num_p if import_callees_num_p <= 99 else 99, 2)
    feature[29:31] = int_to_array(import_callees_sum_p if import_callees_sum_p <= 99 else 99, 2)
    feature[31] = 0
    # 将feature转为list
    feature = feature.tolist()
    return feature



# 根据函数中已有的callee值构建调用特征，以字典形式存储在data[i]["callfeature"]中，键为调用函数名，值为调用特征
def add_bb_callfeature(data):
    if not data:
        return data
    # 如果data中没有callfeature key，则添加callfeature键值对
    for i in range(len(data)):
        data[i]["callfeature"] = []
        callees_list = remove_duplicates(data[i]["callees"])
        callers_list = remove_duplicates(data[i]["callers"])
        import_callees_list = remove_duplicates(data[i]["imported_callees"])
        callees_num = len(callees_list[0])
        callees_sum = sum(callees_list[1])
        callers_num = len(callers_list[0])
        callers_sum = sum(callers_list[1])
        import_callees_num = len(import_callees_list[0])
        import_callees_sum = sum(import_callees_list[1])
        index_from = data[i]["cfg_size"]
        # 为dict格式的data[i]增加一个键值对，键为callfeature，值为一个list，list中每个元素为一个array，array的长度为32
        data[i]["callfeature"] = {}
        # 遍历data[i]代表函数中的所有调用函数，生成调用特征，以字典形式存储在data[i]["callfeature"]中，键为调用函数名，值为调用特征
        for j in range(callees_num):
            callee_name = callees_list[0][j]
            callee_count_in_func = callees_list[1][j]
            callee_index = index_from + j
            feature = generate_feature(callee_index, callee_name, callee_count_in_func, callees_num, callees_sum, callers_num, callers_sum, import_callees_num, import_callees_sum)
            data[i]["callfeature"][callee_name] = feature
    return data

#根据调用特征，重构edge_index和x
def edge_index_x_update(data):
    # 遍历data[i]
    for i in range(len(data)):
        if not data[i]["callfeature"]:
            continue
        # 遍历data[i]["bb_data"][j]
        for j in range(len(data[i]["bb_data"])):
            if not data[i]["bb_data"][j]["callees"]:
                continue
            for k in range(len(data[i]["bb_data"][j]["callees"])):
                callee_name = data[i]["bb_data"][j]["callees"][k]
                callee_feature = data[i]["callfeature"][callee_name]
                # 将调用节点特征加入特征矩阵x中
                data[i]["features"].append(callee_feature)
                # 更新edge_index
                # data[i]["edge_index"]第一行添加data[i]["bb_data"][j][block_id]和callee_feature前3位表示的int
                # data[i]["edge_index"]第二行添加callee_feature前3位表示的int和data[i]["bb_data"][j][block_id]
                call_index = callee_feature[0]*100+callee_feature[1]*10+callee_feature[2]
                # 如果data[i]["edge_index"]为空或不存在这个key，则continue
                if not data[i]["edge_index"]:
                    continue
                if not data[i]["edge_index"][0]:
                    continue
                if not data[i]["edge_index"][1]:
                    continue

                data[i]["edge_index"][0].append(data[i]["bb_data"][j]["block_id"])
                data[i]["edge_index"][1].append(call_index)
                data[i]["edge_index"][0].append(call_index)
                data[i]["edge_index"][1].append(data[i]["bb_data"][j]["block_id"])
    return data

# datas中的feature顺序整理
def feature_order(datas):
    for data in datas:
        # 非空检查
        if not data:
            continue
        if not data['features']:
            continue
        # 对data中的features按照第一位，第二位，第三位排序
        sorted_lists = sorted(data['features'], key=lambda x: (x[0], x[1], x[2]))
        data['features'] = sorted_lists
    return datas


# data对象处理一共有3个工作,1.生成edge_index 2.生成x 3.合并x为features
# 分别对应的方法为insert_edge_index, add_x, add_features,现在用一个函数包装起来
# data处理函数,为原json对象生成edge_index和bb_data的x和data的features
def data_process(data):
    # 为data对象生成edge_index
    data = insert_edge_index(data)
    # 为data对象生成bb_data的x
    data = add_x(data)
    # 为data对象生成data的features
    data = add_features(data)
    # 清空data对象中的基本块的callee值
    data = clear_callee(data)
    # 根据data对象中函数对象的callers值，构建每个基本块中的callees值。
    data = add_bb_callees(data)
    # 根据函数中已有的callee值构建调用特征，以字典形式存储在data[i]["callfeature"]中，键为调用函数名，值为调用特征
    data = add_bb_callfeature(data)
    # 根据调用特征，重构edge_index和x
    data = edge_index_x_update(data)
    # datas中的feature顺序整理
    data = feature_order(data)
    # data数据对象处理完成
    return data


# data筛选处理方法，留下需要的数据
def create_dict(data):
    dict_tmp = {}
    dict_tmp['binname'] = data['bin_name']
    dict_tmp['name'] = data['name']
    dict_tmp['edge_index'] = data['edge_index']
    dict_tmp['features'] = data['features']
    return dict_tmp

# 生成不带call信息的原始cfg数据
def data_process_nocall(data):
    # 为data对象生成edge_index
    data = insert_edge_index(data)
    # 为data对象生成bb_data的x
    data = add_x(data)
    # 为data对象生成data的features
    data = add_features(data)
    return data