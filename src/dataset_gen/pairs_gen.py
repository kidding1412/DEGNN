import json
import os
import random

json_path = r'/home/dj/dataset/jsonfilefiltered'
posi_path = r'/home/dj/dataset/pairs_big_dataset/positive'
neg_path = r'/home/dj/dataset/pairs_big_dataset/negative'
no_call_path = r'/home/dj/dataset/nocall_sum_json'
no_call_pos_pair_path = r'/home/dj/dataset/nocall_pos'
no_call_neg_pair_path = r'/home/dj/dataset/nocall_neg'

# 将json_path中所有文件绝度路径存入一个list
def json_files_list(json_path:str)->list:
    json_file_list = []
    for root, dirs, files in os.walk(json_path):
        for file in files:
            json_file_list.append(os.path.join(root, file))
    return json_file_list

# 小工具，根据路径提取文件名。对输入的str类型绝度路径进行处理，对str用'/'分割取最后一个，再删除'.json'后缀
def get_filename_from_path(path):
    return path.split('/')[-1].split('.json')[0]

#print(get_filename_from_path(json_files_list(json_path)[10]))

# 生成负例,输入参数为：数据集路径，输出路径和数量，默认10000.
def neg_pair_gen(dataset_path,negpath,num=10000):
    #跳过计数
    skip = 0
    #获取dataset_path中的所有json文件绝对路径list
    json_file_list = json_files_list(dataset_path)
    #获取dataset_path中的所有json文件总数
    file_count = len(json_file_list)
    #已生成的正例计数
    done = 0
    #在正例达到num之前，循环
    while(done < num):
        #获得第一个函数
        #生成一个0到file_count-1的随机数
        rand1 = random.randint(0,file_count-1)
        #根据随机数，从dataset_path中取出一个json文件
        json_file1 = json_file_list[rand1]
        #打开json文件
        with open(json_file1,'r') as f:
            datas1 = json.load(f)
        #获得datas长度
        datas_len1 = len(datas1)
        #生成一个0到datas_len-1的随机数
        rand2 = random.randint(0,datas_len1-1)
        #按照两个随机数提取函数
        func1 = datas1[rand2]
        #获得第二个函数
        #生成一个0到file_count-1的随机数
        rand3 = random.randint(0,file_count-1)
        #根据随机数，从dataset_path中取出一个json文件
        json_file2 = json_file_list[rand3]
        #打开json文件
        with open(json_file2,'r') as f:
            datas2 = json.load(f)
        #获得datas长度
        datas_len2 = len(datas2)
        #生成一个0到datas_len-1的随机数
        rand4 = random.randint(0,datas_len2-1)
        #按照两个随机数提取函数
        func2 = datas2[rand4]
        #判断两个函数是否相同
        if((func1['binname'] != func2['binname']) or (func1['name'] != func2['name'])):
            #创建一个空字典
            new_dict = {}
            new_dict['feature_1'] = func1['features']
            new_dict['feature_2'] = func2['features']
            new_dict['edge_index1'] = func1['edge_index']
            new_dict['edge_index2'] = func2['edge_index']
            new_dict['sim_flag'] = 0
            #生成新文件名
            func1_name = get_filename_from_path(json_file1) + '_' + func1['name']
            func2_name = get_filename_from_path(json_file2) + '_' + func2['name']
            new_name = func1_name + '_' + func2_name + '.json'
            new_path = os.path.join(negpath,new_name)
            #写入新文件
            #如果文件已存在，跳过
            if(os.path.exists(new_path)):
                skip += 1
                continue
            with open(new_path,'w') as f:
                json.dump(new_dict,f)
            done += 1
            print('neg done:',done,'/',num,'skip:',skip)
    #循环结束
    print('neg_pair_gen done!')

# 生成正例,输入参数为：数据集路径，输出路径和数量，默认10000.
def posi_pair_gen(dataset_path,posi_path,num=10000):
    #获取dataset_path中的所有json文件绝对路径list
    json_file_list = json_files_list(dataset_path)
    #获取dataset_path中的所有json文件总数
    file_count = len(json_file_list)
    #已生成的正例计数
    done = 0
    #跳过计数
    skip = 0
    #在正例达到num之前，循环
    while(done < num):
        #获得第一个函数
        #生成一个0到file_count-1的随机数
        rand1 = random.randint(0,file_count-1)
        #根据随机数，从dataset_path中取出一个json文件
        json_file1 = json_file_list[rand1]
        #打开json文件
        with open(json_file1,'r') as f:
            datas1 = json.load(f)
        #获得datas长度
        datas_len1 = len(datas1)
        #生成一个0到datas_len-1的随机数
        rand2 = random.randint(0,datas_len1-1)
        #按照两个随机数提取函数
        func1 = datas1[rand2]
        #获取binname和name
        func1_bin_name = func1['binname']
        #如果func1_bin_name是'.elf'结尾，删除'.elf'
        if(func1_bin_name.endswith('.elf')):
            func1_bin_name = func1_bin_name[:-4]
        func1_name = func1['name']
        #创建一个choosen_list，将json_file_list中所有str元素中有func1_bin_name的存入choosen_list
        choosen_list = []
        for i in json_file_list:
            if(func1_bin_name in i):
                choosen_list.append(i)
        #从choosen_list中随机选取一个json文件
        rand3 = random.randint(0,len(choosen_list)-1)
        json_file2 = choosen_list[rand3]
        #打开json文件
        with open(json_file2,'r') as f:
            datas2 = json.load(f)
        #遍历datas2，找到binname和name都和func1相同的函数，如果没找到，continue
        func2 = {}
        for i in datas2:
            func2_binname = i['binname']
            #如果func2_binname是'.elf'结尾，删除'.elf'
            if(func2_binname.endswith('.elf')):
                func2_binname = func2_binname[:-4]
            if((func2_binname == func1_bin_name) and (i['name'] == func1_name)):
                func2 = i
                break
        if(func2 == {}):
            print('func2 not found!')
            continue
        #创建一个空字典
        new_dict = {}
        new_dict['feature_1'] = func1['features']
        new_dict['feature_2'] = func2['features']
        new_dict['edge_index1'] = func1['edge_index']
        new_dict['edge_index2'] = func2['edge_index']
        new_dict['sim_flag'] = 1
        #生成新文件名
        func1_name = get_filename_from_path(json_file1) + '_' + func1['name']
        func2_name = get_filename_from_path(json_file2) + '_' + func2['name']
        new_name = func1_name + '_' + func2_name + '.json'
        new_path = os.path.join(posi_path,new_name)
        #写入新文件
        #如果文件已存在，跳过
        if(os.path.exists(new_path)):
            skip += 1
            continue
        with open(new_path,'w') as f:
            json.dump(new_dict,f)
        done += 1
        print('posi done:',done,'/',num,'skip:',skip)
    #循环结束
    print('posi_pair_gen done!')

# 正负例生成个数设置
posi_num = 100000
neg_num = 100000
neg_pair_gen(no_call_path,no_call_neg_pair_path,neg_num)
posi_pair_gen(no_call_path,no_call_pos_pair_path,posi_num)