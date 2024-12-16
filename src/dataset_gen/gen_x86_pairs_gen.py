output_path = r'/home/dj/dataset/x86_naglfar_pairs'
input_path = r'/home/dj/dataset/x86_naglfar_file_struc'

# 生成一个正例
import os
import random
import json

"""
- input_path is the path to read the dataset from
- output_path is the path to output positive and negative examples

Output conditions:
- Returns 1 if successfully generated, otherwise returns 0
"""

def generate_positive_example(input_path, output_path):
    # 从输入路径中随机选择一个子文件夹
    sub_folders = [f.path for f in os.scandir(input_path) if f.is_dir()]
    chosen_folder = random.choice(sub_folders)
    
    # 在选中的文件夹中随机选择两个json文件
    json_files = [f for f in os.listdir(chosen_folder) if f.endswith('.json')]
    if len(json_files) < 2:
        return 0
    file_a, file_b = random.sample(json_files, 2)
    
    # 分别读取两个文件内容到对象
    with open(os.path.join(chosen_folder, file_a), 'r', encoding='utf-8') as fa:
        obj_a = json.load(fa)
    with open(os.path.join(chosen_folder, file_b), 'r', encoding='utf-8') as fb:
        obj_b = json.load(fb)
    
    # print(len(obj_a),type(obj_a))
    # print(len(obj_b),type(obj_b))
    
    attempt_count = 0
    dict_b = None
    while dict_b is None and attempt_count < 10:
        # Randomly select an element from obj_a
        item = random.choice(obj_a)
        item_name = item['name']
        
        # Search for an element in obj_b with the same name value
        for element in obj_b:
            if element['name'] == item_name:
                dict_b = element
                break
        attempt_count += 1
    if attempt_count == 10:
        return 0
    dict_a = item

    # 数据拼接
    dict_output = {
        "feature_1": dict_a["features"],
        "feature_2": dict_b["features"],
        "edge_index1": dict_a["edge_index"],
        "edge_index2": dict_b["edge_index"],
        "sim_flag": 1
    }
    #print(dict_output)
    # 文件名拼接
    output_file_name = f'{dict_a["binname"]}_{dict_a["name"]}_{dict_b["binname"]}_{dict_b["name"]}_pos.json'
    output_file_path = os.path.join(output_path, output_file_name)
    # 文件输出
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(dict_output, output_file, ensure_ascii=False, indent=4)
        #print(f"生成的文件已保存至: {output_file_path}")
        return 1
    else:
        #print(f"文件 {output_file_path} 已存在，未生成新文件。")
        return 0



def generate_negative_example(chosen_folder, output_path):
    # 随机选择一个file_a
    chosen_subfolder_a = random.choice([f.path for f in os.scandir(chosen_folder) if f.is_dir()])
    json_files_in_subfolder_a = [f for f in os.listdir(chosen_subfolder_a) if f.endswith('.json')]
    file_a = random.choice(json_files_in_subfolder_a)
    # 随机选择一个file_b
    chosen_subfolder_b = random.choice([f.path for f in os.scandir(chosen_folder) if f.is_dir()])
    json_files_in_subfolder_b = [f for f in os.listdir(chosen_subfolder_b) if f.endswith('.json')]
    file_b = random.choice(json_files_in_subfolder_b)
    # 分别读取两个文件内容到对象
    with open(os.path.join(chosen_subfolder_a, file_a), 'r', encoding='utf-8') as fa:
        obj_a = json.load(fa)
    with open(os.path.join(chosen_subfolder_b, file_b), 'r', encoding='utf-8') as fb:
        obj_b = json.load(fb)
    
    # 随机选择obj_a和obj_b中的元素作为dict_a和dict_b
    dict_a = random.choice(obj_a)
    dict_b = random.choice(obj_b)
    
    # 确保dict_a和dict_b的binname或name不同
    while dict_a["binname"] == dict_b["binname"] and dict_a["name"] == dict_b["name"]:
        dict_b = random.choice(obj_b)
    
    # 数据拼接
    dict_output = {
        "feature_1": dict_a["features"],
        "feature_2": dict_b["features"],
        "edge_index1": dict_a["edge_index"],
        "edge_index2": dict_b["edge_index"],
        "sim_flag": 0  # 负例标记为0
    }
    #print(dict_output)
    # 文件名拼接
    output_file_name = f'{dict_a["binname"]}_{dict_a["name"]}_{dict_b["binname"]}_{dict_b["name"]}_neg.json'
    output_file_path = os.path.join(output_path, output_file_name)
    # 文件输出
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(dict_output, output_file, ensure_ascii=False, indent=4)
        #print(f"生成的负例文件已保存至: {output_file_path}")
        return 1
    else:
        #print(f"负例文件 {output_file_path} 已存在，未生成新文件。")
        return 0


from tqdm import tqdm

def generate_examples(num_positive, num_negative, input_path, output_path):
    positive_count = 0
    negative_count = 0
    # 进度条
    with tqdm(total=num_positive + num_negative) as pbar:
        while positive_count < num_positive:
            result = generate_positive_example(input_path, output_path)
            positive_count += result
            pbar.update(result)
        while negative_count < num_negative:
            result = generate_negative_example(input_path, output_path)
            negative_count += result
            pbar.update(result)
# 调用函数生成10000个正例和10000个负例
generate_examples(10000, 10000, input_path, output_path)
