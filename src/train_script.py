import subprocess
import yaml


yaml_path = r'/home/dj/gnn2/config/conf.yaml'
drop_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
main_path = r'/home/dj/gnn2/src/main.py'

# 写一个方法，读取yaml文件中的Parameters，epochs值
def get_epochs(path):
    with open(path, 'r') as f:
        args = yaml.safe_load(f)
    return args['Parameters']['dropout']

# 写一个方法，修改yaml文件中的Parameters，epochs值
def modify_epochs(yaml_path, new_epochs):
    # 读取并修改yaml文件
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # 修改Parameters中的epochs值
    data['Parameters']['dropout'] = new_epochs

    # 将修改后的数据写回yaml文件
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

for i in drop_list:
    # 修改yaml文件中的epochs值为i
    modify_epochs(yaml_path, i)
    # 运行main.py
    subprocess.run(['python', main_path])