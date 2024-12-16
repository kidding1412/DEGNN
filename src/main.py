import yaml
from gnn2 import GNNTrainer
#import torch

yaml_path = r'/home/dj/gnn2/config/conf.yaml'

def get_args(path):
    #读取yaml文件
    with open(path, 'r') as f:
        args = yaml.safe_load(f)
    return args

def main():
    # 获得配置参数
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args(yaml_path)
    # 判断模型模式
    # 训练模式
    if args['run_type'] == 'train':
        print('train model start.')
        trainer = GNNTrainer(args)
        # 训练模型
        trainer.fit()
        # 模型评估
        #trainer.score()
        # 保存模型
        trainer.save()
    # 测试模式
    elif args['run_type'] == 'test':
        print('test model start.')
        # 评估模型
        trainer = GNNTrainer(args)
        trainer.load()
        trainer.score(0)
    # 完成通知
    if args['Parameters']['notify']:
        import os
        import sys

        if sys.platform == "linux":
            os.system('notify-send GNN "Program is finished."')
        elif sys.platform == "posix":
            os.system(
                """
                        osascript -e 'display notification "GNN" with title "Program is finished."'
                        """
            )
        else:
            raise NotImplementedError("No notification support for this OS.")

if __name__ == '__main__':
    main()