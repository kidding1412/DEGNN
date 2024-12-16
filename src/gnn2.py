import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt

import json
import glob
import datetime
from evaluate import *
from evaluate_per_epoch import *
import csv

# 输入x和edge_index，检测正确返回1，否则返回0
def check_lenth(x, edge_index):
    len_x = len(x)
    len_edge_index = len(set(edge_index[0]+edge_index[1]))
    if len_x == len_edge_index:
        return 1
    else:
        return 0

class GNN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args['Parameters']['tensor_neurons'] + self.args['Parameters']['bins']

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args['Parameters']['filters_1'])
        self.convolution_2 = GCNConv(self.args['Parameters']['filters_1'], self.args['Parameters']['filters_2'])
        self.convolution_3 = GCNConv(self.args['Parameters']['filters_2'], self.args['Parameters']['filters_3'])
        

        if self.args['Parameters']['diffpool']:
            self.attention = DiffPool(self.args)
        else:
            self.attention = AttentionModule(self.args)

        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(
            self.feature_count, self.args['Parameters']['bottle_neck_neurons']
        )
        self.scoring_layer = torch.nn.Linear(self.args['Parameters']['bottle_neck_neurons'], 1)

    def calculate_histogram(
        self, abstract_features_1, abstract_features_2, batch_1, batch_2
    ):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        #(abstract_features_1.shape)
        #print(abstract_features_2.shape)
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        #print(abstract_features_1.shape)
        #print(mask_1.shape)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)
        #print(abstract_features_2.shape)
        #print(mask_2.shape)
        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(
            abstract_features_1, abstract_features_2.permute([0, 2, 1])
        ).detach()
        #(scores.shape)
        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[: num_nodes[i], : num_nodes[i]]).view(-1)
            #print(mat.shape)
            hist = torch.histc(mat, bins=self.args['Parameters']['bins'])
            #print(hist.shape)
            hist = hist / torch.sum(hist)
            hist = hist.view(1, -1)
            #print(hist.shape)
            hist_list.append(hist)
        #print(torch.stack(hist_list).view(-1, self.args['Parameters']['bins']).shape)
        return torch.stack(hist_list).view(-1, self.args['Parameters']['bins'])

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args['Parameters']['dropout'], training=self.training)
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args['Parameters']['dropout'], training=self.training)
        features = self.convolution_3(features, edge_index)
        return features

    def diffpool(self, abstract_features, edge_index, batch):
        """
        Making differentiable pooling.
        :param abstract_features: Node feature matrix.
        :param edge_index: Edge indices
        :param batch: Batch vector, which assigns each node to a specific example
        :return pooled_features: Graph feature matrix.]
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1_tmp = data["g1"]['edge_index']
        edge_index_2_tmp = data["g2"]['edge_index']
        # 如果edge_index的类型是list，转换成tensor
        if isinstance(edge_index_1_tmp, list):
            edge_index_1 = torch.tensor(edge_index_1_tmp,dtype=torch.long)
        else:
            edge_index_1 = edge_index_1_tmp
        #print(edge_index_1_tmp)
        #print(edge_index_1)
        if isinstance(edge_index_2_tmp, list):
            edge_index_2 = torch.tensor(edge_index_2_tmp,dtype=torch.long)
        else:
            edge_index_2 = edge_index_2_tmp
        # print(type(edge_index_1))
        features_1_tmp = data["g1"]['x']
        # print(features_1.shape)
        features_2_tmp = data["g2"]['x']
        # 如果features的类型是list，转换成tensor
        if isinstance(features_1_tmp, list):
            features_1 = torch.tensor(features_1_tmp,dtype=torch.float)
        else:
            features_1 = features_1_tmp
        if isinstance(features_2_tmp, list):
            features_2 = torch.tensor(features_2_tmp,dtype=torch.float)
        else:
            features_2 = features_2_tmp
        #测试用
        # print(features_2.shape)
        # print(len(set(edge_index_1[0]+edge_index_1[1])))
        # print(len(set(edge_index_2[0]+edge_index_2[1])))
        # print(len(set(edge_index_1[0]).union(set(edge_index_1[1]))))
        # print(len(set(edge_index_2[0]).union(set(edge_index_2[1]))))
        # list_tmp1 = []
        # list_tmp2 = []
        # for i in range(len(edge_index_1)):
        #     for j in range(len(edge_index_1[i])):
        #         list_tmp1.append(edge_index_1[i][j])
        # for i in range(len(edge_index_2)):
        #     for j in range(len(edge_index_2[i])):
        #         list_tmp2.append(edge_index_2[i][j])
        # print(len(set(list_tmp1)))
        # print(len(set(list_tmp2)))
        batch_1_tmp = data["g1"]['batch']
        batch_2_tmp = data["g2"]['batch']
        # 如果batch的类型是list，转换成tensor
        if isinstance(batch_1_tmp, list):
            batch_1 = torch.tensor(batch_1_tmp,dtype=torch.long)
        else:
            batch_1 = batch_1_tmp
        if isinstance(batch_2_tmp, list):
            batch_2 = torch.tensor(batch_2_tmp,dtype=torch.long)
        else:
            batch_2 = batch_2_tmp
        # 将tensor移动到GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index_1 = edge_index_1.to(device)
        edge_index_2 = edge_index_2.to(device)
        features_1 = features_1.to(device)
        features_2 = features_2.to(device)
        batch_1 = batch_1.to(device)
        batch_2 = batch_2.to(device)

        #print(batch_1.shape)
        #print(batch_2.shape)
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        #print(abstract_features_1.shape)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if True:
            hist = self.calculate_histogram(
                abstract_features_1, abstract_features_2, batch_1, batch_2
            )

        if self.args['Parameters']['diffpool']:
            pooled_features_1 = self.diffpool(
                abstract_features_1, edge_index_1, batch_1
            )
            pooled_features_2 = self.diffpool(
                abstract_features_2, edge_index_2, batch_2
            )
        else:
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            #print(pooled_features_1.shape)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        #print(scores.shape)
        if True:
            scores = torch.cat((scores, hist), dim=1)

        scores = F.relu(self.fully_connected_first(scores))
        #print(scores.shape)
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        #print(score.shape)
        return score
    

# 构建一个train dataset类，从args中获取参数进行初始化
class train_datasets(object):
    def __init__(self, args):
        self.args = args
        if self.args['Parameters']['nocall']:
            self.train_path = args['paths']['nocall_train']
        elif self.args['Parameters']['big']:
            self.train_path = args['paths']['train_big_path']
        else:
            self.train_path = args['paths']['train_path']
        self.data_list = []
        # 读取train_path路径下的所有“*.json”文件，将文件内容append到data_list中
        for file in glob.glob(self.train_path + "/*.json"):
            with open(file) as f:
                self.data_list.append(json.load(f))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
# 构建一个test dataset类，从args中获取参数进行初始化
class test_datasets(object):
    def __init__(self, args):
        self.args = args
        if self.args['Parameters']['nocall']:
            self.test_path = args['paths']['nocall_test']
        elif self.args['Parameters']['big']:
            self.test_path = args['paths']['test_big_path']
        else:
            self.test_path = args['paths']['test_path']
        self.data_list = []
        # 读取test_path路径下的所有“*.json”文件，将文件内容append到data_list中
        for file in glob.glob(self.test_path + "/*.json"):
            with open(file) as f:
                self.data_list.append(json.load(f))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    


class GNNTrainer(object):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.setup_model()
        # 初始化评估list
        self.train_loss_list = []
        self.test_loss_list = []
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.auc_list = []
        self.epoch_list = []

    def setup_model(self):
        self.model = GNN(self.args, self.number_of_labels)

    def save(self):
        """
        Saving model.
        """
        current_time = datetime.datetime.now()
        file_name = self.args['paths']['model_path'] + '/' + current_time.strftime("%Y-%m-%d_%H-%M-%S") + '.pth'
        torch.save(self.model.state_dict(), file_name)
        print(f"Model is saved under {file_name}.")

    def load(self):
        """
        Loading model.
        """
        if self.args['Parameters']['nocall']:
            self.model.load_state_dict(torch.load(self.args['paths']['nocall_model']))
            print(f"Model is loaded from {self.args['paths']['nocall_model']}.")
        else:
            self.model.load_state_dict(torch.load(self.args['paths']['best_model']))
            print(f"Model is loaded from {self.args['paths']['best_model']}.")

    def process_dataset(self):
        """
        processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.training_graphs = train_datasets(self.args)
        self.testing_graphs = test_datasets(self.args)
        # 原代码中nged_matrix是图编辑距离矩阵，用作标签作用。在gnn2中我们使用list作为标签
        #self.nged_matrix = self.training_graphs.norm_ged
        self.train_y = []
        self.test_y = []
        for i in range(len(self.training_graphs)):
            self.train_y.append(self.training_graphs.data_list[i]['sim_flag'])
        self.test_y = []
        for i in range(len(self.testing_graphs)):
            self.test_y.append(self.testing_graphs.data_list[i]['sim_flag'])
        self.real_data_size = len(self.training_graphs)+len(self.testing_graphs)
        self.number_of_labels = self.args['Parameters']['num_features']

    def create_batches(self):
        """
        构造batches，为了适应原结构，无法直接使用DataLoader，需要自己构造
        """
        # 洗牌
        random.shuffle(self.training_graphs.data_list)
        # 将所有数据按照batch_size进行划分
        list_of_batches = []
        batch_size = self.args['Parameters']['batch_size']
        for i in range(0, len(self.training_graphs), batch_size):
            list_of_batches.append(self.training_graphs[i:i + batch_size])
        if len(list_of_batches[-1]) < batch_size:
            list_of_batches.pop(-1)
        # 对list_of_batches中的每个batch进行处理，将其每个batch中的所有数据进行拼接
        output_list = []
        # 遍历list_of_batches大表中的每个batch
        for little_batch in list_of_batches:
            # g1/2汇总这一个batch中所有的edge_index
            g1 = []
            g2 = []
            # i1/2汇总这一个batch的标号,暂时无用
            i1 = []
            i2 = []
            # num_nodes1/2汇总这一个batch中所有的节点数
            num_nodes1 = 0
            num_nodes2 = 0
            # target汇总这一个batch中所有的标签
            target = []
            # feature1/2汇总这一个batch中所有的feature
            feature1 = []
            feature2 = []
            # batch长度和节点数对应，用来标记对位节点是第几个图
            batch1 = []
            batch2 = []
            # ptr长度为batchsize+1，ptr[i]表示第i个图的第一个节点在feature中的位置，最后一位是最后一个节点的位置
            ptr1 = [0,]
            ptr2 = [0,]
            # print(len(little_batch))
            # 遍历batch中的每个数据
            for i in range(len(little_batch)):
                # 提取单个图的特征
                l_feature_1 = little_batch[i]['feature_1']
                l_feature_2 = little_batch[i]['feature_2']
                # 提取单个图的edge_index
                l_edge_index_1 = little_batch[i]['edge_index1']
                l_edge_index_2 = little_batch[i]['edge_index2']
                # 提取单个图的标签
                l_y = little_batch[i]['sim_flag']
                # 记录单个图的节点个数(即特征长度)
                len_fea1 = len(l_feature_1)
                len_fea2 = len(l_feature_2)
                # 记录单个图的edge_index长度
                len_edge1 = len(l_edge_index_1[0])
                len_edge2 = len(l_edge_index_2[0])
                # 记录当前g中的节点数
                len_g1 = 0 if g1 == [] else len(g1[0])
                len_g2 = 0 if g2 == [] else len(g2[0])
                # 汇总填写batch中的数据
                # 填写g1g2
                # 更新这一轮edge_index的序号
                new_edge_index_1 = [[x + num_nodes1 for x in sublist] for sublist in l_edge_index_1]
                new_edge_index_2 = [[x + num_nodes2 for x in sublist] for sublist in l_edge_index_2]
                # 转换为array,方便做拼接
                # 先判断g1g2是否为空，在第一次循环中空list拼接要单独处理
                # 处理g1
                if len_g1 == 0:
                    g1 = new_edge_index_1
                else:
                    array_g1 = np.array(g1)
                    array_new_edge_index_1 = np.array(new_edge_index_1)
                    concatenated_array_g1 = np.concatenate((array_g1, array_new_edge_index_1), axis=1)
                    g1 = concatenated_array_g1.tolist()
                # 处理g2
                if len_g2 == 0:
                    g2 = new_edge_index_2
                else:
                    array_g2 = np.array(g2)
                    array_new_edge_index_2 = np.array(new_edge_index_2)
                    concatenated_array_g2 = np.concatenate((array_g2, array_new_edge_index_2), axis=1)
                    g2 = concatenated_array_g2.tolist()
                # 填写num_nodes12
                num_nodes1 += len_fea1
                num_nodes2 += len_fea2
                # 填写target
                target.append(l_y)
                # 填写feature12
                feature1 = feature1 + l_feature_1
                feature2 = feature2 + l_feature_2
                # 填写batch12
                batch1.extend([i] * len_fea1)
                batch2.extend([i] * len_fea2)
                # 填写ptr12
                ptr1.append(ptr1[-1] + len_fea1)
                ptr2.append(ptr2[-1] + len_fea2)
            data_batch1 = {
                'edge_index': torch.tensor(g1, dtype=torch.long),
                'x': torch.tensor(feature1, dtype=torch.float),
                'batch': torch.tensor(batch1, dtype=torch.long),
                'ptr': torch.tensor(ptr1, dtype=torch.long),
                'num_nodes': num_nodes1
            }
            data_batch2 = {
                'edge_index': torch.tensor(g2, dtype=torch.long),
                'x': torch.tensor(feature2, dtype=torch.float),
                'batch': torch.tensor(batch2, dtype=torch.long),
                'ptr': torch.tensor(ptr2, dtype=torch.long),
                'num_nodes': num_nodes2,
            }
            target = torch.tensor(target, dtype=torch.float)
            tuple_batch = (data_batch1, data_batch2, target)
            output_list.append(tuple_batch)
        return output_list

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        new_data["target"] = data[2]
        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        self.optimizer.zero_grad()
        data = self.transform(data)
        target = data["target"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        prediction = self.model(data)
        loss = F.mse_loss(prediction, target, reduction="sum")
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        # # 初始化评估list
        # self.train_loss_list = []
        # self.test_loss_list = []
        # self.accuracy_list = []
        # self.precision_list = []
        # self.recall_list = []
        # self.f1_list = []
        # self.auc_list = []
        # self.epoch_list = []



        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args['Parameters']['learning_rate'],
            weight_decay=self.args['Parameters']['weight_decay'],
        )
        self.model.train()

        # 如果GPU可用，将模型放入GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        epochs = trange(self.args['Parameters']['epochs'], leave=True, desc="Epoch")
        loss_list = []
        loss_list_test = []
        for epoch in epochs:

            if self.args['Parameters']['plot']:
                if epoch % 10 == 0:
                    self.score(epoch)
                    self.model.train(True)

            batches = self.create_batches()

            main_index = 0

            loss_sum = 0
            for index, batch_pair in tqdm(
                enumerate(batches), total=len(batches), desc="Batches", leave=False
            ):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0]['batch'][-1]
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item(), 5))
            loss_list.append(loss)
            self.train_loss_list.append(loss)

        # 在所有epoch完成后，记录所有数据list到csv文件
        if self.args['Parameters']['plot']:
            # 创建一个csv文件对象
            csv_path = self.args['paths']['evaluate']
            csv_name = 'eva_result_' + 'epoch_' + str(self.args['Parameters']['epochs']) + '_dropout' + str(self.args['Parameters']['dropout']) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            with open(os.path.join(csv_path,csv_name), 'w', newline='') as csv_file:
                # 基于文件对象构建 csv写入对象
                csv_writer = csv.writer(csv_file)
                # 写入csv文件内容
                csv_writer.writerow(['epoch', 'test_loss', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
                csv_writer.writerows(zip(self.epoch_list, self.test_loss_list, self.accuracy_list, self.precision_list, self.recall_list, self.f1_list, self.auc_list))
                print("Evaluate result is saved under: ", os.path.join(csv_path,csv_name))
            csv_trainloss_name = 'trainloss_' + 'epoch_' + str(self.args['Parameters']['epochs']) + '_dropout' + str(self.args['Parameters']['dropout']) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            with open(os.path.join(csv_path,csv_trainloss_name), 'w', newline='') as csv_file:
                # 基于文件对象构建 csv写入对象
                csv_writer = csv.writer(csv_file)
                # 写入csv文件内容
                csv_writer.writerow(['train_loss'])
                float_list = [tensor.item() for tensor in self.train_loss_list]
                # 将float_list写入csv文件中
                csv_writer.writerows([[x] for x in float_list])
                print("Train loss is saved under: ", os.path.join(csv_path,csv_trainloss_name))

    def measure_time(self):
        import time

        self.model.eval()
        count = len(self.testing_graphs) * len(self.training_graphs)

        t = np.empty(count)
        i = 0
        tq = tqdm(total=count, desc="Graph pairs")
        for g1 in self.testing_graphs:
            for g2 in self.training_graphs:
                source_batch = Batch.from_data_list([g1])
                target_batch = Batch.from_data_list([g2])
                data = self.transform((source_batch, target_batch))

                start = time.process_time()
                self.model(data)
                t[i] = time.process_time() - start
                i += 1
                tq.update()
        tq.close()

        print(
            "Average time (ms): {}; Standard deviation: {}".format(
                round(t.mean() * 1000, 5), round(t.std() * 1000, 5)
            )
        )

    def score(self, epoch_int):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")

        self.model.eval()
        test_list = self.testing_graphs.data_list
        # 提取数据集中关键信息，为送入模型做准备
        test_data = []
        for i in range(len(test_list)):
            edge_index1 = test_list[i]['edge_index1']
            edge_index2 = test_list[i]['edge_index2']
            x1 = test_list[i]['feature_1']
            x2 = test_list[i]['feature_2']
            y = test_list[i]['sim_flag']
            batch1 = [0] * len(x1)
            batch2 = [0] * len(x2)
            ptr1 = [0, len(x1)]
            ptr2 = [0, len(x2)]
            num_nodes1 = len(x1)
            num_nodes2 = len(x2)
            # 构建dict数据结构，方便后续送入模型
            g1 = {
                "edge_index": edge_index1,
                "x": x1,
                "num_nodes": num_nodes1,
                "batch": batch1,
                "ptr": ptr1
            }
            g2 = {
                "edge_index": edge_index2,
                "x": x2,
                "num_nodes": num_nodes2,
                "batch": batch2,
                "ptr": ptr2
            }
            target = torch.tensor(y, dtype=torch.float)
            output_dict = {
                'g1': g1,
                'g2': g2,
                'target': target
            }
            test_data.append(output_dict)
        # scores为测试集中loss的表现
        scores = np.zeros(len(test_list))
        # ground_truth
        ground_truth = np.zeros(len(test_list))
        prediction_mat = []

        # 如果GPU可用，将模型放入GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        t = tqdm(total=len(test_list),desc="testinging, dropout:" + str(self.args['Parameters']['dropout']))
        # 开始迭代测试集
        for i, g in enumerate(test_data):
            target = g["target"]
            ground_truth[i] = target
            prediction = self.model(g)
            prediction_mat.append(prediction.item())
            target_tensor = torch.unsqueeze(target, 0)
            scores[i] = (
                F.mse_loss(prediction.cpu(), target_tensor, reduction="none").detach().numpy()
            )
            t.update(1)
        t.close()
        prediction_output = np.array(prediction_mat)
        self.model_error = np.mean(scores).item()
        # 计算测试结果
        accuracy_per_epoch, precision_per_epoch, recall_per_epoch, f1_per_epoch, auc_per_epoch = evaluate_and_log_per_epoch(prediction_output,ground_truth)
        self.accuracy_list.append(accuracy_per_epoch)
        self.precision_list.append(precision_per_epoch)
        self.recall_list.append(recall_per_epoch)
        self.f1_list.append(f1_per_epoch)
        self.auc_list.append(auc_per_epoch)
        self.test_loss_list.append(self.model_error)
        self.epoch_list.append(epoch_int)

        # 如果是最后一次被调用，记录prediction和ground_truth到csv文件
        if epoch_int == self.args['Parameters']['epochs'] - 1 or self.args['run_type'] == 'test':
            result_path = os.path.join(self.args['paths']['evaluate'], 'truth_pred_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv')
            with open(result_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['prediction', 'ground_truth'])
                for i in range(len(prediction_output)):
                    writer.writerow([prediction_output[i], ground_truth[i]])

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        # print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        # print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        # print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        # print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
