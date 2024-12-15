import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops


class GATConv(MessagePassing):
    def __init__(self, in_feats, out_feats, alpha, drop_prob=0.0):
        super().__init__(aggr="add")
        self.drop_prob = drop_prob
        self.lin = nn.Linear(in_feats, out_feats, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        # 计算 Wh
        h = self.lin(x)
        # 启动消息传播
        h_prime = self.propagate(edge_index, x=h)
        return h_prime

    def message(self, x_i, x_j, edge_index_i):
        # 计算a(Wh_i || wh_j)
        e = torch.matmul((torch.cat([x_i, x_j], dim=-1)), self.a)
        e = self.leakrelu(e)
        alpha = softmax(e, edge_index_i)
        alpha = F.dropout(alpha, self.drop_prob, self.training)
        return x_j * alpha


if __name__ == "__main__":
    conv = GATConv(in_feats=3, out_feats=3, alpha=0.2)
    x = torch.rand(4, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2, 0, 3], [1, 0, 2, 1, 2, 0, 3, 0]], dtype=torch.long)
    x = conv(x, edge_index)
    print(x.shape)