import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # PyG 中的 GAT 卷积
from torch_geometric.utils import dense_to_sparse
from torch_sparse import spmm
class GAT(nn.Module):
    def __init__(self, in_ft, out_ft, nheads=2, concat=True, dropout=0.6, alpha=0.2, bias=True):
        super(GAT, self).__init__()
        #self.attentions = nn.ModuleList([GraphAttentionLayer(in_ft, out_ft, concat=concat, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.gat = GATConv(in_ft, out_ft, heads=nheads, concat=concat, dropout=dropout, bias=bias)
        self.act = nn.PReLU()  
        self.dropout = nn.Dropout(dropout)#dropout
        #self.out_att = GraphAttentionLayer(out_ft * nheads, out_ft, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, input):
        x = input[0]  
        adj = input[1]

        return self.dropout(self.act(self.gat(x, adj)))

        #x = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        

class GraphAttentionLayer(nn.Module):

    def __init__(self, input_dim, out_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        N = x.size()[0]   # 图中节点数
        edge = adj._indices()   # 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        if x.is_sparse:   # 判断特征是否为稀疏矩阵
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        values = self.attn.mm(edge_h).squeeze()   # 使用注意力参数对特征进行投射
        edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge，对应原论文中的添加leakyrelu操作
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        e_rowsum = spmm(edge, edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).cuda())  # e_rowsum: N x 1，spmm是稀疏矩阵和非稀疏矩阵的乘法操作
        h_prime = spmm(edge, edge_e, n=N,m=N, matrix=h)   # 把注意力评分与每个节点对应的特征相乘
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())  # h_prime: N x out，div一看就是除，并且每一行的和要加一个9e-15防止除数为0
        # softmax结束
        if self.concat:
            # if this layer is not last layer
            return F.elu(h_prime)
        else:
            # if this layer is last layer
            return h_prime