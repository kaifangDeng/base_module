import torch
import torch.nn as nn
from torch.nn import functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数,通常为0.01左右
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [Batch_size, Seqencelen, in_features]  in_feature输入维度
        adj: 图的邻接矩阵 维度[Batch_size, Seqlen, Seqlen]
        """
        h = torch.matmul(inp, self.W)  # [B, S, in]
        B = h.size()[0]  # Batch_size大小
        N = h.size()[1]  # N 序列长度

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(B, N, -1, 2 * self.out_features)
        # [B, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=-1)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#在图注意层的基础上加入多头机制
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, alpha=0.01, n_heads=3):
        """Dense version of GAT
        n_heads 表示有几个GAT层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # 将每个head得到的表示进行拼接
        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        # x = self.out_att(x,adj)
        # return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定
        return x

if __name__ == "__main__":
    xx = GAT(5,5,5)
    inp = torch.randn(3, 3, 5)

    # 邻接矩阵
    adj = torch.tensor([[0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0]])
    adj = adj.repeat(3, 1, 1)
    # .unsqueeze(-1).repeat(1, 1, 1, 5)

    out = xx.forward(inp,adj)
    print(out)
