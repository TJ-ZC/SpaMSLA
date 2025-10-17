import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv

EPS = 1e-8

class SpaMSLA(Module):
    def __init__(self, in_feat1, out_feat1, in_feat2, out_feat2, hidden_feat, dropout):
        super(SpaMSLA, self).__init__()

        self.encoder_omics1 = MSLA(in_channels=in_feat1, out_channels=out_feat1, hidden_channels=hidden_feat, dropout=dropout)
        self.encoder_omics2 = MSLA(in_channels=in_feat2, out_channels=out_feat2, hidden_channels=hidden_feat, dropout=dropout)

        self.decoder_out1 = Decoder(out_feat1, in_feat1)
        self.decoder_out2 = Decoder(out_feat2, in_feat2)

        self.graphfusion = GraphFusion(out_feat1, out_feat2, out_feat1)
        self.dis_weight = Parameter(torch.Tensor(out_feat1, out_feat2))
        torch.nn.init.xavier_uniform_(self.dis_weight)


    def forward(self, features_omics1, features_omics2, adj_spatial1, adj_spatial2, spatial_sp):
        encoder_omics1 = self.encoder_omics1(features_omics1)
        encoder_omics2 = self.encoder_omics2(features_omics2)

        pos_cross_omics = self.graphfusion(encoder_omics1, encoder_omics2, spatial_sp)
        neg_omics1, neg_omics2 = encoder_omics1[torch.randperm(encoder_omics1.size(0))], encoder_omics2[torch.randperm(encoder_omics2.size(0))]
        neg_cross_omics = self.graphfusion(neg_omics1, neg_omics2, spatial_sp)
        pos_readout = torch.sigmoid(pos_cross_omics.mean(dim=0))

        loss_dis = -torch.log(self.discriminator(pos_cross_omics, pos_readout) + EPS).mean() + \
                   -torch.log(1 - self.discriminator(neg_cross_omics, pos_readout) + EPS).mean()

        decoder_out1 = self.decoder_out1(pos_cross_omics, adj_spatial1)
        decoder_out2 = self.decoder_out2(pos_cross_omics, adj_spatial2)

        results = {
                   'decoder_out1': decoder_out1,
                   'decoder_out2': decoder_out2,
                   'attn_omics': pos_cross_omics,
                   'ctr': loss_dis,
                   }

        return results

    def discriminator(self, h, s):
        summary = s.t() if s.dim() > 1 else s
        value = torch.matmul(h, torch.matmul(self.dis_weight, summary))
        return torch.sigmoid(value)

'''
---------------------
Decoder functions
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
AGPL-3.0 LICENSE
---------------------
'''
class Decoder(Module):
    """
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    def __init__(self, in_feat, out_feat):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x

'''
MSLA Module
Ref: https://arxiv.org/abs/2306.08385
code: https://github.com/qitianwu/NodeFormer
'''
def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("nhd,md->nhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                            dim=attention_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    return data_dash


class MSLA(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, num_heads=1, dropout=0.1,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, nb_gumbel_sample=10):
        super(MSLA, self).__init__()

        self.convs = nn.ModuleList([NodeFormerConv(hidden_channels, hidden_channels, num_heads, kernel_transformation,
                                                   nb_random_features, nb_gumbel_sample)
                                    for _ in range(num_layers)])
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels), nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers + 1)])
        self.dropout = dropout
        self.activation = F.elu

    def forward(self, x, tau=1.0):
        layer_ = []
        z = self.fcs[0](x)
        z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)

        for i, conv in enumerate(self.convs):
            z = conv(z, tau)
            z += layer_[i]
            z = self.bns[i + 1](z)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)

        z = torch.cat(layer_, dim=-1)
        return self.fcs[1](z)


class NodeFormerConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation,
                 nb_random_features=10, nb_gumbel_sample=10):
        super(NodeFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.nb_random_features = nb_random_features
        self.nb_gumbel_sample = nb_gumbel_sample

    def forward(self, z, tau):
        query, key, value = [layer(z).reshape(-1, self.num_heads, self.out_channels)
                             for layer in [self.Wq, self.Wk, self.Wv]]

        projection_matrix = create_projection_matrix(self.nb_random_features, query.shape[-1], seed=int(torch.ceil(torch.abs(torch.sum(query) * 1e8)).item())).to(query.device)

        z_next = kernelized_gumbel_softmax(query, key, value, self.kernel_transformation, projection_matrix, self.nb_gumbel_sample, tau)

        return self.Wo(z_next.flatten(-2, -1))


def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix, K, tau):
    query, key = [q / math.sqrt(tau) for q in [query, key]]
    query_prime = kernel_transformation(query, True, projection_matrix)
    key_prime = kernel_transformation(key, False, projection_matrix)

    gumbels = (-torch.empty(key_prime.shape[:-1] + (K,)).exponential_().log()).to(query.device) / tau
    key_t_gumbel = key_prime.unsqueeze(2) * gumbels.exp().unsqueeze(3)

    z_num = numerator_gumbel(query_prime, key_t_gumbel, value)
    z_den = denominator_gumbel(query_prime, key_t_gumbel)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    return torch.mean(z_num / z_den, dim=2)


def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nhkm,nhd->hkmd", ks, vs)
    return torch.einsum("nhm,hkmd->nhkd", qs, kvs)


def denominator_gumbel(qs, ks):
    ks_sum = torch.einsum("nhkm,n->hkm", ks, torch.ones([ks.shape[0]]).to(qs.device))
    return torch.einsum("nhm,hkm->nhk", qs, ks_sum)

def create_projection_matrix(m, d, seed=0):
    nb_full_blocks = m // d
    block_list = []
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(seed)
        q = torch.randn((d, d))
        q, _ = torch.qr(q)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.vstack(block_list)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)


'''
GraphFusion
Ref: https://www.nature.com/articles/s41467-024-55204-y
code: https://github.com/Lin-Xu-lab/COSMOS
'''
class GraphFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels):
        super(GraphFusion, self).__init__()
        self.conv1 = GCNConv(in_channels1, hidden_channels, cached=False)
        self.conv2 = GCNConv(in_channels2, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.prelu3 = nn.PReLU(hidden_channels)
        self.prelu4 = nn.PReLU(hidden_channels)
        self.alpha = Parameter(torch.ones(2))
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

    def forward(self, x1, x2, edge_index):
        x1 = self.conv1(x1, edge_index)
        x1 = self.prelu(x1)
        x1 = self.conv3(x1, edge_index)
        x1 = self.prelu2(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1)

        x2 = self.conv2(x2, edge_index)
        x2 = self.prelu3(x2)
        x2 = self.conv4(x2, edge_index)
        x2 = self.prelu4(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1)

        weights = F.softmax(self.alpha, dim=0)

        x1 = x1 * weights[0]
        x2 = x2 * weights[1]
        x = x1 + x2
        return x
