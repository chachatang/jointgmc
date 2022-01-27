import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import time
import random
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, pipeline
import logging
from utils.train_utils import format_metrics
from torch_geometric.utils import degree, to_undirected
from copy import deepcopy
import numpy as np
from torch.nn.modules.module import Module
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score,normalized_mutual_info_score,accuracy_score
from elastic import Elastic_Choose

from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy import sparse
from util import accuracy,sparse_mx_to_sparse_tensor, prob_to_adj,evaluate
import manifolds
import layers.hyp_layers as hyp_layers
import scipy.sparse as sp
from termcolor import cprint
from torch_sparse import SparseTensor


def knn(num_node,k,feature):
    from sklearn.metrics.pairwise import cosine_similarity as cos
    adj = np.zeros((num_node, num_node), dtype = np.int64)
    dist = cos(feature.detach().cpu().numpy())
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    return adj

def knn_overlap(num_node,k,feature,emb):
    feature_index = []
    emb_index = []
    overlap = 0
    feature_knn = knn(num_node, k, feature)
    emb_knn = knn(num_node, k, emb)

    for i in range(num_node):
        index1 = list(np.where(feature_knn[i] != 0)[0])
        index2 = list(np.where(emb_knn[i] != 0)[0])
        feature_index.append(index1)
        emb_index.append(index2)

    for j in range(num_node):
        o = set(feature_index[j]) & set(emb_index[j])
        overlap = overlap + len(o)
    overlap_ratio = overlap / (num_node * (k + 1))
    return overlap_ratio


class GCN(nn.Module):
    def __init__(self, num_feature, hidden_size, dim, num_class,dropout=0.5, activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_size)
        self.conv2 = GCNConv(hidden_size, dim)
        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, adj)
        return x1, F.log_softmax(x2, dim=1)

class HGCN(nn.Module):
    def __init__(self, c, args):
        super(HGCN, self).__init__()
        self.c = c
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        self.num_layers = args.num_layers
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(self.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,args
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def normalize(self,mx):
        mx_ = mx.to_dense().cpu().numpy()
        a = sp.lil_matrix(mx_)
        r_sum = np.array(a.sum(1).astype(float))
        r_inv = np.power(r_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx_)
        return mx

    def sparse(self,sparse_mx):
        sparse_mx = sparse.coo_matrix(sparse_mx).astype(np.float32)
        values = sparse_mx.data
        indices = np.vstack((sparse_mx.row, sparse_mx.col))
        i = torch.tensor(indices)
        v = torch.tensor(values)
        shape = sparse_mx.shape
        adj = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        return adj

    def forward(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        adj__ = self.normalize(adj)
        adj_h = self.sparse(adj__)
        input = (x_hyp, adj_h)
        x_hidden = []
        for i in range(len(self.layers)):
            out_hidden = (self.layers[i](input))
            input = out_hidden
            x_hidden.append(out_hidden[0])
        return x_hidden


class Base_Module(nn.Module):
    def __init__(self,num_feature, hidden_size,dim,num_class,args,c):
        super(Base_Module, self).__init__()
        self.args = args
        self.device = args.device
        self.gcn = GCN(num_feature, hidden_size,dim,num_class).to(self.device)
        self.hgch = HGCN(c,args).to(self.device)
        self.c = torch.tensor(args.c)
        self.tau: float = args.tau
        self.dim = dim
        self.elastic = np.random.random()
        self.elastic_choose = Elastic_Choose(args,self.elastic)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def compare_loss(self, e, h):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(e, h))
        zero = torch.zeros(between_sim.size()).to(self.device)
        between_p = torch.where(between_sim > 0.5, between_sim, zero)
        between_n = torch.where(between_sim < 0.5, between_sim, zero)
        e_sim = f(self.sim(e,e))
        e_p = torch.where(e_sim > 0.5, e_sim, zero)
        e_n = torch.where(e_sim < 0.5, e_sim, zero)
        h_sim = f(self.sim(h,h))
        h_p = torch.where(h_sim > 0.5, h_sim, zero)
        h_n = torch.where(h_sim < 0.5, h_sim, zero)
        comp_loss = -torch.log((between_p.sum(1) +e_p.sum(1) +h_p.sum(1))/ (between_p.sum(1) +e_p.sum(1) +h_p.sum(1)+between_n.sum(1) +e_n.sum(1) +h_n.sum(1)))
        ret = comp_loss.mean()
        return ret



    """Euclidean Feature Fusion from hyperbolic space"""
    def hyper_eucl_mix(self,x_e,x_h):
        dist = (pmath.logmap0(x_h, k=self.c) - x_e).pow(2).sum(dim=-1)
        x_h = dist.view([-1, 1]) * pmath.logmap0(x_h, k=self.c)
        x_h = F.dropout(x_h, p=0)
        x_h = torch.sigmoid(x_h)
        x_mix_dot = x_e*x_h
        x_mix_scaled_dot = torch.sigmoid(x_mix_dot)
        x_mix = torch.relu(x_mix_dot * x_mix_scaled_dot)
        self.z = [x_e, x_mix]
        x_mix = x_e +self.elastic * x_mix
        comp_loss_h_mix = self.compare_loss(x_mix, x_h)
        return x_mix,comp_loss_h_mix

    def hyper_eucl_mix_revise(self, x_e, x_mix,elastic):
        x_mix = x_e + elastic * x_mix
        return x_mix


    def elastic_revise(self,acc_train,data,compute_metrics0823):
        self.elastic = self.elastic_choose.step(acc_train,data,self.z,self.hyper_eucl_mix_revise,compute_metrics0823)
        return self.elastic

    def forward(self,x,adj):
        x_out_1,x_out_2 = self.gcn(x, adj)
        x_hypout = self.hgch(x, adj)
        x_hypout_1 = x_hypout[0]
        x_hypout_2 = x_hypout[1]
        x_mix_1,comp_loss_1 = self.hyper_eucl_mix(x_out_1,x_hypout_1)
        x_mix_2,comp_loss_2 = self.hyper_eucl_mix(x_out_2,x_hypout_2)
        self.z = [x_out_2, x_hypout_2]
        compare_loss = comp_loss_1+comp_loss_2
        return x_mix_1,x_mix_2,compare_loss














