import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

class EAGCN(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, structure = 'Concate', molfp_mode = 'sum', pool_num = 5):
        super(EAGCN, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.molfp_mode = molfp_mode

        if structure == 'Concate' or structure == 'GCN' or structure == 'GAT':
            self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        elif structure == 'Weighted_sum':
            outdim_sum_1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc1 = outdim_sum_1
            n_sgc1_1 = outdim_sum_1
            n_sgc1_2 = outdim_sum_1
            n_sgc1_3 = outdim_sum_1
            n_sgc1_4 = outdim_sum_1
            n_sgc1_5 = outdim_sum_1
            outdim_sum_2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
            self.ngc2 = outdim_sum_2
            n_sgc2_1 = outdim_sum_2
            n_sgc2_2 = outdim_sum_2
            n_sgc2_3 = outdim_sum_2
            n_sgc2_4 = outdim_sum_2
            n_sgc2_5 = outdim_sum_2

        if structure == 'Concate' or structure == 'Weighted_sum':
            self.layer1 = GraphConv_Layer(node_feature_in = n_afeat, bond_feature_num = n_bfeat, node_out_1 = n_sgc1_1,
                                      node_out_2 = n_sgc1_2, node_out_3 = n_sgc1_3, node_out_4 = n_sgc1_4,
                                      node_out_5 = n_sgc1_5, dropout = dropout, structure = structure)
            self.layer2 = GraphConv_Layer(node_feature_in = self.ngc1, bond_feature_num = n_bfeat, node_out_1 = n_sgc2_1,
                                      node_out_2 = n_sgc2_2, node_out_3 = n_sgc2_3, node_out_4 = n_sgc2_4,
                                      node_out_5 = n_sgc2_5, dropout = dropout, structure = structure)
            self.layer3 = GraphConv_Layer(node_feature_in=self.ngc2, bond_feature_num=n_bfeat, node_out_1=2*n_sgc2_1,
                                          node_out_2=2*n_sgc2_2, node_out_3=2*n_sgc2_3, node_out_4=2*n_sgc2_4,
                                          node_out_5=2*n_sgc2_5, dropout=dropout, structure=structure)
            self.layer4 = GraphConv_Layer(node_feature_in=2*self.ngc2, bond_feature_num=n_bfeat, node_out_1=2 * n_sgc2_1,
                                          node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
                                          node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last = True)

        elif structure == 'GCN':
            self.layer1 = Vanilla_GCN(node_feature_in = n_afeat, node_feature_out = self.ngc1, dropout = dropout)
            self.layer2 = Vanilla_GCN(node_feature_in = self.ngc1, node_feature_out = self.ngc2, dropout = dropout)
            self.layer3 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=self.ngc2, dropout=dropout)
            self.layer4 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=2*self.ngc2, dropout=dropout)

        elif structure == 'GAT':
            self.layer1 = GAT(node_feature_in = n_afeat, node_feature_out = self.ngc1, dropout = dropout)
            self.layer2 = GAT(node_feature_in = self.ngc1, node_feature_out = self.ngc2, dropout = dropout)
            self.layer3 = GAT(node_feature_in=self.ngc2, node_feature_out=self.ngc2, dropout=dropout)
            self.layer4 = GAT(node_feature_in=self.ngc2, node_feature_out=2*self.ngc2, dropout=dropout)

        # Compress atom representations to mol representation
        #self.molfp = MolFP(self.ngc2)

        #self.den1 = Dense(self.ngc2, n_den1)
        self.den1 = Dense(2*self.ngc2, n_den1)
        #self.den2 = Dense(n_den1, nclass)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if self.use_cuda:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2).cuda()
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2)
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        if self.molfp_mode == 'pool':
            self.pool1 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, pool_num)
            #self.pool2 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 4)
            self.pool3 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 1)
        self.dropout = dropout


    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt, size):
        x1, A = self.layer1(adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer2(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer3(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer4(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)

        #atom_representations = x2.view(-1, 2*self.ngc2).data.cpu()
        atom_representations = x2.data.cpu()

        #x = self.molfp(x2)

        if self.molfp_mode == 'pool':
            A, x = self.pool1(A, x2)
            #A, x = self.pool2(A, x)
            # A, x = self.pool3(A, x)
            # x = x.squeeze()
            x = torch.sum(x, 1)
        else:
            x = torch.sum(x2, 1)
            if self.molfp_mode == 'ave':
                size = size.view(x2.data.shape[0], 1).type(FloatTensor).expand(x2.data.shape[0], x2.data.shape[2])
                # size = size.view(x3.data.shape[0], 1).type(FloatTensor).expand(x3.data.shape[0], x3.data.shape[2])
                x = x / size
        x = self.Graph_BN(x)

        x = self.den1(x)
        #graph_representation = x
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        graph_representation = x
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x, atom_representations, graph_representation

'''
class EAGCN_6(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, structure = 'Concate', molfp_mode = 'sum', pool_num = 5):
        super(EAGCN_6, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.molfp_mode = molfp_mode

        if structure == 'Concate' or structure == 'GCN':
            self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        elif structure == 'Weighted_sum':
            outdim_sum_1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc1 = outdim_sum_1
            n_sgc1_1 = outdim_sum_1
            n_sgc1_2 = outdim_sum_1
            n_sgc1_3 = outdim_sum_1
            n_sgc1_4 = outdim_sum_1
            n_sgc1_5 = outdim_sum_1
            outdim_sum_2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
            self.ngc2 = outdim_sum_2
            n_sgc2_1 = outdim_sum_2
            n_sgc2_2 = outdim_sum_2
            n_sgc2_3 = outdim_sum_2
            n_sgc2_4 = outdim_sum_2
            n_sgc2_5 = outdim_sum_2

        if structure == 'Concate' or structure == 'Weighted_sum':
            self.layer1 = GraphConv_Layer(node_feature_in = n_afeat, bond_feature_num = n_bfeat, node_out_1 = n_sgc1_1,
                                      node_out_2 = n_sgc1_2, node_out_3 = n_sgc1_3, node_out_4 = n_sgc1_4,
                                      node_out_5 = n_sgc1_5, dropout = dropout, structure = structure)
            self.layer2 = GraphConv_Layer(node_feature_in = self.ngc1, bond_feature_num = n_bfeat, node_out_1 = n_sgc2_1,
                                      node_out_2 = n_sgc2_2, node_out_3 = n_sgc2_3, node_out_4 = n_sgc2_4,
                                      node_out_5 = n_sgc2_5, dropout = dropout, structure = structure)
            self.layer3 = GraphConv_Layer(node_feature_in=self.ngc2, bond_feature_num=n_bfeat, node_out_1=2*n_sgc2_1,
                                          node_out_2=2*n_sgc2_2, node_out_3=2*n_sgc2_3, node_out_4=2*n_sgc2_4,
                                          node_out_5=2*n_sgc2_5, dropout=dropout, structure=structure)
            self.layer4 = GraphConv_Layer(node_feature_in=2*self.ngc2, bond_feature_num=n_bfeat, node_out_1=2 * n_sgc2_1,
                                          node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
                                          node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last = False)
            self.layer5 = GraphConv_Layer(node_feature_in=2 * self.ngc2, bond_feature_num=n_bfeat,
                                          node_out_1=4 * n_sgc2_1,
                                          node_out_2=4 * n_sgc2_2, node_out_3=4 * n_sgc2_3, node_out_4=4 * n_sgc2_4,
                                          node_out_5=4 * n_sgc2_5, dropout=dropout, structure=structure, last=False)
            self.layer6 = GraphConv_Layer(node_feature_in=4 * self.ngc2, bond_feature_num=n_bfeat,
                                          node_out_1=4 * n_sgc2_1,
                                          node_out_2=4 * n_sgc2_2, node_out_3=4 * n_sgc2_3, node_out_4=4 * n_sgc2_4,
                                          node_out_5=4 * n_sgc2_5, dropout=dropout, structure=structure, last=True)

        elif structure == 'GCN':
            self.layer1 = Vanilla_GCN(node_feature_in = n_afeat, node_feature_out = self.ngc1, dropout = dropout)
            self.layer2 = Vanilla_GCN(node_feature_in = self.ngc1, node_feature_out = self.ngc2, dropout = dropout)
            self.layer3 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=self.ngc2, dropout=dropout)
            self.layer4 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=2*self.ngc2, dropout=dropout)

        # Compress atom representations to mol representation
        #self.molfp = MolFP(self.ngc2)

        #self.den1 = Dense(self.ngc2, n_den1)
        self.den1 = Dense(4*self.ngc2, n_den1)
        #self.den2 = Dense(n_den1, nclass)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if self.use_cuda:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2).cuda()
            self.Graph_BN = nn.BatchNorm1d(4*self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2)
            self.Graph_BN = nn.BatchNorm1d(4*self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        if self.molfp_mode == 'pool':
            self.pool1 = Diff_Pooling(4 * self.ngc2, 4 * self.ngc2, pool_num)
            #self.pool2 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 4)
            self.pool3 = Diff_Pooling(4 * self.ngc2, 4 * self.ngc2, 1)
        self.dropout = dropout


    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt, size):
        x1, A = self.layer1(adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer2(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer3(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer4(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer5(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer6(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)

        #atom_representations = x2.view(-1, 2*self.ngc2).data.cpu()
        atom_representations = x2.data.cpu()

        #x = self.molfp(x2)

        if self.molfp_mode == 'pool':
            A, x = self.pool1(A, x2)
            #A, x = self.pool2(A, x)
            # A, x = self.pool3(A, x)
            # x = x.squeeze()
            x = torch.sum(x, 1)
        else:
            x = torch.sum(x2, 1)
            if self.molfp_mode == 'ave':
                size = size.view(x2.data.shape[0], 1).type(FloatTensor).expand(x2.data.shape[0], x2.data.shape[2])
                # size = size.view(x3.data.shape[0], 1).type(FloatTensor).expand(x3.data.shape[0], x3.data.shape[2])
                x = x / size
        x = self.Graph_BN(x)

        x = self.den1(x)
        #graph_representation = x
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        graph_representation = x
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x, atom_representations, graph_representation

class EAGCN_res(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, structure = 'Concate', molfp_mode = 'sum', pool_num = 5):
        super(EAGCN_res, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.molfp_mode = molfp_mode

        if structure == 'Concate' or structure == 'GCN':
            self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        elif structure == 'Weighted_sum':
            outdim_sum_1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc1 = outdim_sum_1
            n_sgc1_1 = outdim_sum_1
            n_sgc1_2 = outdim_sum_1
            n_sgc1_3 = outdim_sum_1
            n_sgc1_4 = outdim_sum_1
            n_sgc1_5 = outdim_sum_1
            outdim_sum_2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
            self.ngc2 = outdim_sum_2
            n_sgc2_1 = outdim_sum_2
            n_sgc2_2 = outdim_sum_2
            n_sgc2_3 = outdim_sum_2
            n_sgc2_4 = outdim_sum_2
            n_sgc2_5 = outdim_sum_2

        if structure == 'Concate' or structure == 'Weighted_sum':
            self.layer1 = GraphConv_Layer(node_feature_in = n_afeat, bond_feature_num = n_bfeat, node_out_1 = n_sgc1_1,
                                      node_out_2 = n_sgc1_2, node_out_3 = n_sgc1_3, node_out_4 = n_sgc1_4,
                                      node_out_5 = n_sgc1_5, dropout = dropout, structure = structure)
            self.layer2 = GraphConv_Layer(node_feature_in = self.ngc1, bond_feature_num = n_bfeat, node_out_1 = n_sgc2_1,
                                      node_out_2 = n_sgc2_2, node_out_3 = n_sgc2_3, node_out_4 = n_sgc2_4,
                                      node_out_5 = n_sgc2_5, dropout = dropout, structure = structure)
            self.layer3 = GraphConv_Layer(node_feature_in=self.ngc2, bond_feature_num=n_bfeat, node_out_1=2*n_sgc2_1,
                                          node_out_2=2*n_sgc2_2, node_out_3=2*n_sgc2_3, node_out_4=2*n_sgc2_4,
                                          node_out_5=2*n_sgc2_5, dropout=dropout, structure=structure)
            self.layer4 = GraphConv_Layer(node_feature_in=2*self.ngc2, bond_feature_num=n_bfeat, node_out_1=2 * n_sgc2_1,
                                          node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
                                          node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last = False)
            self.layer5 = GraphConv_Layer(node_feature_in=2 * self.ngc2, bond_feature_num=n_bfeat,
                                          node_out_1=2 * n_sgc2_1,
                                          node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
                                          node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last=False)
            self.layer6 = GraphConv_Layer(node_feature_in=2 * self.ngc2, bond_feature_num=n_bfeat,
                                          node_out_1=2 * n_sgc2_1,
                                          node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
                                          node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last=True)

        elif structure == 'GCN':
            self.layer1 = Vanilla_GCN(node_feature_in = n_afeat, node_feature_out = self.ngc1, dropout = dropout)
            self.layer2 = Vanilla_GCN(node_feature_in = self.ngc1, node_feature_out = self.ngc2, dropout = dropout)
            self.layer3 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=self.ngc2, dropout=dropout)
            self.layer4 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=2*self.ngc2, dropout=dropout)

        # Compress atom representations to mol representation
        #self.molfp = MolFP(self.ngc2)

        #self.den1 = Dense(self.ngc2, n_den1)
        self.den1 = Dense(2*self.ngc2, n_den1)
        #self.den2 = Dense(n_den1, nclass)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if self.use_cuda:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2).cuda()
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2)
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        if self.molfp_mode == 'pool':
            self.pool1 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, pool_num)
            #self.pool2 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 4)
            self.pool3 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 1)
        self.dropout = dropout


    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt, size):
        x1, A = self.layer1(adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x1, A = self.layer2(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x1, A = self.layer3(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer4(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer5(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer6(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)

        x2 = x1+ x2
        #atom_representations = x2.view(-1, 2*self.ngc2).data.cpu()
        atom_representations = x2.data.cpu()

        #x = self.molfp(x2)

        if self.molfp_mode == 'pool':
            A, x = self.pool1(A, x2)
            #A, x = self.pool2(A, x)
            # A, x = self.pool3(A, x)
            # x = x.squeeze()
            x = torch.sum(x, 1)
        else:
            x = torch.sum(x2, 1)
            if self.molfp_mode == 'ave':
                size = size.view(x2.data.shape[0], 1).type(FloatTensor).expand(x2.data.shape[0], x2.data.shape[2])
                # size = size.view(x3.data.shape[0], 1).type(FloatTensor).expand(x3.data.shape[0], x3.data.shape[2])
                x = x / size
        x = self.Graph_BN(x)

        x = self.den1(x)
        #graph_representation = x
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        graph_representation = x
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x, atom_representations, graph_representation


class EAGCN_pool(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, structure = 'Concate', molfp_mode = 'sum', pool_num = 4):
        super(EAGCN_pool, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.molfp_mode = molfp_mode

        if structure == 'Concate' or structure == 'GCN':
            self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        elif structure == 'Weighted_sum':
            outdim_sum_1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
            self.ngc1 = outdim_sum_1
            n_sgc1_1 = outdim_sum_1
            n_sgc1_2 = outdim_sum_1
            n_sgc1_3 = outdim_sum_1
            n_sgc1_4 = outdim_sum_1
            n_sgc1_5 = outdim_sum_1
            outdim_sum_2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
            self.ngc2 = outdim_sum_2
            n_sgc2_1 = outdim_sum_2
            n_sgc2_2 = outdim_sum_2
            n_sgc2_3 = outdim_sum_2
            n_sgc2_4 = outdim_sum_2
            n_sgc2_5 = outdim_sum_2

        if structure == 'Concate' or structure == 'Weighted_sum':
            self.layer1 = GraphConv_Layer(node_feature_in = n_afeat, bond_feature_num = n_bfeat, node_out_1 = n_sgc1_1,
                                      node_out_2 = n_sgc1_2, node_out_3 = n_sgc1_3, node_out_4 = n_sgc1_4,
                                      node_out_5 = n_sgc1_5, dropout = dropout, structure = structure)
            self.layer2 = GraphConv_Layer(node_feature_in = self.ngc1, bond_feature_num = n_bfeat, node_out_1 = n_sgc2_1,
                                      node_out_2 = n_sgc2_2, node_out_3 = n_sgc2_3, node_out_4 = n_sgc2_4,
                                      node_out_5 = n_sgc2_5, dropout = dropout, structure = structure)
            self.layer3 = GraphConv_Layer(node_feature_in=self.ngc2, bond_feature_num=n_bfeat, node_out_1=2*n_sgc2_1,
                                          node_out_2=2*n_sgc2_2, node_out_3=2*n_sgc2_3, node_out_4=2*n_sgc2_4,
                                          node_out_5=2*n_sgc2_5, dropout=dropout, structure=structure, last = True)
            # self.layer4 = GraphConv_Layer(node_feature_in=2*self.ngc2, bond_feature_num=n_bfeat, node_out_1=2 * n_sgc2_1,
            #                               node_out_2=2 * n_sgc2_2, node_out_3=2 * n_sgc2_3, node_out_4=2 * n_sgc2_4,
            #                               node_out_5=2 * n_sgc2_5, dropout=dropout, structure=structure, last = True)

        elif structure == 'GCN':
            self.layer1 = Vanilla_GCN(node_feature_in = n_afeat, node_feature_out = self.ngc1, dropout = dropout)
            self.layer2 = Vanilla_GCN(node_feature_in = self.ngc1, node_feature_out = self.ngc2, dropout = dropout)
            self.layer3 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=self.ngc2, dropout=dropout)
            self.layer4 = Vanilla_GCN(node_feature_in=self.ngc2, node_feature_out=2*self.ngc2, dropout=dropout)

        # Compress atom representations to mol representation
        #self.molfp = MolFP(self.ngc2)

        #self.den1 = Dense(self.ngc2, n_den1)
        self.den1 = Dense(2*self.ngc2, n_den1)
        #self.den2 = Dense(n_den1, nclass)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if self.use_cuda:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2).cuda()
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            #self.Graph_BN = nn.BatchNorm1d(self.ngc2)
            self.Graph_BN = nn.BatchNorm1d(2*self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        if self.molfp_mode == 'pool':
            self.pool1 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, pool_num)
            self.pool2 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 1)
            # self.pool3 = Diff_Pooling(2 * self.ngc2, 2 * self.ngc2, 1)
        self.dropout = dropout


    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt, size):
        x1, A = self.layer1(adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer2(adjs, x1, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        x2, A = self.layer3(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)
        # x2, A = self.layer4(adjs, x2, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt)

        #atom_representations = x2.view(-1, 2*self.ngc2).data.cpu()
        atom_representations = x2.data.cpu()

        #x = self.molfp(x2)

        if self.molfp_mode == 'pool':
            A, x = self.pool1(A, x2)
            A, x = self.pool2(A, x)
            # A, x = self.pool3(A, x)
            # x = x.squeeze()
            x = torch.sum(x, 1)
        else:
            x = torch.sum(x2, 1)
            if self.molfp_mode == 'ave':
                size = size.view(x2.data.shape[0], 1).type(FloatTensor).expand(x2.data.shape[0], x2.data.shape[2])
                # size = size.view(x3.data.shape[0], 1).type(FloatTensor).expand(x3.data.shape[0], x3.data.shape[2])
                x = x / size
        x = self.Graph_BN(x)

        x = self.den1(x)
        #graph_representation = x
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        graph_representation = x
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x, atom_representations, graph_representation
'''