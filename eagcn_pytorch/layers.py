import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

class GraphConv_base(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv_base, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjs, afms):
        support = torch.bmm(adjs, afms)
        result = torch.mm(support.view(-1, self.in_features), self.weight)
        output = result.view(-1, adjs.data.shape[1], self.out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConv_block(Module):
    """
    A block deal with one bond relation
    """

    def __init__(self, node_feature_in, bond_feature_num, node_feature_out, dropout):
        super(GraphConv_block, self).__init__()
        self.node_feature_in = node_feature_in
        self.bond_feature_num = bond_feature_num
        self.node_feature_out = node_feature_out
        self.dropout = dropout

        self.att = nn.Conv2d(bond_feature_num, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        #self.soft = nn.Softmax(dim=2)
        self.graph_conv = GraphConv_base(node_feature_in, node_feature_out, bias=True)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.batch_norm = AFM_BatchNorm(node_feature_out).cuda()
        else:
            self.batch_norm = AFM_BatchNorm(node_feature_out)
        self.dropout = dropout

        self.self_r = Parameter(FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1.# / math.sqrt(self.self_r.size(0))
        self.self_r.data.uniform_(-0.01, 0.01)

    def forward(self, adjs, afms, bond_relation_tensor, mask_tiny, mask2, identity):
        A1 = self.att(bond_relation_tensor.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1 = torch.sigmoid(A1) * adjs
        A =  A1 + (torch.sigmoid(self.self_r)) * identity  + mask_tiny
        # A = torch.sigmoid(A) * adjs + (torch.sigmoid(self.self_r) + 1) * identity + mask_tiny
        #A = torch.sigmoid(A) * adjs +  identity + mask_tiny
        A_rowsum = torch.sum(A, dim=2, keepdim=True).expand(adjs.data.shape[0], adjs.data.shape[1],
                                                                  adjs.data.shape[2])
        #r_inv = torch.pow(A_rowsum, -1)
        A = (A / A_rowsum) * mask2

        x = self.graph_conv(A, afms)
        x = F.relu(self.batch_norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, A1



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        #print(input.size())
        #print(input)
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        #zero_vec = 0 * torch.ones_like(e)
        attention1 = torch.where(adj > 0, e, zero_vec)
        attention2 = F.softmax(attention1, dim=1)
        zero = 0 * torch.ones_like(e)
        attention = torch.where(adj > 0, attention2, zero)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return h_prime
        #if self.concat:
        #    return F.elu(h_prime)
        #else:
        #    return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(Module):
    """
    The GCN by THOMAS KIPF
    """
    def __init__(self, node_feature_in, node_feature_out, dropout):
        super(GAT, self).__init__()
        self.node_feature_in = node_feature_in
        #self.bond_feature_num = bond_feature_num
        self.node_feature_out = node_feature_out
        self.dropout = dropout
        #self.self_r = Parameter(FloatTensor(1))
        #self.reset_parameters()

        #self.att = nn.Conv2d(bond_feature_num, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        #self.soft = nn.Softmax(dim=2)
        self.graph_conv = GraphAttentionLayer(node_feature_in, node_feature_out)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.batch_norm = AFM_BatchNorm(node_feature_out).cuda()
        else:
            self.batch_norm = AFM_BatchNorm(node_feature_out)
        self.dropout = dropout

    #def reset_parameters(self):
    #    stdv = 1. / math.sqrt(self.self_r.size(0))
    #    self.self_r.data.uniform_(-stdv, stdv)

    #def forward(self, adjs, afms, bond_relation_tensor, mask_tiny, mask2, identity):
    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt):
        #A = self.att(bond_relation_tensor.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        #A = torch.sigmoid(A) * adjs + (torch.sigmoid(self.self_r) + 1) * identity  + mask_tiny
        #A = torch.sigmoid(A) * adjs +  identity + mask_tiny
        #mask_tiny = (1. - adjs) * 1e-9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        #mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.total_output)
        if self.use_cuda:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1],
                                         adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1])).cuda()
        else:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1],
                                         adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1]))

        A = adjs + identity #+ mask_tiny
        #A_rowsum = torch.sum(A, dim=2, keepdim=True).expand(adjs.data.shape[0], adjs.data.shape[1],adjs.data.shape[2])
        #A = (A / A_rowsum) * mask2

        bz = A.size()[0]
        x = torch.stack([self.graph_conv(afms[i], A[i]) for i in range(bz)])
        #x = self.graph_conv(A, afms)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)

        return x, A




class Vanilla_GCN(Module):
    """
    The GCN by THOMAS KIPF
    """
    def __init__(self, node_feature_in, node_feature_out, dropout):
        super(Vanilla_GCN, self).__init__()
        self.node_feature_in = node_feature_in
        #self.bond_feature_num = bond_feature_num
        self.node_feature_out = node_feature_out
        self.dropout = dropout
        #self.self_r = Parameter(FloatTensor(1))
        #self.reset_parameters()

        #self.att = nn.Conv2d(bond_feature_num, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        #self.soft = nn.Softmax(dim=2)
        self.graph_conv = GraphConv_base(node_feature_in, node_feature_out, bias=True)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.batch_norm = AFM_BatchNorm(node_feature_out).cuda()
        else:
            self.batch_norm = AFM_BatchNorm(node_feature_out)
        self.dropout = dropout

    #def reset_parameters(self):
    #    stdv = 1. / math.sqrt(self.self_r.size(0))
    #    self.self_r.data.uniform_(-stdv, stdv)

    #def forward(self, adjs, afms, bond_relation_tensor, mask_tiny, mask2, identity):
    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt):
        #A = self.att(bond_relation_tensor.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        #A = torch.sigmoid(A) * adjs + (torch.sigmoid(self.self_r) + 1) * identity  + mask_tiny
        #A = torch.sigmoid(A) * adjs +  identity + mask_tiny
        mask_tiny = (1. - adjs) * 1e-9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        #mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.total_output)
        if self.use_cuda:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1],
                                         adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1])).cuda()
        else:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1],
                                         adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1]))

        A = adjs + identity + mask_tiny
        A_rowsum = torch.sum(A, dim=2, keepdim=True).expand(adjs.data.shape[0], adjs.data.shape[1],
                                                                  adjs.data.shape[2])
        A = (A / A_rowsum) * mask2

        x = self.graph_conv(A, afms)
        x = F.relu(self.batch_norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, A



class GraphConv_Layer(Module):
    """
    A layer to deal with all bond relations (multiple blocks)
    """
    def __init__(self, node_feature_in, bond_feature_num, node_out_1, node_out_2, node_out_3, node_out_4,
                 node_out_5, dropout, structure, last = False, adj_size = 0):
        super(GraphConv_Layer, self).__init__()
        self.block1 = GraphConv_block(node_feature_in, bond_feature_num, node_out_1, dropout)
        self.block2 = GraphConv_block(node_feature_in, 4, node_out_2, dropout)
        self.block3 = GraphConv_block(node_feature_in, 2, node_out_3, dropout)
        self.block4 = GraphConv_block(node_feature_in, 2, node_out_4, dropout)
        self.block5 = GraphConv_block(node_feature_in, 2, node_out_5, dropout)
        self.use_cuda = torch.cuda.is_available()
        self.structure = structure
        self.last = last
        if structure == 'Concate':
            self.total_output = node_out_1 + node_out_2 + node_out_3 + node_out_4 + node_out_5
        elif structure == 'Weighted_sum':
            self.total_output = node_out_1
            self.ave = Ave_multi_view(5)
        else:
            print('error, structure not support')

        # need to revise self.total_output
        self.ave_A = Ave_multi_view(5)
        self.self_r = Parameter(FloatTensor(1))
        self.reset_parameters()
    #
    def reset_parameters(self):
        self.self_r.data.uniform_(-0.01, 0.01)

    def forward(self, adjs, afms, TypeAtt, OrderAtt, AromAtt, ConjAtt, RingAtt):
        mask_tiny = (1. - adjs) * 1e-9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.total_output)
        if self.use_cuda:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1])).cuda()
        else:
            identity = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1],
                                         adjs.data.shape[2]) * torch.FloatTensor(
                torch.eye(n=adjs.data.shape[1]))

        x1, A1 = self.block1(adjs, afms, TypeAtt, mask_tiny, mask2, identity)
        x2, A2 = self.block2(adjs, afms, OrderAtt, mask_tiny, mask2, identity)
        x3, A3 = self.block3(adjs, afms, AromAtt, mask_tiny, mask2, identity)
        x4, A4 = self.block4(adjs, afms, ConjAtt, mask_tiny, mask2, identity)
        x5, A5 = self.block5(adjs, afms, RingAtt, mask_tiny, mask2, identity)

        if self.structure == 'Concate':
            x = torch.cat((x1, x2, x3, x4, x5), dim=2) * mask3
        elif self.structure == 'Weighted_sum':
            x = torch.stack((x1, x2, x3, x4, x5), dim=0)
            x = self.ave(x)

        A_weight = torch.stack((A1, A2, A3, A4, A5), dim=0)
        if self.last:
            A_weight = self.ave_A(A_weight)
            A_weight = torch.sigmoid(A_weight) * adjs + (torch.sigmoid(self.self_r)) * identity + mask_tiny
            A_rowsum = torch.sum(A_weight, dim=2, keepdim=True).expand(adjs.data.shape[0], adjs.data.shape[1],
                                                                       adjs.data.shape[2])
            A_weight = (A_weight / A_rowsum) * mask2
        return x, A_weight

class MolFP(Module):
    """
    Add node repressentations within a graph to get the representation of graph
    """

    def __init__(self, out_features, bias=True):
        super(MolFP, self).__init__()
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(1, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.sum(input, 1) # 0 is batchsize, 1 is atom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Dense(Module):
    """
    Full Connected Layer, take graph representation as input, compressed to target length at final step.
    """

    def __init__(self, in_features, out_features, bias=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class AFM_BatchNorm(Module):
    """
        For a 3D tensor about atom feature, batch_size * node_num * feature_size
        Batchnorm along feature_size for atom feature tensor.
    """
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True, bias = True):
        super(AFM_BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.weight = Parameter(FloatTensor(1, 1))
        if bias:
            self.bias = Parameter(FloatTensor(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn(x.contiguous())
        x = x.permute(0,2,1)
        return x

class Ave_multi_view(Module):
    """
    Weighted average the Adjacent matrix or atom features updated by different relations.
    """

    def __init__(self, ave_source_num, feature_size = 0, bias=False):
        super(Ave_multi_view, self).__init__()
        self.ave_source_num = ave_source_num
        self.feature_size = feature_size
        self.weight = Parameter(FloatTensor(ave_source_num))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        # self.weight.data.uniform_(0.99, 1.01)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.data.shape[1]
        mol_size = input.data.shape[2]
        feature_size = input.data.shape[3]
        weight_expand = self.weight.view(self.ave_source_num, 1, 1, 1).expand(self.ave_source_num, batch_size, mol_size, feature_size)
        output = torch.sum(input * weight_expand, 0)
        return output

class AX_W(Module):
    """
    update the AX by multipling a weight matrix, second part of GCN layer.
    """

    def __init__(self, in_features, out_features, bias=False):
        super(AX_W, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ax):
        result = torch.mm(ax.view(-1, self.in_features), self.weight)
        output = result.view(-1, ax.data.shape[1], self.out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


'''
class Diff_Pooling(Module):
    def __init__(self, in_feature, out_feature, out_size):
        super(Diff_Pooling, self).__init__()
        self.feature_layer = GraphConv_base(in_feature, out_feature)
        self.adjacent_layer = GraphConv_base(in_feature, out_size)

    def forward(self, A, X):
        X_feature = F.relu(self.feature_layer(A, X))
        S = F.softmax(self.adjacent_layer(A, X), dim = 2)
        S_T = torch.transpose(S, 1, 2)
        X_feature =  F.relu(torch.bmm(S_T, X_feature))
        A_update = torch.bmm(torch.bmm(S_T, A), S)
        A_update = F.dropout(A_update, p = 0.0)
        return A_update, X_feature
'''

class Diff_Pooling(Module):
    def __init__(self, in_feature, out_feature, out_size):
        super(Diff_Pooling, self).__init__()
        self.feature_layer = GraphConv_base(in_feature, out_feature)
        self.adjacent_layer = GraphConv_base(in_feature, out_size)

    def forward(self, A, X):
        X_feature = F.relu(self.feature_layer(A, X))
        S = F.softmax(self.adjacent_layer(A, X), dim = 2)
        S_T = torch.transpose(S, 1, 2)
        #X_feature = torch.bmm(S_T, X_feature)
        X_feature = F.relu(torch.bmm(S_T, X_feature))
        A_update = torch.bmm(torch.bmm(S_T, A), S)
        A_update = F.dropout(A_update, p = 0.3)
        return A_update, X_feature




