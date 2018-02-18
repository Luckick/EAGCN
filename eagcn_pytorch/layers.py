import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
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
    Weighted average the atom features updated by different relations.
    """

    def __init__(self, ave_source_num, feature_size, bias=False):
        super(Ave_multi_view, self).__init__()
        self.ave_source_num = ave_source_num
        self.feature_size = feature_size
        self.weight = Parameter(FloatTensor(ave_source_num))
        if bias:
            self.bias = Parameter(FloatTensor(feature_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.data.shape[1]
        mol_size = input.data.shape[2]
        weight_expand = self.weight.view(self.ave_source_num, 1, 1, 1).expand(self.ave_source_num, batch_size, mol_size, self.feature_size)
        output = torch.sum(input * weight_expand, 0)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str('{} * {}'.format(self.ave_source_num, self.feature_size)) + ' -> ' \
               + str('1 * {}'.format(self.feature_size)) + ')'
