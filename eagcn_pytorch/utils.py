import numpy as np
import scipy.sparse as sp
import torch
import csv
from rdkit.Chem import MolFromSmiles, SDMolSupplier
from torch.utils.data import Dataset
from neural_fp import *
import math
import os
import scipy
from sklearn.utils import shuffle, resample
import pickle
import openbabel
import pybel
import operator
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def load_data(dataset, path = '../data/'):
    if dataset == 'tox21':
        x_all, y_all, target, sizes = load_dc_tox21(path=path, keep_nan=True)
    elif dataset == 'hiv':
        x_all, y_all, target, sizes = load_hiv(path=path, keep_nan=True)
    elif dataset == 'lipo':
        x_all, y_all, target, sizes = load_lipo()
    elif dataset == 'freesolv':
        x_all, y_all, target, sizes = load_freesolv()
    return(x_all, y_all, target, sizes)

def load_lipo(path='../data/', dataset = 'Lipophilicity.csv', bondtype_freq = 10,
                    atomtype_freq=10):
    print('Loading {} dataset...'.format(dataset))
    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        for row in reader:
            data.append(row)
    print('done')

    target = data[0][1]
    labels = []
    mol_sizes = []
    error_row = []
    bondtype_dic, atomtype_dic = got_all_Type_solu_dic(dataset)

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    x_all = []
    count_1 = 0
    count_2 = 0
    for i in range(1, len(data)):
        mol = MolFromSmiles(data[i][2])
        count_1 += 1
        try:
            (afm, adj, bft, adjTensor_OrderAtt,
             adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt) = molToGraph(mol, filted_bondtype_list_order,
                                                                                   filted_atomtype_list_order).dump_as_matrices_Att()
            mol_sizes.append(adj.shape[0])
            labels.append([np.float32(data[i][1])])
            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt])
            count_2 += 1
        except AttributeError:
            print('the {}th row has an error'.format(i))
            error_row.append(i)
        except TypeError:
            print('the {}th row smile is: {}, can not convert to graph structure'.format(i, data[i][2]))
            error_row.append(i)
        else:
            pass
        i += 1

    x_all = feature_normalize(x_all)
    print('Done.')

    return(x_all, labels, target, mol_sizes)

def load_freesolv(path='../data/', dataset = 'SAMPL.csv', bondtype_freq = 3,
                    atomtype_freq=3):
    print('Loading {} dataset...'.format(dataset))
    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        for row in reader:
            data.append(row)
    print('done')

    target = data[0][2]
    labels = []
    mol_sizes = []
    error_row = []
    bondtype_dic, atomtype_dic = got_all_Type_solu_dic(dataset)

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    x_all = []
    count_1 = 0
    count_2 = 0
    for i in range(1, len(data)):
        mol = MolFromSmiles(data[i][1])
        count_1 += 1
        try:
            (afm, adj, bft, adjTensor_OrderAtt,
             adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt) = molToGraph(mol, filted_bondtype_list_order,
                                         filted_atomtype_list_order).dump_as_matrices_Att()
            mol_sizes.append(adj.shape[0])
            labels.append([np.float32(data[i][2])])
            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt])
            count_2 +=1
        except AttributeError:
            print('the {}th row has an error'.format(i))
            error_row.append(i)
        except TypeError:
            print('the {}th row smile is: {}, can not convert to graph structure'.format(i, data[i][1]))
            error_row.append(i)
        i += 1

    x_all = feature_normalize(x_all)
    print('Done.')

    return(x_all, labels, target, mol_sizes)

def load_dc_tox21(path='../data/', dataset = 'tox21.csv', bondtype_freq =20, atomtype_freq =10, keep_nan=True):
    print('Loading {} dataset...'.format(dataset))
    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        for row in reader:
            data.append(row)

    label_name = data[0][0:12]
    target = label_name

    bondtype_dic, atomtype_dic = got_all_Type_solu_dic(dataset)

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order,
                                                                                     filted_bondtype_list_order))

    # mol to graph
    i = 0
    mol_sizes = []
    x_all = []
    y_all = []
    print('Transfer mol to matrices')
    for row in data[1:]:
        smile = row[13]
        mol = MolFromSmiles(smile)

        label = row[0:12]
        label = ['nan' if ele=='' else ele for ele in label]
        num_label = [float(x) for x in label]
        num_label = [-1 if math.isnan(x) else x for x in num_label]

        idx = i+1
        i = i + 1
        try:
            (afm, adj, bft, adjTensor_OrderAtt,
             adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt) = molToGraph(mol, filted_bondtype_list_order,
                                                                                   filted_atomtype_list_order).dump_as_matrices_Att()
            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt])
            y_all.append(num_label)
            mol_sizes.append(adj.shape[0])
            # feature matrices of mols, include Adj Matrix, Atom Feature, Bond Feature.
        except AttributeError:
            print('the {}th row has an error'.format(i))
        except TypeError:
            print('the {}th row smile is: {}, can not convert to graph structure'.format(i, smile))
        else:
            pass
    x_all = feature_normalize(x_all)
    print('Done.')
    return (x_all, y_all, target, mol_sizes)

def load_hiv(path='../data/', dataset = 'HIV.csv', bondtype_freq =20, atomtype_freq =10, keep_nan=True):
    print('Loading {} dataset...'.format(dataset))
    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        for row in reader:
            data.append(row)

    label_name = data[0][2]
    target = [label_name]

    bondtype_dic, atomtype_dic = got_all_Type_solu_dic(dataset)

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    # mol to graph
    i = 0
    mol_sizes = []
    x_all = []
    y_all = []
    print('Transfer mol to matrices')
    for row in data[1:]:
        i+=1
        if len(row) ==0:
            continue
        smile = row[0]
        label = row[2]
        label = [label]
        label = ['nan' if ele=='' else ele for ele in label]
        num_label = [float(x) for x in label]
        num_label = [-1 if math.isnan(x) else x for x in num_label]
        try:
            mol = MolFromSmiles(smile)
            (afm, adj, bft, adjTensor_OrderAtt,
             adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt) = molToGraph(mol, filted_bondtype_list_order,
                                                                                   filted_atomtype_list_order).dump_as_matrices_Att()
            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt])
            y_all.append(num_label)
            mol_sizes.append(adj.shape[0])
            # feature matrices of mols, include Adj Matrix, Atom Feature, Bond Feature.
        except AttributeError:
            print('the {}th row has an error'.format(i))
        except ValueError:
            print('the {}th row smile is: {}, can not convert to graph structure'.format(i, smile))
        else:
            pass
    x_all = feature_normalize(x_all)
    print('Done.')
    return (x_all, y_all, target, mol_sizes)

def data_filter(x_all, y_all, target, sizes, tasks, size_cutoff=1000):
    idx_row = []
    for i in range(0, len(sizes)):
        if sizes[i] <= size_cutoff:
            idx_row.append(i)
    x_select = [x_all[i] for i in idx_row]
    y_select = [y_all[i] for i in idx_row]

    idx_col = []
    for task in tasks:
        for i in range(0, len(target)):
            if task == target[i]:
                idx_col.append(i)
    y_task = [[each_list[i] for i in idx_col] for each_list in y_select]

    return(x_select, y_task)

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx_abs = np.absolute(mx)
    #rowsum = np.array(mx.sum(1))
    rowsum = np.array(mx_abs.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    feature_num = x_all[0][0].shape[1]
    feature_min_dic = {}
    feature_max_dic = {}
    for i in range(len(x_all)):
        afm = x_all[i][0]
        afm_min = afm.min(0)
        afm_max = afm.max(0)
        for j in range(feature_num):
            if j not in feature_max_dic.keys():
                feature_max_dic[j] = afm_max[j]
                feature_min_dic[j] = afm_min[j]
            else:
                if feature_max_dic[j] < afm_max[j]:
                    feature_max_dic[j] = afm_max[j]
                if feature_min_dic[j] > afm_min[j]:
                    feature_min_dic[j] = afm_min[j]

    for i in range(len(x_all)):
        afm = x_all[i][0]
        feature_diff_dic = {}
        for j in range(feature_num):
            feature_diff_dic[j] = feature_max_dic[j]-feature_min_dic[j]
            if feature_diff_dic[j] ==0:
                feature_diff_dic[j] = 1
            afm[:,j] = (afm[:,j] - feature_min_dic[j])/(feature_diff_dic[j])
        x_all[i][0] = afm
    return x_all

class MolDatum():
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x, label, target, index):
        self.adj = x[1]
        self.afm = x[0]
        self.bft = x[2]
        self.orderAtt = x[3]
        self.aromAtt = x[4]
        self.conjAtt = x[5]
        self.ringAtt = x[6]
        self.label = label
        self.target = target
        self.index = index

def construct_dataset(x_all, y_all, target):
    output = []
    for i in range(len(x_all)):
        output.append(MolDatum(x_all[i], y_all[i], target, i))
    return(output)

class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of MolDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        adj, afm, bft, orderAtt, aromAtt, conjAtt, ringAtt  = self.data_list[key].adj, self.data_list[key].afm, self.data_list[key].bft, \
                                                              self.data_list[key].orderAtt, self.data_list[key].aromAtt, self.data_list[key].conjAtt, self.data_list[key].ringAtt
        label = self.data_list[key].label
        return (adj, afm, bft, orderAtt, aromAtt, conjAtt, ringAtt, label)

def mol_collate_func_reg(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    bft_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []
    for datum in batch:
        label_list.append(datum[7])
        size_list.append(datum[0].shape[0])
    max_size = np.max(size_list) # max of batch. 55 for solu, 115 for lipo, 24 for freesolv
    #max_size = max_molsize #max_molsize 132
    btf_len = datum[2].shape[0]
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 25), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((5, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        bft_list.append(filled_bft)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)


    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(bft_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 torch.from_numpy(np.array(label_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
                 torch.from_numpy(np.array(bft_list)), torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 torch.from_numpy(np.array(label_list))])

def mol_collate_func_class(batch):

    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    bft_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []

    for datum in batch:
        label_list.append(datum[7])
        size_list.append(datum[0].shape[0])
    max_size = np.max(size_list) # max of batch    222 for hiv, 132 for tox21,
    btf_len = datum[2].shape[0]
    #max_size = max_molsize #max_molsize 132
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 25), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((5, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        bft_list.append(filled_bft)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)

    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(bft_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 FloatTensor(label_list)])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
             torch.from_numpy(np.array(bft_list)),torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 FloatTensor(label_list)])

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def weight_tensor(weights, labels):
    # when labels is variable
    weight_tensor = []
    a = IntTensor([1])
    b = IntTensor([0])
    for i in range(0, labels.data.shape[0]):
        for j in range(0, labels.data.shape[1]):
            if torch.equal(IntTensor([int(labels.data[i][j])]), a):
                weight_tensor.append(weights[j][0])
            elif torch.equal(IntTensor([int(labels.data[i][j])]), b):
                weight_tensor.append(weights[j][1])
            else:
                weight_tensor.append(0)
    if use_cuda:
        return (torch.from_numpy(np.array(weight_tensor, dtype=np.float32)).cuda())
    else:
        return(torch.from_numpy(np.array(weight_tensor, dtype=np.float32)))

def set_weight(y_all):
    weight_dic = {}
    pos_dic ={}
    neg_dic = {}
    for i in range(len(y_all)):
        for j in range(len(y_all[0])):
            if y_all[i][j] == 1:
                if pos_dic.get(j) is None:
                    pos_dic[j] = 1
                else:
                    pos_dic[j] += 1
            elif y_all[i][j] == 0:
                if neg_dic.get(j) is None:
                    neg_dic[j] = 1
                else:
                    neg_dic[j] += 1

    for key in pos_dic.keys():
        weight_dic[key] = [5000/pos_dic[key], 5000/neg_dic[key]]
    return(weight_dic)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(1.0)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def got_all_bondType_tox21(dataset, path='../data/'):

    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter='\t', quotechar="'")
        for row in reader:
            data.append(row)

    bondtype_list = []
    for row in data[1:11746]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        smile = row[0]
        mol = MolFromSmiles(smile)
        bondtype_list = fillBondType(mol, bondtype_list)
    bondtype_list.sort()
    return(bondtype_list)

def got_all_bondType_tox21_dic(dataset, path='../data/'):
    if dataset == 'coley_tox21.tdf':
        delimiter = '\t'
        quotechar = "'"
        smile_idx = 0
        len_data = 11746
    elif dataset == 'tox21.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 13
        len_data = 7832
    elif dataset =='HIV.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 0
        len_data = 82255

    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            data.append(row)

    bondtype_dic = {}
    for row in data[1:len_data]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        if len(row) == 0:
            continue
        smile = row[smile_idx]
        try:
            mol = MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
        except AttributeError:
            pass
    return(bondtype_dic)

def got_all_atomType_tox21_dic(dataset, path='../data/'):

    data = []
    if dataset == 'coley_tox21.tdf':
        delimiter = '\t'
        quotechar = "'"
        smile_idx = 0
        len_data = 11746
    elif dataset == 'tox21.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 13
        len_data = 7832
    elif dataset =='HIV.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 0
        len_data = 82255

    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            data.append(row)

    atomtype_dic = {}
    for row in data[1:len_data]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        if len(row) == 0:
            continue
        smile = row[smile_idx]
        try:
            mol = MolFromSmiles(smile)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
    return(atomtype_dic)

def got_all_Type_solu_dic(dataset, path='../data/'):
    if dataset == 'Lipophilicity.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 2
        len_data = 4201
    elif dataset =='HIV.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 0
        len_data = 82255
    elif dataset == 'SAMPL.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 1
        len_data = 643
    elif dataset == 'tox21.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 13
        len_data = 7832

    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            data.append(row)

    bondtype_dic = {}
    atomtype_dic = {}
    for row in data[1:len_data]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        if len(row) == 0:
            continue
        smile = row[smile_idx]
        try:
            mol = MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
        else:
            pass
    return(bondtype_dic, atomtype_dic)

def data_padding(x, max_size):
    btf_len = x[0][2].shape[0]
    x_padded = []
    for data in x:  # afm, adj, bft
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:data[1].shape[0], 0:data[1].shape[0]] = data[1]
        filled_afm = np.zeros((max_size, 25), dtype=np.float32)
        filled_afm[0:data[0].shape[0], :] = data[0]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:data[0].shape[0], 0:data[0].shape[0]] = data[2]

        x_padded.append([filled_adj, filled_afm, filled_bft])
    return(x_padded)

def construct_loader(x, y, target, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target)
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_class,
                                               shuffle=shuffle)
    return loader

def construct_loader_reg(x, y, target, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target)
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_reg,
                                               shuffle=shuffle)
    return loader

def earily_stop(val_acc_history, tasks, early_stop_step_single,
                early_stop_step_multi, required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here
    if len(tasks) == 1:
        t = early_stop_step_single
    else:
        t = early_stop_step_multi

    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False
