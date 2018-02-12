from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from models import *
from torch.utils.data import Dataset

from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import os

import matplotlib.pyplot as plt
from time import gmtime, strftime

# Training settings
dataset = 'tox21'   # 'tox21', 'hiv'
EAGCN_structure = 'concate' #  'concate', 'weighted_ave'
write_file = True
n_den1, n_den2= 64, 32

if dataset == 'tox21':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 256
    weight_decay = 0.0001  # L-2 Norm
    dropout  = 0.3
    random_state = 2
    num_epochs = 80
    learning_rate = 0.0005
if dataset == 'hiv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 64
    weight_decay = 0.00001  # L-2 Norm
    dropout = 0.3
    random_state = 1
    num_epochs = 50
    learning_rate = 0.0005

# Early Stopping:
early_stop_step_single = 3
early_stop_step_multi = 5
early_stop_required_progress = 0.001
early_stop_diff = 0.11

experiment_date = strftime("%b_%d_%H:%M", gmtime()) +'N'
print(experiment_date)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

# targets for  tox21
if dataset == 'tox21':
    all_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
if dataset == 'hiv':
    all_tasks = ['HIV_active']

def test_model(loader, model, tasks):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    true_value = []
    all_out = []
    model.eval()
    out_value_dic = {}
    true_value_dic = {}
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        probs = F.sigmoid(outputs)

        if use_cuda:
            out_list = probs.cpu().data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.cpu().numpy().tolist()
            true_value.extend([item for sublist in label_list for item in sublist])
            out_sep_list = probs.cpu().data.view(-1, len(tasks)).numpy().tolist()
        else:
            out_list = probs.data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.numpy().tolist()
            true_value.extend([item for sublist in label_list for item in sublist])
            out_sep_list = probs.data.view(-1, len(tasks)).numpy().tolist()

        for i in range(0, len(out_sep_list)):
            for j in list(range(0, len(tasks))):
                if label_list[i][j] == -1:
                    #print('Ignore {},{} case: nan'.format(i,j))
                    continue
                if j not in true_value_dic.keys():
                    out_value_dic[j] = [out_sep_list[i][j]]
                    true_value_dic[j] = [int(label_list[i][j])]
                else:
                    out_value_dic[j].extend([out_sep_list[i][j]])
                    true_value_dic[j].extend([int(label_list[i][j])])
    model.train()

    aucs = []
    for key in list(range(0, len(tasks))):
        fpr, tpr, threshold = metrics.roc_curve(true_value_dic[key], out_value_dic[key], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if math.isnan(auc):
            print('the {}th label has no postive samples, max value {}'.format(key, max(true_value_dic[key])))
        aucs.append(auc)
    return (aucs, sum(aucs)/len(aucs))

def train(tasks, EAGCN_structure, n_den1, n_den2, file_name):
    x_all, y_all, target, sizes = load_data(dataset)
    max_size = max(sizes)
    x_all, y_all = data_filter(x_all, y_all, target, sizes, tasks)
    x_all, y_all = shuffle(x_all, y_all, random_state=random_state)

    X, x_test, y, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=random_state)
    del x_all, y_all
    test_loader = construct_loader(x_test, y_test, target, batch_size)
    del x_test, y_test

    n_bfeat = X[0][2].shape[0]

    if EAGCN_structure == 'concate':
        model = Concate_GCN(n_bfeat=n_bfeat, n_afeat=25,
                            n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4,
                            n_sgc1_5=n_sgc1_5,
                            n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4,
                            n_sgc2_5=n_sgc2_5,
                            n_den1=n_den1, n_den2=n_den2,
                            nclass=len(tasks), dropout=dropout)
    else:
        model = Weighted_GCN(n_bfeat=n_bfeat, n_afeat=25,
                             n_sgc1_1 = n_sgc1_1, n_sgc1_2 = n_sgc1_2, n_sgc1_3= n_sgc1_3, n_sgc1_4 = n_sgc1_4, n_sgc1_5 = n_sgc1_5,
                             n_sgc2_1 = n_sgc2_1, n_sgc2_2 = n_sgc2_2, n_sgc2_3= n_sgc2_3, n_sgc2_4 = n_sgc2_4, n_sgc2_5 = n_sgc2_5,
                             n_den1=n_den1, n_den2=n_den2, nclass=len(tasks), dropout=dropout)
    if use_cuda:
        # lgr.info("Using the GPU")
        model.cuda()

    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    validation_acc_history = []
    stop_training = False
    BCE_weight = set_weight(y)

    X = np.array(X)
    y = np.array(y)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
    train_loader = construct_loader(x_train, y_train, target, batch_size)
    validation_loader = construct_loader(x_val, y_val, target, batch_size)
    len_train = len(x_train)
    del x_train, y_train, x_val, y_val

    for epoch in range(num_epochs):

        for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels) in enumerate(train_loader):
            adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
            orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                aromAtt), Variable(
                conjAtt), Variable(ringAtt)
            optimizer.zero_grad()
            outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                            ringAtt_batch)
            weights = Variable(weight_tensor(BCE_weight, labels=label_batch))
            non_nan_num = Variable(FloatTensor([(labels == 1).sum() + (labels == 0).sum()]))
            loss = F.binary_cross_entropy_with_logits(outputs.view(-1), \
                                                      label_batch.float().view(-1), \
                                                      weight=weights, size_average=False) / non_nan_num
            loss.backward()
            optimizer.step()

        # report performance
        if True:
            train_acc_sep, train_acc_tot = test_model(train_loader, model, tasks)
            val_acc_sep, val_acc_tot = test_model(validation_loader, model, tasks)
            print(
                'Epoch: [{}/{}], '
                'Step: [{}/{}], '
                'Loss: {}, \n'
                'Train AUC seperate: {}, \n'
                'Train AUC total: {}, \n'
                'Validation AUC seperate: {}, \n'
                'Validation AUC total: {} \n'.format(
                    epoch + 1, num_epochs, i + 1,
                    math.ceil(len_train / batch_size), loss.data[0], \
                    train_acc_sep, train_acc_tot, val_acc_sep,
                    val_acc_tot))
            if write_file:
                with open(file_name, 'a') as fp:
                    fp.write(
                        'Epoch: [{}/{}], '
                        'Step: [{}/{}], '
                        'Loss: {}, \n'
                        'Train AUC seperate: {}, \n'
                        'Train AUC total: {}, \n'
                        'Validation AUC seperate: {}, \n'
                        'Validation AUC total: {} \n'.format(
                            epoch + 1, num_epochs, i + 1,
                            math.ceil(len_train / batch_size),
                            loss.data[0], \
                            train_acc_sep, train_acc_tot, val_acc_sep,
                            val_acc_tot))
            validation_acc_history.append(val_acc_tot)
            # check if we need to earily stop the model
            stop_training = earily_stop(validation_acc_history, tasks, early_stop_step_single,
                                        early_stop_step_multi, early_stop_required_progress) and (train_acc_tot > 0.99)
            if stop_training:  # early stopping
                print("{}th epoch: earily stop triggered".format(epoch))
                if write_file:
                    with open(file_name, 'a') as fp:
                        fp.write("{}th epoch: earily stop triggered".format(epoch))
                break

        # because of the the nested loop
        if stop_training:
            break

    test_auc_sep, test_auc_tot = test_model(test_loader, model, tasks)
    torch.save(model.state_dict(), '{}.pkl'.format(file_name))
    torch.save(model, '{}.pt'.format(file_name))

    print('AUC of the model on the test set for single task: {}\n'
          'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))
    if write_file:
        with open(file_name, 'a') as fp:
            fp.write('AUC of the model on the test set for single task: {}\n'
                     'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))

    return(test_auc_tot)

tasks = all_tasks # [task]
print(' learning_rate: {},\n batch_size: {}, \n '
          'tasks: {},\n random_state: {}, \n EAGCN_structure: {}\n'.format(
        learning_rate, batch_size, tasks, random_state, EAGCN_structure))
print('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))
print('n_den1, nden2: {}, {}'.format(n_den1, n_den2))
if use_cuda:
    position = 'server'
else:
    position = 'local'
if len(tasks) == 1:
    directory = '../experiment_result/{}/{}/{}/'.format(position, dataset, tasks)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}'.format(directory, experiment_date)
else:
    directory = "../experiment_result/{}/{}/['all_tasks']/".format(position, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}'.format(directory, experiment_date)


if write_file:
    with open(file_name, 'w') as fp:
        fp.write(' learning_rate: {},\n batch_size: {}, \n '
                     'tasks: {},\n random_state: {} \n,'
                     ' EAGCN_structure: {}\n'.format(learning_rate, batch_size,
                                           tasks, random_state, EAGCN_structure))
        fp.write('early_stop_step_single: {}, early_stop_step_multi: {}, \n'
                     'early_stop_required_progress: {},\n early_stop_diff: {}, \n'
                     'weight_decay: {}, dropout: {}\n'.format(early_stop_step_single, early_stop_step_multi,
                                                             early_stop_required_progress, early_stop_diff,
                                                             weight_decay, dropout))
        fp.write('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))

result = train(tasks, EAGCN_structure, n_den1, n_den2,file_name)
