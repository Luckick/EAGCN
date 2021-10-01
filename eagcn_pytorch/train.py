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
import argparse
from EAGCN_dataset import *
# CUDA_VISIBLE_DEVICES=1

model_names = ['Concate', 'Weighted_sum', 'GCN', 'GAT']
molfp_mode = ['sum', 'ave', 'pool']
datasets = ['tox21', 'hiv', 'nih', 'lipo', 'esol', 'freesolv']

parser = argparse.ArgumentParser(description='PyTorch EAGCN Training')
parser.add_argument('--dataset', default= 'freesolv', help='dataset name', choices=datasets)
parser.add_argument('--arch', '-a', metavar='ARCH', default='Concate',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: Concate)')
parser.add_argument('--molfp', metavar='FP', default='sum',
                    choices=molfp_mode,
                    help='molecule fingure print mode: ' +
                        ' | '.join(molfp_mode) +
                        ' (default: sum)')
parser.add_argument('--rs','--random_state', default=0, type=int)

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--dr', '--dropout-rate', default=0.3, type=float,
                    metavar='DR', help='Dropout Rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                     metavar='N', help='print frequency (default: 20)')
parser.add_argument('-t', '--eval_train_loader', default=False, type=bool)


args = parser.parse_args()
# Training settings
write_file = True
n_den1, n_den2= 128, 64

# Classification
if args.dataset == 'tox21':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 80, 80, 80, 80, 80  # 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 140, 140, 140, 140, 140  # 60, 20, 20, 20, 20
    weight_decay = 0.0001  # L-2 Norm
    num_epochs = 100
    learning_rate = 0.0005
    exp_data = Tox21()
    n_den1, n_den2 = 256, 64
if args.dataset == 'hiv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 100, 100, 100, 100, 100  # 40, 15, 15, 15, 15
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 250, 250, 250, 250, 250  # 80, 30, 30, 30, 30
    weight_decay = 0.00001  # L-2 Norm
    num_epochs = 300
    learning_rate = 0.001 #0.0005
    n_den1, n_den2 = 256 * 2, 128
    exp_data = HIV()
if args.dataset == 'nih':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 15, 12, 12, 12, 12  # 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 30, 20, 20, 20, 20  # 60, 20, 20, 20, 20
    weight_decay = 0.001  # L-2 Norm 0.0001
    num_epochs = 200
    learning_rate = 0.0005
    exp_data = NIH()
    n_den1, n_den2 = 32, 16

# Regression
if args.dataset == 'lipo':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 60, 60, 60, 60, 60  # 30, 30, 30, 30, 30
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 100, 100, 100, 100, 100  # 60, 60, 60, 60, 60
    n_den1, n_den2 = 128, 64
    early_stop_val_rmse = 0.50
    learning_rate = 0.0001  # 0.001
    num_epochs = 500 #800
    weight_decay = 0.001  # 0.0001
    exp_data = Lipo()
if args.dataset == 'esol':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 50, 50, 50, 50, 50  # 30, 30, 30, 30, 30
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 80, 80, 80, 80, 80  # 60, 60, 60, 60, 60
    n_den1, n_den2 = 128, 64
    early_stop_val_rmse = 0.50
    learning_rate = 0.001  # 0.001
    num_epochs = 800
    weight_decay = 0.001  # 0.0001
    num_epochs = 2500
    exp_data = ESOL()
if args.dataset == 'freesolv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 40, 40, 40, 40, 40  # 15, 15, 15, 15, 15
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 60, 60, 60, 60  # 25, 25, 25, 25, 25
    n_den1, n_den2 = 128, 64  # 32, 12
    early_stop_val_rmse = 0.70
    num_epochs = 1500
    learning_rate = 0.0001
    weight_decay = 0.01 # 0.001
    exp_data = Freesolv()

# Early Stopping:
early_stop_step_single = 3
early_stop_step_multi = 5
early_stop_required_progress = 0.001
early_stop_diff = 0.11

experiment_date = strftime("%y_%b_%d_%H:%M", gmtime()) +'New'
print(experiment_date)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

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
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels, subtype, size, index in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        size_batch = Variable(size)
        outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch, size_batch)
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
    auc_sum = 0
    count = 0
    for auc in aucs:
        if not math.isnan(auc):
            auc_sum += auc
            count += 1
    #return (aucs, sum(aucs)/len(aucs))
    return(aucs, auc_sum/count)

def test_model_reg(loader, model):
    """
    Help function that tests the model's regression performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    true_value = []
    all_out = []
    model.eval()
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels, subtype, size, index in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        size_batch = Variable(size)
        #outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch, size_batch)

        all_out.extend(outputs.data.view(-1).cpu().numpy().tolist())

        label_list = labels.cpu().numpy().tolist()
        true_value.extend([item for sublist in label_list for item in sublist])
    model.train()
    return (math.sqrt(metrics.mean_squared_error(all_out, true_value)))

def dump_atom_rep(loader, model, epoch):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    model.eval()
    i = 0
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels, subtype, size, index in loader:
        size_batch = Variable(size)
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        outputs, all_atom_rep, graph_rep = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch, size_batch)

        if i == 0:
            non_empty_rep = np.array([])
            care_subtype = np.array([])
            belong_mol_index = np.array([])
            graph_rep_array = np.array([])
            label_mol_array = np.array([])
            index_mol_array = np.array([])
            pred_label_array = np.array([])

        # mol_rep = torch.sum(all_atom_rep, 1)
        # size = size.view(all_atom_rep.data.shape[0], 1).type(FloatTensor).expand(all_atom_rep.data.shape[0],
        #                                                                          all_atom_rep.data.shape[2]).data.cpu()
        atom_rep = all_atom_rep.view(-1, all_atom_rep.data.shape[2])
        # mol_rep = mol_rep / size

        # index_ is the molecule index for each atom.
        index_ = index.view(index.data.shape[0], 1)
        index_ = index_.expand(subtype.data.shape[0], subtype.data.shape[1])
        index_list = index_.reshape(-1, 1)
        # subtype is an index of real subtype.
        subtypes_list = subtype.view(-1, 1)

        if i == 0:
            non_empty_rep = []
            care_subtype = []
            belong_mol_index = []

            graph_rep_array = graph_rep.cpu().data.numpy()
            index_mol_array= index.view(index.data.shape[0], 1).cpu().data.numpy()
            label_mol_array= labels.cpu().data.numpy()
            pred_label_array = outputs.cpu().data.numpy()
        else:
            graph_rep_array = np.concatenate((graph_rep_array, graph_rep.cpu().data.numpy()))
            index_mol_array  = np.concatenate((index_mol_array, index.view(index.data.shape[0], 1).cpu().data.numpy()))
            label_mol_array = np.concatenate((label_mol_array, labels.cpu().data.numpy()))
            pred_label_array = np.concatenate((pred_label_array, outputs.cpu().data.numpy()))

        for idx in range(subtypes_list.shape[0]):
            if subtypes_list[idx][0] in range(1, 19):
                non_empty_rep.append(atom_rep[idx, :].numpy())
                care_subtype.append(subtypes_list[idx][0].cpu().data.numpy())
                belong_mol_index.append(index_list[idx].cpu().data.numpy())

        i += 1

    subtype_tsne_data = [non_empty_rep, care_subtype, belong_mol_index]
    tsne_dir_atom = '../tsne/data/atom/'
    tsne_file_name = '{}{}_{}_{}'.format(tsne_dir_atom, args.dataset, args.arch, epoch)
    if not os.path.exists(tsne_dir_atom):
        os.makedirs(tsne_dir_atom)
    torch.save(subtype_tsne_data, tsne_file_name)

    # Dump graph representation
    mol_rep_data = [graph_rep_array, index_mol_array, label_mol_array, pred_label_array]
    tsne_dir_mol = '../tsne/data/mol/'
    tsne_file_name = '{}{}_{}_{}'.format(tsne_dir_mol, args.dataset, args.arch, epoch)
    if not os.path.exists(tsne_dir_mol):
        os.makedirs(tsne_dir_mol)
    torch.save(mol_rep_data, tsne_file_name)

    model.train()

def train(tasks, n_den1, n_den2, file_name):
    atom_types = ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    train_loader, validation_loader, test_loader = exp_data.get_train_val_test_loader(args.rs, args.batch_size)
    #print('construct model.')
    model = EAGCN(n_bfeat=exp_data.n_bfeat, n_afeat=exp_data.n_afeat,
                  n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4, n_sgc1_5=n_sgc1_5,
                  n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4, n_sgc2_5=n_sgc2_5,
                  n_den1=n_den1, n_den2=n_den2, nclass=len(tasks),
                  dropout=args.dr, structure=args.arch, molfp_mode = args.molfp)
    #print(model)
    if use_cuda:
        model.cuda()
    #print('initialize model.')
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    validation_acc_history = []
    stop_training = False

    max_i = 0
    #print('Start training.')
    for epoch in range(num_epochs):
        for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels, subtypes, size, index) in enumerate(train_loader):
            adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
            orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                aromAtt), Variable(
                conjAtt), Variable(ringAtt)
            size_batch = Variable(size)
            optimizer.zero_grad()
            outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                            ringAtt_batch, size_batch)

            if exp_data.type == 'reg':
                criterion = torch.nn.MSELoss()
                if use_cuda:
                    criterion.cuda()
                loss = criterion(outputs.view(-1), label_batch.float().view(-1))
            else:
                weights = Variable(weight_tensor(exp_data.BCE_weight, labels=label_batch))
                non_nan_num = Variable(FloatTensor([(labels == 1).sum() + (labels == 0).sum()]))
                loss = F.binary_cross_entropy_with_logits(outputs.view(-1), \
                                                          label_batch.float().view(-1), \
                                                          weight=weights, size_average=False) / non_nan_num

            loss.backward()
            optimizer.step()
            # dump_atom_rep(train_loader, model, epoch)

            if i > max_i:
                max_i = i

            # report performance
            if exp_data.type == 'class' and i%args.print_freq == 0:
                train_acc_sep, train_acc_tot = 'Not_Eval', 'Not_Eval'
                if args.eval_train_loader or epoch%10 == 0:
                    train_acc_sep, train_acc_tot = test_model(train_loader, model, tasks)
                val_acc_sep, val_acc_tot = test_model(validation_loader, model, tasks)
                test_acc_sep, test_acc_tot = test_model(test_loader, model, tasks)
                print(
                    'Epoch: [{}/{}], '
                    'Step: [{}/{}], '
                    'Loss: {}, \n'
                    'Train AUC seperate: {}, \n'
                    'Train AUC total: {}, \n'
                    'Validation AUC seperate: {}, \n'
                    'Validation AUC total: {} \n'
                    'test AUC seperate: {}, \n'
                    'test AUC total: {}, \n'.format(
                        epoch + 1, num_epochs, i + 1,
                        math.ceil(exp_data.len_train / args.batch_size), loss.data[0], \
                        train_acc_sep, train_acc_tot, val_acc_sep,
                        val_acc_tot, test_acc_sep, test_acc_tot))
                if write_file:
                    with open(file_name, 'a') as fp:
                        fp.write(
                            'Epoch: [{}/{}], '
                            'Step: [{}/{}], '
                            'Loss: {}, \n'
                            'Train AUC seperate: {}, \n'
                            'Train AUC total: {}, \n'
                            'Validation AUC seperate: {}, \n'
                            'Validation AUC total: {} \n'
                            'test AUC seperate: {}, \n'
                            'test AUC total: {}, \n'
                                .format(
                                epoch + 1, num_epochs, i + 1,
                                math.ceil(exp_data.len_train / args.batch_size),
                                loss.data[0], \
                                train_acc_sep, train_acc_tot, val_acc_sep,
                                val_acc_tot, test_acc_sep, test_acc_tot))
                if epoch < num_epochs * 4 / 5:
                    validation_acc_history.append(val_acc_tot)
                else:
                    validation_acc_history.sort()
                    compare_list = validation_acc_history[-30:]
                    compare_value = sum(compare_list)/float(len(compare_list))
                    if val_acc_tot > compare_value:
                        stop_training = True
                    # validation_acc_history
                # check if we need to earily stop the model
                # stop_training = earily_stop(validation_acc_history, tasks, early_stop_step_single,
                #                             early_stop_step_multi, early_stop_required_progress) and (
                #                             train_acc_tot > 0.99)
                if stop_training:  # early stopping
                    print("{}th epoch: earily stop triggered".format(epoch))
                    if write_file:
                        with open(file_name, 'a') as fp:
                            fp.write("{}th epoch: earily stop triggered".format(epoch))
                    break

            if exp_data.type == 'reg' and i%args.print_freq == 0:
                train_acc = 'Not_Eval'
                if args.eval_train_loader or epoch%10 == 0:
                    train_acc = test_model_reg(train_loader, model)
                val_acc = test_model_reg(validation_loader, model)
                test_acc = test_model_reg(test_loader, model)
                print(
                    'Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train RMSE: {5}, Validation RMSE:{6}, Test RMSE: {7}'.format(
                        epoch + 1, num_epochs, i + 1, math.ceil(exp_data.len_train / args.batch_size), loss.item(),
                        train_acc,
                        val_acc, test_acc))
                with open(file_name, 'a') as fp:
                    fp.write(
                        'Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train RMSE: {5}, Validation RMSE:{6}, Test RMSE: {7}\n'.format(
                            epoch + 1, num_epochs, i + 1, math.ceil(exp_data.len_train / args.batch_size), loss.item(),
                            train_acc,
                            val_acc, test_acc))

                if epoch < num_epochs * 4 / 5:
                    validation_acc_history.append(val_acc)
                else:
                    validation_acc_history.sort()
                    compare_list = validation_acc_history[0:30]
                    compare_value = sum(compare_list)/float(len(compare_list))
                    if val_acc < compare_value:
                        stop_training = True
                # check if we need to earily stop the model
                # stop_training = earily_stop(validation_acc_history) and val_acc < early_stop_val_rmse
                if stop_training:  # early stopping
                    print("earily stop triggered\n")
                    break
        # because of the the nested loop
        if stop_training:
            break

    if exp_data.type == 'class':
        train_auc_sep, train_auc_tot = test_model(train_loader, model, tasks)
        val_auc_sep, val_auc_tot = test_model(validation_loader, model, tasks)
        test_auc_sep, test_auc_tot = test_model(test_loader, model, tasks)

        torch.save(model.state_dict(), '{}.pkl'.format(file_name))
        torch.save(model, '{}.pt'.format(file_name))

        print('AUC of the model on the train set for single task: {}\n'
              'AUC of the model on the train set for all tasks: {}'.format(train_auc_sep, train_auc_tot))

        print('AUC of the model on the val set for single task: {}\n'
              'AUC of the model on the val set for all tasks: {}'.format(val_auc_sep, val_auc_tot))

        print('AUC of the model on the test set for single task: {}\n'
              'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))
        if write_file:
            with open(file_name, 'a') as fp:
                fp.write('AUC of the model on the train set for single task: {}\n'
                         'AUC of the model on the train set for all tasks: {}'.format(train_auc_sep, train_auc_tot))
                fp.write('AUC of the model on the val set for single task: {}\n'
                         'AUC of the model on the val set for all tasks: {}'.format(val_auc_sep, val_auc_tot))
                fp.write('AUC of the model on the test set for single task: {}\n'
                         'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))


    elif exp_data.type == 'reg':
        # Test the Model
        train_acc = test_model_reg(train_loader, model)
        val_acc = test_model_reg(validation_loader, model)
        test_acc = test_model_reg(test_loader, model)
        print('RMSE of the model on the train set: {}'.format(train_acc))
        print('RMSE of the model on the val set: {}'.format(val_acc))
        print('RMSE of the model on the test set: {}'.format(test_acc))
        with open(file_name, 'a') as fp:
            fp.write('RMSE of the model on the train set: {}'.format(train_acc))
            fp.write('RMSE of the model on the val set: {}'.format(val_acc))
            fp.write('RMSE of the model on the test set: {}\n'.format(test_acc))

        # Save the Model
        torch.save(model.state_dict(), '{}.pkl'.format(file_name))
        torch.save(model, '{}.pt'.format(file_name))

    dump_atom_rep(train_loader, model, num_epochs)
    return

tasks = exp_data.all_tasks    # or [task] if want to focus on one single task.
print(' learning_rate: {},\n batch_size: {}, \n '
                     'tasks: {},\n random_state: {} \n,'
                     ' EAGCN_structure: {}\n'.format(learning_rate, args.batch_size,
                                           tasks, args.rs, args.arch))
print('Finger Print Mode: {}.\n'.format(args.molfp))
print('early_stop_step_single: {}, early_stop_step_multi: {}, \n'
                     'early_stop_required_progress: {},\n early_stop_diff: {}, \n'
                     'weight_decay: {}, dropout: {}\n'.format(early_stop_step_single, early_stop_step_multi,
                                                             early_stop_required_progress, early_stop_diff,
                                                             weight_decay, args.dr))
print('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))
print('den_1, den_2: {},{}'.format(n_den1, n_den2))

if use_cuda:
    position = 'server'
else:
    position = 'local'
if len(tasks) == 1:
    directory = '../experiment_result/{}/{}/{}/'.format(position, args.dataset, tasks[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}{}{}{}'.format(directory, args.arch, args.molfp, args.rs, experiment_date)
else:
    directory = "../experiment_result/{}/{}/all_tasks/".format(position, args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}{}{}{}'.format(directory, args.arch, args.molfp, args.rs, experiment_date)


if write_file:
    with open(file_name, 'w') as fp:
        fp.write(' learning_rate: {},\n batch_size: {}, \n '
                     'tasks: {},\n random_state: {} \n,'
                     ' EAGCN_structure: {}\n'.format(learning_rate, args.batch_size,
                                           tasks, args.rs, args.arch))
        fp.write('Finger Print Mode: {}.\n'.format(args.molfp))
        fp.write('early_stop_step_single: {}, early_stop_step_multi: {}, \n'
                     'early_stop_required_progress: {},\n early_stop_diff: {}, \n'
                     'weight_decay: {}, dropout: {}\n'.format(early_stop_step_single, early_stop_step_multi,
                                                             early_stop_required_progress, early_stop_diff,
                                                             weight_decay, args.dr))
        fp.write('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}\n'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))
        fp.write('den_1, den_2: {},{}\n'.format(n_den1, n_den2))



train(tasks, n_den1, n_den2, file_name)

if write_file:
    with open(file_name, 'a') as fp:
        fp.write('Experiment End.')