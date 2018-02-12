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
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import os

from time import gmtime, strftime

dataset = 'freesolv' # 'lipo', 'freesolv'

# Training settings
early_stop_step = 6
early_stop_required_progress = 0.001
batch_size = 256 if dataset == 'lipo' else 1024
EAGCN_structure = 'concate' #  'concate', 'weighted_ave'

if dataset == 'lipo':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 15, 15, 15, 15
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 30, 30, 30, 30
    n_den1, n_den2, n_den3 = 64, 32, 12
    early_stop_val_rmse = 0.50
    learning_rate = 0.01   #0.001
    random_state = 1  # was 1
    num_epochs = 500
    dropout = 0.3
    weight_decay = 0.0001 # 0.0001
elif dataset == 'freesolv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 5, 5, 5
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 30, 10, 5, 5, 5
    n_den1, n_den2, n_den3 = 32, 12, 12
    early_stop_val_rmse = 0.70
    random_state = 2
    num_epochs = 1000
    learning_rate = 0.001
    dropout = 0.3
    weight_decay = 0.0001

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

if use_cuda:
    position = 'server'
else:
    position = 'local'
experiment_date = strftime("%b_%d_%H:%M", gmtime()) +'N'
print(experiment_date)
directory = '../experiment_result/{}/{}/'.format(position, dataset)
if not os.path.exists(directory):
    os.makedirs(directory)
file_name = '{}{}'.format(directory, experiment_date)

print('learning_rate: {}, random_state: {},  '
      'n_den1, n_den2,{}, {}'.format(learning_rate, random_state, n_den1, n_den2))
print('dropout: {}, weight_dacay {}, EAGCN_structure: {}'.format(dropout, weight_decay, EAGCN_structure))

with open(file_name, 'w') as fp:
    fp.write('learning_rate: {}, random_state: {}, EAGCN_structure: {}\n,'
      'n_den1, n_den2: {}, {}\n'.format(learning_rate, random_state, EAGCN_structure,
                                                  n_den1, n_den2))
    fp.write('dropout: {}, weight_dacay {}\n'.format(dropout, weight_decay))
    fp.write('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
             'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
             '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}\n'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                              n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))

# Load data
x_all, y_all, target, sizes = load_data(dataset)
max_size = np.max(sizes)
n_bfeat = x_all[0][2].shape[0]

X, x_test, y, y_test, idx_train, idx_test = train_test_split(x_all, y_all, range(len(x_all)), test_size=0.1, random_state=random_state)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_len = len(x_train)
train_loader = construct_loader_reg(x_train, y_train, target, batch_size=batch_size)
validation_loader = construct_loader_reg(x_val, y_val, target, batch_size=batch_size)
test_loader = construct_loader_reg(x_test, y_test, target, batch_size=batch_size)

if EAGCN_structure == 'concate':
    model = Concate_GCN(n_bfeat=n_bfeat, n_afeat=25,
                        n_sgc1_1 = n_sgc1_1, n_sgc1_2 = n_sgc1_2, n_sgc1_3= n_sgc1_3, n_sgc1_4 = n_sgc1_4, n_sgc1_5 = n_sgc1_5,
                        n_sgc2_1 = n_sgc2_1, n_sgc2_2= n_sgc2_2, n_sgc2_3= n_sgc2_3, n_sgc2_4 = n_sgc2_4, n_sgc2_5= n_sgc2_5,
                        n_den1=n_den1, n_den2=n_den2,
                        nclass=1, dropout=dropout)
elif EAGCN_structure == 'weighted':
    model = Weighted_GCN(n_bfeat=n_bfeat, n_afeat=25,
                         n_sgc1_1 = n_sgc1_1, n_sgc1_2 = n_sgc1_2, n_sgc1_3= n_sgc1_3, n_sgc1_4 = n_sgc1_4, n_sgc1_5 = n_sgc1_5,
                        n_sgc2_1 = n_sgc2_1, n_sgc2_2= n_sgc2_2, n_sgc2_3= n_sgc2_3, n_sgc2_4 = n_sgc2_4, n_sgc2_5= n_sgc2_5,
                         n_den1=n_den1, n_den2=n_den2, nclass=1, dropout=dropout)
criterion = torch.nn.MSELoss()

if use_cuda:
    model.cuda()
    criterion.cuda()

model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    true_value = []
    all_out = []
    model.eval()
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        all_out.extend(outputs.data.view(-1).cpu().numpy().tolist())

        label_list = labels.cpu().numpy().tolist()
        true_value.extend([item for sublist in label_list for item in sublist])
    model.train()
    return (math.sqrt(metrics.mean_squared_error(all_out, true_value)))

def earily_stop(val_acc_history, t=early_stop_step, required_progress=early_stop_required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False

# Training the Model
validation_acc_history = []
stop_training = False
for epoch in range(num_epochs):
    for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels) in enumerate(train_loader):
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(conjAtt), Variable(ringAtt)
        optimizer.zero_grad()
        outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        loss = criterion(outputs, label_batch.float())
        loss.backward()
        optimizer.step()
        # report performance
        if True:
            train_acc = test_model(train_loader, model)
            val_acc = test_model(validation_loader, model)
            print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train RMSE: {5}, Validation RMSE:{6}'.format(
                epoch + 1, num_epochs, i + 1, math.ceil(train_len / batch_size), loss.data[0], train_acc,
                val_acc))
            with open(file_name, 'a') as fp:
                fp.write('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train RMSE: {5}, Validation RMSE:{6}\n'.format(
                    epoch + 1, num_epochs, i + 1, math.ceil(train_len / batch_size), loss.data[0], train_acc,
                    val_acc))
            validation_acc_history.append(val_acc)
            # check if we need to earily stop the model
            stop_training = earily_stop(validation_acc_history) and val_acc < early_stop_val_rmse
            if stop_training: # early stopping
                print("earily stop triggered\n")
                break
    if stop_training:
        break

# Test the Model
print('RMSE of the model on the test set: {}'.format(test_model(test_loader, model)))
with open(file_name, 'a') as fp:
    fp.write('RMSE of the model on the test set: {}\n'.format(test_model(test_loader, model)))

# Save the Model
torch.save(model.state_dict(), '{}.pkl'.format(file_name))
torch.save(model, '{}.pt'.format(file_name))
#model = torch.load('filename.pt')