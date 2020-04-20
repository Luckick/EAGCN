import matplotlib.pyplot as plt
import os
import re

#dataset = 'dc_tox21'
dataset ='hiv'
path = '../experiment_result/{}/{}/HIV_active'.format('server', 'hiv')
#path = '../experiment_result/{}/{}/all_tasks'.format('server', 'nih')
#path = '../experiment_result/{}/{}/exp'.format('server', 'lipo')

if dataset == 'tox21':
    all_tasks = ['nr-ahr', 'nr-ar', 'nr-ar-lbd', 'nr-aromatase', 'nr-er', 'nr-er-lbd',
                 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmpp', 'sr-p53']
    all_tasks.append('all_tasks')
if dataset == 'dc_tox21':
    all_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    all_tasks.append('all_tasks')

if dataset == 'hiv':
    all_tasks = ['HIV_active']

def plot_auc_change(path, file):
    val_auc = []
    train_auc = []
    test_auc = []
    try:
        with open('{}/{}'.format(path, file), 'r') as fh:
            idx = 0
            for line in fh:
                if 'Validation AUC total' in line:
                    content = line.replace(",", "")
                    content = content.split(' ')
                    val_auc.append(float(content[-2]))
                elif 'Train AUC total' in line:
                    content = line.replace(",", "")
                    content = content.split(' ')
                    train_auc.append(float(content[-2]))
                elif 'test AUC total' in line:
                    content = line.replace(",", "")
                    content = content.split(' ')
                    test_auc.append(float(content[-2]))
                idx += 1
    except UnicodeDecodeError:
        print('check the {} file'.format(file))
    else:
        pass

    if idx > 200 and len(train_auc)>30:
        plt.plot(train_auc, label='train auc')
        plt.plot(val_auc, label='val auc')
        plt.plot(test_auc, label='test auc')
        plt.title(file)
        plt.show()

def plot_RMSE_change(path, file, start = 0):
    val_rmse = []
    train_rmse = []
    test_rmse =[]
    try:
        with open('{}/{}'.format(path, file), 'r') as fh:
            idx = 0
            for line in fh:
                if 'Train RMSE:' in line and len(line) < 200 and len(line) > 80:
                    content = re.sub('[a-zA-Z|:|,]', '', line)
                    content = content.split('  ')
                    val_rmse.append(float(content[-2]))
                    train_rmse.append(float(content[-3]))
                    test_rmse.append(float(content[-1]))
                idx += 1
    except UnicodeDecodeError:
        print('check the {} file'.format(file))

    if idx > 100:
        plt.plot(train_rmse[start:], label='train rmse')
        plt.plot(val_rmse[start:], label='val rmse')
        plt.plot(test_rmse[start:], label='test rmse')
        plt.title(file)
        plt.legend()
        plt.show()
        #plt.savefig('freesolv_rmse_{}.png'.format(file))
        #plt.close()
        #print('finished lipo_rmse_{}.png'.format(file))

#path = '../experiment_result/{}/{}/exp'.format('server', 'lipo')

#path = '../experiment_result/{}/{}/measured log solubility in mols per litre'.format('server', 'esol')
files = os.listdir(path)

for file in files:
    if 'Concate' in file:
        print(file)
        #plot_RMSE_change(path, file, 100)
        plot_auc_change(path, file)


# For Classification
"""
for task in ['HIV_active']:
    tasks = ['HIV_active']#[task]
    results = []
    for position in ['server', 'local']:
        print('{} result: '.format(position))
        path = '../experiment_result/{}/{}/{}'.format(position, dataset, tasks)
        files = os.listdir(path)

        for file in files:
            if 'Feb_06' in file or 'Jan_25' in file:
                plot_auc_change(path, file)
"""

