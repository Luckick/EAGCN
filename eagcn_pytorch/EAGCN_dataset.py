import csv
import operator
from neural_fp import *
import numpy as np
from utils import *
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import train_test_split
import os
import torch

class EAGCN_Dataset():
    """
    The class that define the process of loading data from csv file, clean data by deleting molecules contain
    some atom types, pre-process data to transform molecule into graph etc.
    """
    def __init__(self):
        """
        We only hope to deal with the molecules which includes atoms whose index are in [5, 6, 7, 8, 9, 15, 16, 17, 35, 53].
        :param name:
        """
        self.name = None
        self.path ='../data/'
        self.filename = None
        self.smile_col_num = None
        self.label_col_num = None
        self.delimiter = ','
        self.quotechar = '"'
        self.selected_atom_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        self.bondtype_freq = 0
        self.atomtype_freq = 0
        self.size_cutoff = 1000

    def load_from_csv(self):
        check_clean = os.path.isfile('{}{}_cleaned.csv'.format(self.path, self.name))
        if check_clean:
            self.load_from_clean_csv()
            return True
        print('Loading {} dataset...'.format(self.name))
        data = []
        with open('{}{}'.format(self.path, self.filename), 'r') as data_fid:
            reader = csv.reader(data_fid, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                data.append(row)

        filted_atomtype_list_order, filted_bondtype_list_order = self.get_filt_types()
        print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order,
                                                                                         filted_bondtype_list_order))

        #target = data[0][self.label_col_num]
        target = [data[0][i] for i in list(self.label_col_num)]
        if not check_clean:
            with open('{}{}_cleaned.csv'.format(self.path, self.name), 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=self.delimiter, quotechar=self.quotechar,
                                        quoting=csv.QUOTE_MINIMAL)
                print(data[0])
                spamwriter.writerow(data[0])
        labels = []
        mol_sizes = []
        smile_list = []
        x_all = []
        count_1 = 0     # Number of smile record
        count_2 = 0     # Number of valid data
        for i in range(1, len(data)):
            if len(data[i]) == 0:
                continue
            smile = data[i][self.smile_col_num]
            mol = MolFromSmiles(smile)
            count_1 += 1
            remove, reason = mol_remover(smile, mol)
            if remove:
                #print('the {}th row smile is: {}, {}'.format(i, smile, reason))
                continue
            try:
                (afm, adj, adjTensor_TypeAtt, adjTensor_OrderAtt,
                 adjTensor_AromAtt, adjTensor_ConjAtt,
                 adjTensor_RingAtt, nodeSubtypes) = molToGraph(mol, filted_bondtype_list_order,
                                                                                       filted_atomtype_list_order).dump_as_matrices_Att()
                mol_sizes.append(adj.shape[0])
                # labels.append([np.float32(data[i][self.label_col_num])])
                if self.name != 'NIH':
                    label = [data[i][j] for j in self.label_col_num]
                    if self.type == 'class':
                        label = ['-1' if ele == '' else ele for ele in label]
                    labels.append(np.float32(label))
                else:
                    label = []
                    for j in self.label_col_num:
                        if data[i][j] == 'Inactive':
                            label.append(0)
                        elif data[i][j] == 'Active':
                            label.append(1)
                        elif data[i][j] == '':
                            label.append(-1)
                        else:
                            #print('Check the {}th case {} column: {}'.format(i, j, data[i][j]))
                            label.append(-1)
                    labels.append(np.float32(label))
                    pass
                x_all.append(
                    [afm, adj, adjTensor_TypeAtt, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt,
                     adjTensor_RingAtt, nodeSubtypes, count_2])
                smile_list.append(smile)
                count_2 += 1
                if not check_clean:
                    with open('{}{}_cleaned.csv'.format(self.path, self.name), 'a') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=self.delimiter, quotechar=self.quotechar,
                                                quoting=csv.QUOTE_MINIMAL)
                        spamwriter.writerow(data[i])
            except AttributeError:
                print('the {}th row has an error'.format(i))
            except GraphError:
                print('the {}th row smile is: {}, can not convert to graph structure'.format(i, data[i][self.smile_col_num]))
            except AtomError:
                print('the {}th row smile is: {}, has uncommon atom types, ignore'.format(i, data[i][self.smile_col_num]))
                #error_row.append(i)
            except SubtypeError:
                print('the {}th row smile is: {}, atom can not find key subtype'.format(i, data[i][self.smile_col_num]))
            else:
                pass

        self.x_all = self.feature_normalize(x_all)
        print('Done. Total line is {}. Valid line is {}'.format(count_1, count_2))
        self.cleaned_data_length = len(x_all)
        self.labels = labels
        self.target = target
        self.mol_sizes = mol_sizes
        self.smile_list = smile_list
        self.max_size = max(mol_sizes)
        #return (self.x_all, labels, target, mol_sizes, smile_list)

    def get_filt_types(self):
        atomtype_dic, bondtype_dic = self.get_type_dics()

        sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
        sorted_atom_types_dic.reverse()
        atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
        atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

        filted_atomtype_list_order = []
        append_other = False
        for i in range(0, len(atomtype_list_order)):
            if atomtype_list_number[i] > self.atomtype_freq:
                filted_atomtype_list_order.append(atomtype_list_order[i])
            else:
                append_other = True
            #print(atomtype_list_order[i], atomtype_list_number[i])
        if append_other:
            filted_atomtype_list_order.append('Others')

        sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
        sorted_bondtype_dic.reverse()
        bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
        bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

        filted_bondtype_list_order = []
        append_other = False
        for i in range(0, len(bondtype_list_order)):
            if bondtype_list_number[i] > self.bondtype_freq:
                filted_bondtype_list_order.append(bondtype_list_order[i])
            else:
                append_other = True
            #print(bondtype_list_order[i], bondtype_list_number[i])
        if append_other:
            filted_bondtype_list_order.append('Others')

        return filted_atomtype_list_order, filted_bondtype_list_order

    def get_type_dics(self):
        """
        :return: return the atom type dictionary and bond type dictionary
        """
        data = []
        with open('{}{}'.format(self.path, self.filename), 'r') as data_fid:
            reader = csv.reader(data_fid, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                data.append(row)

        bondtype_dic = {}
        atomtype_dic = {}
        for row in data[1:]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
            if len(row) == 0:
                continue
            smile = row[self.smile_col_num]
            try:
                mol = MolFromSmiles(smile)
                atomtype_dic = fillAtomType_dic(mol, atomtype_dic, selected_atom_list = self.selected_atom_list)
                bondtype_dic = fillBondType_dic(mol, bondtype_dic, selected_atom_list=self.selected_atom_list)
            except AtomError:
                pass
                #print('{}th row include uncommon type, delete {} when make type dictionaries'.format(row, smile))
            except AttributeError:
                #print('{}th row have error, the smile is {}, skip'.format(row[0], smile))
                pass
            else:
                pass
        # print(atomtype_dic, bondtype_dic)
        return (atomtype_dic, bondtype_dic)

    def get_train_val_test(self, random_state):
        self.load_from_csv()
        self.data_filter()
        X_train, X_vt, y_train, y_vt, smiles_train, smiles_vt = train_test_split(self.x_select, self.y_task, self.smile_select, test_size=0.2, random_state=random_state)
        X_val, X_test, y_val, y_test, smiles_val, smiles_test = train_test_split(X_vt, y_vt, smiles_vt, test_size=0.5, random_state=random_state)
        return(X_train, X_val, X_test, y_train, y_val, y_test, smiles_train, smiles_val, smiles_test)

    def feature_normalize(self, x_all):
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
                feature_diff_dic[j] = feature_max_dic[j] - feature_min_dic[j]
                if feature_diff_dic[j] == 0:
                    feature_diff_dic[j] = 1
                afm[:, j] = (afm[:, j] - feature_min_dic[j]) / (feature_diff_dic[j])
            x_all[i][0] = afm
        return x_all

    def data_filter(self):
        idx_row = []
        for i in range(0, len(self.mol_sizes)):
            if self.mol_sizes[i] <= self.size_cutoff:
                idx_row.append(i)
        x_select = [self.x_all[i] for i in idx_row]
        y_select = [self.labels[i] for i in idx_row]
        smile_select = [self.smile_list[i] for i in idx_row]

        idx_col = []
        for task in self.all_tasks:
            for i in range(0, len(self.target)):
                if task == self.target[i]:
                    idx_col.append(i)
        y_task = [[each_list[i] for i in idx_col] for each_list in y_select]
        self.x_select = x_select
        self.y_task = y_task
        self.smile_select = smile_select
        #return (x_select, y_task, smile_select)

    def load_from_clean_csv(self):
        print('Loading {} cleaned dataset...'.format(self.name))
        data = []
        with open('{}{}_cleaned.csv'.format(self.path, self.name), 'r') as data_fid:
            reader = csv.reader(data_fid, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                data.append(row)

        filted_atomtype_list_order, filted_bondtype_list_order = self.get_filt_types()
        print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order,
                                                                                         filted_bondtype_list_order))

        target = [data[0][i] for i in list(self.label_col_num)]
        labels = []
        mol_sizes = []
        error_row = []
        correct_row = []
        smile_list = []
        x_all = []
        count_1 = 0  # Number of smile record
        count_2 = 0  # Number of valid data
        for i in range(1, len(data)):
            if self.name == 'hiv':
                if i%2 == 0:
                    print(data[i])
                    #continue
            if len(data[i]) == 0:
                continue

            smile = data[i][self.smile_col_num]
            mol = MolFromSmiles(smile)
            count_1 += 1
            remove, reason = mol_remover(smile, mol)
            if remove:
                #print('the {}th row smile is: {}, {}'.format(i, smile, reason))
                continue
            try:
                (afm, adj, adjTensor_TypeAtt, adjTensor_OrderAtt,
                 adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, nodeSubtypes) = molToGraph(mol, filted_bondtype_list_order,
                                                                                       filted_atomtype_list_order).dump_as_matrices_Att()
                mol_sizes.append(adj.shape[0])
                # labels.append([np.float32(data[i][self.label_col_num])])
                if self.name != 'NIH':
                    label = [data[i][j] for j in self.label_col_num]
                    if self.type == 'class':
                        label = ['-1' if ele == '' else ele for ele in label]
                    labels.append(np.float32(label))
                else:
                    label = []
                    for j in self.label_col_num:
                        if data[i][j] == 'Inactive':
                            label.append(0)
                        elif data[i][j] == 'Active':
                            label.append(1)
                        elif data[i][j] == '':
                            label.append(-1)
                        else:
                            #print('Check the {}th case {} column: {}'.format(i, j, data[i][j]))
                            label.append(-1)
                    labels.append(np.float32(label))
                    pass

                x_all.append(
                    [afm, adj, adjTensor_TypeAtt, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt,
                     adjTensor_RingAtt, nodeSubtypes, count_2])
                smile_list.append(smile)
                count_2 += 1
                correct_row.append(i)
            except AttributeError:
                #print('the {}th row has an error'.format(i))
                error_row.append(i)
            except GraphError:
                #print('the {}th row smile is: {}, can not convert to graph structure'.format(i, data[i][1]))
                error_row.append(i)
            except AtomError:
                #print('the {}th row smile is: {}, has uncommon atom types, ignore'.format(i, data[i][1]))
                pass
            except SubtypeError:
                print('the {}th row smile is: {}, has Subtype Error, ignore'.format(i, data[i][1]))
                # error_row.append(i)
                error_row.append(i)
            else:
                pass
                #print('the {}th row smile is: {}, has some other errors, ignore'.format(i, data[i][self.smile_col_num]))

        self.x_all = self.feature_normalize(x_all)
        print('Done. Total line is {}. Valid line is {}'.format(count_1, count_2))
        self.cleaned_data_length = len(x_all)
        self.labels = labels
        self.target = target
        self.mol_sizes = mol_sizes
        self.smile_list = smile_list
        self.max_size = max(mol_sizes)
        # return (self.x_all, labels, target, mol_sizes, smile_list)

class Reg_Dataset(EAGCN_Dataset):
    def __init__(self):
        super(Reg_Dataset, self).__init__()
        self.type = 'reg'


    def get_train_val_test_loader(self, random_state, batch_size):
        self.load_from_csv()
        self.data_filter()
        X_train, X_vt, y_train, y_vt, smiles_train, smiles_vt = train_test_split(self.x_select, self.y_task, self.smile_select, test_size=0.2, random_state=random_state)
        self.n_bfeat = X_train[0][2].shape[0]
        self.n_afeat = X_train[0][0].shape[1]
        train_loader = construct_loader_reg(X_train, y_train, self.target, smiles_train, batch_size=batch_size)
        self.len_train = len(X_train)
        del X_train, y_train, smiles_train

        X_val, X_test, y_val, y_test, smiles_val, smiles_test = train_test_split(X_vt, y_vt, smiles_vt, test_size=0.5,
                                                                                 random_state=random_state)
        del X_vt, y_vt, smiles_vt
        validation_loader = construct_loader_reg(X_val, y_val, self.target, smiles_val, batch_size=batch_size)
        test_loader = construct_loader_reg(X_test, y_test, self.target, smiles_test, batch_size=batch_size)
        return(train_loader, validation_loader, test_loader)

class Class_Dataset(EAGCN_Dataset):
    def __init__(self):
        super(Class_Dataset, self).__init__()
        self.type = 'class'

    def get_train_val_test_loader(self, random_state, batch_size):
        self.load_from_csv()
        self.data_filter()
        X_train, X_vt, y_train, y_vt, smiles_train, smiles_vt = train_test_split(self.x_select, self.y_task, self.smile_select, test_size=0.2, random_state=random_state)
        self.n_bfeat = X_train[0][2].shape[0]
        self.n_afeat = X_train[0][0].shape[1]
        train_loader = construct_loader(X_train, y_train, self.target, smiles_train, batch_size=batch_size)
        self.len_train = len(X_train)
        self.BCE_weight = set_weight(y_train)
        del X_train, y_train, smiles_train

        X_val, X_test, y_val, y_test, smiles_val, smiles_test = train_test_split(X_vt, y_vt, smiles_vt, test_size=0.5,
                                                                                 random_state=random_state)
        del X_vt, y_vt, smiles_vt
        validation_loader = construct_loader(X_val, y_val, self.target, smiles_val, batch_size=batch_size)
        test_loader = construct_loader(X_test, y_test, self.target, smiles_test, batch_size=batch_size)
        return(train_loader, validation_loader, test_loader)


class Lipo(Reg_Dataset):
    def __init__(self):
        super(Lipo, self).__init__()
        self.name = 'Lipo'
        self.filename = 'Lipophilicity.csv'
        self.smile_col_num = 2
        self.label_col_num = [1]
        self.all_tasks = ['exp']

class Freesolv(Reg_Dataset):
    def __init__(self):
        super(Freesolv, self).__init__()
        self.name = 'Freesolv'
        self.filename = 'SAMPL.csv'
        self.smile_col_num = 1
        self.label_col_num = [2]
        self.all_tasks = ['expt']

class Tox21(Class_Dataset):
    def __init__(self):
        super(Tox21, self).__init__()
        self.name = 'Tox21'
        self.filename = 'tox21.csv'
        self.smile_col_num = 13
        self.label_col_num = list(range(0,12))
        self.all_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

class HIV(Class_Dataset):
    def __init__(self):
        super(HIV, self).__init__()
        self.name = 'HIV'
        self.filename = 'HIV.csv'
        self.smile_col_num = 0
        self.label_col_num = [2]
        self.all_tasks = ['HIV_active']

class NIH(Class_Dataset):
    def __init__(self):
        super(NIH, self).__init__()
        self.name = 'NIH'
        self.filename = 'pubchem_data.csv'
        self.smile_col_num = 44
        self.label_col_num = [4, 12, 20, 28, 36]
        self.all_tasks = ['HEK293-Outcome', 'KB-3-1-Outcome','NIH3T3-Outcome', 'CRL-7250-Outcome',
                          'HaCat-Outcome']
        self.delimiter = '\t'

class ESOL(Reg_Dataset):
    def __init__(self):
        super(ESOL, self).__init__()
        self.name = 'ESOL'
        self.filename = 'delaney-processed.csv'
        self.smile_col_num = 9
        self.label_col_num = [8]
        self.all_tasks = ['measured log solubility in mols per litre']

class MUV(Class_Dataset):
    def __init__(self):
        super(MUV, self).__init__()
        self.name = 'MUV'
        self.filename = 'muv.csv'
        self.smile_col_num = 18
        self.label_col_num = list(range(0,17))
        self.all_tasks = ['MUV-466', 'MUV-548', 'MUV-600','MUV-644', 'MUV-652', 'MUV-689', 'MUV-692',
                          'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846',
                          'MUV-852', 'MUV-858', 'MUV-859']

class PCBA(Class_Dataset):
    def __init__(self):
        super(PCBA, self).__init__()
        self.name = 'PCBA'
        self.filename = 'pcba.csv'
        self.smile_col_num = 129
        self.label_col_num = list(range(0,128))
        self.all_tasks = ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458',
                          'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479',
                          'PCBA-1631', 'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101',
                          'PCBA-2147', 'PCBA-2242', 'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528',
                          'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
                          'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294',
                          'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349',
                          'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947',
                          'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339',
                          'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842',
                          'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317',
                          'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590',
                          'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233',
                          'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171',
                          'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288',
                          'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635',
                          'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104',
                          'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979',
                          'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553',
                          'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709',
                          'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881',
                          'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899',
                          'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915',
                          'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995']