import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
import torch
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage
# https://github.com/iwatobipen/chemo_info/blob/master/rdkit_notebook/drawmol_with_idx.ipynb


def kmeans_for_atom(atom_rep_data, target_index_list, select_atom = 'C'):
    X = np.array(atom_rep_data[0])
    # y = np.array(atom_rep_data[1])
    y = np.array([int(ele - 1) for ele in atom_rep_data[1]])
    index = np.array(atom_rep_data[2])

    if select_atom == 'O':
        numbers = list(range(1, 5))
        classes = ['oh', 'oa', 'o', 'unO']
        y = y
    elif select_atom == 'C':
        numbers = np.array(list(range(5, 11))) - 5
        classes = ['c3', 'c1', 'ca', 'c', 'c2', 'unC']
        y = y - 5
    elif select_atom == 'N':
        numbers = np.array(list(range(11, 18))) - 11
        classes = ['n1', 'n3', 'na', 'n', 'no', 'nh', 'unN']
        y = y - 11

    row = np.zeros((y.shape), dtype=bool)
    for j in numbers:
        row = np.logical_or(row, y == j)

    #y_ohehot = np.zeros((y.shape[0], 18))
    #y_ohehot[np.arange(y.shape[0]), y] = 1
    X_select = X[row]
    y_select = y[row]
    index_select = index[row]

    kmeans = KMeans(n_clusters=len(numbers), random_state=0).fit(X_select)
    y_pred = kmeans.labels_
    print(confusion_matrix(y_select, y_pred))

    plot_confusion_matrix(y_select, np.array(y_pred), classes, normalize=True)

    smile_list = []
    with open('{}{}'.format('../data/', 'ESOL_cleaned.csv'), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        i = -1
        for row in reader:
            if i in target_index_list:
                smile = row[9]
                smile_list.append(smile)
            i += 1

    # p = index.argsort()
    # sorted_X = X[p]
    # sorted_y = y[p]
    # sorted_y_pred = y_pred[p]

    for idx in range(len(target_index_list)):
        row2 = np.zeros((y_select.shape), dtype=bool)
        row2 = np.logical_or(row2, index_select[:, 0] == target_index_list[idx])
        if row2.sum() == 0:
            print('{}th index data not in dumped data.'.format(target_index_list[idx]))
            continue
        index_check = index_select[row2]
        y_idx = y_select[row2]
        y_pred_idx = y_pred[row2]

        print('The {}th intereted smile is {}.'.format(idx, smile_list[idx]))
        mol = mol_with_atom_index(smile_list[idx])
        plt = MolToImage(mol)
        plt.show()
        print('The {} atoms includes {}, the kmeans predicted labels are {}'.format(select_atom, y_idx, y_pred_idx))
        print('\n')

    # data = pairwise_distances(sort_X)
    # ax = sns.heatmap(data)
    # fig = ax.get_figure()
    # #fig.savefig("output.png")
    # fig.savefig("similarity_plot.png")

def plot_confusion_matrix(y_true, y_pred,
                          classes ,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    print('Classification Report')
    target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    # print(classification_report(y_true, y_pred, target_names=target_names))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def mol_with_atom_index(smile):
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


atom_rep_data = torch.load('../tsne/data/atom/esol_Concate_2500')
kmeans_for_atom(atom_rep_data, list(range(0,20)), 'C')