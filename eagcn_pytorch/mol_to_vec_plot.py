import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
import torch
import csv
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict

compare_pair_list= [
    ### C, O, 10
    # [619, 130],
    [336, 164],
    # [409, 454, 211],
    # [111, 68, 156],
    # [336, 164],
    [193, 342],
    [41, 356],
    [2, 468],
    # [517, 97],
    [263, 458],
    ### C, N
    # [130, 619],
    # [25, 193],
    # [83, 627],
    # [374, 484]
    ### O, N : not good
    # [281, 469],
    # [363, 467],
    # [60, 434]
    ## Br, C: good, 6
    # [368, 618],
    [318, 96],
    # [319, 30],
    [470, 83],
    [364, 129],
    # ### Cl, I: good, but too close
    # [621, 288],
    # [545, 99],
    # [596, 259],
    # # [259, ]

    ## Cl, O: ok, 12
    # [253, 535],
    #### [147, 26],
    [382, 9],
    # [176, 467],
    [637, 4],
    [516, 202],
    [596, 569],

    ## I, O, 8
    [621, 215],
    [545, 616],
    # [314, 462],
    [623, 565]

]

def mol_to_vec_plot(mol_rep_data, compare_pair_list, method = 'pca'):
    X = np.array(mol_rep_data[0])
    y = mol_rep_data[2].reshape(-1)
    index = mol_rep_data[1].reshape(-1)
    y_pred = mol_rep_data[3].reshape(-1)

    index_list = [item for sublist in compare_pair_list for item in sublist]
    index_smile_dic = {}
    with open('{}{}'.format('../data/', 'Freesolv_cleaned.csv'), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        i = -1
        for row in reader:
            if i in index_list:
                index_smile_dic[i] = row[1]
            i += 1

    if method == 'pca':
        pca = PCA(n_components=2)
        pca.fit(X)
        X_trans = pca.transform(X)
    elif method =='tsne':
        tsne = TSNE(n_components=2)
        X_trans = tsne.fit_transform(X)

    coord1, coord2, smiles = [], [], []
    for idx in index_list:
        coord1.append(X_trans[index == idx, 0])
        coord2.append(X_trans[index == idx, 1])
        smiles.append(index_smile_dic[idx])

    # fig, ax = plt.subplots()
    # ax.scatter(coord1, coord2)

    # plt.plot(coord1, coord2)
    plt.figure(figsize=(10, 6))
    if method == 'pca':
        plt.xlim(-6.2, -2.2)
        plt.ylim(-2, 6)
    for i, txt in enumerate(smiles):
        # ax.annotate(txt, (coord1[i], coord2[i]))
        if i< 10:
            plt.scatter(coord1[i], coord2[i], c = 'y')
            if i ==9:
                plt.annotate(txt, (coord1[i], coord2[i]-0.13), xycoords='data')
            else:
                plt.annotate(txt, (coord1[i], coord2[i]), xycoords = 'data')
            if i % 2 == 0:
                plt.plot(coord1[i:i + 2], coord2[i:i + 2], 'r--', label = 'C and O')
        elif i < 16:
            plt.scatter(coord1[i], coord2[i], c = 'b')
            if i ==14:
                plt.annotate(txt, (coord1[i] - 0.35, coord2[i]), xycoords='data')
            elif i ==10:
                plt.annotate(txt, (coord1[i] - 0.40, coord2[i] - 0.08), xycoords='data')
            elif i == 12:
                plt.annotate(txt, (coord1[i] - 0.30, coord2[i] - 0.15), xycoords='data')
            else:
                plt.annotate(txt, (coord1[i]-0.35, coord2[i]-0.1), xycoords='data')
            if i % 2 == 0:
                plt.plot(coord1[i:i + 2], coord2[i:i + 2], 'g--', label = 'C and Br')
        elif i < 24:
            plt.scatter(coord1[i], coord2[i], c = 'k')
            plt.annotate(txt, (coord1[i], coord2[i]), xycoords='data')
            if i % 2 == 0:
                plt.plot(coord1[i:i + 2], coord2[i:i + 2], 'b--', label = 'Cl and O')
        else:
            plt.scatter(coord1[i], coord2[i], c = 'm')
            plt.annotate(txt, (coord1[i], coord2[i]), xycoords='data')
            if i % 2 == 0:
                plt.plot(coord1[i:i + 2], coord2[i:i + 2], 'y--', label = 'I and O')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # plt.legend()
    legend = plt.legend(by_label.values(), by_label.keys(), fontsize = 14, frameon=True)
    legend.get_frame().set_edgecolor('b')
    # fig.savefig("mol_to_vec.png")
    plt.savefig("mol_to_vec_{}.png".format(method))

mol_rep_data = torch.load('../tsne/data/mol/freesolv_Concate_1500')
mol_to_vec_plot(mol_rep_data, compare_pair_list, 'tsne')

