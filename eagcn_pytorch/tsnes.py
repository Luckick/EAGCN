import torch
from utils import tsne_plot
from sklearn.manifold import TSNE, Isomap, MDS, locally_linear_embedding, SpectralEmbedding
from os import listdir
import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.markers
#from umap import UMAP
from collections import OrderedDict
import itertools
from sklearn.decomposition import PCA

data_path_atom = '../tsne/data/atom/'
data_path_mol = '../tsne/data/mol/'
dim_reduc_method = 'tsne' # 'umap'
type_care = 'N' # C, N, O, CNO
dataset = 'lipo'

print(type_care)

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

"""
def tsne_from_file(file_name, number = 10, n_components=2, random_state=2, perplexity=50, n_iter=500, early_exaggeration = 50):
    X_all = torch.load(file_name)
    dataset = file_name.split('_')[-2]
    epoch = file_name.split('_')[-1]
    tsne_plot(X_all, dataset, epoch, number = number, n_components=n_components, random_state=random_state,
              early_exaggeration = early_exaggeration, perplexity=perplexity, n_iter=n_iter)

path = './atom_rep'
for f in listdir(path):
    chars = f.split('_')
    if chars[2] == 'hiv': # and int(chars[3]) > 800:
        print(f)
        tsne_from_file(os.path.join(path, f))
"""

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_subtype(subtype_tsne_data, dataset, epoch, method, type_care, select_mol_index = 0):
    X = np.array(subtype_tsne_data[0])
    #belong_mol_index = np.array(subtype_tsne_data[2])
    y = np.array([int(ele-1) for ele in subtype_tsne_data[1]])
    if type_care == 'O':
        plot_range = range(0, 4)
    elif type_care == 'C':
        plot_range = range(5,10)
    elif type_care == 'N':
        plot_range = list(range(11,15)) + [16] # list(range(11,17))#
    elif type_care == 'CNO':
        plot_range = list(range(1, 4)) + list(range(5, 10)) + list(range(11, 17))

    X_rand_select, y_rand_select = np.array([]), np.array([])
    for k in plot_range:
        filter_select = y == k
        #must_select = belong_mol_index == select_mol_index
        if sum(filter_select) > 1000:
            rand = np.random.uniform(size = len(filter_select))
            rand_select = rand < 1000/sum(filter_select)
            row = np.logical_and(filter_select, rand_select)
            #row = np.logical_or(row, must_select)
            if len(X_rand_select) > 0:
                X_rand_select = np.concatenate((X_rand_select, X[row]), axis=0)
                y_rand_select = np.concatenate((y_rand_select, y[row]), axis=0)
            else:
                X_rand_select = X[row]
                y_rand_select = y[row]
        else:
            row = filter_select
            if sum(row) == 0:
                pass
            elif len(X_rand_select) > 0:
                X_rand_select = np.concatenate((X_rand_select, X[row]), axis=0)
                y_rand_select = np.concatenate((y_rand_select, y[row]), axis=0)
            else:
                X_rand_select = X[row]
                y_rand_select = y[row]

    if method == 'tsne':
        # Some good try:
        # perplexity=50.0, early_exaggeration=18.0, learning_rate=800.0,
        # tsne = TSNE()
        tsne = TSNE(n_components=2, perplexity=100.0, early_exaggeration=30.0, learning_rate=800.0,
                    n_iter=2000, n_iter_without_progress=300, min_grad_norm=1e-07,
                    metric='euclidean', init='random', verbose=0, random_state=None,
                    method='barnes_hut', angle=0.5)
        #X_2d = tsne.fit_transform(X)
        X_2d = tsne.fit_transform(X_rand_select)
    elif method == 'umap':
        umap = UMAP() # n_components=components, metric='cosine', init='random', n_neighbors=5
        X_2d = umap.fit_transform(X_rand_select)
    elif method == 'pca':
        pca = PCA(n_components=2)
        pca.fit(X)
        X_2d = pca.transform(X_rand_select)
    plt.figure(figsize=(30, 24))
    if 'lipo' in dataset:
        alpha = 0.8
    else:
        alpha = 0.8

    sub_type = ['os', 'oh', 'oa', 'o', 'un O', 'c3', 'c1', 'ca', 'c', 'c2', 'un C', 'n1', 'n3', 'na', 'n', 'no',
					'nh', 'un N']

    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lightpink",  "cyan", "gold",
              (0.1, 0.2, 0.3),  (0.2, 0.3, 0.4),  (0.3, 0.4, 0.5), (0.1, 0.2, 0.8),
              (0.4, 0.5, 0.6), (0.1, 0.3, 0.5), (0.7, 0.3, 0.2), (0.6,0.7, 0.8)]
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lightpink", "cyan", "gold",
              'tan', 'teal', 'olive', 'khaki', 'coral', 'goldenrod', 'maroon', 'orchid']

    markers = ['8', 'P' , '^' , 's' , '|' ,
               '*', 'P', '^',  's', 'o', '3',
               '*', 'o', 'H', 'X', '<', 'P', '4']
    # markers = ['8', 'P', '^', 's', '|',
    #            9, 'p', 'D', 'd', 'o', '3',
    #            '*', 'h', 'H', 'X', '<', 7, '4']
    #markers = ['o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd']

    dots = list(range(18))
    #fig, ax = plt.subplots()
    #plt.figure()
    if method == 'tsne':
        plt.xlim(X_2d[:, 0].min()-5, X_2d[:, 0].max()+20)
    for k in plot_range:
        row = y_rand_select == k
        print(sub_type[k], sum(row))
        #plt.scatter(X_2d[row, 0], X_2d[row, 1], alpha=0.5, c=cmap(k-5), label=sub_type[k])
        plt.scatter(X_2d[row, 0], X_2d[row, 1],
                    s = 800,
                    # c= colors[k],
                    marker = markers[k],
                    alpha=alpha, label=sub_type[k])
    plt.legend(markerscale=3, loc='right', bbox_to_anchor=(1.1,  0.5), fontsize = 70)
    # plt.show()
    if method == 'tsne':
        title_method = 't-SNE'
    elif method == 'umap':
        title_method = 'UMAP'
    elif method == 'pca':
        title_method = 'PCA'
    # plt.xlabel('X', fontsize = 80);
    # plt.ylabel('Y', fontsize = 80);
    plt.title('{} Visualization of Atom Representations'.format(title_method), fontsize = 80)

    save_plot_dir = '../{}/subtype_plots/'.format(method)
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    tsne_plot_name = '{}{}_{}.jpg'.format(save_plot_dir, dataset, type_care)
    plt.savefig(tsne_plot_name, dpi = 200)
    plt.close()

    if type_care == 'CNO':
        colors = ['b', 'g', 'r']
        sub_type = ['O', 'C', 'N']
        plt.figure(figsize=(30, 24))
        # for k in plot_range:
        for k in list(range(0, 3)):
            if k == 0:
                numbers = list(range(1, 4))
            elif k == 1:
                numbers = list(range(5, 10))
            elif k == 2:
                numbers = list(range(11, 17))
            row = np.zeros((y_rand_select.shape), dtype=bool)
            for j in numbers:
                row = np.logical_or(row, y_rand_select == j)
            print(sub_type[k], sum(row))
            # plt.scatter(X_2d[row, 0], X_2d[row, 1], alpha=0.5, c=cmap(k-5), label=sub_type[k])
            plt.scatter(X_2d[row, 0], X_2d[row, 1],
                        s=800,
                        # c=colors[k],
                        marker = markers[k+1],
                        alpha=alpha, label=sub_type[k])
        plt.legend(markerscale=3, loc='right', bbox_to_anchor=(1.1, 0.5), fontsize=60)
        # plt.legend(markerscale=3, loc='right', bbox_to_anchor=(1.1, 0.5), fontsize=45)
        # plt.show()
        if method == 'tsne':
            title_method = 't-SNE'
        elif method == 'umap':
            title_method = 'UMAP'
        # plt.xlabel('X', fontsize=80);
        # plt.ylabel('Y', fontsize=80);
        plt.title('{} Visualization of Atom Representations'.format(title_method), fontsize=80)

        save_plot_dir = '../{}/NoSub_plots/'.format(method)
        if not os.path.exists(save_plot_dir):
            os.makedirs(save_plot_dir)
        tsne_plot_name = '{}{}_NoSub.jpg'.format(save_plot_dir, dataset)
        plt.savefig(tsne_plot_name, dpi=200)
        plt.close()


    #for key, value in count_dic:
    #    print(sub_type[int(key)], value)

def plot_graph(mol_rep_data, dataset, epoch, method, label_index = 1, select_mol_index = 0):
    X = np.array(mol_rep_data[0])
    mol_index = np.array(mol_rep_data[1])
    #y = np.array([int(ele-1) for ele in mol_rep_data[1]])
    y = mol_rep_data[2]

    # Every time, need to update real label_index.
    flag1 = y[:, 3] == 1
    flag2 = y[:, 3] == 0

    if sum(flag1) > 1000:
        rand = np.random.uniform(size=flag1.shape[0])
        rand_select = rand < 1000 / sum(flag1)
        flag1 = np.logical_and(flag1, rand_select)

    if sum(flag2) > 2000:
        rand = np.random.uniform(size=flag2.shape[0])
        rand_select = rand < 2000 / sum(flag2)
        flag2 = np.logical_and(flag2, rand_select)

    filter_select = np.logical_or(flag1, flag2)

    if method == 'tsne':
        # Some good try:
        # perplexity=50.0, early_exaggeration=18.0, learning_rate=800.0,
        tsne = TSNE()
        tsne = TSNE(n_components=2, perplexity=50.0, early_exaggeration=100.0, learning_rate=800.0,
                    n_iter=2000, n_iter_without_progress=300, min_grad_norm=1e-07,
                    metric='euclidean', init='random', verbose=0, random_state=None,
                    method='barnes_hut', angle=0.5)
        # X_2d = tsne.fit_transform(X)
        X_2d = tsne.fit_transform(X[filter_select])
    elif method == 'umap':
        umap = UMAP() # n_components=components, metric='cosine', init='random', n_neighbors=5
        X_2d = umap.fit_transform(X[filter_select])
    elif method =='pca':
        pca = PCA(n_components=2)
        pca.fit(X)
        X_2d = pca.transform(X[filter_select])
    plt.figure(figsize=(30, 24))
    if 'lipo' in dataset:
        alpha = 0.8
    else:
        alpha = 0.8

    sub_type = ['os', 'oh', 'oa', 'o', 'un O', 'c3', 'c1', 'ca', 'c', 'c2', 'un C', 'n1', 'n3', 'na', 'n', 'no',
					'nh', 'un N']

    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lightpink",  "cyan", "gold",
              (0.1, 0.2, 0.3),  (0.2, 0.3, 0.4),  (0.3, 0.4, 0.5), (0.1, 0.2, 0.8),
              (0.4, 0.5, 0.6), (0.1, 0.3, 0.5), (0.7, 0.3, 0.2), (0.6,0.7, 0.8)]
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lightpink", "cyan", "gold",
              'tan', 'teal', 'olive', 'khaki', 'coral', 'goldenrod', 'maroon', 'orchid']

    markers = ['o' , '^' , 's' , '|' , 9, 'P', 'D',  'd', 'o', '3', '*', 'h', 'H', 'X', '<', 7, '4']
    #markers = ['o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd', 'o', '*', '+', 'x', 'd']

    labels = ['Inactive', 'Active']
    dots = list(range(18))

    #y_filtered = y[filter_select]
    for k in [0, 1]:
        row = y[filter_select, 3] == k
        #print(sub_type[k], sum(row))
        plt.scatter(X_2d[row, 0], X_2d[row, 1],
                    s = 800,
                    # c= colors[k],
                    marker = markers[k],
                    alpha=alpha, label=k)
    plt.legend(markerscale=3, loc='right', bbox_to_anchor=(1.1, 0.5), fontsize = 70)
    # plt.show()
    if method == 'tsne':
        title_method = 't-SNE'
    elif method == 'umap':
        title_method = 'UMAP'
    elif method =='pca':
        title_method = 'PCA'
    # plt.xlabel(title_method + ' 1', fontsize = 'x-large');
    # plt.ylabel(title_method + ' 2', fontsize = 'x-large');
    plt.title('{} Visualization of Molecule Representations'.format(title_method), fontsize = 80)

    save_plot_dir = '../{}/graph_plots/'.format(method)
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    tsne_plot_name = '{}{}.jpg'.format(save_plot_dir, dataset)
    plt.savefig(tsne_plot_name, dpi = 200)
    plt.close()


def plot_subtype_from_file(file_name, method, type_care):
    subtype_tsne_data = torch.load(file_name)
    dataset = file_name.split('/')[-1]
    epoch = file_name.split('_')[-1]
    plot_subtype(subtype_tsne_data, dataset, epoch, method, type_care)

def plot_graph_from_file(file_name, method):
    graph_data = torch.load(file_name)
    dataset = file_name.split('/')[-1]
    epoch = file_name.split('_')[-1]
    plot_graph(graph_data, dataset, epoch, method, type_care)

for f in listdir(data_path_atom):
    if 'GCN' in f:
        continue
    chars = f.split('_')
    if chars[0] == dataset:
        if (dataset == 'freesolv' and int(chars[2]) == 1500) or \
                (dataset == 'lipo' and int(chars[2]) == 500) or \
                (dataset == 'nih' and int(chars[2]) == 200) or \
                (dataset == 'esol' and int(chars[2]) == 2500) or \
                (dataset == 'tox21' and int(chars[2]) ==100) or \
                (dataset == 'hiv' and int(chars[2]) == 200):
            print(f)
            plot_subtype_from_file(os.path.join(data_path_atom, f), method = dim_reduc_method, type_care=type_care)

# if dataset == 'tox21' or dataset == 'nih' or dataset == 'hiv':
#     for f in listdir(data_path_mol):
#         if 'GCN' in f:
#             continue
#         chars = f.split('_')
#         if chars[0] == dataset:
#             print(f)
#             plot_graph_from_file(os.path.join(data_path_mol, f), method=dim_reduc_method)
#             # for hiv, label index = 3 is good one.
#             # TOX: 0, 8, 10,
#
