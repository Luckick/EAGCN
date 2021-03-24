

# EAGCN

This is a PyTorch implementation of paper "[Multi-View Spectral Graph Convolution with Consistent Edge Attention for Molecular Modeling](https://www.sciencedirect.com/science/article/abs/pii/S092523122100271X)" published at Neurocomputing. We also released a previous [arXiv version](https://arxiv.org/abs/1802.04944v1).

## Installation

Install pytorch and torchvision. 

## Train EAGCN model

### Dataset

Four benchmark datasets ([Tox21, HIV, Freesolv and Lipophilicity](http://moleculenet.ai/datasets-1)) are utilized in this study to evaluate the predictive performance of built graph convolutional networks.  They are all downloaded from the [MoleculeNet](http://moleculenet.ai/) that hold various benchmark datasets for molecular machine learning.

Datasets are also provided in folder "Data".

### Train the model
Open the folder "eagcn_pytorch".

When you train the model, you can use:

    python train.py

support files:    
EAGCN_dataset.py: pre-processing data      
neural_fp.py: from smiles to graph     
layers.py: define layers     
models.py: define models     
utils.py: other tools     


### Visualization Tools
check_model.py: check parameters (edge attention for each layer).     
mol_to_vec.py: visualize the molecule in 2D space, compare with other molecules which have similiar SMILEs.      
plot.py: show model training process.      
tsnes.py: tsne visualization about atom subtype, also provide umap option.     
kmeans_atomrep.py: kmeans clustering for atom subtype.     
plot_molecule.py: plot single molecule.     


## Acknowledgments
Code is inspired by [GCN](https://github.com/tkipf/gcn) and [conv_qsar_fast](https://github.com/connorcoley/conv_qsar_fast)


