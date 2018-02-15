# EAGCN
Implementation of [Edge Attention based Multi-relational Graph Convolutional Networks](https://arxiv.org/pdf/1802.04944.pdf), which is submitted to KDD 2018


## Model Structure
### Attention Layer:
Element Attention Machanism:
![Element Attention Machanism](./Chart/layers.png)

Graph Convolution:
![Graph Convolution](./Chart/axw.png)


## Experiment Result
Classification Performance on Tox21 Dataset
![Tox21 Classification AUC](./Chart/Tox21_12tasks.png)

RMSE for Regression tasks on  Freesolv and Lipo:
![](./Chart/RMSE.jpeg)

ROC-AUC for Classification tasks on HIV and Tox21:
![](./Chart/AUC.jpeg)


## Acknowledgments
Code is inspired by [GCN](https://github.com/tkipf/gcn) and [conv_qsar_fast](https://github.com/connorcoley/conv_qsar_fast)


