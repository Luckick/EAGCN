# This is the code for visualizing the model parameters.

import torch
import os
from models import *
#import ggplot

# path = '../experiment_result/{}/{}/exp'.format('server', 'lipo')
# path = '../experiment_result/{}/{}/expt'.format('server', 'freesolv')

dataset = 'Esol'
plot_heatmap = True
plot_self = False
#model = torch.load()
normal_relation = False
if dataset == 'hiv':
    path = '../experiment_result/{}/{}/HIV_active'.format('server', 'HIV')
if dataset == 'Freesolv':
    path = '../experiment_result/{}/{}/expt'.format('server', 'freesolv')
    bond = ['6_6', '6_8', '6_7', '6_17', '7_8', '6_9', '6_16', '8_15', '6_35', '8_16', '15_16', '6_53', '7_7',
            '16_16', '7_16', '6_15', '16_17']
    bond = ['C_C', 'C_O', 'C_N', 'C_Cl', 'N_O', 'C_F', 'C_S', 'O_P', 'C_Br', 'O_S', 'P_S', 'C_I', 'N_N',
            'S_S', 'N_S', 'C_P', 'S_Cl']
if dataset == 'Lipo':
    path = '../experiment_result/{}/{}/exp'.format('server', 'lipo')
    bond = ['C_C', 'C_N', 'C_O', 'C_S', 'C_F', 'O_S', 'C_Cl', 'N_N', 'N_S', 'N_O', 'C_Br', 'O_P', 'C_I', 'B_O',
            'B_C', 'B_N', 'C_P', 'N_P']
    # bond = ['6_6', '6_7', '6_8', '6_16', '6_9', '8_16', '6_17', '7_7', '7_16', '7_8', '6_35', '8_15', '6_53', '5_8',
    # '5_6', '5_7', '6_15', '7_15']
if dataset == 'Esol':
    path = '../experiment_result/{}/{}/measured log solubility in mols per litre'.format('server', 'esol')
    # bond = ['6_6', '6_8', '6_7', '6_17', '6_16', '7_8', '8_15', '6_9', '6_35', '8_16', '15_16', '7_7',
    #         '7_16', '6_53', '16_16', '7_15', '6_15']
    bond = ['C_C', 'C_O', 'C_N', 'C_Cl', 'C_S', 'N_O', 'O_P', 'C_F', 'C_Br', 'O_S', 'P_S', 'N_N',
            'N_S', 'C_I', 'S_S', 'F_P', 'C_P']
if dataset == 'tox21':
    path = '../experiment_result/{}/{}/all_tasks'.format('server', 'tox21')
files = os.listdir(path)

for file in files:
    if 'New' in file and 'pt' in file and 'pool' not in file:
        new_path = os.path.join(path, file)
        print(new_path)
        model = torch.load(new_path, map_location='cpu')

        try:
            print('layer1')
            print(torch.sigmoid(model.layer1.block1.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block2.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block3.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block4.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block5.self_r.data.view(-1)).numpy())

            print(torch.sigmoid(model.layer1.block1.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block2.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block3.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block4.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer1.block5.att.weight.data.view(-1)).numpy())

            # layer1_data = []


            layer1_rel1 = []
            layer1_rel1 += list(torch.sigmoid(model.layer1.block1.self_r.data.view(-1)).numpy())
            layer1_rel1 += list(torch.sigmoid(model.layer1.block1.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel1 = list(layer1_rel1/sum(layer1_rel1))

            layer1_rel2 = []
            layer1_rel2 += list(torch.sigmoid(model.layer1.block2.self_r.data.view(-1)).numpy())
            layer1_rel2 += list(torch.sigmoid(model.layer1.block2.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel2 = list(layer1_rel2 / sum(layer1_rel2))

            layer1_rel3 = []
            layer1_rel3 += list(torch.sigmoid(model.layer1.block3.self_r.data.view(-1)).numpy())
            layer1_rel3 += list(torch.sigmoid(model.layer1.block3.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel3 = list(layer1_rel3 / sum(layer1_rel3))

            layer1_rel4 = []
            layer1_rel4 += list(torch.sigmoid(model.layer1.block4.self_r.data.view(-1)).numpy())
            layer1_rel4 += list(torch.sigmoid(model.layer1.block4.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel4 = list(layer1_rel4 / sum(layer1_rel4))

            layer1_rel5 = []
            layer1_rel5 += list(torch.sigmoid(model.layer1.block5.self_r.data.view(-1)).numpy())
            layer1_rel5 += list(torch.sigmoid(model.layer1.block5.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel5 = list(layer1_rel5 / sum(layer1_rel5))

            layer1_self = [layer1_rel1[0], layer1_rel2[0], layer1_rel3[0], layer1_rel4[0], layer1_rel5[0]]
            layer1_data = layer1_rel1 + [None] + layer1_rel2 + [None] + layer1_rel3 + [None] + layer1_rel4 + [None] + layer1_rel5
            # print(layer1_data)

            print('layer2')
            print(torch.sigmoid(model.layer2.block1.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block2.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block3.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block4.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block5.self_r.data.view(-1)).numpy())

            print(torch.sigmoid(model.layer2.block1.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block2.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block3.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block4.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer2.block5.att.weight.data.view(-1)).numpy())

            # layer2_data = []
            # layer2_data += list(torch.sigmoid(model.layer2.block1.self_r.data.view(-1)).numpy())
            # layer2_data += list(torch.sigmoid(model.layer2.block1.att.weight.data.view(-1)).numpy())
            # layer2_data.append(0)
            # layer2_data += list(torch.sigmoid(model.layer2.block2.self_r.data.view(-1)).numpy())
            # layer2_data += list(torch.sigmoid(model.layer2.block2.att.weight.data.view(-1)).numpy())
            # layer2_data.append(0)
            # layer2_data += list(torch.sigmoid(model.layer2.block3.self_r.data.view(-1)).numpy())
            # layer2_data += list(torch.sigmoid(model.layer2.block3.att.weight.data.view(-1)).numpy())
            # layer2_data.append(0)
            # layer2_data += list(torch.sigmoid(model.layer2.block4.self_r.data.view(-1)).numpy())
            # layer2_data += list(torch.sigmoid(model.layer2.block4.att.weight.data.view(-1)).numpy())
            # layer2_data.append(0)
            # layer2_data += list(torch.sigmoid(model.layer2.block5.self_r.data.view(-1)).numpy())
            # layer2_data += list(torch.sigmoid(model.layer2.block5.att.weight.data.view(-1)).numpy())

            layer1_rel1 = []
            layer1_rel1 += list(torch.sigmoid(model.layer2.block1.self_r.data.view(-1)).numpy())
            layer1_rel1 += list(torch.sigmoid(model.layer2.block1.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel1 = list(layer1_rel1 / sum(layer1_rel1))

            layer1_rel2 = []
            layer1_rel2 += list(torch.sigmoid(model.layer2.block2.self_r.data.view(-1)).numpy())
            layer1_rel2 += list(torch.sigmoid(model.layer2.block2.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel2 = list(layer1_rel2 / sum(layer1_rel2))

            layer1_rel3 = []
            layer1_rel3 += list(torch.sigmoid(model.layer2.block3.self_r.data.view(-1)).numpy())
            layer1_rel3 += list(torch.sigmoid(model.layer2.block3.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel3 = list(layer1_rel3 / sum(layer1_rel3))

            layer1_rel4 = []
            layer1_rel4 += list(torch.sigmoid(model.layer2.block4.self_r.data.view(-1)).numpy())
            layer1_rel4 += list(torch.sigmoid(model.layer2.block4.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel4 = list(layer1_rel4 / sum(layer1_rel4))

            layer1_rel5 = []
            layer1_rel5 += list(torch.sigmoid(model.layer2.block5.self_r.data.view(-1)).numpy())
            layer1_rel5 += list(torch.sigmoid(model.layer2.block5.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel5 = list(layer1_rel5 / sum(layer1_rel5))

            layer2_self = [layer1_rel1[0], layer1_rel2[0], layer1_rel3[0], layer1_rel4[0], layer1_rel5[0]]
            layer2_data = layer1_rel1 + [None] + layer1_rel2 + [None] + \
                          layer1_rel3 + [None] + layer1_rel4 + [None] + layer1_rel5

            print('layer3')
            print(torch.sigmoid(model.layer3.block1.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block2.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block3.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block4.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block5.self_r.data.view(-1)).numpy())

            print(torch.sigmoid(model.layer3.block1.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block2.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block3.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block4.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer3.block5.att.weight.data.view(-1)).numpy())

            # layer3_data = []
            # layer3_data += list(torch.sigmoid(model.layer3.block1.self_r.data.view(-1)).numpy())
            # layer3_data += list(torch.sigmoid(model.layer3.block1.att.weight.data.view(-1)).numpy())
            # layer3_data.append(0)
            # layer3_data += list(torch.sigmoid(model.layer3.block2.self_r.data.view(-1)).numpy())
            # layer3_data += list(torch.sigmoid(model.layer3.block2.att.weight.data.view(-1)).numpy())
            # layer3_data.append(0)
            # layer3_data += list(torch.sigmoid(model.layer3.block3.self_r.data.view(-1)).numpy())
            # layer3_data += list(torch.sigmoid(model.layer3.block3.att.weight.data.view(-1)).numpy())
            # layer3_data.append(0)
            # layer3_data += list(torch.sigmoid(model.layer3.block4.self_r.data.view(-1)).numpy())
            # layer3_data += list(torch.sigmoid(model.layer3.block4.att.weight.data.view(-1)).numpy())
            # layer3_data.append(0)
            # layer3_data += list(torch.sigmoid(model.layer3.block5.self_r.data.view(-1)).numpy())
            # layer3_data += list(torch.sigmoid(model.layer3.block5.att.weight.data.view(-1)).numpy())

            layer1_rel1 = []
            layer1_rel1 += list(torch.sigmoid(model.layer3.block1.self_r.data.view(-1)).numpy())
            layer1_rel1 += list(torch.sigmoid(model.layer3.block1.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel1 = list(layer1_rel1 / sum(layer1_rel1))

            layer1_rel2 = []
            layer1_rel2 += list(torch.sigmoid(model.layer3.block2.self_r.data.view(-1)).numpy())
            layer1_rel2 += list(torch.sigmoid(model.layer3.block2.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel2 = list(layer1_rel2 / sum(layer1_rel2))

            layer1_rel3 = []
            layer1_rel3 += list(torch.sigmoid(model.layer3.block3.self_r.data.view(-1)).numpy())
            layer1_rel3 += list(torch.sigmoid(model.layer3.block3.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel3 = list(layer1_rel3 / sum(layer1_rel3))

            layer1_rel4 = []
            layer1_rel4 += list(torch.sigmoid(model.layer3.block4.self_r.data.view(-1)).numpy())
            layer1_rel4 += list(torch.sigmoid(model.layer3.block4.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel4 = list(layer1_rel4 / sum(layer1_rel4))

            layer1_rel5 = []
            layer1_rel5 += list(torch.sigmoid(model.layer3.block5.self_r.data.view(-1)).numpy())
            layer1_rel5 += list(torch.sigmoid(model.layer3.block5.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel5 = list(layer1_rel5 / sum(layer1_rel5))

            layer3_self = [layer1_rel1[0], layer1_rel2[0], layer1_rel3[0], layer1_rel4[0], layer1_rel5[0]]
            layer3_data = layer1_rel1 + [None] + layer1_rel2 + [None] + \
                          layer1_rel3 + [None] + layer1_rel4 + [None] + layer1_rel5

            print('layer4')
            print(torch.sigmoid(model.layer4.block1.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block2.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block3.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block4.self_r.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block5.self_r.data.view(-1)).numpy())

            print(torch.sigmoid(model.layer4.block1.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block2.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block3.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block4.att.weight.data.view(-1)).numpy())
            print(torch.sigmoid(model.layer4.block5.att.weight.data.view(-1)).numpy())

            # layer4_data = []
            # layer4_data += list(torch.sigmoid(model.layer4.block1.self_r.data.view(-1)).numpy())
            # layer4_data += list(torch.sigmoid(model.layer4.block1.att.weight.data.view(-1)).numpy())
            # layer4_data.append(0)
            # layer4_data += list(torch.sigmoid(model.layer4.block2.self_r.data.view(-1)).numpy())
            # layer4_data += list(torch.sigmoid(model.layer4.block2.att.weight.data.view(-1)).numpy())
            # layer4_data.append(0)
            # layer4_data += list(torch.sigmoid(model.layer4.block3.self_r.data.view(-1)).numpy())
            # layer4_data += list(torch.sigmoid(model.layer4.block3.att.weight.data.view(-1)).numpy())
            # layer4_data.append(0)
            # layer4_data += list(torch.sigmoid(model.layer4.block4.self_r.data.view(-1)).numpy())
            # layer4_data += list(torch.sigmoid(model.layer4.block4.att.weight.data.view(-1)).numpy())
            # layer4_data.append(0)
            # layer4_data += list(torch.sigmoid(model.layer4.block5.self_r.data.view(-1)).numpy())
            # layer4_data += list(torch.sigmoid(model.layer4.block5.att.weight.data.view(-1)).numpy())

            layer1_rel1 = []
            layer1_rel1 += list(torch.sigmoid(model.layer4.block1.self_r.data.view(-1)).numpy())
            layer1_rel1 += list(torch.sigmoid(model.layer4.block1.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel1 = list(layer1_rel1 / sum(layer1_rel1))

            layer1_rel2 = []
            layer1_rel2 += list(torch.sigmoid(model.layer4.block2.self_r.data.view(-1)).numpy())
            layer1_rel2 += list(torch.sigmoid(model.layer4.block2.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel2 = list(layer1_rel2 / sum(layer1_rel2))

            layer1_rel3 = []
            layer1_rel3 += list(torch.sigmoid(model.layer4.block3.self_r.data.view(-1)).numpy())
            layer1_rel3 += list(torch.sigmoid(model.layer4.block3.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel3 = list(layer1_rel3 / sum(layer1_rel3))

            layer1_rel4 = []
            layer1_rel4 += list(torch.sigmoid(model.layer4.block4.self_r.data.view(-1)).numpy())
            layer1_rel4 += list(torch.sigmoid(model.layer4.block4.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel4 = list(layer1_rel4 / sum(layer1_rel4))

            layer1_rel5 = []
            layer1_rel5 += list(torch.sigmoid(model.layer4.block5.self_r.data.view(-1)).numpy())
            layer1_rel5 += list(torch.sigmoid(model.layer4.block5.att.weight.data.view(-1)).numpy())
            if normal_relation:
                layer1_rel5 = list(layer1_rel5 / sum(layer1_rel5))

            layer4_self = [layer1_rel1[0], layer1_rel2[0], layer1_rel3[0], layer1_rel4[0], layer1_rel5[0]]
            layer4_data = layer1_rel1 + [None] + layer1_rel2 + [None] + \
                          layer1_rel3 + [None] + layer1_rel4 + [None] + layer1_rel5


            if plot_heatmap:
                import plotly.plotly as py
                import plotly.graph_objs as go
                import plotly.io as pio

                trace = go.Heatmap(z=[layer2_data, layer1_data],
                                   x=['BondType_self'] + bond + ['  ', 'BondOrder_self'] +
                                     ['BondOrder_1', 'BondOrder_1.5', 'BondOrder_2', 'BondOrder_3'] +
                                     ['   ', 'Arom_self', 'Is_Arom', 'Not_Arom', '    ', 'Conj_self'] +
                                     ['Is_Conj', 'Not_Conj', '      ', 'Ring_self', 'In_Ring', 'Not_Ring'],
                                   y=['{} layer2'.format(dataset), '{} layer1'.format(dataset)],
                                   colorscale='Blues', reversescale=True,
                                   zmax=1, zmin=0,
                                   opacity=1)
                data = [trace]
                layout = go.Layout(
                    width=1000,
                    height=200,
                    margin=dict(
                        l=120,
                        r=120,
                        b=150,
                        t=5,
                        pad=4
                    ),
                    xaxis=dict(ticks=''),
                )
                fig = go.Figure(data=data, layout=layout)
                # py.iplot(data, filename='labelled-heatmap')
                pio.write_image(fig, './para/{}_{}_parameter.png'.format(dataset, file))

                if plot_self:
                    trace = go.Heatmap(z=[layer4_self, layer3_self, layer2_self, layer1_self],
                                       x= ['BondType_self'] + bond + [' ' + 'BondOrder_self'] +
                                          ['BondOrder_1', 'BondOrder_1.5', 'BondOrder_2', 'BondOrder_3'] +
                                          [' ', 'Arom_self', 'Is_Arom', 'Not_Arom', ' ', 'Conj_self'] +
                                          ['Is_Conj', 'Not_Conj', ' ', 'Ring_self', 'In_Ring', 'Not_Ring'],
                                       y=['layer4', 'layer3', 'layer2', 'layer1'],
                                       colorscale='Blues', reversescale=True,
                                       zmax=1, zmin=0,
                                       opacity=1)
                    data = [trace]
                    layout = go.Layout(
                        margin=dict(
                            l=300,
                            r=120,
                            b=170,
                            t=5,
                            pad=4
                        )
                    )
                    fig = go.Figure(data=data, layout=layout)
                    # py.iplot(data, filename='labelled-heatmap')
                    pio.write_image(fig, './para/{}_{}_self_parameter.png'.format(dataset, file))
        except AttributeError:
            print(file)
            print('Attribute Error, may due to not EAGCN model')
        else:
            pass

