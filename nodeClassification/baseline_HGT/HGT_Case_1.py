import numpy as np
import pandas as pd

import torch
from torch_geometric.nn import HGTConv, Linear

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
rounds = 10
epochs = 3001

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, target, data):
        super().__init__()
        self.target = target
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict[self.target])


def train(model, data, target, optimizer, loss_op) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target].train_mask
    loss = loss_op(out[mask], data[target].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, target):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict)
    predT = torch.sigmoid(pred)
    predT = (predT > 0.49).float()
    mask = data[target]['test_mask']
    y_true = data[target].y[mask]
    y_true = y_true.to('cpu')
    y_pred = predT[mask]
    y_pred = y_pred.to('cpu')

    micro = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    classificationReport = classification_report(y_true, y_pred, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))

    return micro, macro, classificationReport


micro_1_1_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_1_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
micro_1_1_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_1_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
micro_1_2_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_2_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
micro_1_2_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_2_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
micro_1_3_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_3_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
micro_1_3_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_3_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

for roundIDX in range(1, rounds):
    ################################################# Test Case Paper Top ##############################################
    print('Start Test Case Paper_Top')
    target = 'paper'
    data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
    data = data_list[0]

    data.__delitem__(('paper', 'published_in', 'journal'))
    data.__delitem__(('journal', 'published_in', 'paper'))
    data.__delitem__(('journal'))

    # Paper y index
    dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_a/Paper_Label.csv', low_memory=False)
    dfPaperLabel = dfPaperLabel.sort_values('IDPAPER')
    dfPaperLabel = dfPaperLabel.reset_index(drop=True)
    classes = []
    for idx, item in dfPaperLabel.iterrows():
        cls = [int(pClass) for pClass in item['1ST_CAT_ID'].split(',')]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((dfPaperLabel.shape[0], num_classes), dtype=torch.float)
    for i, item in dfPaperLabel.iterrows():
        labelInt = [int(pClass) for pClass in item['1ST_CAT_ID'].split(',')]
        for j in labelInt:
            yClass[i, j] = 1.

    data['paper'].y = yClass
    data['paper'].y_index = torch.from_numpy(dfPaperLabel['IDPAPER'].index.values)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_1_1_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=8, num_heads=8, num_layers=3, target='paper', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Set Train/Test Ratio
        x = data['paper'].y_index
        y = data['paper'].y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)
        mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['paper']['train_mask'] = mask
        mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['paper']['test_mask'] = mask

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_1_1_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_1_1.loc[roundIDX] = data_micro
    macro_1_1_1.loc[roundIDX] = data_macro
    print('End Test Case Paper_Top')


    ################################################# Test Case Paper Sub ##############################################
    print('Start Test Case Paper_Sub')
    target = 'paper'
    data_list = torch.load('../SATGraph_Sub/Hetero_data_sub.pt')
    data = data_list[0]

    data.__delitem__(('paper', 'published_in', 'journal'))
    data.__delitem__(('journal', 'published_in', 'paper'))
    data.__delitem__(('journal'))

    # Paper y index
    dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_b/Paper_Label.csv', low_memory=False)
    dfPaperLabel = dfPaperLabel.sort_values('IDPAPER')
    dfPaperLabel = dfPaperLabel.reset_index(drop=True)
    classes = []
    for idx, item in dfPaperLabel.iterrows():
        cls = [int(pClass) for pClass in item['2ND_CAT_ID_100'].split(',')]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((dfPaperLabel.shape[0], num_classes), dtype=torch.float)
    for i, item in dfPaperLabel.iterrows():
        labelInt = [int(pClass) for pClass in item['2ND_CAT_ID_100'].split(',')]
        for j in labelInt:
            yClass[i, j] = 1.

    data['paper'].y = yClass
    data['paper'].y_index = torch.from_numpy(dfPaperLabel['IDPAPER'].index.values)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_1_2_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=100, num_heads=8, num_layers=3, target='paper', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Set Train/Test Ratio
        x = data['paper'].y_index
        y = data['paper'].y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)
        mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['paper']['train_mask'] = mask
        mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['paper']['test_mask'] = mask

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_1_2_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_1_2.loc[roundIDX] = data_micro
    macro_1_1_2.loc[roundIDX] = data_macro

    print('End Test Case Paper_Sub')

    ####################################################################################################################
    ################################################# Test Case Patent Top #############################################
    print('Start Test Case Patent_Top')
    target = 'patent'
    data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
    data = data_list[0]

    data.__delitem__(('patent', 'is_classified', 'ipc'))
    data.__delitem__(('ipc', 'is_classified', 'patent'))
    data.__delitem__(('ipc'))

    # Patent y index
    dfPatentLabel = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv', low_memory=False)
    dfPatentLabel = dfPatentLabel.sort_values('IDPATENT')
    dfPatentLabel = dfPatentLabel.reset_index(drop=True)
    classes = []
    for idx, item in dfPatentLabel.iterrows():
        cls = [int(pClass) for pClass in item['ClassID'].split(',')]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((data['patent'].num_nodes, num_classes), dtype=torch.float)
    for i, item in dfPatentLabel.iterrows():
        i = item['IDPATENT']
        labelInt = [int(pClass) for pClass in item['ClassID'].split(',')]
        for j in labelInt:
            yClass[i, j] = 1.

    data['patent'].y = yClass
    data['patent'].y_index = torch.arange(0, data['patent'].num_nodes, 1)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_2_1_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=54, num_heads=8, num_layers=3, target='patent', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Split Train/Test Data
        x = dfPatentLabel['IDPATENT'].values
        y = dfPatentLabel['ClassID'].tolist()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        mask = torch.zeros(data['patent'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['patent']['train_mask'] = mask
        mask = torch.zeros(data['patent'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['patent']['test_mask'] = mask
        data, model = data.to(device), model.to(device)

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_2_1_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_2_1.loc[roundIDX] = data_micro
    macro_1_2_1.loc[roundIDX] = data_macro

    print('End Test Case Patent_Top')

    ################################################# Test Case Patent Sub #############################################
    print('Start Test Case Patent_Sub')
    target = 'patent'
    data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
    data = data_list[0]

    data.__delitem__(('patent', 'is_classified', 'ipc'))
    data.__delitem__(('ipc', 'is_classified', 'patent'))
    data.__delitem__(('ipc'))

    # Patent y index
    dfPatentLabel = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv', low_memory=False)
    dfPatentLabel = dfPatentLabel.sort_values('IDPATENT')
    dfPatentLabel = dfPatentLabel.reset_index(drop=True)
    classes = []
    for idx, item in dfPatentLabel.iterrows():
        cls = [int(pClass) for pClass in item['SubClassID'].split(',')]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((data['patent'].num_nodes, num_classes), dtype=torch.float)
    for i, item in dfPatentLabel.iterrows():
        i = item['IDPATENT']
        labelInt = [int(pClass) for pClass in item['SubClassID'].split(',')]
        for j in labelInt:
            yClass[i, j] = 1.

    data['patent'].y = yClass
    data['patent'].y_index = torch.arange(0, data['patent'].num_nodes, 1)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_2_2_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=139, num_heads=8, num_layers=3, target='patent', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Split Train/Test Data
        x = dfPatentLabel['IDPATENT'].values
        y = dfPatentLabel['SubClassID'].tolist()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        mask = torch.zeros(data['patent'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['patent']['train_mask'] = mask
        mask = torch.zeros(data['patent'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['patent']['test_mask'] = mask
        data, model = data.to(device), model.to(device)

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_2_2_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_2_2.loc[roundIDX] = data_micro
    macro_1_2_2.loc[roundIDX] = data_macro

    print('End Test Case Patent_Sub')

    ####################################################################################################################
    ################################################# Test Case Report Top #############################################
    print('Start Test Case Report_Top')
    target = 'report'
    data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
    data = data_list[0]

    dfReportLabel = pd.read_csv('../SATDataset_ver2.1_a/Report_Label.csv', low_memory=False)
    dfReportLabel = dfReportLabel.sort_values('IDREPORT')
    dfReportLabel = dfReportLabel.reset_index(drop=True)

    # Report y index
    classes = []
    for idx, item in dfReportLabel.iterrows():
        cls = [item['ClassID']]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((data['report'].num_nodes, num_classes), dtype=torch.float)
    for i, item in dfReportLabel.iterrows():
        i = item['IDREPORT']
        labelInt = [item['ClassID']]
        for j in labelInt:
            yClass[i, j] = 1.

    data['report'].y = yClass
    data['report'].y_index = torch.arange(0, data['report'].num_nodes, 1)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_3_1_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=24, num_heads=8, num_layers=3, target='report', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Split Train/Test Data
        num_classes = max(classes) + 1
        yClass = torch.zeros((data['report'].num_nodes, num_classes), dtype=torch.float)
        for i, item in dfReportLabel.iterrows():
            i = item['IDREPORT']
            labelInt = [item['ClassID']]
            for j in labelInt:
                yClass[i, j] = 1.

        # Split Train/Test Data
        x = dfReportLabel['IDREPORT'].values
        y = dfReportLabel['ClassID'].tolist()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        mask = torch.zeros(data['report'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['report']['train_mask'] = mask
        mask = torch.zeros(data['report'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['report']['test_mask'] = mask
        data, model = data.to(device), model.to(device)

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_3_1_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_3_1.loc[roundIDX] = data_micro
    macro_1_3_1.loc[roundIDX] = data_macro

    print('End Test Case Report_Top')

    ################################################# Test Case Report Sub #############################################
    print('Start Test Case Report_Sub')
    target = 'report'
    data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
    data = data_list[0]

    dfReportLabel = pd.read_csv('../SATDataset_ver2.1_a/Report_Label.csv', low_memory=False)
    dfReportLabel = dfReportLabel.sort_values('IDREPORT')
    dfReportLabel = dfReportLabel.reset_index(drop=True)

    classes = []
    for idx, item in dfReportLabel.iterrows():
        cls = [item['SubClassID']]
        classes.extend(cls)

    num_classes = max(classes) + 1
    yClass = torch.zeros((data['report'].num_nodes, num_classes), dtype=torch.float)
    for i, item in dfReportLabel.iterrows():
        i = item['IDREPORT']
        labelInt = [item['SubClassID']]
        for j in labelInt:
            yClass[i, j] = 1.

    data['report'].y = yClass
    data['report'].y_index = torch.arange(0, data['report'].num_nodes, 1)

    # Test Setup
    data_micro = []
    data_macro = []
    testsizes = [0.2, 0.4, 0.6, 0.8]
    for testsize in testsizes:
        print('Test Size: ', testsize)
        file = open("./model/model_HGT_Case_1_3_2_" + str(testsize) + '_' + str(roundIDX) + ".txt", "wt")

        # Set Model
        model = HGT(hidden_channels=224, out_channels=100, num_heads=8, num_layers=3, target='report', data=data)
        data, model = data.to(device), model.to(device)
        print(data)
        print(model)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # write file
        file.writelines('Test Ratio : ')
        file.writelines(str(testsize))
        file.writelines('\n')

        # Split Train/Test Data
        x = dfReportLabel['IDREPORT'].values
        y = dfReportLabel['SubClassID'].tolist()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        mask = torch.zeros(data['report'].num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data['report']['train_mask'] = mask
        mask = torch.zeros(data['report'].num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data['report']['test_mask'] = mask
        data, model = data.to(device), model.to(device)

        for epoch in range(1, epochs):
            loss = train(model, data, target, optimizer, loss_op)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data, target)
                file.writelines('Epoch: ' + str(epoch) + ', ')
                file.writelines('Loss: ' + str(loss) + ', ')
                file.writelines('F1-Micro: ' + str(micro) + ', ')
                file.writelines('F1-Macro: ' + str(macro) + ', ')
                file.writelines('\n')

        file.writelines('---------------------------------------------------------------------------------------------')
        file.close()

        #PATH = './model/model_HGT_Case_1_3_2_' + str(testsize) + '_' + str(roundIDX) + '.pt'
        #torch.save(model, PATH)  # 전체 모델 저장

        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_3_2.loc[roundIDX] = data_micro
    macro_1_3_2.loc[roundIDX] = data_macro
    print('End Test Case Report_Top')

    micro_1_1_1.to_csv('./model/HGT_micro_1_1_1.csv', encoding='utf-8-sig', index=False)
    macro_1_1_1.to_csv('./model/HGT_macro_1_1_1.csv', encoding='utf-8-sig', index=False)
    micro_1_1_2.to_csv('./model/HGT_micro_1_1_2.csv', encoding='utf-8-sig', index=False)
    macro_1_1_2.to_csv('./model/HGT_macro_1_1_2.csv', encoding='utf-8-sig', index=False)
    micro_1_2_1.to_csv('./model/HGT_micro_1_2_1.csv', encoding='utf-8-sig', index=False)
    macro_1_2_1.to_csv('./model/HGT_macro_1_2_1.csv', encoding='utf-8-sig', index=False)
    micro_1_2_2.to_csv('./model/HGT_micro_1_2_2.csv', encoding='utf-8-sig', index=False)
    macro_1_2_2.to_csv('./model/HGT_macro_1_2_2.csv', encoding='utf-8-sig', index=False)
    micro_1_3_1.to_csv('./model/HGT_micro_1_3_1.csv', encoding='utf-8-sig', index=False)
    macro_1_3_1.to_csv('./model/HGT_macro_1_3_1.csv', encoding='utf-8-sig', index=False)
    micro_1_3_2.to_csv('./model/HGT_micro_1_3_2.csv', encoding='utf-8-sig', index=False)
    macro_1_3_2.to_csv('./model/HGT_macro_1_3_2.csv', encoding='utf-8-sig', index=False)

micro_1_1_1.to_csv('./model/HGT_micro_1_1_1.csv', encoding='utf-8-sig', index=False)
macro_1_1_1.to_csv('./model/HGT_macro_1_1_1.csv', encoding='utf-8-sig', index=False)
micro_1_1_2.to_csv('./model/HGT_micro_1_1_2.csv', encoding='utf-8-sig', index=False)
macro_1_1_2.to_csv('./model/HGT_macro_1_1_2.csv', encoding='utf-8-sig', index=False)
micro_1_2_1.to_csv('./model/HGT_micro_1_2_1.csv', encoding='utf-8-sig', index=False)
macro_1_2_1.to_csv('./model/HGT_macro_1_2_1.csv', encoding='utf-8-sig', index=False)
micro_1_2_2.to_csv('./model/HGT_micro_1_2_2.csv', encoding='utf-8-sig', index=False)
macro_1_2_2.to_csv('./model/HGT_macro_1_2_2.csv', encoding='utf-8-sig', index=False)
micro_1_3_1.to_csv('./model/HGT_micro_1_3_1.csv', encoding='utf-8-sig', index=False)
macro_1_3_1.to_csv('./model/HGT_macro_1_3_1.csv', encoding='utf-8-sig', index=False)
micro_1_3_2.to_csv('./model/HGT_micro_1_3_2.csv', encoding='utf-8-sig', index=False)
macro_1_3_2.to_csv('./model/HGT_macro_1_3_2.csv', encoding='utf-8-sig', index=False)
print('end')

