import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, data) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.train_mask
    loss = loss_op(out[mask], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index)
    predT = torch.sigmoid(pred)
    predT = (predT > 0.49).float()
    mask = data.test_mask
    y_pred = predT[mask]
    y_pred = y_pred.to('cpu')
    y_true = data.test_y
    y_true = y_true.to('cpu')

    micro = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    classificationReport = classification_report(y_true, y_pred, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))

    return micro, macro, classificationReport



rounds = 6
epochs = 21
hidden_channel = 512
testsizes = [0.2, 0.4, 0.6, 0.8]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


############################################## Test Case Paper Top ##############################################
print('Start Test Case Paper_Top')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Top/Homo_data_a1.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Top/HomoGraph_A_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_a/Paper_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNPAPER', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDPAPER')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'paper'
targetColumn = '1ST_CAT_ID'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [int(pClass) for pClass in item[targetColumn].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [int(pClass) for pClass in item[targetColumn].split(',')]
    for j in labelInt:
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
data_micro = []
data_macro = []

for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []
    
    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro

dfMicro.to_csv('./model/GCN_micro_1_1_1.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_1_1.csv', encoding='utf-8-sig', index=False)

print('End Test Case Paper_Top')


############################################## Test Case Paper Sub ###########################################
print('Start Test Case Paper_Sub')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Sub/Homo_data_b1.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Sub/HomoGraph_B_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_b/Paper_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNPAPER', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDPAPER')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'paper'
targetColumn = '2ND_CAT_ID_100'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [int(pClass) for pClass in item[targetColumn].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [int(pClass) for pClass in item[targetColumn].split(',')]
    for j in labelInt:
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []

    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro

dfMicro.to_csv('./model/GCN_micro_1_1_2.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_1_2.csv', encoding='utf-8-sig', index=False)

print('End Test Case Paper_Sub')

################################################# Test Case Patent Top #############################################
print('Start Test Case Patent_Top')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Top/Homo_data_a2.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Top/HomoGraph_A_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNPATENT', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDPATENT')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'patent'
targetColumn = 'ClassID'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [int(pClass) for pClass in item[targetColumn].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [int(pClass) for pClass in item[targetColumn].split(',')]
    for j in labelInt:
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []

    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro
    
dfMicro.to_csv('./model/GCN_micro_1_2_1.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_2_1.csv', encoding='utf-8-sig', index=False)

print('End Test Case Patent_Top')


################################################# Test Case Patent Sub #############################################
print('Start Test Case Patent_Sub')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Top/Homo_data_a2.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Top/HomoGraph_A_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNPATENT', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDPATENT')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'patent'
targetColumn = 'SubClassID'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [int(pClass) for pClass in item[targetColumn].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [int(pClass) for pClass in item[targetColumn].split(',')]
    for j in labelInt:
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []

    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro

dfMicro.to_csv('./model/GCN_micro_1_2_2.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_2_2.csv', encoding='utf-8-sig', index=False)

print('End Test Case Patent_Top')


################################################# Test Case Report Top #############################################
print('Start Test Case Report_Top')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Top/Homo_data_a3.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Top/HomoGraph_A_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_a/Report_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNREPORT', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDREPORT')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'report'
targetColumn = 'ClassID'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

# Report y index
classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [item[targetColumn]]
    classes.extend(cls)

num_classes = int(max(classes) + 1)
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [item[targetColumn]]
    for j in labelInt:
        j = int(j)
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []

    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro
    
dfMicro.to_csv('./model/GCN_micro_1_3_1.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_3_1.csv', encoding='utf-8-sig', index=False)

print('End Test Case Report_Top')


################################################# Test Case Report Sub #############################################
print('Start Test Case Report_Sub')
dfMicro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
dfMacro = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

# Load Graph
data_list = torch.load('../SATGraph_Top/Homo_data_a3.pt')
data = data_list[0]
data = data.to(device)

# Merge Node ID
dfNode = pd.read_csv('../SATGraph_Top/HomoGraph_A_DataID.csv')
dfContentLabel = pd.read_csv('../SATDataset_ver2.1_a/Report_Label.csv', low_memory=False)
dfContentLabel = pd.merge(left=dfNode, right=dfContentLabel, left_on='CN', right_on='CNREPORT', how='left')
dfContentLabel = dfContentLabel.dropna()
dfContentLabel = dfContentLabel.sort_values('IDREPORT')
dfContentLabel = dfContentLabel.reset_index(drop=True)

# Search Targets Node ID
target = 'report'
targetColumn = 'SubClassID'
dfTargetContent = dfContentLabel[dfContentLabel['type'] == target]

# Report y index
classes = []
for idx, item in dfTargetContent.iterrows():
    cls = [item[targetColumn]]
    classes.extend(cls)

num_classes = int(max(classes) + 1)
yClass = torch.zeros((dfTargetContent.shape[0], num_classes), dtype=torch.float)
for i, item in dfTargetContent.iterrows():
    labelInt = [item[targetColumn]]
    for j in labelInt:
        j = int(j)
        yClass[i, j] = 1.

data.y = yClass
data.y_index = torch.from_numpy(dfTargetContent['ID'].values)

# Test Setup
for roundIDX in range(1, rounds):
    data_micro = []
    data_macro = []

    for testsize in testsizes:
        print('Test Size: ', testsize)

        # Set Train/Test Ratio
        x = data.y_index
        y = data.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, shuffle=True, random_state=1)

        # Define train/test mask on DATA
        temp = dict(zip(x_train.tolist(), y_train.tolist()))
        temp = dict(sorted(temp.items()))
        x_train = list(temp.keys())
        x_train = torch.tensor(x_train)
        y_train = list(temp.values())
        y_train = torch.tensor(y_train)
        data.train_y_index = x_train
        data.train_y = y_train

        temp = dict(zip(x_test.tolist(), y_test.tolist()))
        temp = dict(sorted(temp.items()))
        x_test = list(temp.keys())
        x_test = torch.tensor(x_test)
        y_test = list(temp.values())
        y_test = torch.tensor(y_test)
        data.test_y_index = x_test
        data.test_y = y_test

        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_train] = True
        data.train_mask = mask
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[x_test] = True
        data.test_mask = mask

        # Define Model
        model = GCN(data.num_features, hidden_channels=hidden_channel, out_channels=num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.001)
        loss_op = torch.nn.BCEWithLogitsLoss()

        # Train
        for epoch in range(1, epochs):
            loss = train(model, data)
            print("epoch: ", epoch, "loss: ", loss)
            if epoch % 20 == 0:
                micro, macro, classificationReport = test(model, data)

        data_micro.append(micro)
        data_macro.append(macro)

    dfMicro.loc[roundIDX] = data_micro
    dfMacro.loc[roundIDX] = data_macro

dfMicro.to_csv('./model/GCN_micro_1_3_2.csv', encoding='utf-8-sig', index=False)
dfMacro.to_csv('./model/GCN_macro_1_3_2.csv', encoding='utf-8-sig', index=False)

print('End Test Case Report_Sub')
