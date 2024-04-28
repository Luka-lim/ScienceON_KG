import pandas as pd
import torch
from torch_geometric.nn import HGTConv, Linear
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random


# Select Paper
dfJournalTarget = pd.read_csv('journalEC.csv')
dfJournalTarget = dfJournalTarget[['CNJOURNAL']]
dfGoogleCategory = pd.read_csv('../SATDataset_ver2.1_b/Google_Category.csv')
dfJournalTarget = pd.merge(left=dfJournalTarget, right=dfGoogleCategory, on='CNJOURNAL', how='inner')
dfJournalTarget = dfJournalTarget[['CNJOURNAL', 'JOURNALTITLE', '1ST_CAT', '2ND_CAT']]

dfJournalCategory = pd.read_csv('../SATDataset_ver2.1_b/Journal_Category.csv')
dfJournal = pd.merge(left=dfJournalTarget, right=dfJournalCategory, on='CNJOURNAL', how='inner')
dfPaperJournalEdge = pd.read_csv('../SATDataset_ver2.1_b/Paper_Journal_edge.csv')
dfPaper = pd.merge(left=dfPaperJournalEdge, right=dfJournal, on=['CNJOURNAL', 'IDJOURNAL'], how='inner')

dfGroupTop = dfPaper.groupby('1ST_CAT_ID').count()
dfGroupTop = dfGroupTop.sort_values('1ST_CAT', ascending=False)
dfGroupSub = dfPaper.groupby('2ND_CAT_ID_100').count()
dfGroupSub = dfGroupSub.sort_values('2ND_CAT', ascending=False)

dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_b/Paper_Label.csv')
dfPaperJournal = pd.merge(left=dfPaperJournalEdge, right=dfPaperLabel, on=['CNPAPER', 'IDPAPER'], how='inner')
dfGoogleCategory = dfGoogleCategory[['CNJOURNAL', 'JOURNALTITLE', '1ST_CAT', '2ND_CAT']]
dfPaperJournalCategory = pd.merge(left=dfPaperJournal, right=dfGoogleCategory, on='CNJOURNAL', how='inner')

'''
# For dfTargetCategory
# Top 1, 2
# Sub (10,12) 40 25 18 27 (3,25) (8,40)
targets = ['3', '8', '10', '12', '18', '19', '20', '22', '25', '27', '28', '30', '34', '38', '39', '40', \
          '3,25', '8,40', '10,12', '22,38', '20,39']
topCategory = ['Chemical & Material Sciences', 'Engineering & Computer Science',
               'Chemical & Material Sciences / Engineering & Computer Science']
subCategory = []

dfPickPaper = pd.DataFrame()
for target in targets:
    temp = dfPaperJournalCategory[dfPaperJournalCategory['2ND_CAT_ID_100'] == target]
    dfPickPaper = pd.concat([dfPickPaper, temp], ignore_index=True)

dfTargetCategory = dfPickPaper[['1ST_CAT_ID', '2ND_CAT_ID_100', '1ST_CAT', '2ND_CAT']]
dfTargetCategory = dfTargetCategory.drop_duplicates()
dfTargetCategory.to_csv('temp.csv', encoding='utf-8-sig', index=False)
'''

# read dfTargetCategory
dfTargetCategory = pd.read_csv('targetCategory_5th.csv')

# Paper y data
dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_b/Paper_Label.csv', low_memory=False)
dfPaperLabel = dfPaperLabel.sort_values('IDPAPER')
dfPaperLabel = dfPaperLabel.reset_index(drop=True)
classes = []
for idx, item in dfPaperLabel.iterrows():
    cls = [int(pClass) for pClass in item['2ND_CAT_ID_100'].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
Y_data = torch.zeros((dfPaperLabel.shape[0], num_classes), dtype=torch.int)
for i, item in dfPaperLabel.iterrows():
    labelInt = [int(pClass) for pClass in item['2ND_CAT_ID_100'].split(',')]
    for j in labelInt:
        Y_data[i, j] = 1

Y_data = np.array(Y_data)
temp = []
classList = [x for x in Y_data.tolist() if x not in temp and not temp.append(x)]

# Paper y Index which is selected
targetIDs = []
for idx, item in dfTargetCategory.iterrows():
    idx = item['2ND_CAT_ID_100']
    temp = [0 for x in range(0, num_classes)]
    if ',' in idx:
        idxs = idx.split(',')
        for i in idxs:
            i = int(i)
            temp[i] = 1
        if idx == '5,29':
            temp[39] = 1

    else:
        idx = int(idx)
        temp[idx] = 1

    targetIDs.append(temp)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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


# Load Model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_list = torch.load('../SATGraph_Sub/Hetero_data_sub.pt')
data = data_list[0]
data.__delitem__(('paper', 'published_in', 'journal'))
data.__delitem__(('journal', 'published_in', 'paper'))
data.__delitem__(('journal'))
data.to(device)

model = HGT(hidden_channels=224, out_channels=100, num_heads=8, num_layers=3, target='paper', data=data)
model.load_state_dict(torch.load('../baseline_HGT/model/model_HGT_Case_1_1_2_0.2_1.pt_model_state_dict.pt'))
model.to(device)
model.lin = Identity()
model.eval()

# Paper X data
X_data = model(data.x_dict, data.edge_index_dict)
tsne = TSNE(n_components=2, random_state=0)
cluster = np.array(tsne.fit_transform(np.array(X_data.detach().cpu().numpy())))

# visualization
# 1. all data
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.nipy_spectral(np.linspace(0, 1, len(targetIDs)))))

for i, label in zip(range(len(targetIDs)), targetIDs):
    idx = []

    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s")

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o")

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", s=50)


plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_all.png')

# 2.1 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i == 3:
        color = 'blue'
    elif i == 4:
        color = 'purple'
    elif i == 17:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s1.png')

# 2.2 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i == 2:
        color = 'blue'
    elif i == 10:
        color = 'purple'
    elif i == 18:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s2.png')

# 2.3 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i == 7:
        color = 'blue'
    elif i == 8:
        color = 'purple'
    elif i == 15:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s3.png')

# 2.4 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i == 13:
        color = 'blue'
    elif i == 14:
        color = 'purple'
    elif i == 16:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s4.png')

# 2.5 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i in [0,1,2,3]:
        color = 'blue'
    elif i in [4,5,6,7,8,9,10,11,12,13,14,15,16]:
        color = 'purple'
    elif i in [17,18]:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s5.png')

# 2.6 grey data
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(len(targetIDs)), targetIDs):
    if i in [0,1,2,3]:
        color = 'blue'
    elif i in [4,5,6]:
        color = 'green'
    elif i in [4,5,6,7,8,9,10,11,12,13,14,15,16]:
        color = 'purple'
    elif i in [17,18]:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)))

    if dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="s", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '2':
        if ',' in dfTargetCategory['2ND_CAT_ID_100'].iloc[i]:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="x", c=color, s=50)
        else:
            labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
            plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="o", c=color)

    elif dfTargetCategory['1ST_CAT_ID'].iloc[i] == '1,2':
        labelNM = dfTargetCategory['1ST_CAT'].iloc[i] + '->' + dfTargetCategory['2ND_CAT'].iloc[i]
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=labelNM, alpha=.6, marker="*", c=color, s=50)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_sub_s6.png')

print('end')