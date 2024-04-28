import pandas as pd
import torch
from torch_geometric.nn import HGTConv, Linear
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

# https://hongl.tistory.com/157
# https://velog.io/@pppanghyun/%EC%8B%9C%EA%B0%81%ED%99%94-t-SNE-t-Stochastic-Neighbor-Embedding
# https://trycolors.com/
# https://www.kaggle.com/code/parulpandey/visualizing-kannada-mnist-with-t-sne
# https://www.w3schools.com/python/matplotlib_scatter.asp
# https://stackoverflow.com/questions/4805048/how-to-get-different-colored-lines-for-different-plots-in-a-single-figure

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
data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
data = data_list[0]
data.__delitem__(('paper', 'published_in', 'journal'))
data.__delitem__(('journal', 'published_in', 'paper'))
data.__delitem__(('journal'))
data.to(device)

model = HGT(hidden_channels=224, out_channels=8, num_heads=8, num_layers=3, target='paper', data=data)
model.load_state_dict(torch.load('../baseline_HGT/model/model_HGT_Case_1_1_1_0.2_1.pt_model_state_dict.pt'))
model.to(device)
model.lin = Identity()
model.eval()

# Paper X data
X_data = model(data.x_dict, data.edge_index_dict)
tsne = TSNE(n_components=2, random_state=0)
cluster = np.array(tsne.fit_transform(np.array(X_data.detach().cpu().numpy())))

# Target INDEX -> Engineering & Computer Science
targetIDX = 2

# Paper y data
dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_a/Paper_Label.csv', low_memory=False)
dfPaperLabel = dfPaperLabel.sort_values('IDPAPER')
dfPaperLabel = dfPaperLabel.reset_index(drop=True)
classes = []
for idx, item in dfPaperLabel.iterrows():
    cls = [int(pClass) for pClass in item['1ST_CAT_ID'].split(',')]
    classes.extend(cls)

num_classes = max(classes) + 1
Y_data = torch.zeros((dfPaperLabel.shape[0], num_classes), dtype=torch.int)
for i, item in dfPaperLabel.iterrows():
    labelInt = [int(pClass) for pClass in item['1ST_CAT_ID'].split(',')]
    for j in labelInt:
        Y_data[i, j] = 1

Y_data = np.array(Y_data)
temp = []
classList = [x for x in Y_data.tolist() if x not in temp and not temp.append(x)]
target = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]
pickClassList = [x for x in classList if x[targetIDX] == 1]
pickClass = [(np.where(np.array(x) == 1)[0]).tolist() for x in pickClassList]
pickClass = sum(pickClass, [])
for i in pickClass:
    pickClassList.append(target[i])

temp = []
pickClassList = [x for x in pickClassList if x not in temp and not temp.append(x)]
temp = []
pickClassList = [x for x in pickClassList if sum(x) < 3 and not temp.append(x)]
pickClassList.sort(reverse=True)
classList = pickClassList
classNum = len(classList)
className = []
category = ['Business, Economics & Management', 'Chemical & Material Sciences', 'Engineering & Computer Science',
            'Health & Medical Sciences', 'Humanities, Literature & Arts', 'Life Sciences & Earth Sciences',
            'Physics & Mathematics', 'Social Sciences']

doubleLableIDX = []
for i in classList:
    idx = np.where(np.array(i) == 1)[0]
    idx = idx.tolist()
    if len(idx) < 2:
        cName = category[idx[0]]
        className.append(cName)
    else:
        cName = ''.join([category[j] for j in idx if not j == targetIDX]) + ' / ' + category[targetIDX]
        className.append(cName)
        doubleLableIDX.append(className.index(cName))

classNameIDX = []
for i, item in enumerate(className):
    title = str(i) + ' - ' + item
    classNameIDX.append(title)

# visualization
# 1. all data
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.gist_ncar(np.linspace(0, 1, len(className)))))

for i, label in zip(range(classNum), classList):
    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)/10))
    if i in doubleLableIDX:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], alpha=.6, marker="*")

    else:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], alpha=.6)

plt.legend(loc='best')
plt.show()
plt.savefig('vis_top_all.png')

# 2. grey data
# i 3:blue, 2:purple, 8:red, else:silver
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(classNum), classList):
    if i == 3:
        color = 'blue'
    elif i == 2:
        color = 'purple'
    elif i == 8:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)/10))
    if i in doubleLableIDX:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6, marker="*")

    else:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6)

plt.show()
plt.savefig('vis_top_s1.png')

# 2. grey data
# i 11:blue, 5:purple, 8:red, else:silver
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(classNum), classList):
    if i == 11:
        color = 'blue'
    elif i == 5:
        color = 'purple'
    elif i == 8:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)/10))
    if i in doubleLableIDX:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6, marker="*")

    else:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6)

plt.show()
plt.savefig('vis_top_s2.png')

# 2. grey data
# i 12:blue, 6:purple, 8:red, else:silver
plt.figure(figsize=(20, 20))
plt.axis('off')

for i, label in zip(range(classNum), classList):
    if i == 12:
        color = 'blue'
    elif i == 6:
        color = 'purple'
    elif i == 8:
        color = 'red'
    else:
        color = 'silver'

    idx = []
    for j, k in enumerate(Y_data):
        if k.tolist() == label:
            idx.append(j)

    idx = random.sample(idx, int(len(idx)/10))
    if i in doubleLableIDX:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6, marker="*")

    else:
        plt.scatter(cluster[idx, 0], cluster[idx, 1], label=classNameIDX[i], c=color, alpha=.6)

plt.show()
plt.savefig('vis_top_s3.png')

print('end')
