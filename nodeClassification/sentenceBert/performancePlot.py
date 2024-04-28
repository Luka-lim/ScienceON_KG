import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
from torch_geometric.nn import HGTConv, Linear


# Test Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param_max_iter = 1000
testSize = 0.2
target = 'paper'
plt.figure(figsize=(120, 60))

## Graph Embedding + sentenceBERT
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

    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return f1


print("Paper Top Category Classification with sentenceBERT + graph embedding")
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


max_iter_range = list(range(0, param_max_iter, 1))  # 적절한 범위 및 간격 설정
f1_scores = []

model = HGT(hidden_channels=176, out_channels=8, num_heads=8, num_layers=3, target=target, data=data)
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set Train/Test Ratio
x = data['paper'].y_index
y = data['paper'].y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, shuffle=True, random_state=1)
mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
mask[x_train] = True
data['paper']['train_mask'] = mask
mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
mask[x_test] = True
data['paper']['test_mask'] = mask

for max_iter in max_iter_range:
    loss = train(model, data, target, optimizer, loss_op)
    f1 = test(model, data, target)
    f1_scores.append(f1)
    print("iter: ", max_iter, "f1: ", f1)

# 결과를 텍스트 파일로 저장
with open('result_sentenceBERT_Graph.txt', 'w') as file:
    file.write('Training Iteration\tLoss\n')
    for i, loss in enumerate(f1_scores):
        file.write(f'{i+1}\t{loss}\n')

# 그래프 설정
plt.plot(max_iter_range, f1_scores, marker='o', color='red', label=f'sentenceBERT + graph embedding')
plt.xlabel('Training Iterations')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)


## Only sentenceBERT
print("Paper Top Category Classification with sentenceBERT")

def load_node_csv(df, index_col, encoders=None):
    df = df[[index_col, 'FEATURE']]
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x

class LangEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name=None, device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()


# Load paper Label
dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_a/Paper_Label.csv', low_memory=False)
dfPaperLabel = dfPaperLabel.sort_values('IDPAPER')
dfPaperLabel = dfPaperLabel.reset_index(drop=True)

dfPaperBLabel = pd.read_csv('../SATDataset_ver2.1_b/Paper_Label.csv', low_memory=False)
dfPaperBLabel = dfPaperBLabel.sort_values('IDPAPER')
dfPaperBLabel = dfPaperBLabel.reset_index(drop=True)

# Load paper Feature
dfPaperFeat = pd.read_csv('../SATDataset_ver2.1_a/Paper_feat.csv', low_memory=False)
dfPaperFeat = dfPaperFeat.sort_values('IDPAPER')
dfPaperFeat = dfPaperFeat.reset_index(drop=True)
dfPaperFeat = dfPaperFeat[['CNPAPER', 'FEATURE']]

# Converge paper Data
dfPaper = pd.merge(left=dfPaperLabel, right=dfPaperFeat, how='inner', on='CNPAPER')
dfPaper = dfPaper.drop(columns=['IDPAPER'])
dfPaper = dfPaper.sort_values('CNPAPER')
dfPaper = dfPaper.reset_index(drop=True)
dfPaper = dfPaper.reset_index(drop=False)

dfPaperB = pd.merge(left=dfPaperBLabel, right=dfPaperFeat, how='inner', on='CNPAPER')
dfPaperB = dfPaperB.drop(columns=['IDPAPER'])
dfPaperB = dfPaperB.sort_values('CNPAPER')
dfPaperB = dfPaperB.reset_index(drop=True)
dfPaperB = dfPaperB.reset_index(drop=False)

# Y top / Y sub
num_classes = 8
Y_top = torch.zeros((dfPaper.shape[0], num_classes), dtype=torch.float)
for i, item in dfPaper.iterrows():
    labelInt = [int(pClass) for pClass in item['1ST_CAT_ID'].split(',')]
    for j in labelInt:
        Y_top[i, j] = 1.

X = load_node_csv(dfPaper, index_col='index',
                  encoders={'FEATURE': LangEncoder(model_name='../graph/distiluse-base-multilingual-cased-v1')})

X_top_train, X_top_test, Y_top_train, Y_top_test = train_test_split(X, Y_top, test_size=testSize, random_state=123)
base_model = LogisticRegression(max_iter=param_max_iter, random_state=27, warm_start=True)
model = MultiOutputClassifier(base_model)
max_iter_range = list(range(0, param_max_iter, 1))  # 적절한 범위 및 간격 설정
f1_scores = []

for max_iter in max_iter_range:
    model.estimator.max_iter = max_iter
    model.fit(X_top_train, Y_top_train)
    y_pred = model.predict(X_top_test)
    f1 = f1_score(Y_top_test, y_pred, average='weighted')  # 다중 출력에 대한 평균 F1 점수 계산
    f1_scores.append(f1)
    print("iter: ", max_iter, "f1: ", f1)

# 결과를 텍스트 파일로 저장
with open('result_sentenceBERT.txt', 'w') as file:
    file.write('Training Iteration\tLoss\n')
    for i, loss in enumerate(f1_scores):
        file.write(f'{i+1}\t{loss}\n')

# 그래프 그리기
plt.plot(max_iter_range, f1_scores, marker='*', color='blue', label=f'sentenceBERT')
plt.xlabel('Training Iterations')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.savefig('figure.png')  # 파일 경로 및 확장자 설정

print('end')
