import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
data = data_list[0]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = data.to(device)
data.__delitem__(('paper', 'published_in', 'journal'))
data.__delitem__(('journal', 'published_in', 'paper'))
data.__delitem__(('journal'))

metapath = [
    ('project', 'in', 'Tproject'),
    ('Tproject', 'in', 'project'),
    ('project', 'results_from', 'paper'),
    ('paper', 'cites', 'paper'),
    ('paper', 'cites', 'report'),
    ('report', 'cites', 'paper'),
    ('paper', 'writes', 'author'),
    ('author', 'writes', 'paper'),
    ('paper', 'writes', 'organ'),
    ('organ', 'writes', 'paper'),
    ('paper', 'cites', 'copaper'),
    ('copaper', 'cites', 'report'),
    ('report', 'cites', 'copaper'),
    ('copaper', 'cites', 'paper'),
    ('paper', 'results_from', 'project'),
    ('project', 'results_from', 'report'),
    ('report', 'writes', 'author'),
    ('author', 'writes', 'report'),
    ('report', 'writes', 'organ'),
    ('organ', 'writes', 'report'),
    ('report', 'results_from', 'project'),
    ('project', 'results_from', 'patent'),
    ('patent', 'writes', 'author'),
    ('author', 'writes', 'patent'),
    ('patent', 'writes', 'organ'),
    ('organ', 'writes', 'patent'),
    ('patent', 'is_classified', 'ipc'),
    ('ipc', 'is_classified', 'patent'),
    ('patent', 'results_from', 'project'),
]

model = MetaPath2Vec(data.edge_index_dict,
                     embedding_dim=128,
                     metapath=metapath,
                     walk_length=100,
                     context_size=7,
                     walks_per_node=500,
                     num_negative_samples=5,
                     sparse=True).to(device)

PATH = './model/metapath2vec_case_1_1_idx_150.pt'
model.load_state_dict(torch.load(PATH, map_location=device))
print('set model')

loader = model.loader(batch_size=8, shuffle=True, num_workers=16)
print('set loader')

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.0005)
print('set optimizer')


def train(epoch, log_steps=1000, eval_steps=10000):
    model.train()
    print(f'Epoch: {epoch}, model train')

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, Loss: {total_loss / log_steps:.4f}')
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            micro, macro = test()
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, F1-Micro: {micro:.4f}, F1-Macro: {macro:.4f}')


@torch.no_grad()
def test(test_size=0.2):
    model.eval()
    X_data = model('paper', batch=data['paper'].y_index.to(device))
    Y_data = data['paper'].y
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(X_data, Y_data, test_size=test_size, random_state=10)
    print('Train classifier')
    forest = RandomForestClassifier(random_state=10)
    classifier = MultiOutputClassifier(forest, n_jobs=24)
    classifier.fit(Train_X.detach().cpu().numpy(), Train_Y.detach().cpu().numpy())
    y_pred = classifier.predict(Test_X.detach().cpu().numpy())

    micro = f1_score(Test_Y.detach().cpu().numpy(), y_pred, average='micro', zero_division=0)
    macro = f1_score(Test_Y.detach().cpu().numpy(), y_pred, average='macro', zero_division=0)

    return micro, macro


for epoch in range(151, 301):
    train(epoch)

    if epoch % 20 == 0:
        micro, macro = test()
        print(f'Epoch: {epoch},F1-Micro: {micro:.4f}, F1-Macro: {macro:.4f}')

        PATH = './model/'
        torch.save(model, PATH + 'metapath2vec_case_1_1_idx_{0}.pt'.format(epoch))  # 전체 모델 저장
        '''
        torch.save(model.state_dict(), PATH + 'metapath2vec_state_dict_{0}.pt'.format(epoch))  #모델 객체의 state_dict 저장
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, PATH + 'metapath2vec_{0}.tar'.format(epoch))
        '''

print('end')
