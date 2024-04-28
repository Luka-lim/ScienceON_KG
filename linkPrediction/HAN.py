import os
import numpy as np
from typing import Dict, Union

import torch
from torch import nn
from torch_geometric.nn import HANConv

import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

# Load dataset
data_list = torch.load('../SATGraph_Top/Hetero_data_top.pt')
data = data_list[0]
data = data.to(device)
data.__delitem__(('copaper', 'cites', 'paper'))

# Defining Edge-level Training Splits
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=True,
    edge_types=("paper", "cites", "copaper"),
)
train_data, val_data, test_data = transform(data)


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]], data, hidden_channels=896, heads=128):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads, dropout=0.6, metadata=data.metadata())

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        return out


class Classifier(torch.nn.Module):
    def forward(self, x_src, x_dst, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_src = x_src[edge_label_index[0]]
        edge_feat_dst = x_dst[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_src * edge_feat_dst).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, in_channels, data):
        super().__init__()
        self.encoder = HAN(in_channels, data)
        self.classifier = Classifier()

    def forward(self, data):
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.encoder(data.x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["paper"],
            x_dict["copaper"],
            data["paper", "cites", "copaper"].edge_label_index,
        )
        return pred


model = Model(in_channels=-1, data=data)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data)
    ground_truth = train_data["paper", "cites", "copaper"].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(testData):
    model.eval()
    pred = model(testData).view(-1).sigmoid()
    ground_truth = testData["paper", "cites", "copaper"].edge_label

    aucScore = roc_auc_score(ground_truth.cpu().numpy(), pred.cpu().numpy())
    apScore = average_precision_score(ground_truth.cpu().numpy(), pred.cpu().numpy())
    return [aucScore, apScore]


best_val_auc = 0
final_test_auc = 0
final_test_ap = 0
file = open("./model/HAN_LinkPrediction.txt", "wt")
for epoch in range(1, 501):
    loss = train()
    valResult = test(val_data)
    valAUC = valResult[0]
    valAP = valResult[1]

    testResult = test(test_data)
    testAUC = testResult[0]
    testAP = testResult[1]

    if valAUC > best_val_auc:
        best_val_auc = valAUC
        final_test_auc = testAUC
        final_test_ap = testAP

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_AUC: {valAUC:.4f}, Test_ACU: {testAUC:.4f}, '
          f'VAL_AP: {valAP:.4f}, TEST_AP: {testAP:.4f}')

    file.writelines('Epoch: ' + str(epoch) + ', ')
    file.writelines('Loss: ' + str(loss) + ', ')
    file.writelines('Val_AUC: ' + str(valAUC) + ', ')
    file.writelines('Test_ACU: ' + str(testAUC) + ', ')
    file.writelines('VAL_AP: ' + str(valAP) + ', ')
    file.writelines('TEST_AP: ' + str(testAP) + ', ')
    file.writelines('\n')

print(f'Final Test -> AUC: {final_test_auc:.4f}, AP: {testAP:.4f}')
file.writelines('AUC: ' + str(final_test_auc) + ', ')
file.writelines('AP: ' + str(testAP))
file.close()
