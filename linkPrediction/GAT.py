import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_list = torch.load('../SATGraph_Top/Homo_data_a3.pt')
data = data_list[0]
data = data.to(device)

transform = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=True,
)
train_data, val_data, test_data = transform(data)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads, heads=1, concat=False, dropout=0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(in_channels=data.num_features, hidden_channels=8, heads=8).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    aucScore = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    apScore = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    return [aucScore, apScore]


best_val_auc = 0
final_test_auc = 0
final_test_ap = 0
file = open("./model/GAT_LinkPrediction.txt", "wt")
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

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_AUC: {valAUC:.4f}, Test_ACU: {testAUC:.4f}'
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
