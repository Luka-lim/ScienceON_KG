import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


device = torch.device("cuda:0")
# device = torch.device("cpu")


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


num_classes = 100
Y_sub = torch.zeros((dfPaperB.shape[0], num_classes), dtype=torch.float)
for i, item in dfPaperB.iterrows():
    labelInt = [int(pClass) for pClass in item['2ND_CAT_ID_100'].split(',')]
    for j in labelInt:
        Y_sub[i, j] = 1.

print("Sub category")
X_B = load_node_csv(dfPaperB, index_col='index',
                  encoders={'FEATURE': LangEncoder(model_name='../graph/distiluse-base-multilingual-cased-v1')})

for i in [0.2, 0.4, 0.6, 0.8]:
    print(i)
    # train/test split
    X_sub_train, X_sub_test, Y_sub_train, Y_sub_test = train_test_split(X_B, Y_sub, test_size=i, random_state=123)

    # train classifier
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=5000)).fit(X_sub_train, Y_sub_train)

    # test classifier
    predicted = classifier.predict(X_sub_test)
    print("F1-micro:", metrics.f1_score(Y_sub_test, predicted, average='micro', zero_division=0))
    print("F1-macro:", metrics.f1_score(Y_sub_test, predicted, average='macro', zero_division=0))

print("Top category")
X = load_node_csv(dfPaper, index_col='index',
                  encoders={'FEATURE': LangEncoder(model_name='../graph/distiluse-base-multilingual-cased-v1')})

for i in [0.2, 0.4, 0.6, 0.8]:
    print(i)
    # train/test split
    X_top_train, X_top_test, Y_top_train, Y_top_test = train_test_split(X, Y_top, test_size=i, random_state=123)

    # train classifier
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=5000)).fit(X_top_train, Y_top_train)

    # test classifier
    predicted = classifier.predict(X_top_test)
    print("F1-micro:", metrics.f1_score(Y_top_test, predicted, average='micro', zero_division=0))
    print("F1-macro:", metrics.f1_score(Y_top_test, predicted, average='macro', zero_division=0))

print('end')
