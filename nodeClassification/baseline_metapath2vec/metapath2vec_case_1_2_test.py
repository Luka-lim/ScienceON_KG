import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC


rounds = 3
testsizes = [0.2, 0.4, 0.6, 0.8]
device = torch.device('cpu')

######################################## Test Case Patent Top ########################################
print('Start Test Case Patent_Top')
micro_1_2_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_2_1 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

model = torch.load('./model/metapath2vec_case_1_2_idx_200.pt', map_location=device)

print('load train/test data')
dfPatentGT = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv')
X_data = model('patent', batch=torch.from_numpy(dfPatentGT['IDPATENT'].values))

# patent y index
classes = []
for idx, item in dfPatentGT.iterrows():
    cls = [int(pClass) for pClass in item['ClassID'].split(',')]
    classes.extend(cls)

classes = list(set(classes))
classes.sort()

num_classes = len(classes)
Y_data = torch.zeros((dfPatentGT.shape[0], num_classes), dtype=torch.float)
for i, item in dfPatentGT.iterrows():
    labelInt = [int(pClass) for pClass in item['ClassID'].split(',')]
    labeltoIdx = [classes.index(k) for k in labelInt]
    for j in labeltoIdx:
        Y_data[i, j] = 1.

for roundIDX in range(1, rounds):
    model.eval()

    data_micro = []
    data_macro = []
    for testsize in testsizes:
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(X_data, Y_data, test_size=testsize, random_state=10)
        print('Train classifier')
        #forest = RandomForestClassifier(random_state=10)
        #classifier = MultiOutputClassifier(forest, n_jobs=4)
        svc = SVC(gamma="scale")
        classifier = MultiOutputClassifier(estimator=svc, n_jobs=24)
        classifier.fit(Train_X.detach().cpu().numpy(), Train_Y.detach().cpu().numpy())
        y_pred = classifier.predict(Test_X.detach().cpu().numpy())
        y_true = Test_Y.detach().cpu().numpy()

        print(classification_report(y_true, y_pred, zero_division=0))
        print("F1 None: {}".format(metrics.f1_score(y_true, y_pred, average=None, zero_division=0)))
        print("F1 micro: {:.2f}".format(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)))
        print("F1 macro: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)))
        print("F1 weighted: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)))
        print("F1 samples: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='samples', zero_division=0)))

        micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_2_1.loc[roundIDX] = data_micro
    macro_1_2_1.loc[roundIDX] = data_macro

    micro_1_2_1.to_csv('./model/metapath2vec_micro_1_2_1.csv', encoding='utf-8-sig', index=False)
    macro_1_2_1.to_csv('./model/metapath2vec_macro_1_2_1.csv', encoding='utf-8-sig', index=False)
print('End Test Case Patent_Top')

######################################## Test Case Patent Sub ########################################
print('Start Test Case Patent_Sub')
micro_1_2_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])
macro_1_2_2 = pd.DataFrame(columns=['0.2', '0.4', '0.6', '0.8'])

model = torch.load('./model/metapath2vec_case_1_2_idx_200.pt', map_location=device)

print('load train/test data')
dfPatentGT = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv')
X_data = model('patent', batch=torch.from_numpy(dfPatentGT['IDPATENT'].values))

# patent y index
classes = []
for idx, item in dfPatentGT.iterrows():
    cls = [int(pClass) for pClass in item['SubClassID'].split(',')]
    classes.extend(cls)

classes = list(set(classes))
classes.sort()

num_classes = len(classes)
Y_data = torch.zeros((dfPatentGT.shape[0], num_classes), dtype=torch.float)
for i, item in dfPatentGT.iterrows():
    labelInt = [int(pClass) for pClass in item['SubClassID'].split(',')]
    labeltoIdx = [classes.index(k) for k in labelInt]
    for j in labeltoIdx:
        Y_data[i, j] = 1.

for roundIDX in range(1, rounds):
    model.eval()

    data_micro = []
    data_macro = []
    for testsize in testsizes:
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(X_data, Y_data, test_size=testsize, random_state=10)
        print('Train classifier')
        #forest = RandomForestClassifier(random_state=10)
        #classifier = MultiOutputClassifier(forest, n_jobs=4)
        svc = SVC(gamma="scale")
        classifier = MultiOutputClassifier(estimator=svc, n_jobs=24)
        classifier.fit(Train_X.detach().cpu().numpy(), Train_Y.detach().cpu().numpy())
        y_pred = classifier.predict(Test_X.detach().cpu().numpy())
        y_true = Test_Y.detach().cpu().numpy()

        print(classification_report(y_true, y_pred, zero_division=0))
        print("F1 None: {}".format(metrics.f1_score(y_true, y_pred, average=None, zero_division=0)))
        print("F1 micro: {:.2f}".format(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)))
        print("F1 macro: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)))
        print("F1 weighted: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)))
        print("F1 samples: {:.2f} ".format(metrics.f1_score(y_true, y_pred, average='samples', zero_division=0)))

        micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        data_micro.append(micro)
        data_macro.append(macro)

    micro_1_2_2.loc[roundIDX] = data_micro
    macro_1_2_2.loc[roundIDX] = data_macro
    print('End Test Case Patent_Sub')

    micro_1_2_2.to_csv('./model/metapath2vec_micro_1_2_2.csv', encoding='utf-8-sig', index=False)
    macro_1_2_2.to_csv('./model/metapath2vec_macro_1_2_2.csv', encoding='utf-8-sig', index=False)
print('end')


