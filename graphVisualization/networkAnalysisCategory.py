import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


targetATT = 'JOURNALTITLE'
#targetATT = '2ND_CAT'

# Paper
dfProjectTop = pd.read_csv('../SATDataset_ver2.1_a/Project_Top_edge.csv')
dfProjectPaper = pd.read_csv('../SATDataset_ver2.1_a/Project_Paper_edge.csv')
dfGoogleCategory = pd.read_csv('../SATDataset_ver2.1_a/Google_Category.csv')
dfPaperJournal = pd.read_csv('../SATDataset_ver2.1_a/Paper_Journal_edge.csv')

dfProjectPaper = dfProjectPaper[['CNPJT', 'CNPAPER']]
dfPaperJournal = dfPaperJournal[['CNPAPER', 'CNJOURNAL']]
dfGoogleCategory = dfGoogleCategory[['CNJOURNAL', targetATT]]

dfPaperCategory = pd.merge(left=dfPaperJournal, right=dfGoogleCategory, on='CNJOURNAL', how='inner')
dfPaperCategory = dfPaperCategory[['CNPAPER', targetATT]]

dfProjectPaperCategory = pd.merge(left=dfProjectPaper, right=dfPaperCategory, on='CNPAPER', how='inner')
dfProjectPaperCategory = dfProjectPaperCategory[['CNPJT', targetATT]]
dfProjectPaperCategory = dfProjectPaperCategory.drop_duplicates()

dfTProjectPaperCategory = pd.merge(left=dfProjectTop, right=dfProjectPaperCategory, on='CNPJT', how='inner')
dfTProjectPaperCategory = dfTProjectPaperCategory[['TopID', targetATT]]
dfTProjectPaperCategory = dfTProjectPaperCategory.drop_duplicates()

# Patent
dfProjectTop = pd.read_csv('../SATDataset_ver2.1_a/Project_Top_edge.csv')
dfProjectPatent = pd.read_csv('../SATDataset_ver2.1_a/Project_Patent_edge.csv')
dfPatentLabel = pd.read_csv('../SATDataset_ver2.1_a/Patent_Label.csv')
dfIPC = pd.read_csv('../SATDataset_ver2.1_b/Patent_Label_Title.csv')

dfPatentIPC = pd.merge(left=dfPatentLabel, right=dfIPC, on='SubClass', how='inner')
dfPatentIPC = dfPatentIPC[['CNPATENT', 'TITLE_ENG']]
dfPatentIPC = dfPatentIPC.drop_duplicates()

dfProjectPatentCategory = pd.merge(left=dfProjectPatent, right=dfPatentIPC, on='CNPATENT', how='inner')
dfProjectPatentCategory = dfProjectPatentCategory[['CNPJT', 'TITLE_ENG']]
dfProjectPatentCategory = dfProjectPatentCategory.drop_duplicates()

dfTProjectPatentCategory = pd.merge(left=dfProjectTop, right=dfProjectPatentCategory, on='CNPJT', how='inner')
dfTProjectPatentCategory = dfTProjectPatentCategory[['TopID', 'TITLE_ENG']]
dfTProjectPatentCategory = dfTProjectPatentCategory.drop_duplicates()

# Target
threadhold = 24

# Test1
targetIPC = 'ELECTRIC COMMUNICATION TECHNIQUE'
# Test2
targetIPC = 'AIRCRAFT; AVIATION; COSMONAUTICS'
# Test3
# PHYSICS INFORMATION AND COMMUNICATION TECHNOLOGY [ICT] SPECIALLY ADAPTED FOR SPECIFIC APPLICATION FIELDS HEALTHCARE INFORMATICS, i.e. INFORMATION AND COMMUNICATION TECHNOLOGY [ICT] SPECIALLY ADAPTED FOR THE HANDLING OR PROCESSING OF MEDICAL OR HEALTHCARE DATA
targetIPC = 'HEALTHCARE INFORMATICS, i.e. INFORMATION AND COMMUNICATION TECHNOLOGY [ICT] SPECIALLY ADAPTED FOR THE HANDLING OR PROCESSING OF MEDICAL OR HEALTHCARE DATA'
targetIPCNM = 'G16H'
# Test4
# PHYSICS COMPUTING; CALCULATING OR COUNTING IMAGE DATA PROCESSING OR GENERATION, IN GENERAL
targetIPC = 'IMAGE DATA PROCESSING OR GENERATION, IN GENERAL'
targetIPCNM = 'G06T'


# Pick Patent
dfPickPatent = dfTProjectPatentCategory[dfTProjectPatentCategory['TITLE_ENG'] == targetIPC]
dfPickProjectPaper = pd.merge(left=dfTProjectPaperCategory, right=dfPickPatent[['TopID']], on='TopID', how='inner')
dfPickPatent = dfPickPatent.rename(columns={'TITLE_ENG': 'Source'})
dfPickProjectPaper = dfPickProjectPaper.rename(columns={targetATT: 'Target'})

dfPickMerge = pd.merge(left=dfPickPatent, right=dfPickProjectPaper, on='TopID', how='inner')
dfPickMerge = dfPickMerge[['Source', 'Target']]
dfPickMerge = dfPickMerge.sort_values('Target')
dfPickMerge = dfPickMerge.reset_index(drop=True)

dfWeight = dfPickMerge.groupby(['Target']).size().to_frame()
dfWeight = dfWeight.rename(columns={0: 'Weight'})
dfPickMerge = dfPickMerge.drop_duplicates()
dfPickMerge = pd.merge(left=dfPickMerge, right=dfWeight, on='Target', how='inner')
dfPickMerge = dfPickMerge[dfPickMerge['Weight'] > threadhold]
dfPickMerge = dfPickMerge.reset_index(drop=True)
dfPickMerge['Source'] = targetIPCNM

# Graph
plt.figure(figsize=(15, 15))
G = nx.from_pandas_edgelist(dfPickMerge, 'Source', 'Target', 'Weight')
pos = nx.spring_layout(G, seed=0)

for i, node in enumerate(G.nodes):
    if node == targetIPCNM:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=15000, node_color="darkorange", alpha=0.8)

    else:
        weight = dfPickMerge[dfPickMerge['Target'] == node]
        size = weight['Weight'].iloc[0]
        size = size*25
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_color="skyblue", alpha=0.8)

edges, weights = zip(*nx.get_edge_attributes(G, 'Weight').items())
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, alpha=0.8, width=10.0, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(G, pos=pos, verticalalignment='top', horizontalalignment='center')

plt.tight_layout()
plt.savefig('./sample.png')
