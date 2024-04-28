import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random


dfEdge = pd.read_csv('../SATDataset_ver2.1_a/Paper_Paper_edge.csv', low_memory=False)
dfEdgeUni1 = dfEdge[['CitingPaper']]
dfEdgeUni1 = dfEdgeUni1.drop_duplicates().reset_index(drop=True)
dfEdgeUni2 = dfEdge[['CitedPaper']]
dfEdgeUni2 = dfEdgeUni2.drop_duplicates().reset_index(drop=True)
dfEdgeUni1.rename(columns={'CitingPaper': 'CNPAPER'}, inplace=True)
dfEdgeUni2.rename(columns={'CitedPaper': 'CNPAPER'}, inplace=True)
dfNode = pd.concat([dfEdgeUni1, dfEdgeUni2], ignore_index=True)
dfNode = dfNode.drop_duplicates().reset_index(drop=True)

dfEdgeUni = pd.concat([dfEdgeUni1, dfEdgeUni2], ignore_index=True)
dfEdgeUni = dfEdgeUni.drop_duplicates().reset_index(drop=True)
dfEdgeUni = dfEdgeUni.assign(CitedPaper=dfEdgeUni['CNPAPER'])
dfEdgeUni.rename(columns={'CNPAPER': 'CitingPaper'}, inplace=True)
dfEdge = dfEdge[['CitingPaper', 'CitedPaper']]
dfEdgeExtra = pd.concat([dfEdge, dfEdgeUni], ignore_index=True)

dfKeyword = pd.read_csv('../SATDataset_ver0.9/Paper_keywords_G.csv', low_memory=False)
dfKeyword['KEYWORD_eng'] = dfKeyword['KEYWORD_eng'].str.lower()
dfKeyword = dfKeyword[['1ST_CAT', '2ND_CAT', 'CNPAPER', 'KEYWORD_eng']]
dfKeyword = dfKeyword[~dfKeyword['KEYWORD_eng'].str.contains(r'[^\w\s]', regex=True)]
dfKeyword = dfKeyword[dfKeyword['KEYWORD_eng'].str.len() > 5]
dfKeyword['ID'] = pd.factorize(dfKeyword['KEYWORD_eng'])[0]
dfKeywordDup = dfKeyword[['CNPAPER', 'KEYWORD_eng', 'ID']]
dfKeywordDup = dfKeywordDup.drop_duplicates().reset_index(drop=True)

dfKeywordEdge1 = pd.merge(dfEdgeExtra, dfKeywordDup, left_on='CitingPaper', right_on='CNPAPER', how='inner')
dfKeywordEdge1 = dfKeywordEdge1[['CNPAPER', 'CitedPaper', 'KEYWORD_eng', 'ID']]
dfKeywordEdge1.rename(columns={'CNPAPER': 'ToPaper', 'CitedPaper': 'FromPaper',
                              'KEYWORD_eng': 'ToKeyword', 'ID': 'ToID'}, inplace=True)
dfKeywordEdge2 = pd.merge(dfKeywordEdge1, dfKeywordDup, left_on='FromPaper', right_on='CNPAPER', how='inner')
dfKeywordEdge = dfKeywordEdge2[['ToPaper', 'FromPaper', 'ToKeyword', 'KEYWORD_eng', 'ToID', 'ID']]
dfKeywordEdge.rename(columns={'KEYWORD_eng': 'FromKeyword', 'ID': "FromID"}, inplace=True)
dfKeywordEdge = dfKeywordEdge[dfKeywordEdge['ToID'] != dfKeywordEdge['FromID']]
dfKeywordEdge = dfKeywordEdge.drop_duplicates().reset_index(drop=True)

dfPaperLabel = pd.read_csv('../SATDataset_ver2.1_a/Paper_Label.csv', low_memory=False)
dfNode = pd.merge(dfNode, dfPaperLabel, on='CNPAPER')
dfNode = dfNode[['CNPAPER', '1ST_CAT_ID']]
dfNode = pd.merge(dfNode, dfKeyword, on='CNPAPER')
dfNode = dfNode[['CNPAPER', '1ST_CAT_ID', '1ST_CAT', 'KEYWORD_eng']]
dfNode.rename(columns={'KEYWORD_eng': 'Keyword'}, inplace=True)

categoryName = ['0 : Business, Economics & Management',
                '1 : Chemical & Material Sciences',
                '2 : Engineering & Computer Science',
                '3 : Health & Medical Sciences',
                '4 : Humanities, Literature & Arts',
                '5 : Life Sciences & Earth Sciences',
                '6 : Physics & Mathematics',
                '7 : Social Sciences']

G = nx.from_pandas_edgelist(dfKeywordEdge, 'ToKeyword', 'FromKeyword')

# Ego graph 생성 (반경 설정)
#center_node = 'knowledge sharing'
#center_node = 'feedback control'
center_node = 'open innovation'
radius = 2  # 반경 설정
ego_graph = nx.ego_graph(G, center_node, radius=radius)
nodes_to_remove = []

# 다중 레이블 노드 제거
for node, data in ego_graph.nodes(data=True):
    category = dfNode[dfNode['Keyword'] == node]
    categoryIDs = category['1ST_CAT_ID'].drop_duplicates().tolist()
    categoryIDs = [int(num) for x in categoryIDs for num in x.split(',')]
    categoryIDs = list(set(categoryIDs))
    categoryIDs.sort()

    if len(categoryIDs) > 3:
        nodes_to_remove.append(node)
    else:
        categoryNameString = ' + '.join(map(str, categoryIDs))
        data['Type'] = categoryNameString

# Remove nodes from the ego_graph
for node in nodes_to_remove:
    if ego_graph.has_node(node):
        ego_graph.remove_node(node)


# Specify the number of nodes and edges to randomly remove
num_nodes_to_remove = int(nx.number_of_nodes(ego_graph) * 0.53)  # Adjust the number of nodes to remove

# Randomly select nodes and edges to remove
nodes_to_remove = random.sample(list(ego_graph.nodes()), k=num_nodes_to_remove)

# Remove randomly selected nodes and edges
ego_graph.remove_nodes_from(nodes_to_remove)

# Find connected components
connected_components = list(nx.connected_components(ego_graph))

# Identify the largest connected component
largest_component = max(connected_components, key=len)

# Create a subgraph containing only the largest connected component
largest_subgraph = ego_graph.subgraph(largest_component)

# Update ego_graph to contain only the largest connected component
ego_graph = largest_subgraph

# 색상 팔레트 선택 (Set1 팔레트 활용)
palette = plt.get_cmap('Set1')

# 각 노드 타입에 대해 팔레트에서 색상 할당
node_types = nx.get_node_attributes(ego_graph, 'Type')
unique_types = set(node_types.values())
color_mapping = {node_type: palette(i / len(unique_types)) for i, node_type in enumerate(unique_types)}

for node, data in ego_graph.nodes(data=True):
    data['color'] = color_mapping[data['Type']]

# 노드의 색상을 설정
node_colors = [data['color'] for _, data in ego_graph.nodes(data=True)]

# 노드의 크기를 설정
node_degrees = dict(ego_graph.degree())
nx.set_node_attributes(ego_graph, node_degrees, 'degree')
node_sizes = [data['degree'] * 150 for node, data in ego_graph.nodes(data=True)]

# Create a legend for the node types
sorted_legend_labels = {node_type: node_type for node_type in sorted(unique_types)}
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[node_type], markersize=10, label=label)
                  for node_type, label in sorted_legend_labels.items()]

# Customize edge color and width
edge_color = (0.5, 0.5, 0.5, 0.5)  # RGBA format with alpha set to 0.5
edge_width = 1.5  # Specify the width you want

# Ego graph 시각화
plt.figure(figsize=(15, 15))
#pos = nx.spring_layout(ego_graph)
pos = nx.spring_layout(ego_graph, k=0.6)
nx.draw(ego_graph, pos, with_labels=True, font_weight='bold', font_size=12,
        node_color=node_colors, node_size=node_sizes, cmap=plt.cm.Blues,
        edge_color=edge_color, width=edge_width)

# Add custom text at the bottom-left corner
categoryName = ['0 : Business, Economics & Management',
                '1 : Chemical & Material Sciences',
                '2 : Engineering & Computer Science',
                '3 : Health & Medical Sciences',
                '4 : Humanities, Literature & Arts',
                '5 : Life Sciences & Earth Sciences',
                '6 : Physics & Mathematics',
                '7 : Social Sciences']

categoryName = sorted(categoryName, reverse=True)
text_location = (0.6, -0.8)  # Adjust the position of the text
'''
for i, text_line in enumerate(categoryName):
    plt.text(text_location[0], text_location[1] + i * 0.03, text_line,
             fontsize=12, color='black', ha='left', va='top')
'''

plt.legend(handles=legend_handles, title='Node Types', loc='best')
plt.title('Ego Graph Centered around Node {} with Radius {}'.format(center_node, radius))

# 시각화된 그래프 보기
plt.show()
