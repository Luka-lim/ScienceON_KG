import pandas as pd
import matplotlib.pyplot as plt

# 텍스트 파일 읽기 (예제 데이터)
file_path = 'result_sentenceBERT.txt'  # 파일 경로를 실제 파일 경로로 변경해주세요.
dfBert = pd.read_csv(file_path, sep='\t')
dfBert.rename(columns={'Loss': 'F1 Score'}, inplace=True)

file_path = 'result_sentenceBERT_Graph.txt'  # 파일 경로를 실제 파일 경로로 변경해주세요.
dfGraph = pd.read_csv(file_path, sep='\t')
dfGraph.rename(columns={'Loss': 'F1 Score'}, inplace=True)

plt.figure(figsize=(7, 7))

plt.plot(dfBert['Training Iteration'], dfBert['F1 Score'], marker='*', markersize=2,
         color='blue', label=f'sentenceBERT')
plt.plot(dfGraph['Training Iteration'], dfGraph['F1 Score'], marker='o', markersize=2,
         color='red', label=f'sentenceBERT + Graph Embedding')

plt.xlabel('Training Iteration')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()
plt.show()

plt.savefig('sentenceBertResultPlot.png')  # 파일 경로 및 확장자 설정
