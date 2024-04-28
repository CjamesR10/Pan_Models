import numpy as np
import matplotlib.pyplot as plt

labels = ['Easy', 'Medium', 'Hard']
robertas = [0.809, 0.843, 0.817]
orcas = [0.505, 0.643, 0.515]
llamas = [0.540, 0.659, 0.421]

plt.figure(figsize=(8, 6))

edge_styles = ['dotted', 'dashed', 'solid']

bar_width = 0.30

x = np.arange(len(labels))

bars = []
for i, (roberta, orca, llama) in enumerate(zip(robertas, orcas, llamas)):
    bar1 = plt.bar(x[i] - bar_width, roberta, color='grey', width=bar_width, edgecolor='black', linestyle=edge_styles[0])
    bar2 = plt.bar(x[i], orca, color='lightsteelblue', width=bar_width, edgecolor='black', linestyle=edge_styles[1])
    bar3 = plt.bar(x[i] + bar_width, llama, color='royalblue', width=bar_width, edgecolor='black', linestyle=edge_styles[2])
    bars.extend([bar1[0], bar2[0], bar3[0]])
    plt.text(x[i] - bar_width, roberta, f'{roberta:.3f}', ha='center', va='bottom', fontsize=10)
    plt.text(x[i], orca, f'{orca:.3f}', ha='center', va='bottom', fontsize=10)
    plt.text(x[i] + bar_width, llama, f'{llama:.3f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('Comparison of Model Results')

plt.xticks(x, labels)

plt.ylim(0, max(robertas + orcas + llamas) * 1.2)

plt.legend(bars, ['RoBERTa', 'Orca-2', 'Llama-3'], loc='upper left')

plt.show()