import pandas as pd
# import numpy as np
import ast
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def calculate_metrics(group):
    TP = group['TP'].sum()
    FP = group['FP'].sum()
    FN = group['FN'].sum()
    group['IoUs'] = group['IoUs'].apply(lambda x: ast.literal_eval(x))
    IoUs = group['IoUs'].sum()

    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_IoU = sum(IoUs) / len(IoUs) if len(IoUs) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return pd.Series({
        'precision': precision,
        'recall': recall,
        'mean_IoU': mean_IoU,
        'F1': F1
    })

def hierarchical_benchmark(csv_path):
    df = pd.read_csv(csv_path)
    
    results = {}
    for level in ['class', 'order', 'family', 'genus', 'specie']:
        grouped = df.groupby(level).apply(calculate_metrics).reset_index()
        results[level] = grouped
    
    return results

def plot_metrics(results, level, output, sort=False):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    data = results[level]
    
    # F1 score
    if sort:
        data = data.sort_values(by='F1', ascending=False)
    data.plot(kind='bar', x=level, y='F1', ax=axs[0, 0])
    axs[0, 0].set_title(f'F1 score at {level} level')
    axs[0, 0].set_ylim(0, 1)
    
    # Precision
    if sort:
        data = data.sort_values(by='precision', ascending=False)
    data.plot(kind='bar', x=level, y='precision', ax=axs[0, 1])
    axs[0, 1].set_title(f'Precision at {level} level')
    axs[0, 1].set_ylim(0, 1)
    
    # Recall
    if sort:
        data = data.sort_values(by='recall', ascending=False)
    data.plot(kind='bar', x=level, y='recall', ax=axs[1, 0])
    axs[1, 0].set_title(f'Recall at {level} level')
    axs[1, 0].set_ylim(0, 1)
    
    # Mean IoU
    if sort:
        data = data.sort_values(by='mean_IoU', ascending=False)
    data.plot(kind='bar', x=level, y='mean_IoU', ax=axs[1, 1])
    axs[1, 1].set_title(f'Mean IoU at {level} level')
    axs[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output)

# Example usage:
results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_FULL_DATASET.csv')
for level, metrics in results.items():
    print(f"Metrics for {level}:")
    print(metrics)

# Plotting the results for class level:
plot_metrics(results, level='class', output='arthropods_dataset_scripts/benchmark/plot_class.png')
plot_metrics(results, level='class', output='arthropods_dataset_scripts/benchmark/plot_class_sorted.png', sort=True)

# Plotting the results for order level:
plot_metrics(results, level='order', output='arthropods_dataset_scripts/benchmark/plot_order.png')
plot_metrics(results, level='order', output='arthropods_dataset_scripts/benchmark/plot_order_sorted.png', sort=True)

# # Plotting the results
# def plot_metrics(results):
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#     for i, (level, metrics) in enumerate(results.items()):
#         ax = axs[i // 3, i % 3]
#         metrics.plot(kind='bar', x=level, y='F1', ax=ax)
#         ax.set_title(f'F1 score at {level} level')
#         ax.set_ylim(0, 1)
#     plt.tight_layout()
#     plt.savefig("arthropods_dataset_scripts/benchmark/plot.png")

# plot_metrics(results)