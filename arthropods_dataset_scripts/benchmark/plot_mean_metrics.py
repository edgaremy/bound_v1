import pandas as pd
import ast
import matplotlib
# matplotlib.use('agg')  # Use non-interactive backend
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

def hierarchical_benchmark(csv_path, blacklist=None):
    df = pd.read_csv(csv_path)
    
    # Remove blacklisted entries for all levels
    if blacklist is not None:
        for level, items in blacklist.items():
            df = df[~df[level].isin(items)]

    results = {}
    for level in ['class', 'order', 'family', 'genus', 'specie']:
        grouped = df.groupby(level).apply(calculate_metrics).reset_index()
        grouped['count'] = df.groupby(level).size().values
        results[level] = grouped
    
    # add image level to results:
    image_level = calculate_metrics(df)
    image_level['count'] = len(df)
    results['image'] = image_level

    return results

def plot_mean_metrics(results, output, figsize=(15, 11), bar_width=0.8):
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    levels = ['class', 'order', 'family', 'genus', 'specie', 'image']
    metrics = ['F1', 'precision', 'recall', 'mean_IoU']
    
    for i, level in enumerate(levels):
        ax = axs[i // 3, i % 3]
        if level == 'image':
            data = results[level].to_frame().T
        else:
            data = results[level]
        
        means = data[metrics].mean()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        means.plot(kind='bar', ax=ax, width=bar_width, color=colors)
        ax.set_title(f'Mean metrics at {level} level')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(means.index, rotation=45)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_yticks([i/10 for i in range(11)])  # Set y-axis ticks at 0.1 increments

        # Remove the black rectangle around the plot
        for ax in axs.flat:
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output)

# Example usage:
blacklist = {'class': ['Ostracoda', 'Ichthyostraca']}

results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_full_conf0.444yolo8n.csv', blacklist=blacklist)
plot_mean_metrics(results, 'arthropods_dataset_scripts/benchmark/mean_metrics/mean_metrics_full_conf0.444yolo8n.png')

results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_full_conf0.337yolo11m.csv', blacklist=blacklist)
plot_mean_metrics(results, 'arthropods_dataset_scripts/benchmark/mean_metrics/mean_metrics_full_conf0.337yolo11m.png')

# Display mean results for each taxonomic level
# for level, metrics in results.items():
#     if level == 'image':
#         continue
#     print(f'Level: {level}')
#     print(metrics[['F1', 'precision', 'recall', 'mean_IoU']].mean().to_string())
#     print()
# print("Level: image")
# print(results["image"][['F1', 'precision', 'recall', 'mean_IoU']].to_string())

