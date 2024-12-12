import pandas as pd
# import numpy as np
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
    
    return results

def plot_metrics(results, level, output, sort=False, figsize=(15, 15), bar_width=0.8):
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 12
    # slightly less black text:
    ratio = '0.15'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    data = results[level]
    data_max_count = data['count'].max()
    # Use log scale for color bar
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=data_max_count)
    color_palette = plt.cm.viridis
    
    # F1 score
    if sort:
        data = data.sort_values(by='F1', ascending=True)
    color = color_palette(norm(data['count'])) # color bar plot depending on the number of samples
    data.plot(kind='barh', x=level, y='F1', ax=axs[0, 0], color=color, legend=False, width=bar_width)
    axs[0, 0].set_title(f'F1 score at {level} level')
    axs[0, 0].set_xlim(0, 1)
    
    # Precision
    if sort:
        data = data.sort_values(by='precision', ascending=True)
    color = color_palette(norm(data['count']))
    data.plot(kind='barh', x=level, y='precision', ax=axs[0, 1], color=color, legend=False, width=bar_width)
    axs[0, 1].set_title(f'Precision at {level} level')
    axs[0, 1].set_xlim(0, 1)
    
    # Recall
    if sort:
        data = data.sort_values(by='recall', ascending=True)
    color = color_palette(norm(data['count']))
    data.plot(kind='barh', x=level, y='recall', ax=axs[1, 0], color=color, legend=False, width=bar_width)
    axs[1, 0].set_title(f'Recall at {level} level')
    axs[1, 0].set_xlim(0, 1)
    
    # Mean IoU
    if sort:
        data = data.sort_values(by='mean_IoU', ascending=True)
    color = color_palette(norm(data['count']))
    data.plot(kind='barh', x=level, y='mean_IoU', ax=axs[1, 1], color=color, legend=False, width=bar_width)
    axs[1, 1].set_title(f'Mean IoU at {level} level')
    axs[1, 1].set_xlim(0, 1)
    
    # Make the plot look nicer
    for ax in axs.flat:
        ax.tick_params(which='both', direction='in', length=0) # remove small ticks next to the numbers    
        ax.tick_params(axis='both', which='major', pad=6) # number further from the axis
    
    plt.tight_layout()

    # Adjust layout to make room for color bar
    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=color_palette, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Number of images')
    cbar.outline.set_visible(False)

    # Remove the black rectangle around the plot
    for spine in axs[0, 0].spines.values():
        spine.set_visible(False)
    for spine in axs[0, 1].spines.values():
        spine.set_visible(False)
    for spine in axs[1, 0].spines.values():
        spine.set_visible(False)
    for spine in axs[1, 1].spines.values():
        spine.set_visible(False)
    # Add grid to each subplot
    for ax in axs.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)  # Ensure grid is below the bars
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    plt.savefig(output)

# Example usage:
blacklist = {'class': ['Ostracoda', 'Ichthyostraca']}
# results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_FULL_DATASET.csv')
results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_conf0.444yolo8n.csv', blacklist=blacklist)
# results = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/validation_full_0.5_conf0.372yolo11s.csv', blacklist=blacklist)

# # Display results for each taxonomic level
# for level, metrics in results.items():
#     print(f"Metrics for {level}:")
#     print(metrics)

# Plotting the results for class level:
plot_metrics(results, level='class', output='arthropods_dataset_scripts/benchmark/hierarchical_perfs/plot_class_sorted_yolo8n.png', sort=True, figsize=(12, 13), bar_width=0.7)
# Plotting the results for order level:
plot_metrics(results, level='order', output='arthropods_dataset_scripts/benchmark/hierarchical_perfs/plot_order_sorted_yolo8n.png', sort=True, figsize=(15, 22), bar_width=0.8)