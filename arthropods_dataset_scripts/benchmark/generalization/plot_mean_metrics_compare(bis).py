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
    df['line'] = range(len(df)) # add line number to the dataframe
    grouped = df.groupby('line').apply(calculate_metrics).reset_index()
    grouped['count'] = df.groupby('line').size().values
    results['image'] = grouped

    return results

def plot_mean_metrics(list_of_results, labels, output, figsize=(17, 8), bar_width=0.8):
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 12
    # slightly less black text:
    ratio = '0.15'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    fig, axs = plt.subplots(2, 3, figsize=figsize)
    levels = ['class', 'order', 'family', 'genus', 'specie', 'image']
    metrics = ['F1', 'precision', 'recall', 'mean_IoU']
    
    for i, level in enumerate(levels):
        ax = axs[i // 3, i % 3]
        data_list = []
        
        for results in list_of_results:
            data = results[level]
            data_list.append(data[metrics].mean())
        
        means_df = pd.DataFrame(data_list, index=labels)
        colors = plt.get_cmap('tab10').colors
        means_df.T.plot(kind='bar', ax=ax, width=bar_width, color=colors, legend=False)
        ax.set_title(f'Mean metrics at {level} level')
        ax.set_ylim(0.5, 1)
        ax.set_xticklabels(metrics, rotation=0)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Set y-axis ticks at 0.1 increments

        # Remove the black rectangle around the plot
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Make the plot look nicer
    for ax in axs.flat:
        ax.tick_params(which='both', direction='in', length=0) # remove small ticks next to the numbers    
        ax.tick_params(axis='both', which='major', pad=6) # number further from the axis

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for the legend

    # Add legend of models:
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors[:len(labels)]]
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 1.1))

    plt.savefig(output)

# Example usage:
blacklist = {}

same_genus = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/generalization/same_genus_conf0.444yolo8n.csv', blacklist=blacklist)
next_genus = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/generalization/next_genus_conf0.444yolo8n.csv', blacklist=blacklist)
south_american_nowater = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/generalization/south_american_nowater_conf0.444yolo8n.csv', blacklist=blacklist)
south_american_wateronly = hierarchical_benchmark('arthropods_dataset_scripts/benchmark/generalization/south_american_wateronly_conf0.444yolo8n.csv', blacklist=blacklist)

list_of_results = [same_genus, next_genus, south_american_nowater, south_american_wateronly]
labels = ['same Genus', 'next Genus', 'South American\n(no water)', 'South American\n(water only)']
plot_mean_metrics(list_of_results, labels, '/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/mean_metrics_compare_ generalization.png')

