import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('ggplot')

from matplotlib.patches import Patch

def plot_bbox_number_metrics(csv_path, output, log_scale=False):
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 18
    # slightly less black text:
    ratio = '0.2'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    df = pd.read_csv(csv_path)
    IoUs_with_zeros = df['IoUs_with_zeros'].map(ast.literal_eval)
    
    # Calculate number of bboxes per image
    num_bboxes = [len(ious) for ious in IoUs_with_zeros]
    
    # Calculate average IoU per image
    avg_IoUs = [sum(ious)/len(ious) if len(ious) > 0 else 0 for ious in IoUs_with_zeros]

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'num_bboxes': num_bboxes,
        'avg_IoUs': avg_IoUs
    })

    # Plot IoU performance depending on the number of bounding boxes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(num_bboxes, avg_IoUs, alpha=0.5, s=20, color='#3b5998')  # Jean blue color
    
    # Calculate Spearman correlation
    X = np.array(num_bboxes)
    y = np.array(avg_IoUs)

    # Filter out any NaN values
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # pingouin approach for Spearman correlation
    import pingouin as pg
    res = pg.corr(x=X, y=y, method='spearman')
    print(res)
    rho = res['r'].iloc[0]
    p_value = res['p-val'].iloc[0]

    # Use seaborn for regression with 95% CI
    sns.regplot(
        x='num_bboxes', 
        y='avg_IoUs', 
        data=plot_df, 
        scatter=False,
        ax=ax,
        line_kws={'color': 'red', 'linewidth': 2.5},
        ci=95,
        color='red'
    )
    
    ax.set_xlabel('Number of bounding boxes per image')
    ax.set_ylabel('Average IoU')

    # Remove borders of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if log_scale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    
    ax.set_ylim(0, 1)

    # Make the plot look nicer
    ax.set_facecolor('#f0f0f0')  # grey background
    ax.grid(which='major', linestyle='-', linewidth='2.2', color='white')  # thicker white grid lines
    ax.set_axisbelow(True)  # grid lines are behind the plot
    ax.tick_params(which='both', direction='in', length=0)  # remove small ticks next to the numbers
    ax.tick_params(axis='both', which='major', pad=10)  # number further from the axis
    
    # Add the red line, CI patch, and correlation text to the legend
    line_patch = plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2.5, label='Regression line')
    ci_patch = Patch(color='red', alpha=0.2, label='95% CI')
    ax.legend([line_patch, ci_patch, plt.Line2D([0], [0], linestyle='none')], 
              ['Regression line', '95% CI', f'Spearman rho = {rho:.2f}\np-value = {p_value:.2f}'],
              loc='upper right')

    plt.tight_layout()
    plt.savefig(output)

# Usage examples
plot_bbox_number_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_conf0.437yolo11n.csv',
                  output = 'arthropods_dataset_scripts/benchmark/other_plots/bbox_number_performance_log_yolo11n.png',
                  log_scale = True)
plot_bbox_number_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_conf0.437yolo11n.csv',
                  output = 'arthropods_dataset_scripts/benchmark/other_plots/bbox_number_performance_yolo11n.png',
                  log_scale = False)
