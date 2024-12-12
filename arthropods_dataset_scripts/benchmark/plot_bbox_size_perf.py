import pandas as pd
import numpy as np
import ast
import matplotlib
# matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.style.use('ggplot')

def plot_bbox_metrics(csv_path, output, log_scale=False):
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
    bbox_relative_sizes = df['bbox_sizes'].map(ast.literal_eval)

    # Flatten the lists
    IoUs_with_zeros = [item for sublist in IoUs_with_zeros for item in sublist]
    bbox_relative_sizes = [item for sublist in bbox_relative_sizes for item in sublist]

    # Sort based on the size of the bounding box
    sorted_indices = np.argsort(bbox_relative_sizes)
    IoUs_with_zeros = np.array(IoUs_with_zeros)[sorted_indices]
    bbox_relative_sizes = np.array(bbox_relative_sizes)[sorted_indices]

    # Plot IoU performance depending on the size of the bounding box
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(IoUs_with_zeros, bbox_relative_sizes, alpha=0.5, s=20, color='#3b5998')  # Jean blue color
    ax.set_xlabel('Bounding box relative size')
    ax.set_ylabel('IoU')
    # ax.set_title('IoU performance depending on the size of the bounding box')

    # Remove borders of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if log_scale:
        ax.set_xscale('log')
        # replace 10^2 with 1%, 10^1 with 10%, 10^0 with 100%
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(['1%', '10%', '100%'])
    else:
        ax.set_xscale('linear')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Make the plot look nicer
    ax.set_facecolor('#f0f0f0') # grey background
    ax.grid(which='major', linestyle='-', linewidth='2.2', color='white') # thicker white grid lines
    ax.set_axisbelow(True) # grid lines are behind the plot
    ax.tick_params(which='both', direction='in', length=0) # remove small ticks next to the numbers    
    ax.tick_params(axis='both', which='major', pad=10) # number further from the axis

    plt.tight_layout()

    # Calculate and plot Spearman correlation
    X = np.array(bbox_relative_sizes)
    y = np.array(IoUs_with_zeros)

    # Bootstrap estimation for confidence intervals
    n_bootstraps = 1000
    bootstrapped_lines = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(X)), len(X), replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        slope_sample, intercept_sample, _, _, _ = stats.linregress(X_sample, y_sample)
        bootstrapped_lines.append(slope_sample * X + intercept_sample)

    # Calculate 95% confidence interval for the regression lines
    lower_bound = np.percentile(bootstrapped_lines, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_lines, 97.5, axis=0)

    # Calculate and print Spearman correlation
    rho, p_value = stats.spearmanr(X, y)
    print(f'Spearman correlation coefficient: {rho}, p_value: {p_value}')

    # Plot the regression line
    slope, intercept, _, _, _ = stats.linregress(X, y)
    ax.plot(X, slope * X + intercept, color='red', linewidth=2.5, label=f'Spearman rho = {rho:.2f}')

    # Add confidence interval to the plot
    ax.fill_between(X, lower_bound, upper_bound, color='red', alpha=0.2, label='95% CI')

    ax.legend()

    plt.savefig(output)

# Example usage
plot_bbox_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_conf0.444yolo8n.csv',
                  output = 'arthropods_dataset_scripts/benchmark/other_plots/bbox_size_performance_log.png',
                  log_scale = True)
plot_bbox_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_conf0.444yolo8n.csv',
                  output = 'arthropods_dataset_scripts/benchmark/other_plots/bbox_size_performance.png',
                  log_scale = False)
