import pandas as pd
import numpy as np
import ast
import matplotlib
# matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

def plot_bbox_metrics(csv_path, output, log_scale=False):
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
    fig, ax = plt.subplots()
    ax.scatter(IoUs_with_zeros, bbox_relative_sizes, alpha=0.5)
    ax.set_xlabel('Bounding box relative size')
    ax.set_ylabel('IoU')
    # ax.set_title('IoU performance depending on the size of the bounding box')

    if log_scale:
        ax.set_xscale('log')
        # replace 10^2 with 1%, 10^1 with 10%, 10^0 with 100%
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(['1%', '10%', '100%'])
    else:
        ax.set_xscale('linear')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Make the plot look nicer
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Make regression line using scipy
    X = np.array(bbox_relative_sizes)
    y = np.array(IoUs_with_zeros)

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    print(f'slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}')
    y_pred = intercept + slope * X
    ax.plot(X, y_pred, color='red', linewidth=2)

    plt.savefig(output)

# Example usage
plot_bbox_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_FULL_DATASET.csv',
                  output = 'arthropods_dataset_scripts/benchmark/bbox_size_performance_log.png',
                  log_scale = True)
plot_bbox_metrics(csv_path = 'arthropods_dataset_scripts/benchmark/validation_FULL_DATASET.csv',
                  output = 'arthropods_dataset_scripts/benchmark/bbox_size_performance.png',
                  log_scale = False)

