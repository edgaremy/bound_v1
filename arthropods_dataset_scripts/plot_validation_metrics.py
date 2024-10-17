import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def smooth_curve(x, y, poly_degree=4):
    z = np.polyfit(x, y, poly_degree)
    return np.polyval(z, x)

def plot_metrics(csv_file, smoothing_degree=4):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract the wave numbers and metric values
    wave_numbers = df['wave'].tolist()
    precision_values = df['metrics/precision(B)'].tolist()
    recall_values = df['metrics/recall(B)'].tolist()
    # mAP50_values = df['metrics/mAP50(B)'].tolist()
    # mAP5_values = df['metrics/mAP50-95(B)'].tolist()
    # fitness_values = df['fitness'].tolist()
    mean_IoU_values = df['mean_IoU'].tolist()
    test_f1_score_values = df['F1-score'].tolist()

    # Create a 2 by 3 subplot grid
    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(16)
    fig.set_figheight(10)
    ymin = 0.7
    ymax = 1.0
    xmin = min(df['wave'].tolist())
    xmax = max(df['wave'].tolist())
 
    # Plot precision
    axs[0, 0].plot(wave_numbers, precision_values, label='Original')
    axs[0, 0].plot(wave_numbers, smooth_curve(wave_numbers, precision_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[0, 0].set_xlabel('Wave Number')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision Evolution')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(ymin, ymax)
    axs[0, 0].set_xlim(xmin, xmax)

    # Plot recall
    axs[0, 1].plot(wave_numbers, recall_values, label='Original')
    axs[0, 1].plot(wave_numbers, smooth_curve(wave_numbers, recall_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[0, 1].set_xlabel('Wave Number')
    axs[0, 1].set_ylabel('Recall')
    axs[0, 1].set_title('Recall Evolution')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(ymin, ymax)
    axs[0, 1].set_xlim(xmin, xmax)

    # Plot Mean IoU
    axs[1, 0].plot(wave_numbers, mean_IoU_values, label='Original')
    axs[1, 0].plot(wave_numbers, smooth_curve(wave_numbers, mean_IoU_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[1, 0].set_xlabel('Wave Number')
    axs[1, 0].set_ylabel('IoU')
    axs[1, 0].set_title('Mean IoU Evolution')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(ymin, ymax)
    axs[1, 0].set_xlim(xmin, xmax)

    # Plot F1-score (test)
    axs[1, 1].plot(wave_numbers, test_f1_score_values, label='Original')
    axs[1, 1].plot(wave_numbers, smooth_curve(wave_numbers, test_f1_score_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[1, 1].set_xlabel('Wave Number')
    axs[1, 1].set_ylabel('F1-score')
    axs[1, 1].set_title('F1-score(test) Evolution')
    axs[1, 1].legend()
    axs[1, 1].set_ylim(ymin, ymax)
    axs[1, 1].set_xlim(xmin, xmax)

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example:
csv_file = "arthropods_dataset_scripts/test_metrics_test17.csv"
plot_metrics(csv_file, smoothing_degree=4)
