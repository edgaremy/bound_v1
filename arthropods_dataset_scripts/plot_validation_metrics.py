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
    mAP50_values = df['metrics/mAP50(B)'].tolist()
    mAP5_values = df['metrics/mAP50-95(B)'].tolist()
    # fitness_values = df['fitness'].tolist()
    mean_IoU_values = df['mean_IoU'].tolist()
    # f1_score_values = df['metrics/F1-score(B)'].tolist()
    # val_f1_score_values = [0.72, 0.78, 0.78, 0.82, 0.81, 0.81, 0.83, 0.75, 0.77, 0.78, 0.79, 0.80, 0.81, 0.81]
    test_f1_score_values = [0.79, 0.80, 0.82, 0.82, 0.83, 0.84, 0.85, 0.84, 0.86, 0.85, 0.86, 0.86, 0.87, 0.87]

    # Create a 2 by 3 subplot grid
    fig, axs = plt.subplots(2, 3)
    fig.set_figwidth(16)
    fig.set_figheight(10)

   # Plot precision
    axs[0, 0].plot(wave_numbers, precision_values, label='Original')
    axs[0, 0].plot(wave_numbers, smooth_curve(wave_numbers, precision_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[0, 0].set_xlabel('Wave Number')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision Evolution')
    axs[0, 0].legend()

    # Plot recall
    axs[0, 1].plot(wave_numbers, recall_values, label='Original')
    axs[0, 1].plot(wave_numbers, smooth_curve(wave_numbers, recall_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[0, 1].set_xlabel('Wave Number')
    axs[0, 1].set_ylabel('Recall')
    axs[0, 1].set_title('Recall Evolution')
    axs[0, 1].legend()

    # Plot mAP50
    axs[1, 0].plot(wave_numbers, mAP50_values, label='Original')
    axs[1, 0].plot(wave_numbers, smooth_curve(wave_numbers, mAP50_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[1, 0].set_xlabel('Wave Number')
    axs[1, 0].set_ylabel('mAP50')
    axs[1, 0].set_title('mAP50 Evolution')
    axs[1, 0].legend()

    # Plot mAP50-95
    axs[1, 1].plot(wave_numbers, mAP5_values, label='Original')
    axs[1, 1].plot(wave_numbers, smooth_curve(wave_numbers, mAP5_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[1, 1].set_xlabel('Wave Number')
    axs[1, 1].set_ylabel('mAP50-95')
    axs[1, 1].set_title('mAP50-95 Evolution')
    axs[1, 1].legend()

    # Plot Mean IoU
    axs[0, 2].plot(wave_numbers, mean_IoU_values, label='Original')
    axs[0, 2].plot(wave_numbers, smooth_curve(wave_numbers, mean_IoU_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[0, 2].set_xlabel('Wave Number')
    axs[0, 2].set_ylabel('IoU')
    axs[0, 2].set_title('Mean IoU Evolution')
    axs[0, 2].legend()

    # Plot F1-score (test)
    axs[1, 2].plot(wave_numbers, test_f1_score_values, label='Original')
    axs[1, 2].plot(wave_numbers, smooth_curve(wave_numbers, test_f1_score_values, smoothing_degree),
                   label=f'Smoothed (deg={smoothing_degree})')
    axs[1, 2].set_xlabel('Wave Number')
    axs[1, 2].set_ylabel('F1-score')
    axs[1, 2].set_title('F1-score(test) Evolution')
    axs[1, 2].legend()

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example:
csv_file = "arthropods_dataset_scripts/test_metrics_test15.csv"
plot_metrics(csv_file, smoothing_degree=4)
