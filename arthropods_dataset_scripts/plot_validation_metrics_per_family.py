import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(x, y, poly_degree=4):
    z = np.polyfit(x, y, poly_degree)
    return np.polyval(z, x)

def plot_metrics(csv_file, smoothing_degree=4):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create a 2 by 2 subplot grid
    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(16)
    fig.set_figheight(10)
    ymin = 0.0
    ymax = 1.0
    xmin = min(df['wave'].tolist())
    xmax = max(df['wave'].tolist())

    # Plot precision
    axs[0, 0].set_xlabel('Wave Number')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision Evolution')
    axs[0, 0].set_ylim(ymin, ymax)
    axs[0, 0].set_xlim(xmin, xmax)
    # Plot recall
    axs[0, 1].set_xlabel('Wave Number')
    axs[0, 1].set_ylabel('Recall')
    axs[0, 1].set_title('Recall Evolution')
    axs[0, 1].set_ylim(ymin, ymax)
    axs[0, 1].set_xlim(xmin, xmax)
    # Plot Mean IoU
    axs[1, 0].set_xlabel('Wave Number')
    axs[1, 0].set_ylabel('IoU')
    axs[1, 0].set_title('Mean IoU Evolution')
    axs[1, 0].set_ylim(ymin, ymax)
    axs[1, 0].set_xlim(xmin, xmax)
    # Plot F1-score (test)
    axs[1, 1].set_xlabel('Wave Number')
    axs[1, 1].set_ylabel('F1-score')
    axs[1, 1].set_title('F1-score(test) Evolution')
    axs[1, 1].set_ylim(ymin, ymax)
    axs[1, 1].set_xlim(xmin, xmax)

    subtypes = df['subtype'].unique()

    for subtype in subtypes:
        # print(f"Plotting metrics for subtype: {subtype}")

        subtype_df = df[df['subtype'] == subtype]
        wave_numbers = subtype_df['wave'].tolist()
        precision_values = subtype_df['metrics/precision(B)'].tolist()
        recall_values = subtype_df['metrics/recall(B)'].tolist()
        mean_IoU_values = subtype_df['mean_IoU'].tolist()
        test_f1_score_values = subtype_df['F1-score'].tolist()

        # Plot precision
        # axs[0, 0].plot(wave_numbers, precision_values, label=f'Original ({subtype})')
        axs[0, 0].plot(wave_numbers, smooth_curve(wave_numbers, precision_values, smoothing_degree),
                       label=f'Smoothed ({subtype}, deg={smoothing_degree})')

        # Plot recall
        # axs[0, 1].plot(wave_numbers, recall_values, label=f'Original ({subtype})')
        axs[0, 1].plot(wave_numbers, smooth_curve(wave_numbers, recall_values, smoothing_degree),
                       label=f'Smoothed ({subtype}, deg={smoothing_degree})')
        
        # Plot Mean IoU
        # axs[1, 0].plot(wave_numbers, mean_IoU_values, label=f'Original ({subtype})')
        axs[1, 0].plot(wave_numbers, smooth_curve(wave_numbers, mean_IoU_values, smoothing_degree),
                       label=f'Smoothed ({subtype}, deg={smoothing_degree})')

        # Plot F1-score (test)
        # axs[1, 1].plot(wave_numbers, test_f1_score_values, label=f'Original ({subtype})')
        axs[1, 1].plot(wave_numbers, smooth_curve(wave_numbers, test_f1_score_values, smoothing_degree),
                       label=f'Smoothed ({subtype}, deg={smoothing_degree})')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example:
csv_file = "arthropods_dataset_scripts/test_metrics_test17_families.csv"
plot_metrics(csv_file, smoothing_degree=4)
